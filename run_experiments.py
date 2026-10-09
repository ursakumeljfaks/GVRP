"""Run both methods on every instance and seed, and save everything.

Written to results/:
    runs.csv           one row per Pareto point      -> analyse_results.ipynb
    routes.csv         route decomposition of every saved Pareto solution
    ws_weights.csv     weighted sum, one row per weight, before Pareto filtering
    convergence.csv    front size and best f1 / f2 over time
    validation.csv     feasibility check of every saved solution
    run_metadata.json  seeds, budgets, parameters, versions, machine

Set PILOT = False for a quick shape check, then False for the real run.
Resuming is on by default: finished (instance, algorithm, seed) triples are
read back from runs.csv and skipped.
"""

import csv
import json
import os
import platform
import time

import numpy as np

# ---------------------------------------------------------------
# WHAT TO RUN
# ---------------------------------------------------------------
INSTANCES = ["X-n106-k14", "X-n101-k25", "X-n110-k13", "X-n115-k10", "X-n200-k36", "X-n313-k71", "X-n143-k7", "X-n172-k51", "X-n237-k14", "X-n261-k13"]#["X-n106-k14", "X-n101-k25", "X-n110-k13", "X-n115-k10"] #["X-n115-k10"]#["X-n106-k14", "X-n101-k25", "X-n110-k13", "X-n115-k10"]

PILOT = False

if PILOT:
    SEEDS, BUDGET_SECONDS, N_WEIGHTS = [1, 2], 20, 40
else:
    SEEDS, BUDGET_SECONDS, N_WEIGHTS = list(range(1, 21)), 120, 40

# FINAL configuration from the ablation study (notebook 10.11).

# The intensification callbacks in gvrp.nsga2 (corner intensification, elite
# polishing, extreme preservation). The Pareto archive is always kept.
USE_INTENSIFICATION = False

# Solve the two single-objective corners with PyVRP and seed them into the
# initial population.
USE_ANCHORS = True

# Normalization: scale distance / CO2 by the front spans (first from the
# anchors, then RecalibrateScales). False: scales stay at 1.0.
CALIBRATE_SCALES = False

OUT_DIR = "results"
RESUME = True

os.makedirs(OUT_DIR, exist_ok=True)
RUNS_CSV = f"{OUT_DIR}/runs.csv"
ROUTES_CSV = f"{OUT_DIR}/routes.csv"
WEIGHTS_CSV = f"{OUT_DIR}/ws_weights.csv"
TRACE_CSV = f"{OUT_DIR}/convergence.csv"
VALID_CSV = f"{OUT_DIR}/validation.csv"
META_JSON = f"{OUT_DIR}/run_metadata.json"

# ---------------------------------------------------------------
# config must be set BEFORE the solver modules are imported: both of them
# bind values from it at import time.
# ---------------------------------------------------------------
from gvrp import config as C, nsga2 as G

C.N_WEIGHTS = N_WEIGHTS
C.BUDGET_SECONDS = BUDGET_SECONDS

from gvrp import emissions, weighted_sum as W

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.termination import Termination
from pymoo.optimize import minimize
from pyvrp import Model, read


# ---------------------------------------------------------------
# INSTANCE SETUP
# ---------------------------------------------------------------

def instance_path(name):
    return f"{C.INSTANCE_DIR}/{name}.vrp"


def setup(name):
    """Load an instance and derive its fleet. Returns (data, capacities, fleet)."""
    data = read(instance_path(name), round_func="round")
    base_cap = int(Model.from_data(data).vehicle_types[0].capacity[0])

    Q_by_type = C.derived_capacities(base_cap)
    fleet = C.derived_fleet_size(C.customers_in(instance_path(name)))

    # weighted_sum reads the fleet from config at call time.
    C.N_DIESEL, C.N_CLEAN = fleet["diesel"], fleet["clean"]
    return data, Q_by_type, fleet


def nondominated_idx(points):
    """Indices of the distinct, non-dominated points."""
    P = np.asarray(points, float)
    keep, seen = [], set()

    for i in range(len(P)):
        dom = np.all(P <= P[i], axis=1) & np.any(P < P[i], axis=1)
        dom[i] = False
        key = (P[i, 0], P[i, 1])
        if not dom.any() and key not in seen:
            seen.add(key)
            keep.append(i)

    return keep


# ---------------------------------------------------------------
# ROUTE-LEVEL METRICS AND VALIDATION
# ---------------------------------------------------------------

def route_metrics(visits, type_name, data, Q_by_type):
    """Load, capacity, distance and true CO2 of one route."""
    D = data.distance_matrix(0)
    clients = data.clients()

    load = sum(emissions.demand_at_visit_id(i, clients) for i in visits)
    seq = [0] + list(visits) + [0]
    dist = sum(float(D[a][b]) for a, b in zip(seq[:-1], seq[1:]))

    vt_idx = 0 if type_name == "diesel" else 1
    co2 = emissions.true_co2([(list(visits), vt_idx)], data,
                             {vt_idx: type_name}, Q_by_type,
                             C.F0, C.ALPHA, C.BETA, C.GAMMA, C.MULT)

    return dict(load=load, capacity=Q_by_type[type_name], distance=dist, co2=co2)


def solution_routes(sol):
    """[(visits, 'diesel'|'clean')] from a PyVRP Solution."""
    return [(list(r.visits()), C.TYPE_LABELS[int(r.vehicle_type())])
            for r in sol.routes() if list(r.visits())]


def validate(routes, data, Q_by_type, fleet):
    """Feasibility of a whole solution. Returns (ok, reason)."""
    n = len(data.clients())
    visited = sorted(c for r, _ in routes for c in r)
    if visited != list(range(1, n + 1)):
        return False, "customers not visited exactly once"

    types = [t for _, t in routes]
    if types.count("diesel") > fleet["diesel"]:
        return False, "too many diesel routes"
    if types.count("clean") > fleet["clean"]:
        return False, "too many clean routes"

    clients = data.clients()
    for visits, t in routes:
        load = sum(emissions.demand_at_visit_id(i, clients) for i in visits)
        if load > Q_by_type[t]:
            return False, f"{t} route load {load} > capacity {Q_by_type[t]}"

    return True, ""


# ---------------------------------------------------------------
# WEIGHTED SUM
# ---------------------------------------------------------------

def run_weighted_sum(name, seed, per_weight_rows, trace_rows):
    data, Q_by_type, fleet = setup(name)
    C.SEED = seed                       # solve_weighted_surrogate reads C.SEED

    per_weight = BUDGET_SECONDS / N_WEIGHTS
    results = []
    t0 = time.perf_counter()

    for j, (w_dist, w_co2) in enumerate(W.weights):
        if time.perf_counter() - t0 >= BUDGET_SECONDS:
            break

        tw = time.perf_counter()
        sol = W.solve_weighted_surrogate(data, w_dist, w_co2,
                                         Q_by_type["diesel"], Q_by_type["clean"],
                                         per_weight)
        f1 = float(sol.distance())
        f2 = W.true_co2_of_solution(sol, data, Q_by_type)
        results.append((f1, f2, solution_routes(sol)))

        # Every weight, before any Pareto filtering.
        per_weight_rows.append(dict(
            instance=name, seed=seed, weight_index=j,
            w_dist=round(w_dist, 6), w_co2=round(w_co2, 6),
            runtime_s=round(time.perf_counter() - tw, 3),
            f1=round(f1), f2=round(f2, 4)))
        trace_rows.append(dict(
            instance=name, algorithm="Weighted-sum", seed=seed,
            elapsed_s=round(time.perf_counter() - t0, 2),
            n_points=len(results),
            best_f1=round(min(r[0] for r in results)),
            best_f2=round(min(r[1] for r in results), 4)))

    return results, time.perf_counter() - t0, data, Q_by_type, fleet


# ---------------------------------------------------------------
# MEMETIC NSGA-II
# ---------------------------------------------------------------

class Trace(Callback):
    """One convergence snapshot per generation."""

    def __init__(self, rows, name, seed, t0):
        super().__init__()
        self.rows, self.name, self.seed, self.t0 = rows, name, seed, t0

    def notify(self, algorithm):
        F = algorithm.pop.get("F")
        if F is None or not len(F):
            return
        self.rows.append(dict(
            instance=self.name, algorithm="NSGA-II", seed=self.seed,
            elapsed_s=round(time.perf_counter() - self.t0, 2),
            n_points=int(len(F)),
            best_f1=round(float(F[:, 0].min())),
            best_f2=round(float(F[:, 1].min()), 4)))


class WallClock(Termination):
    """Stop so the WHOLE run fits the budget.

    pymoo's ("time", ...) clock starts inside minimize() and can only stop on a
    generation boundary, so setup and sampling fall outside the budget and the
    last generation overruns it. This measures from the true start of the run
    and stops once one more generation would cross the deadline.
    """

    def __init__(self, t_start, budget):
        super().__init__()
        self.t_start = t_start
        self.budget = budget
        self.deadline = t_start + budget
        self._last = None
        self._gen_times = []

    def _update(self, algorithm):
        now = time.perf_counter()
        if self._last is not None:
            self._gen_times.append(now - self._last)
        self._last = now

        next_gen = float(np.median(self._gen_times)) if self._gen_times else 0.0
        if now + next_gen >= self.deadline:
            return 1.0
        return min(0.999, (now - self.t_start) / self.budget)


def run_nsga2(name, seed, per_weight_rows, trace_rows):
    data, Q_by_type, fleet = setup(name)
    Kd, Kc = fleet["diesel"], fleet["clean"]

    t0 = time.perf_counter()

    dist_data, co2_data = G.build_problem_datas(data, Kd, Kc, Q_by_type)
    dls = G.DualLocalSearch(dist_data, co2_data, Q_by_type, Kd, Kc,
                            seed=seed, nb_size=G.LS_NEIGHBOURHOOD_SIZE,
                            time_budget=float(BUDGET_SECONDS))
    problem = G.GreenVRPProblem(dist_data, Q_by_type, Kd, Kc)

    # Solve both single-objective corners with PyVRP's full solver (4s of the
    # 120s budget). 
    anchor_d = anchor_c = None
    # Seed numpy before the initial population is built: minimize() only seeds
    # it later, so without this line the initial population was not
    # reproducible from the seed.
    np.random.seed(seed)
    if USE_ANCHORS:
        anchor_d, anchor_c, m_d, m_c = G.calibrate_from_anchors(dls, problem, seed=seed)
        if not CALIBRATE_SCALES:
            # Anchors kept, normalization off.
            dls.dist_scale = dls.co2_scale = 1.0
        if m_d is not None:
            print(f"    anchor_d: f1={m_d[0]:.0f} f2={m_d[1]:.0f} pen={m_d[2]:.1f}")
            print(f"    anchor_c: f1={m_c[0]:.0f} f2={m_c[1]:.0f} pen={m_c[2]:.1f}")

    init_X = G.VRPSampling(dls)._do(problem, C.POP_SIZE)

    # Seed the corners.
    if anchor_d is not None:
        init_X[0, 0], init_X[0, 1] = anchor_d, 1.0
        init_X[1, 0], init_X[1, 1] = anchor_c, 0.0
        _m = G.evaluate_solution(init_X[0, 0], dist_data, Q_by_type, Kd, Kc)
        print(f"    init_X[0]: f1={_m[0]:.0f} pen={_m[2]:.1f}")

    algorithm = NSGA2(
        pop_size=C.POP_SIZE,
        sampling=init_X,
        crossover=G.VRPCrossover(dls, prob=G.CX_PROB),
        mutation=G.VRPMutation(dls, rate=G.MUT_RATE),
        eliminate_duplicates=G.VRPDuplicateElimination(),
    )

    trace = Trace(trace_rows, name, seed, t0)
    extras = [trace]
    if CALIBRATE_SCALES:
        extras.append(G.RecalibrateScales(dls))

    if USE_INTENSIFICATION:
        callback, archive, _ = G.make_callbacks(dls, problem, extra=extras)
    else:
        # The archive is kept without intensification too, as in the ablation.
        archive = G.ParetoArchive()
        callback = G.FanOut([archive] + extras)

    res = minimize(problem, algorithm, WallClock(t0, BUDGET_SECONDS),
                   seed=seed, verbose=False, copy_algorithm=False,
                   callback=callback)
    elapsed = time.perf_counter() - t0

    results = []
    for row in np.atleast_2d(res.X):
        m = G.evaluate_solution(row[0], dist_data, Q_by_type, Kd, Kc)
        if m is not None and m[2] == 0.0:            # feasible only
            results.append((float(m[0]), float(m[1]), solution_routes(row[0])))

    if archive is not None:
        # Points found earlier and later crowded out. They have no routes
        # attached, so they are reported as objective values only.
        known = {(f1, f2) for f1, f2, _ in results}
        results += [(f1, f2, []) for f1, f2 in archive.front()
                    if (f1, f2) not in known]

    return results, elapsed, data, Q_by_type, fleet


# ---------------------------------------------------------------
# OUTPUT FILES
# ---------------------------------------------------------------

RUNNERS = {"NSGA-II": run_nsga2, "Weighted-sum": run_weighted_sum}

RUNS_COLS = ["instance", "algorithm", "seed", "runtime_s", "f1", "f2",
             "solution_id"]
ROUTES_COLS = ["instance", "algorithm", "seed", "solution_id", "route_index",
               "vehicle_type", "n_diesel", "n_clean", "route_load",
               "route_capacity", "route_distance", "route_co2", "sequence"]
WEIGHTS_COLS = ["instance", "seed", "weight_index", "w_dist", "w_co2",
                "runtime_s", "f1", "f2"]
TRACE_COLS = ["instance", "algorithm", "seed", "elapsed_s", "n_points",
              "best_f1", "best_f2"]
VALID_COLS = ["instance", "algorithm", "seed", "solution_id", "feasible",
              "reason"]


def already_done():
    """(instance, algorithm, seed) triples already in runs.csv."""
    if not (RESUME and os.path.exists(RUNS_CSV)):
        return set()
    with open(RUNS_CSV) as fh:
        return {(r["instance"], r["algorithm"], int(r["seed"]))
                for r in csv.DictReader(fh)}


def writer(path, cols):
    """Append-mode CSV writer, with a header if the file is new."""
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    fh = open(path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=cols)
    if new:
        w.writeheader()
    return fh, w


def save_metadata():
    from importlib.metadata import version

    def _ver(pkg):
        try:
            return version(pkg)
        except Exception:
            return "unknown"

    params = {k: v for k, v in vars(C).items()
              if k.isupper() and isinstance(v, (int, float, str, list, dict))}

    meta = dict(
        instances=INSTANCES, seeds=SEEDS,
        budget_seconds=BUDGET_SECONDS, n_weights=N_WEIGHTS,
        pop_size=C.POP_SIZE, use_intensification=USE_INTENSIFICATION,
        use_anchors=USE_ANCHORS, calibrate_scales=CALIBRATE_SCALES,
        nsga2=dict(mut_rate=G.MUT_RATE, split_prob=G.SPLIT_PROB,
                   use_both_engines=G.USE_BOTH_ENGINES),
        config=params,
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      pyvrp=_ver("pyvrp"), pymoo=_ver("pymoo"),
                      numba=_ver("numba")),
        machine=dict(system=platform.system(), release=platform.release(),
                     machine=platform.machine(), processor=platform.processor()),
        command=f"python {os.path.basename(__file__)}",
    )
    with open(META_JSON, "w") as fh:
        json.dump(meta, fh, indent=2)


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

def main():
    # Compile the numba kernels before anything is timed.
    G.warm_up_split()

    done = already_done()
    n_runs = len(INSTANCES) * len(RUNNERS) * len(SEEDS)
    print(f"{n_runs} runs x {BUDGET_SECONDS}s "
          f"= {n_runs * BUDGET_SECONDS / 60:.0f} min of solving")
    if done:
        print(f"resuming: {len(done)} runs already recorded")

    save_metadata()

    handles = [writer(RUNS_CSV, RUNS_COLS), writer(ROUTES_CSV, ROUTES_COLS),
               writer(WEIGHTS_CSV, WEIGHTS_COLS), writer(TRACE_CSV, TRACE_COLS),
               writer(VALID_CSV, VALID_COLS)]
    (fh_runs, w_runs), (fh_rt, w_rt), (fh_wt, w_wt), (fh_tr, w_tr), (fh_vl, w_vl) = handles
    files = [fh for fh, _ in handles]

    try:
        for name in INSTANCES:
            for algo, runner in RUNNERS.items():
                for seed in SEEDS:
                    if (name, algo, seed) in done:
                        continue

                    weight_rows, trace_rows = [], []
                    results, elapsed, data, Q_by_type, fleet = runner(
                        name, seed, weight_rows, trace_rows)

                    keep = nondominated_idx([(f1, f2) for f1, f2, _ in results])
                    n_bad = 0

                    for sid, i in enumerate(keep, start=1):
                        f1, f2, routes = results[i]
                        sol_id = f"{name}_{algo}_s{seed}_p{sid:03d}"

                        ok, reason = validate(routes, data, Q_by_type, fleet) \
                            if routes else (True, "no routes recorded")
                        n_bad += not ok
                        w_vl.writerow(dict(
                            instance=name, algorithm=algo, seed=seed,
                            solution_id=sol_id, feasible=int(ok), reason=reason))

                        w_runs.writerow(dict(
                            instance=name, algorithm=algo, seed=seed,
                            runtime_s=round(elapsed, 1),
                            f1=f"{f1:.0f}", f2=f"{f2:.4f}", solution_id=sol_id))

                        types = [t for _, t in routes]
                        for k, (visits, t) in enumerate(routes):
                            m = route_metrics(visits, t, data, Q_by_type)
                            w_rt.writerow(dict(
                                instance=name, algorithm=algo, seed=seed,
                                solution_id=sol_id, route_index=k, vehicle_type=t,
                                n_diesel=types.count("diesel"),
                                n_clean=types.count("clean"),
                                route_load=m["load"], route_capacity=m["capacity"],
                                route_distance=f"{m['distance']:.0f}",
                                route_co2=f"{m['co2']:.4f}",
                                sequence=" ".join(map(str, visits))))

                    for r in weight_rows:
                        w_wt.writerow(r)
                    for r in trace_rows:
                        w_tr.writerow(r)
                    for fh in files:
                        fh.flush()

                    flag = f"  {n_bad} INFEASIBLE" if n_bad else ""
                    print(f"  {name:<12} {algo:<13} seed {seed:>2}  "
                          f"{elapsed:5.1f}s  |P|={len(keep):>3}  "
                          f"best f1={min(results[i][0] for i in keep):.0f}  "
                          f"best f2={min(results[i][1] for i in keep):.1f}{flag}")
    finally:
        for fh in files:
            fh.close()

    print(f"\nwrote {OUT_DIR}/: runs.csv routes.csv ws_weights.csv "
          f"convergence.csv validation.csv run_metadata.json")


if __name__ == "__main__":
    main()
