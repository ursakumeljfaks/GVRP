
from __future__ import annotations

import time

import numpy as np

from pymoo.core.callback import Callback
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling

from pyvrp import (
    CostEvaluator,
    ProblemData,
    RandomNumberGenerator,
    Route,
    Solution,
    VehicleType,
)

try:
    from . import config as C
    from .emissions import true_co2
except ImportError:
    import config as C
    from emissions import true_co2


# ============================================================
# OPTIONAL DEPENDENCIES
# ============================================================

# PyVRP's local search. Without it the GA still runs, but far worse.
HAS_SEARCH = False
_SEARCH_ERROR = None
try:
    from pyvrp.search import LocalSearch, NeighbourhoodParams, compute_neighbours

    try:
        from pyvrp.search import NODE_OPERATORS, ROUTE_OPERATORS
    except ImportError:  # pyvrp < 0.10 exposes the operators individually
        from pyvrp.search import (
            Exchange10, Exchange11, Exchange20, Exchange21, Exchange22,
            SwapStar, SwapTails, TwoOpt,
        )
        NODE_OPERATORS = [Exchange10, Exchange11, Exchange20,
                          Exchange21, Exchange22, TwoOpt]
        ROUTE_OPERATORS = [SwapStar, SwapTails]

    HAS_SEARCH = True
except Exception as exc:  # pragma: no cover
    _SEARCH_ERROR = exc

# Split and the intra-route search are the hot inner loops. numba compiles
# them; without it, separate pure-Python versions below are used instead.
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False


# ============================================================
# SETTINGS
# ============================================================

TYPE_LABELS = C.TYPE_LABELS
DIESEL, CLEAN = 0, 1

F0, ALPHA, BETA, GAMMA = C.F0, C.ALPHA, C.BETA, C.GAMMA
MULT, L_NOM = C.MULT, C.L_NOM

# Fleet, read once at import. Set C.N_DIESEL / C.N_CLEAN before importing this
# module; run_experiments.py passes the fleet explicitly instead.
N_DIESEL, N_CLEAN = C.N_DIESEL, C.N_CLEAN

# --- genetic operators ---
CX_PROB = 0.90                # probability a mating recombines
MUT_RATE = 0#0.50               # probability an individual is mutated
LS_PROB_OFFSPRING = 1.0       # probability an offspring is educated

# --- education ---
USE_BOTH_ENGINES = False      # FINAL (ablation 10.11): single engine; True = educate with both engines
P_CO2_ENGINE = 0.50           # if not: probability of picking the CO2 engine
SPLIT_PROB = 0.0              # FINAL (ablation 10.11): no Split; 1.0 = education starts with a re-Split

CO2_INTRA_PASSES = 3          # passes of the intra-route emission search
                              # A move is accepted when the scalarised cost
                              # improves under the individual's own weight w.

# --- search directions ---
WEIGHT_GRID_SIZE = 9          # number of distinct weights in the cycling grid
EXTREME_WEIGHT_REPEATS = 3    # extra mass on w=0 and w=1, the hardest points
WEIGHT_MUTATION_PROB = 0#0.05   # probability mutation redraws an individual's weight

# Weights are ALWAYS inherited: a child keeps the weight of the parent that gave
# it its route skeleton. The only exception is the redraw above.

EXTREME_ELITE_COUNT = 2       # elites at each end given a pure weight
EXTREME_INTENSIFY_PASSES = 5  # extra polishes for each corner solution
EXTREME_INTENSIFY_EVERY = 1   # ... every this many generations

# --- local search ---
LS_NEIGHBOURHOOD_SIZE = 40

# The CO2 matrices are 0.39x to 0.86x the base distances, so rounding them to
# integers would throw away resolution. Scale up first; a common factor across
# both profiles does not change which solution is cheapest.
CO2_SCALE = 100

# Capacity-violation penalty, large relative to the arc costs each engine sees.
LOAD_PENALTY_DIST = 1000.0
LOAD_PENALTY_CO2 = 1000.0 * CO2_SCALE


# ============================================================
# SMALL HELPERS
# ============================================================

_DEMAND_CACHE = {}
_ARRAY_CACHE = {}


def demands_for(data):
    """Cached {customer id: demand}.

    data.clients() rebuilds the whole client list on every call, and this is on
    the hot path (violation, repair, split, both operators), so build it once.
    The ProblemData is held in the cache entry so it cannot be collected while
    its id() is still in use as the key.
    """
    key = id(data)
    cached = _DEMAND_CACHE.get(key)
    if cached is None:
        table = {}
        for index, client in enumerate(data.clients(), start=1):
            d = client.delivery
            if isinstance(d, (list, tuple, np.ndarray)):
                table[index] = int(d[0]) if len(d) > 0 else 0
            else:
                table[index] = int(d)
        cached = (table, data)
        _DEMAND_CACHE[key] = cached
    return cached[0]


def arrays_for(data):
    """Cached (distance matrix, demand vector) as float64 arrays for the JIT."""
    key = id(data)
    cached = _ARRAY_CACHE.get(key)
    if cached is None:
        matrix = np.ascontiguousarray(
            np.asarray(data.distance_matrix(0)), dtype=np.float64)
        table = demands_for(data)
        demand = np.zeros(len(table) + 1, dtype=np.float64)
        for c, q in table.items():
            demand[c] = q
        cached = (matrix, demand, data)
        _ARRAY_CACHE[key] = cached
    return cached[0], cached[1]


def demand_at_customer(customer_id, data):
    return demands_for(data)[customer_id]


def route_demand(visits, data):
    table = demands_for(data)
    return sum(table[c] for c in visits)


def route_list(solution):
    """[(visits, vehicle_type_index), ...] for a Solution."""
    return [(list(r.visits()), int(r.vehicle_type())) for r in solution.routes()]


def rebuild(solution, target_data):
    """Rebuild a Solution against the other ProblemData.

    A PyVRP Solution is bound to the ProblemData it was built with, and the two
    engines use different ones. The two are structurally identical, so route
    contents and vehicle-type indices carry over unchanged.
    """
    routes = [Route(target_data, list(v), int(t))
              for v, t in route_list(solution) if v]
    return Solution(target_data, routes)


def pareto_filter(points):
    """Distinct, non-dominated (distance, co2) pairs."""
    points = sorted(set(points))
    return sorted(
        p for p in points
        if not any(q[0] <= p[0] and q[1] <= p[1] and q != p for q in points)
    )


def solution_key(solution):
    """Structural identity of a solution, ignoring route order."""
    return frozenset((tuple(v), t) for v, t in route_list(solution))


# ============================================================
# PROBLEM DATA: one for distance, one for surrogate CO2
# ============================================================

def build_problem_datas(raw, n_diesel, n_clean, Q_by_type):
    """Two structurally identical ProblemData objects with different costs.

    dist_data carries the real distance matrix for both vehicle profiles, so
    minimising its cost minimises f1. co2_data carries the surrogate CO2 cost
    per profile, so minimising its cost pushes customers off diesel routes onto
    clean ones -- something the distance engine has no reason to do.
    """
    base = np.asarray(raw.distance_matrix(0)).astype(np.int64)
    dur = np.asarray(raw.duration_matrix(0)).astype(np.int64)

    def vehicle_types():
        # Index 0 is diesel (profile 0), index 1 is clean (profile 1). Identical
        # in both datasets, so vehicle_type() indices mean the same thing.
        return [
            VehicleType(num_available=n_diesel, capacity=[Q_by_type["diesel"]],
                        profile=0, name="diesel"),
            VehicleType(num_available=n_clean, capacity=[Q_by_type["clean"]],
                        profile=1, name="clean"),
        ]

    dist_data = ProblemData(
        clients=raw.clients(), depots=raw.depots(),
        vehicle_types=vehicle_types(),
        distance_matrices=[base, base], # because co2_mats has two profiles (co2_mats[0], co2_mats[1]), so this must too
        duration_matrices=[dur, dur],
    )

    co2_mats = []
    for k in ("diesel", "clean"):
        factor = GAMMA * F0 * (1.0 + ALPHA * (L_NOM ** BETA)) * MULT[k]
        co2_mats.append(np.round(base * factor * CO2_SCALE).astype(np.int64))

    co2_data = ProblemData(
        clients=raw.clients(), depots=raw.depots(),
        vehicle_types=vehicle_types(),
        distance_matrices=co2_mats,
        duration_matrices=[dur, dur],
    )

    return dist_data, co2_data


# ============================================================
# EXACT OBJECTIVES AND FEASIBILITY
# ============================================================

def solution_distance(solution, data):
    D = data.distance_matrix(0)
    total = 0.0
    for visits, _ in route_list(solution):
        seq = [0] + visits + [0]
        total += sum(float(D[a][b]) for a, b in zip(seq[:-1], seq[1:]))
    return total


def violation(solution, data, Q_by_type, n_diesel, n_clean):
    """Excess load plus excess vehicles, zero means feasible.

    A magnitude rather than a boolean, so infeasible individuals can still be
    ranked against each other.
    """
    excess = 0.0
    diesel_used = clean_used = 0

    for visits, vt in route_list(solution):
        excess += max(0.0, route_demand(visits, data)
                      - Q_by_type[TYPE_LABELS[vt]])
        if vt == DIESEL:
            diesel_used += 1
        else:
            clean_used += 1

    over_fleet = max(0, diesel_used - n_diesel) + max(0, clean_used - n_clean)
    return excess + 1000.0 * over_fleet


def evaluate_solution(solution, data, Q_by_type, n_diesel, n_clean):
    """(distance, true CO2, violation), or None for an empty solution."""
    if solution is None or not solution.routes():
        return None

    viol = violation(solution, data, Q_by_type, n_diesel, n_clean)
    distance = solution_distance(solution, data)
    co2 = true_co2(solution, data, TYPE_LABELS, Q_by_type,
                   F0, ALPHA, BETA, GAMMA, MULT)
    return distance, co2, viol


# ============================================================
# INTRA-ROUTE EMISSION SEARCH
# ============================================================
# Reordering a route changes its emissions without necessarily changing its
# length: delivering a heavy demand early drops the load carried on every later
# arc. The moves are segment reversal (whole-route reversal is the i=0, j=m-1
# case) and single-customer relocation.

def _route_metrics_impl(route, matrix, demand):
    """(distance, sum d_ij * u_ij) for one route, depot at node 0."""
    m = route.shape[0]

    remaining = 0.0
    for k in range(m):
        remaining += demand[route[k]]

    distance = weighted = 0.0
    prev = 0
    for k in range(m):
        customer = route[k]
        d = matrix[prev, customer]
        distance += d
        weighted += d * remaining
        remaining -= demand[customer]
        prev = customer

    d = matrix[prev, 0]
    distance += d
    weighted += d * remaining      # remaining is 0 on the return leg

    return distance, weighted


def _improve_route_kernel_impl(route, matrix, demand, capacity, factor, alpha,
                               weight, dist_scale, co2_scale, max_passes):
    """Reorder one route. Every candidate is evaluated in full, which is what
    makes the scalarised acceptance rule affordable: m^2 candidates times O(m)
    is only a few thousand float operations per pass on a typical route.

    Plain Python, so it runs correctly with or without numba -- numba just
    compiles it and makes it fast.
    """
    m = route.shape[0]
    if m < 3:
        return route

    best = route.copy()
    cand = np.empty(m, dtype=np.int64)

    best_dist, best_weighted = _route_metrics(best, matrix, demand)
    best_co2 = factor * (best_dist + (alpha / capacity) * best_weighted)

    for _ in range(max_passes):
        improved = False

        # reverse a segment
        for i in range(m - 1):
            for j in range(i + 1, m):
                for t in range(m):
                    cand[t] = best[t]
                lo, hi = i, j
                while lo < hi:
                    tmp = cand[lo]
                    cand[lo] = cand[hi]
                    cand[hi] = tmp
                    lo += 1
                    hi -= 1

                new_dist, new_weighted = _route_metrics(cand, matrix, demand)
                new_co2 = factor * (new_dist + (alpha / capacity) * new_weighted)

                old_score = (weight * (best_dist / dist_scale)
                             + (1.0 - weight) * (best_co2 / co2_scale))
                new_score = (weight * (new_dist / dist_scale)
                             + (1.0 - weight) * (new_co2 / co2_scale))
                ok = new_score < old_score - 1e-12

                if ok:
                    for t in range(m):
                        best[t] = cand[t]
                    best_dist, best_co2, improved = new_dist, new_co2, True
                    break
            if improved:
                break

        if improved:
            continue

        # relocate one customer
        for i in range(m):
            for p in range(m):
                if p == i:
                    continue

                idx = 0
                for t in range(m):
                    if t != i:
                        cand[idx] = best[t]
                        idx += 1
                for t in range(m - 1, p, -1):
                    cand[t] = cand[t - 1]
                cand[p] = best[i]

                new_dist, new_weighted = _route_metrics(cand, matrix, demand)
                new_co2 = factor * (new_dist + (alpha / capacity) * new_weighted)

                old_score = (weight * (best_dist / dist_scale)
                             + (1.0 - weight) * (best_co2 / co2_scale))
                new_score = (weight * (new_dist / dist_scale)
                             + (1.0 - weight) * (new_co2 / co2_scale))
                ok = new_score < old_score - 1e-12

                if ok:
                    for t in range(m):
                        best[t] = cand[t]
                    best_dist, best_co2, improved = new_dist, new_co2, True
                    break
            if improved:
                break

        if not improved:
            break

    return best


if HAS_NUMBA:
    try:
        # cache=True keeps the compiled kernel between runs, but needs a real
        # file to write beside; it raises here if the module was exec'd from a
        # string or lives on a read-only mount.
        _route_metrics = njit(cache=True)(_route_metrics_impl)
        _improve_route_kernel = njit(cache=True)(_improve_route_kernel_impl)
    except Exception:  # pragma: no cover
        _route_metrics = njit(_route_metrics_impl)
        _improve_route_kernel = njit(_improve_route_kernel_impl)
else:  # pragma: no cover
    # Same functions, uncompiled. Identical results, just slower.
    _route_metrics = _route_metrics_impl
    _improve_route_kernel = _improve_route_kernel_impl


def polish_route_emissions(solution, data, Q_by_type, weight=0.5,
                           dist_scale=1.0, co2_scale=1.0):
    """Run the intra-route search on every route of a solution.

    The scales normalise the two objectives so that w=0.5 really does balance
    them: distance and emissions differ by a factor of two or more in raw
    units, so an unnormalised weighted sum would quietly favour distance.
    """
    matrix, demand = arrays_for(data)
    changed = False
    out = []

    for visits, vt in route_list(solution):
        label = TYPE_LABELS[vt]
        capacity = float(Q_by_type[label])

        if len(visits) < 3 or capacity <= 0:
            out.append((visits, vt))
            continue

        result = _improve_route_kernel(
            np.asarray(visits, dtype=np.int64), matrix, demand,
            capacity, GAMMA * F0 * MULT[label], ALPHA,
            float(weight), float(dist_scale), float(co2_scale),
            CO2_INTRA_PASSES)
        improved = [int(c) for c in result]

        if improved != visits:
            changed = True
        out.append((improved, vt))

    if not changed:
        return solution
    return Solution(data, [Route(data, v, t) for v, t in out if v])


# ============================================================
# SPLIT
# ============================================================
# Given a FIXED customer sequence, cut it into routes optimally by shortest-path
# DP (Prins). PyVRP's local search only accepts strictly improving moves, so a
# distance-neutral rearrangement that pushes a route's demand under the clean
# capacity threshold is invisible to it, yet that rearrangement is exactly
# what turns a diesel route into a clean one for free. Split sees every
# partition at once, so it finds those crossings directly.
#
# w=1 gives the classic minimum-distance Split; w=0 minimises true emissions,
# costing each segment under the cheapest vehicle type that can carry it.
#
# The emission integral expands to
#
#     sum d_ij * u_ij  =  D_seg * P  -  Q_acc
#
# where P is the depot-to-current-customer path distance and Q_acc the running
# sum of (arc distance * demand loaded before that arc). Neither depends on the
# segment's total demand, so both update in O(1) as the segment grows and the
# whole DP is O(n * max_route_len). Assumes BETA == 1.

def giant_tour(solution):
    """Flatten a solution into one customer sequence, dropping route boundaries."""
    tour = []
    for visits, _ in route_list(solution):
        tour.extend(visits)
    return tour

def _split_kernel_impl(tour, demand, matrix, q_clean, q_diesel, weight,
                       ef_diesel, ef_clean, alpha, dist_scale, co2_scale):
    """Compiled Split DP. Only scalars and numpy arrays cross the boundary.

    fastmath is deliberately off: the DP was validated against exhaustive
    enumeration, and reassociating the arithmetic could change which partition
    wins on a tie.
    """
    n = tour.shape[0]
    inf = 1e300
    cost = np.full(n + 1, inf)
    pred = np.full(n + 1, -1, dtype=np.int64)
    cost[0] = 0.0

    for i in range(n):
        if cost[i] >= inf:
            continue

        path = 0.0
        weighted = 0.0
        load = 0.0
        prev = 0

        for j in range(i, n):
            customer = tour[j]
            arc = matrix[prev, customer]
            path += arc
            if j > i:
                weighted += arc * load
            load += demand[customer]

            if load > q_diesel:
                break

            prev = customer
            segment_distance = path + matrix[customer, 0]

            if load <= q_clean:
                factor = ef_clean
                capacity = q_clean
            else:
                factor = ef_diesel
                capacity = q_diesel

            segment_co2 = factor * (segment_distance
                                    + (alpha / capacity) * (load * path - weighted))
            segment_cost = (weight * (segment_distance / dist_scale)
                            + (1.0 - weight) * (segment_co2 / co2_scale))

            total = cost[i] + segment_cost
            if total < cost[j + 1]:
                cost[j + 1] = total
                pred[j + 1] = i

    return cost, pred


if HAS_NUMBA:
    try:
        _split_kernel = njit(cache=True)(_split_kernel_impl)
    except Exception:  # pragma: no cover
        _split_kernel = njit(_split_kernel_impl)
else:  # pragma: no cover
    # Same function, uncompiled.
    _split_kernel = _split_kernel_impl


def warm_up_split():
    """Compile the kernels once at startup, so the JIT cost does not land in
    the middle of a timed run. Returns the seconds spent."""
    if not HAS_NUMBA:
        return 0.0
    started = time.perf_counter()
    matrix = np.zeros((3, 3), dtype=np.float64)
    demand = np.zeros(3, dtype=np.float64)
    for w in (1.0, 0.5):
        _split_kernel(np.array([1, 2], dtype=np.int64), demand, matrix,
                      1.0, 2.0, w, 1.0, 1.0, 0.2, 1.0, 1.0)
    return time.perf_counter() - started


def split_tour(tour, data, Q_by_type, weight, dist_scale=1.0, co2_scale=1.0):
    """Cut a customer sequence into routes. Returns a list of visit-lists, or
    None if no valid partition exists."""
    n = len(tour)
    if n == 0:
        return None

    matrix, demand = arrays_for(data)
    args = (demand, matrix,
            float(Q_by_type["clean"]), float(Q_by_type["diesel"]), float(weight),
            GAMMA * F0 * MULT["diesel"], GAMMA * F0 * MULT["clean"], ALPHA,
            float(dist_scale), float(co2_scale))

    cost, pred = _split_kernel(np.asarray(tour, dtype=np.int64), *args)

    if cost[n] >= 1e300:
        return None

    parts = []
    end = n
    while end > 0:
        start = int(pred[end])
        if start < 0:
            return None
        parts.append(list(tour[start:end]))
        end = start

    parts.reverse()
    return parts


def split_solution(solution, data, Q_by_type, n_diesel, n_clean, weight,
                   dist_scale=1.0, co2_scale=1.0):
    """Re-decode a solution through Split, keeping its customer order."""
    parts = split_tour(giant_tour(solution), data, Q_by_type, weight,
                       dist_scale, co2_scale)
    if parts is None or len(parts) > n_diesel + n_clean:
        return solution

    # Clean is cheaper for any route it can carry: the worst-case clean:diesel
    # emission ratio works out at 0.45 * 1.20 = 0.54. So take clean wherever it
    # fits and let repair_fleet deal with any fleet-count overflow.
    routes = [
        (visits, CLEAN if route_demand(visits, data) <= Q_by_type["clean"] else DIESEL)
        for visits in parts
    ]
    routes = repair_fleet(routes, data, Q_by_type, n_diesel, n_clean)

    try:
        return Solution(data, [Route(data, v, t) for v, t in routes if v])
    except Exception:  # pragma: no cover
        return solution


# ============================================================
# REPAIR
# ============================================================

def repair_fleet(routes, data, Q_by_type, n_diesel, n_clean):
    """Fix fleet-count overflow by flipping vehicle types where capacity allows.
    Anything left over is handled by the violation penalty."""
    routes = [(list(v), int(t)) for v, t in routes]

    def counts():
        d = sum(1 for _, t in routes if t == DIESEL)
        return d, len(routes) - d

    _, clean_used = counts()
    if clean_used > n_clean:
        idx = [i for i, (_, t) in enumerate(routes) if t == CLEAN]
        idx.sort(key=lambda i: route_demand(routes[i][0], data))
        for i in idx[:clean_used - n_clean]:
            routes[i] = (routes[i][0], DIESEL)

    diesel_used, _ = counts()
    if diesel_used > n_diesel:
        idx = [i for i, (v, t) in enumerate(routes)
               if t == DIESEL and route_demand(v, data) <= Q_by_type["clean"]]
        idx.sort(key=lambda i: route_demand(routes[i][0], data))
        for i in idx[:diesel_used - n_diesel]:
            routes[i] = (routes[i][0], CLEAN)

    return routes


# ============================================================
# THE TWO LOCAL SEARCH ENGINES
# ============================================================

def build_weight_grid():
    """The cycling grid of search directions, with extra mass at the ends.

    A uniform grid spends 1/9 of its effort on pure distance, but that corner is
    a full CVRP optimum and by far the hardest single point on the front.
    Repeating the two extremes puts effort where the return is highest.
    """
    if WEIGHT_GRID_SIZE <= 1:
        return [0.5]

    grid = [i / (WEIGHT_GRID_SIZE - 1) for i in range(WEIGHT_GRID_SIZE)]
    extra = max(0, EXTREME_WEIGHT_REPEATS - 1)
    return sorted(grid + [0.0] * extra + [1.0] * extra)


def make_neighbourhood_params(size):
    """The granular-neighbourhood argument was renamed in pyvrp 0.10.

    Resolved explicitly rather than by try/except, because passing the wrong
    name raises TypeError inside the engine constructor - which is caught, and
    the run then continues with local search silently disabled.
    """
    import inspect

    names = set(inspect.signature(NeighbourhoodParams).parameters)
    for candidate in ("num_neighbours", "nb_granular"):
        if candidate in names:
            return NeighbourhoodParams(**{candidate: size})

    raise TypeError("NeighbourhoodParams accepts neither 'num_neighbours' nor "
                    f"'nb_granular'; it accepts {sorted(names)}")


class _Engine:
    """One PyVRP LocalSearch bound to one cost matrix."""

    def __init__(self, name, data, seed, nb_size, load_penalty):
        self.name = name
        self.data = data

        rng = RandomNumberGenerator(seed)
        ls = LocalSearch(data, rng,
                         compute_neighbours(data, make_neighbourhood_params(nb_size)))
        for op in NODE_OPERATORS:
            ls.add_node_operator(op(data))
        for op in ROUTE_OPERATORS:
            ls.add_route_operator(op(data))

        self.ls = ls
        self.cost = CostEvaluator([float(load_penalty)], 6.0, 0.0)

    def __call__(self, solution):
        return self.ls(solution, self.cost)


class DualLocalSearch:
    """Both engines behind one interface, sharing a single time budget.

    Always hands back a Solution living in dist_data; the CO2 engine's input
    and output are converted around the call.
    """

    def __init__(self, dist_data, co2_data, Q_by_type, n_diesel, n_clean,
                 seed, nb_size, time_budget):
        self.dist_data = dist_data
        self.co2_data = co2_data
        self.Q_by_type = Q_by_type
        self.n_diesel = n_diesel
        self.n_clean = n_clean
        self.remaining = float(time_budget)

        self.weight_grid = build_weight_grid()
        self._weight_turn = 0

        self.dist_scale = 1.0
        self.co2_scale = 1.0

        self.calls = {"dist": 0, "co2": 0}
        self.splits = {"dist": 0, "co2": 0}
        self.outcomes = {"dist": 0, "co2": 0, "random": 0}
        self.intra_calls = 0
        self.intra_changed = 0
        self.split_time = 0.0
        self.repartitions = 0

        self.enabled = HAS_SEARCH
        self._warned = False

        D = np.asarray(dist_data.distance_matrix(0))
        self.symmetric = bool(np.array_equal(D, D.T))
        if not self.symmetric:
            print("[note] distance matrix is asymmetric; route reversal disabled.")

        if not self.enabled:
            print(f"\n[warning] pyvrp.search unavailable ({_SEARCH_ERROR!r}); "
                  "running NSGA-II without local search.\n")
            return

        try:
            self.engines = {
                "dist": _Engine("dist", dist_data, seed, nb_size, LOAD_PENALTY_DIST),
                "co2": _Engine("co2", co2_data, seed + 1, nb_size, LOAD_PENALTY_CO2),
            }
        except Exception as e:  # pragma: no cover
            self.enabled = False
            print("\n" + "!" * 58)
            print(f"[CRITICAL] could not build LocalSearch: {type(e).__name__}: {e}")
            print("Running NSGA-II WITHOUT local search. Results will be far")
            print("worse than expected -- fix this before trusting any output.")
            print("!" * 58 + "\n")

    def pick(self):
        return "co2" if np.random.random() < P_CO2_ENGINE else "dist"

    def next_weight(self):
        w = float(self.weight_grid[self._weight_turn % len(self.weight_grid)])
        self._weight_turn += 1
        return w

    def set_scales(self, dist_scale, co2_scale, ref=None, floor_frac=0.01, verbose=True):
        """set_scales(self, dist_scale, co2_scale):
        Calibrate the weight normalisation.
        """
        d_floor = c_floor = 1e-9
        if ref is not None:
            d_floor = max(d_floor, floor_frac * abs(float(ref[0])))
            c_floor = max(c_floor, floor_frac * abs(float(ref[1])))

        if dist_scale > 0:
            self.dist_scale = max(float(dist_scale), d_floor)
        if co2_scale > 0:
            self.co2_scale = max(float(co2_scale), c_floor)

        if verbose:
            print(f"    scales: dist={self.dist_scale:.0f} "
                  f"co2={self.co2_scale:.0f} "
                  f"ratio={self.co2_scale / self.dist_scale:.2f}")
    # education

    def improve(self, solution, objective=None, weight=None):
        """Split, then one engine, then the intra-route emission search."""
        if objective is None:
            objective = self.pick()
        if weight is None:
            weight = self.next_weight()

        if SPLIT_PROB > 0.0 and np.random.random() < SPLIT_PROB:
            start = time.perf_counter()
            before = len(solution.routes())
            solution = split_solution(solution, self.dist_data, self.Q_by_type,
                                      self.n_diesel, self.n_clean, weight,
                                      self.dist_scale, self.co2_scale)
            self.split_time += time.perf_counter() - start
            self.splits[objective] += 1
            if len(solution.routes()) != before:
                self.repartitions += 1

        if self.enabled and self.remaining > 0:
            engine = self.engines[objective]
            start = time.perf_counter()
            try:
                work = solution if objective == "dist" else rebuild(solution, self.co2_data)
                work = engine(work)
                solution = work if objective == "dist" else rebuild(work, self.dist_data)
                self.calls[objective] += 1
            except Exception as e:  # pragma: no cover
                if not self._warned:
                    print(f"[warning] local search failed ({type(e).__name__}: {e}); "
                          "disabling it for the rest of the run.")
                    self._warned = True
                self.enabled = False
            finally:
                self.remaining -= time.perf_counter() - start

        if self.symmetric:
            before = solution
            solution = polish_route_emissions(solution, self.dist_data,
                                              self.Q_by_type, weight,
                                              self.dist_scale, self.co2_scale)
            self.intra_calls += 1
            if solution is not before:
                self.intra_changed += 1

        return solution

    def _educate_both(self, solution, weight):
        """Educate the SAME starting solution with each engine.

        PyVRP's LocalSearch returns a new Solution rather than mutating its
        argument, so the second call does not see the first one's output.
        Returns (candidate_dist, metrics_dist, candidate_co2, metrics_co2).
        """
        w = self.next_weight() if weight is None else float(weight)
        cand_d = self.improve(solution, objective="dist", weight=w)
        cand_c = self.improve(solution, objective="co2", weight=w)

        md = evaluate_solution(cand_d, self.dist_data, self.Q_by_type,
                               self.n_diesel, self.n_clean)
        mc = evaluate_solution(cand_c, self.dist_data, self.Q_by_type,
                               self.n_diesel, self.n_clean)
        return cand_d, md, cand_c, mc

    @staticmethod
    def _dominates(a, b):
        return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

    def improve_both(self, solution, weight=None):
        """Educate with both engines and keep the result that dominates; on a
        tie, pick one at random."""
        if not self.enabled or self.remaining <= 0:
            if self.symmetric:
                return polish_route_emissions(solution, self.dist_data,
                                              self.Q_by_type, self.next_weight(),
                                              self.dist_scale, self.co2_scale)
            return solution

        cand_d, md, cand_c, mc = self._educate_both(solution, weight)

        if md is None and mc is None:
            return cand_d
        if md is None:
            self.outcomes["co2"] += 1
            return cand_c
        if mc is None:
            self.outcomes["dist"] += 1
            return cand_d

        # Feasibility outranks dominance: comparing objectives across the
        # feasibility boundary would let a badly overloaded solution look
        # attractive purely because it is short.
        if md[2] != mc[2]:
            winner = "dist" if md[2] < mc[2] else "co2"
            self.outcomes[winner] += 1
            return cand_d if winner == "dist" else cand_c

        if self._dominates(md, mc):
            self.outcomes["dist"] += 1
            return cand_d
        if self._dominates(mc, md):
            self.outcomes["co2"] += 1
            return cand_c

        self.outcomes["random"] += 1
        return cand_d if np.random.random() < 0.5 else cand_c

    def improve_pair(self, solution, weight=None):
        """Like improve_both, but returns EVERY non-dominated result.

        When neither engine's result dominates the other, both are front
        material, and discarding one on a coin flip throws away spread. Let
        NSGA-II's survival step decide instead.
        """
        if not USE_BOTH_ENGINES or not self.enabled or self.remaining <= 0:
            return [self.improve(solution, weight=weight)]

        cand_d, md, cand_c, mc = self._educate_both(solution, weight)

        if md is None:
            return [cand_c] if mc is not None else [cand_d]
        if mc is None:
            return [cand_d]

        if md[2] != mc[2]:
            winner = "dist" if md[2] < mc[2] else "co2"
            self.outcomes[winner] += 1
            return [cand_d] if winner == "dist" else [cand_c]

        if self._dominates(md, mc):
            self.outcomes["dist"] += 1
            return [cand_d]
        if self._dominates(mc, md):
            self.outcomes["co2"] += 1
            return [cand_c]

        self.outcomes["random"] += 1
        return [cand_d, cand_c]

    def polish(self, solution, weight=None):
        """Single entry point, so the both-engines policy can be switched off
        in one place for ablation runs."""
        if USE_BOTH_ENGINES:
            return self.improve_both(solution, weight=weight)
        return self.improve(solution, weight=weight)

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self


# ============================================================
# RANDOM CONSTRUCTION
# ============================================================

def make_random_solution(data, n_diesel, n_clean, Q_by_type, diesel_bias=None):
    """A random giant tour cut into a plausible number of routes.

    diesel_bias in [0, 1] sets the diesel/clean mix and, through it, the route
    count: more diesel means more capacity per route, so fewer routes.
    """
    n_customers = len(data.clients())
    customers = np.random.permutation(np.arange(1, n_customers + 1)).tolist()
    total_demand = sum(demand_at_customer(c, data) for c in customers)

    bias = (float(np.random.uniform(0.0, 1.0)) if diesel_bias is None
            else float(diesel_bias))

    blended_cap = bias * Q_by_type["diesel"] + (1.0 - bias) * Q_by_type["clean"]
    estimated = max(1, int(np.ceil(total_demand / max(blended_cap, 1.0))))

    lo = max(1, estimated - 2)
    hi = max(lo, min(n_diesel + n_clean, n_customers, estimated + 4))
    n_routes = int(np.random.randint(lo, hi + 1))

    if n_routes <= 1:
        chunks = [customers]
    else:
        cuts = sorted(np.random.choice(np.arange(1, n_customers),
                                       size=n_routes - 1, replace=False))
        chunks, start = [], 0
        for c in cuts:
            chunks.append(customers[start:c])
            start = c
        chunks.append(customers[start:])
        chunks = [c for c in chunks if c]

    diesel_left, clean_left = n_diesel, n_clean
    routes = []

    for visits in chunks:
        demand = route_demand(visits, data)
        want_diesel = np.random.random() < bias
        can_diesel = diesel_left > 0 and demand <= Q_by_type["diesel"]
        can_clean = clean_left > 0 and demand <= Q_by_type["clean"]

        if want_diesel and can_diesel:
            vt, diesel_left = DIESEL, diesel_left - 1
        elif (not want_diesel) and can_clean:
            vt, clean_left = CLEAN, clean_left - 1
        elif can_diesel:
            vt, diesel_left = DIESEL, diesel_left - 1
        elif can_clean:
            vt, clean_left = CLEAN, clean_left - 1
        else:
            vt = DIESEL      # let the penalty and repair sort it out

        routes.append(Route(data, visits, vt))

    return Solution(data, routes)


def make_feasible_random_solution(data, Q_by_type, n_diesel, n_clean, dls,
                                  diesel_bias=None, weight=None, max_attempts=15):
    """Retry random construction until education produces a feasible solution."""
    sol = None
    for _ in range(max_attempts):
        sol = make_random_solution(data, n_diesel, n_clean, Q_by_type, diesel_bias)
        sol = dls.polish(sol, weight=weight)
        metrics = evaluate_solution(sol, data, Q_by_type, n_diesel, n_clean)
        if metrics is not None and metrics[2] == 0.0:
            return sol
    return sol

def calibrate_from_anchors(dls, problem, seconds=2.0, seed=1):
    """Solve both single-objective corners with PyVRP's full solver.

    Local search alone lands 1-3% short of the true corners on these
    instances; the weighted-sum baseline reaches them because each of its
    scalarisations is a complete PyVRP run. Two seconds of the same solver
    buys us the corners and, as a side effect, exact objective spans.

    Returns (dist_anchor_solution, co2_anchor_solution, metrics_d, metrics_c).
    """
    from pyvrp import solve as pyvrp_solve
    from pyvrp.stop import MaxRuntime

    p = problem
    res_d = pyvrp_solve(dls.dist_data, stop=MaxRuntime(seconds),
                        seed=seed, display=False)
    res_c = pyvrp_solve(dls.co2_data, stop=MaxRuntime(seconds),
                        seed=seed, display=False)

    sol_d = res_d.best
    sol_c = rebuild(res_c.best, dls.dist_data)   # co2 engine -> distance space

    m_d = evaluate_solution(sol_d, p.vrp_data, p.Q_by_type,
                            p.n_diesel, p.n_clean)
    m_c = evaluate_solution(sol_c, p.vrp_data, p.Q_by_type,
                            p.n_diesel, p.n_clean)
    if m_d is None or m_c is None:
        print("    [warning] anchor solve failed; scales left at 1.0")
        return None, None, None, None

    d_span = abs(m_c[0] - m_d[0])
    c_span = abs(m_d[1] - m_c[1])
    ratio = c_span / d_span if d_span > 1e-9 else float("inf")
    if not (1.5 <= ratio <= 5.5):
        print(f"    [note] anchor ratio {ratio:.2f} out of band; using 3.30")
        ratio = 3.30
    d_span = 0.15 * m_d[0]
    c_span = d_span * ratio

    dls.set_scales(d_span, c_span)
    return sol_d, sol_c, m_d, m_c


# ============================================================
# PYMOO OPERATORS
# ============================================================
# An individual is a 2-element object array: column 0 is the Solution, column 1
# is its search weight. The weight has to live on the individual, not on a
# global counter -- a lineage only converges toward a corner if it is pushed in
# the same direction for many consecutive generations.

class VRPSampling(Sampling):
    """
    This class builds the starting population for the algorithm.
    Each candidate gets two things: a weight (0 to 1, how much it 
    cares about distance vs CO2) and a starting solution built with 
    a matching diesel/clean truck mix.

    The weights are spread evenly across 0 to 1 across the population, 
    so all tradeoff levels are represented from the start.

    The mix is matched to the weight because it makes sense: w=1 means only distance matters
    and minimizing distance favors diesel trucks (bigger capacity, fewer trips). 
    w=0 means only CO2 matters which favors clean trucks. So a candidate built with a high weight 
    starts off mostly diesel, and one with a low weight starts off mostly clean. Each candidate 
    begins already aligned with what it's being optimized for, instead of starting random and 
    needing to be corrected later.
    """

    def __init__(self, dls, jitter=0.03):
        super().__init__()
        self.dls = dls
        self.jitter = jitter

    def _do(self, problem, n_samples, **kwargs):
        # grid is guaranteeing the population is spread evenly across the entire distance-CO2 tradeoff spectrum
        grid = np.array([0.5]) if n_samples == 1 else np.linspace(0.0, 1.0, n_samples)
        # Without this row 0 would deterministically always be the pure-distance individual, row n-1 always pure-co2
        order = np.random.permutation(n_samples)

        X = np.empty((n_samples, 2), dtype=object)
        for slot, i in enumerate(order):
            base = float(grid[slot]) # the individual's official weight
            bias = base if base in (0.0, 1.0) else float(np.clip(
                base + np.random.uniform(-self.jitter, self.jitter), 0.0, 1.0)) # the diesel/clean bias, jittered to avoid duplicates

            X[i, 0] = make_feasible_random_solution(
                problem.vrp_data, problem.Q_by_type,
                problem.n_diesel, problem.n_clean, self.dls,
                diesel_bias=bias, weight=base)
            X[i, 1] = base 
        return X


class VRPCrossover(Crossover):
    """Inherit a random subset of parent A's routes intact, fill the rest from
    parent B with A's customers removed, repair fleet counts, educate."""

    def __init__(self, dls, prob=0.9):
        # Four offspring per mating when both engines run: each of the two
        # recombinations can return two mutually non-dominated children.
        # prob is applied manually in _do, so behaviour does not depend on
        # which pymoo version's base class also applies it.
        super().__init__(2, 4 if USE_BOTH_ENGINES else 2, prob=1.0)
        self.rate = prob
        self.dls = dls

    def _recombine(self, parent_a, parent_b, problem, weight):
        data = problem.vrp_data
        routes_a = route_list(parent_a)
        routes_b = route_list(parent_b)

        if not routes_a:
            return [parent_b]

        k = np.random.randint(1, len(routes_a) + 1)
        chosen = np.random.choice(len(routes_a), size=k, replace=False)
        inherited = [routes_a[i] for i in chosen]
        taken = {c for visits, _ in inherited for c in visits}

        remainder = []
        for visits, vt in routes_b:
            filtered = [c for c in visits if c not in taken]
            if filtered:
                remainder.append((filtered, vt))

        child_routes = repair_fleet(inherited + remainder, data, problem.Q_by_type,
                                    problem.n_diesel, problem.n_clean)

        try:
            child = Solution(data, [Route(data, v, t) for v, t in child_routes if v])
        except Exception:  # pragma: no cover
            return [parent_a]

        if np.random.random() < LS_PROB_OFFSPRING:
            return self.dls.improve_pair(child, weight=weight)
        return [child]

    def _do(self, problem, X, **kwargs):
        n_matings = X.shape[1]
        n_off = self.n_offsprings
        Y = np.empty((n_off, n_matings, 2), dtype=object)

        for k in range(n_matings):
            p1, p2 = X[0, k, 0], X[1, k, 0]
            w1, w2 = float(X[0, k, 1]), float(X[1, k, 1])

            if np.random.random() > self.rate:
                kids = [(p1, w1), (p2, w2)]
            else:
                # Each child keeps the weight of the parent that gave it its
                # route skeleton, so a direction stays with a lineage instead
                # of being redrawn from a global schedule every generation.
                kids = ([(c, w1) for c in self._recombine(p1, p2, problem, w1)]
                        + [(c, w2) for c in self._recombine(p2, p1, problem, w2)])

            if not kids:
                kids = [(p1, w1), (p2, w2)]

            # Pad by cycling when a recombination returned only one child.
            # Duplicates are removed downstream by eliminate_duplicates.
            base, i = list(kids), 0
            while len(kids) < n_off:
                kids.append(base[i % len(base)])
                i += 1

            for slot in range(n_off):
                child, weight = kids[slot]
                Y[slot, k, 0] = child
                Y[slot, k, 1] = weight

        return Y


class VRPMutation(Mutation):
    """Flip a route's vehicle type, merge two routes, split one, or push the
    whole solution toward one fuel type -> then educate."""

    def __init__(self, dls, rate=0.5):
        super().__init__(prob=1.0)
        self.rate = rate
        self.dls = dls

    def _mutate_one(self, solution, problem, weight):
        data = problem.vrp_data
        routes = route_list(solution)
        if not routes:
            return solution

        # move = np.random.choice(["flip_type", "merge", "split", "block_type"],
        #                         p=[0.40, 0.20, 0.20, 0.20])
        move = np.random.choice(["flip_type", "merge", "split", "block_type"],
                                        p=[0.25, 0.25, 0.25, 0.25])

        if move == "flip_type":
            i = np.random.randint(len(routes))
            visits, vt = routes[i]
            target = CLEAN if vt == DIESEL else DIESEL
            if route_demand(visits, data) <= problem.Q_by_type[TYPE_LABELS[target]]:
                routes[i] = (visits, target)

        elif move == "merge" and len(routes) >= 2:
            i, j = np.random.choice(len(routes), size=2, replace=False)
            v1, t1 = routes[i]
            v2, _ = routes[j]
            merged = v1 + v2
            if route_demand(merged, data) <= problem.Q_by_type[TYPE_LABELS[t1]]:
                routes = [r for idx, r in enumerate(routes) if idx not in (i, j)]
                routes.append((merged, t1))

        elif move == "split":
            i = np.random.randint(len(routes))
            visits, vt = routes[i]
            if len(visits) >= 2:
                cut = np.random.randint(1, len(visits))
                routes[i] = (visits[:cut], vt)
                routes.append((visits[cut:], vt))

        elif move == "block_type":
            target = CLEAN if np.random.random() < 0.5 else DIESEL
            fleet_cap = problem.n_diesel if target == DIESEL else problem.n_clean
            if len(routes) <= fleet_cap:
                routes = [
                    (visits, target)
                    if route_demand(visits, data) <= problem.Q_by_type[TYPE_LABELS[target]]
                    else (visits, vt)
                    for visits, vt in routes
                ]

        routes = repair_fleet(routes, data, problem.Q_by_type,
                              problem.n_diesel, problem.n_clean)

        try:
            mutated = Solution(data, [Route(data, v, t) for v, t in routes if v])
        except Exception:  # pragma: no cover
            return solution

        if np.random.random() < LS_PROB_OFFSPRING:
            # Educate in THIS individual's own direction, exactly as crossover
            # does. Passing no weight here would fall back to the cycling grid,
            # so a mutated solution would be pushed in a direction unrelated to
            # the one it carries.
            mutated = self.dls.polish(mutated, weight=weight)
        return mutated

    def _do(self, problem, X, **kwargs):
        Y = X.copy()
        grid = self.dls.weight_grid

        for i in range(len(Y)):
            weight = float(Y[i, 1])

            # Occasionally redraw the weight. Without this, a direction whose
            # carriers all die out is gone for good.
            if np.random.random() < WEIGHT_MUTATION_PROB and grid:
                weight = float(grid[np.random.randint(len(grid))])
                Y[i, 1] = weight

            if np.random.random() < self.rate:
                Y[i, 0] = self._mutate_one(Y[i, 0], problem, weight)

        return Y


class VRPDuplicateElimination(ElementwiseDuplicateElimination):
    """Compares route structure, not objective values: this runs on offspring
    before they are evaluated, so F may not exist yet."""

    def is_equal(self, a, b):
        return solution_key(a.get("X")[0]) == solution_key(b.get("X")[0])


# ============================================================
# PROBLEM
# ============================================================

class GreenVRPProblem(Problem):
    def __init__(self, data, Q_by_type, n_diesel, n_clean):
        self.vrp_data = data
        self.Q_by_type = Q_by_type
        self.n_diesel = n_diesel
        self.n_clean = n_clean
        super().__init__(n_var=2, n_obj=2, vtype=object)

    def _evaluate(self, X, out, *args, **kwargs):
        F, raw = [], []

        for row in X:
            metrics = evaluate_solution(row[0], self.vrp_data, self.Q_by_type,
                                        self.n_diesel, self.n_clean)
            if metrics is None:
                F.append([1e15, 1e15])
                raw.append([np.inf, np.inf, np.inf])
                continue

            distance, co2, viol = metrics
            pen = 1e6 * viol
            F.append([distance + pen, co2 + pen])
            raw.append([distance, co2, viol])

        out["F"] = np.array(F, dtype=float)
        out["F_raw"] = np.array(raw, dtype=float)


# ============================================================
# CALLBACKS
# ============================================================
# Optional. They intensify the front and protect its extremes, which is what
# hypervolume is most sensitive to. Pass make_callbacks(...) to minimize().

def _feasible(pop):
    """Individuals with a finite, feasible F_raw."""
    return [ind for ind in pop
            if (raw := ind.get("F_raw")) is not None
            and np.isfinite(raw[0]) and raw[2] == 0]

class IntensifyElite(Callback):
    """Spend extra local search on the current non-dominated front.

    A result is written back only if it is not dominated by the point it
    replaces, so a solution can slide along the front but local search cannot
    quietly make an elite worse.
    """

    def __init__(self, dls, problem, every=5, passes=1):
        super().__init__()
        self.dls = dls
        self.problem = problem
        self.every = every
        self.passes = passes

    def notify(self, algorithm):
        if algorithm.n_gen % self.every != 0:
            return

        pop = algorithm.pop
        ranks = pop.get("rank")
        if ranks is None:
            return

        p = self.problem
        elite = [i for i in np.where(ranks == 0)[0]
                 if (raw := pop[i].get("F_raw")) is not None
                 and np.isfinite(raw[0]) and raw[2] == 0]
        elite.sort(key=lambda i: pop[i].get("F_raw")[0])
        n_elite = len(elite)
        pure = min(EXTREME_ELITE_COUNT, max(1, n_elite // 3))

        for position, idx in enumerate(elite):
            ind = pop[idx]
            old = ind.get("F_raw")

            # The outermost elites get a pure direction, since they define the
            # ends of the front. The rest keep their own weight: overriding it
            # with a rank-derived value would undo the lineage persistence this
            # is meant to exploit.
            if n_elite > 1 and position < pure:
                weight = 1.0
            elif n_elite > 1 and position >= n_elite - pure:
                weight = 0.0
            else:
                weight = float(ind.get("X")[1])

            sol = ind.get("X")[0]
            for _ in range(self.passes):
                sol = self.dls.improve(sol, objective="dist" if weight >= 0.5 else "co2",
                                       weight=weight)

            metrics = evaluate_solution(sol, p.vrp_data, p.Q_by_type,
                                        p.n_diesel, p.n_clean)
            if metrics is None:
                continue

            distance, co2, viol = metrics
            if viol > 0:
                continue
            if old[0] <= distance and old[1] <= co2 and (old[0] < distance or old[1] < co2):
                continue

            ind.X[0] = sol
            ind.F = np.array([distance, co2], dtype=float)
            ind.set("F_raw", np.array([distance, co2, 0.0]))


class IntensifyExtremes(Callback):
    """Sustained single-objective pressure on the two corner solutions.

    This is the one thing the cycling weight schedule cannot do: reaching a
    CVRP optimum needs repeated w=1 polishes on the SAME solution, and a global
    schedule spreads pure directions thinly across unrelated lineages. Only two
    individuals are touched per generation, so the cost is a handful of extra
    polishes whatever the population size.
    """

    def __init__(self, dls, problem, passes=5, every=1):
        super().__init__()
        self.dls = dls
        self.problem = problem
        self.passes = passes
        self.every = every
        self.attempts = {"dist": 0, "co2": 0}
        self.improvements = {"dist": 0, "co2": 0}
        self.gains = {"dist": 0.0, "co2": 0.0}

    def notify(self, algorithm):
        if self.passes <= 0 or self.every <= 0:
            return
        if algorithm.n_gen % self.every != 0:
            return

        feasible = _feasible(algorithm.pop)
        if not feasible:
            return

        p = self.problem
        for axis, objective, weight in ((0, "dist", 1.0), (1, "co2", 0.0)):
            best = min(feasible, key=lambda ind: ind.get("F_raw")[axis])
            old = best.get("F_raw")

            sol = best.get("X")[0]
            for _ in range(self.passes):
                sol = self.dls.improve(sol, objective=objective, weight=weight)

            self.attempts[objective] += 1

            metrics = evaluate_solution(sol, p.vrp_data, p.Q_by_type,
                                        p.n_diesel, p.n_clean)
            if metrics is None or metrics[2] > 0:
                continue

            # Accept whenever the axis this corner is DEFINED by improves.
            if metrics[axis] < old[axis] - 1e-9:
                self.gains[objective] += float(old[axis] - metrics[axis])
                self.improvements[objective] += 1
                best.X[0] = sol
                best.X[1] = weight
                best.F = np.array([metrics[0], metrics[1]], dtype=float)
                best.set("F_raw", np.array([metrics[0], metrics[1], 0.0]))


class PreserveExtremes(Callback):
    """Remember the best-distance and best-CO2 solutions seen all run, and put
    them back if crowding selection has dropped them."""

    def __init__(self, problem):
        super().__init__()
        self.problem = problem
        self.best_distance = None
        self.best_co2 = None

    def notify(self, algorithm):
        pop = algorithm.pop

        for ind in _feasible(pop):
            raw = ind.get("F_raw")
            distance, co2 = float(raw[0]), float(raw[1])
            sol = ind.get("X")[0]

            if self.best_distance is None or distance < self.best_distance[1]:
                self.best_distance = (sol, distance, co2)
            if self.best_co2 is None or co2 < self.best_co2[2]:
                self.best_co2 = (sol, distance, co2)

        ranks = pop.get("rank")
        if ranks is None:
            return

        for best in (self.best_distance, self.best_co2):
            if best is None:
                continue

            sol, distance, co2 = best
            key = solution_key(sol)
            if any(solution_key(ind.get("X")[0]) == key for ind in pop):
                continue

            worst = int(np.argmax(ranks))
            pop[worst].X[0] = sol
            # Give the reinjected corner the direction that defines it.
            pop[worst].X[1] = 1.0 if best is self.best_distance else 0.0
            pop[worst].F = np.array([distance, co2], dtype=float)
            pop[worst].set("F_raw", np.array([distance, co2, 0.0]))


class ParetoArchive(Callback):
    """Every feasible point ever evaluated, filtered down to the non-dominated
    ones.

    NSGA-II reports only its final population, so a point found in generation 40
    and later crowded out is lost even though it was genuinely non-dominated.
    Filtering periodically keeps the O(n^2) filter off the critical path.
    """

    def __init__(self, prune_every=10):
        super().__init__()
        self.points = []
        self.prune_every = prune_every
        self._gen = 0

    def notify(self, algorithm):
        for ind in _feasible(algorithm.pop):
            raw = ind.get("F_raw")
            self.points.append((float(raw[0]), float(raw[1])))

        self._gen += 1
        if self._gen % self.prune_every == 0:
            self.points = pareto_filter(self.points)

    def front(self):
        return pareto_filter(self.points)


class FanOut(Callback):
    """pymoo's minimize() takes one callback."""

    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = callbacks

    def notify(self, algorithm):
        for cb in self.callbacks:
            cb.notify(algorithm)

class RecalibrateScales(Callback):
    """Refresh the weight normalisation from the front actually found so far.

    The spans that matter are the ones the run can reach, and those are not
    knowable before the run. Measuring them every few generations off the
    non-dominated set is self-correcting and costs nothing: F_raw is already
    computed by GreenVRPProblem._evaluate.
    """

    def __init__(self, dls, every=5, min_points=4):
        super().__init__()
        self.dls = dls
        self.every = every
        self.min_points = min_points

    def notify(self, algorithm):
        if self.every <= 0 or algorithm.n_gen % self.every != 0:
            return

        feasible = _feasible(algorithm.pop)
        if len(feasible) < self.min_points:
            return

        P = np.array([ind.get("F_raw")[:2] for ind in feasible], dtype=float)
        nd = [i for i in range(len(P))
              if not np.any(np.all(P <= P[i], axis=1)
                            & np.any(P < P[i], axis=1)
                            & (np.arange(len(P)) != i))]
        if len(nd) < self.min_points:
            return

        F = P[nd]
        d_span = float(F[:, 0].max() - F[:, 0].min())
        c_span = float(F[:, 1].max() - F[:, 1].min())
        if d_span > 1e-6 and c_span > 1e-6:
            self.dls.set_scales(d_span, c_span,
                                ref=(float(np.median(F[:, 0])),
                                     float(np.median(F[:, 1]))),
                                verbose=False)

def make_callbacks(dls, problem, extra=()):
    """The standard callback stack, plus anything the caller adds.

    The archive runs first, so each generation is recorded as it came out of
    survival, before the intensifiers overwrite individuals in place.
    Returns (FanOut, archive, extremes). The last two carry the counters
    worth printing at the end of a run.
    """
    archive = ParetoArchive()
    extremes = IntensifyExtremes(dls, problem,
                                 passes=EXTREME_INTENSIFY_PASSES,
                                 every=EXTREME_INTENSIFY_EVERY)
    stack = [archive, extremes,
             IntensifyElite(dls, problem, every=5, passes=1),
             PreserveExtremes(problem)]
    return FanOut(stack + list(extra)), archive, extremes
