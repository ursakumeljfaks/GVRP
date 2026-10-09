
import json
import time

import numpy as np
from pyvrp import Model, read
from pyvrp.stop import MaxRuntime

from . import config as C
from .emissions import true_co2

# Surrogate CO2 per unit distance, one value per vehicle type.
co2_per_dist = {
    k: C.GAMMA * C.F0 * (1 + C.ALPHA * (C.L_NOM ** C.BETA)) * C.MULT[k]
    for k in C.MULT
}

# (w_dist, w_co2) pairs, from pure distance to pure CO2. Built at import time,
# so set C.N_WEIGHTS before importing this module.
weights = [(1 - w, w) for w in np.linspace(0, 1, C.N_WEIGHTS)]


# ============================================================
# SOLVING
# ============================================================

def build_model_two_types(data, Q_diesel, Q_clean):
    """Turn the single-type instance into a diesel + clean fleet."""
    model = Model.from_data(data)
    model.vehicle_types[0] = model.vehicle_types[0].replace(
        num_available=C.N_DIESEL, capacity=[Q_diesel], name="diesel",
    )
    model.add_vehicle_type(
        num_available=C.N_CLEAN, capacity=[Q_clean], name="clean",
    )
    return model


def solve_weighted_surrogate(data, w_dist, w_co2, Q_diesel, Q_clean, runtime):
    """Solve one scalarisation and return the best solution found."""
    model = build_model_two_types(data, Q_diesel, Q_clean)

    # The scalarised cost of an arc is proportional to its distance, so it can
    # be carried entirely by unit_distance_cost. Integers, hence C.SCALE.
    unit_diesel = int(round(C.SCALE * (w_dist + w_co2 * co2_per_dist["diesel"])))
    unit_clean = int(round(C.SCALE * (w_dist + w_co2 * co2_per_dist["clean"])))

    model.vehicle_types[0] = model.vehicle_types[0].replace(
        fixed_cost=0, unit_distance_cost=unit_diesel)
    model.vehicle_types[1] = model.vehicle_types[1].replace(
        fixed_cost=0, unit_distance_cost=unit_clean)

    res = model.solve(stop=MaxRuntime(runtime), display=False, seed=C.SEED)
    return res.best


def true_co2_of_solution(sol, data, Q_by_type):
    """f2 of a solved scalarisation."""
    return true_co2(sol, data, C.TYPE_LABELS, Q_by_type,
                    C.F0, C.ALPHA, C.BETA, C.GAMMA, C.MULT)


def pareto_front(points):
    """Keep the distinct, non-dominated (distance, co2) pairs."""
    front = [
        p for p in points
        if not any(q[0] <= p[0] and q[1] <= p[1] and q != p for q in points)
    ]
    return sorted(set(front))


# ============================================================
# STANDALONE RUN (run_experiments.py is the real entry point)
# ============================================================

def main():
    data = read(C.INSTANCE_PATH, round_func="round")
    base_cap = Model.from_data(data).vehicle_types[0].capacity[0]
    Q_by_type = C.derived_capacities(base_cap)

    per_weight = C.BUDGET_SECONDS / C.N_WEIGHTS
    print(f"Weighted sum: {C.N_WEIGHTS} weights x {per_weight:.2f}s "
          f"= {C.BUDGET_SECONDS}s")

    points = []
    start = time.time()

    for i, (w_dist, w_co2) in enumerate(weights, start=1):
        if time.time() - start >= C.BUDGET_SECONDS:
            print(f"  budget exhausted after {i - 1} weights")
            break

        sol = solve_weighted_surrogate(data, w_dist, w_co2,
                                       Q_by_type["diesel"], Q_by_type["clean"],
                                       per_weight)
        dist = float(sol.distance())
        co2 = true_co2_of_solution(sol, data, Q_by_type)
        points.append((dist, co2))
        print(f"  {i:3d}/{C.N_WEIGHTS}  w_co2={w_co2:.3f}  "
              f"dist={dist:.0f}  co2={co2:.1f}")

    elapsed = time.time() - start
    front = pareto_front(points)
    print(f"\nRuntime {elapsed:.1f}s | evaluated {len(points)} | "
          f"nondominated {len(front)}")

    with open(C.FRONT_WS, "w") as fh:
        json.dump({"method": "weighted_sum", "elapsed": elapsed,
                   "n_evaluated": len(points), "front": front}, fh, indent=2)
    print(f"Front written to {C.FRONT_WS}")
    return front


if __name__ == "__main__":
    main()
