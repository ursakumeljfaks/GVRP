"""Distance-only baseline: solve every instance in data/cvrp with pyvrp.solve,
using 1.1 x the capacity from the .vrp file.

Writes results_distance/distance.csv, one row per (instance, seed).
"""

import csv
import glob
import math
import os
import time

from pyvrp import read, solve
from pyvrp.stop import MaxRuntime

INSTANCE_DIR = "data/CVRP"
CAP_FACTOR = 1.1
SEEDS = [1]
RUNTIME_SECONDS = 120

OUT_DIR = "results_distance"
OUT_CSV = f"{OUT_DIR}/distance.csv"
COLS = ["instance", "seed", "orig_capacity", "capacity", "runtime_s",
        "feasible", "distance", "n_routes"]


def scaled_data(path):
    """Read an instance and scale every vehicle type's capacity by CAP_FACTOR."""
    data = read(path, round_func="round")
    orig = data.vehicle_types()[0].capacity[0]
    # Rounded down, so the inflated capacity is never exceeded.
    vtypes = [vt.replace(capacity=[math.floor(CAP_FACTOR * q) for q in vt.capacity])
              for vt in data.vehicle_types()]
    return data.replace(vehicle_types=vtypes), orig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = sorted(glob.glob(f"{INSTANCE_DIR}/*.vrp"))
    print(f"{len(paths)} instances x {len(SEEDS)} seeds x {RUNTIME_SECONDS}s")

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()

        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            data, orig_cap = scaled_data(path)
            cap = data.vehicle_types()[0].capacity[0]

            for seed in SEEDS:
                t0 = time.perf_counter()
                res = solve(data, stop=MaxRuntime(RUNTIME_SECONDS), seed=seed,
                            display=False)
                elapsed = time.perf_counter() - t0
                best = res.best

                w.writerow(dict(
                    instance=name, seed=seed, orig_capacity=orig_cap,
                    capacity=cap, runtime_s=round(elapsed, 1),
                    feasible=int(best.is_feasible()), distance=best.distance(),
                    n_routes=best.num_routes()))
                fh.flush()

                print(f"  {name:<12} seed {seed:>2}  Q {orig_cap}->{cap}  "
                      f"{elapsed:5.1f}s  dist={best.distance()}  "
                      f"routes={best.num_routes()}  "
                      f"{'' if best.is_feasible() else 'INFEASIBLE'}")

    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
