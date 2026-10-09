

import re
from decimal import Decimal, ROUND_HALF_UP

# ---------------------------------------------------------------
# BUDGET
# ---------------------------------------------------------------
BUDGET_SECONDS = 120          # wall-clock seconds PER METHOD

# ---------------------------------------------------------------
# INSTANCE
# ---------------------------------------------------------------
INSTANCE_DIR = "data/CVRP"
INSTANCE_NAME = "X-n115-k10.vrp"
INSTANCE_PATH = f"{INSTANCE_DIR}/{INSTANCE_NAME}"
SEED = 1

# ---------------------------------------------------------------
# EMISSION MODEL
# ---------------------------------------------------------------
# CO2 on an arc = GAMMA * F0 * (1 + ALPHA * load_ratio**BETA) * MULT[type] * distance
F0, ALPHA, BETA, GAMMA = 0.30, 0.20, 1.0, 2.61
MULT = {"diesel": 1.00, "clean": 0.45}
L_NOM = 0.50                  # load ratio assumed by the surrogate

TYPE_LABELS = {0: "diesel", 1: "clean"}

# ---------------------------------------------------------------
# FLEET
# ---------------------------------------------------------------
CLEAN_CAP_FRAC = 0.8          # clean capacity, as a fraction of diesel capacity
DIESEL_CAP_FRAC = 1.1         # diesel capacity, as a fraction of the file's CAPACITY


def customers_in(path):
    """Number of customers = DIMENSION - 1 (the depot)."""
    with open(path) as fh:
        for line in fh:
            m = re.match(r"\s*DIMENSION\s*:\s*(\d+)", line)
            if m:
                return int(m.group(1)) - 1
    raise ValueError(f"no DIMENSION header in {path}")


def derived_fleet_size(n_customers):
    """Half the fleet is clean, rounded so the two halves cover K0 = n."""
    clean = n_customers // 2
    return {"diesel": n_customers - clean, "clean": clean}


def _round_half_up(x):
    return int(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def derived_capacities(base_cap):
    """Qd from the file's CAPACITY, then Qc from Qd -- not from CAPACITY."""
    Qd = _round_half_up(base_cap * DIESEL_CAP_FRAC)
    Qc = _round_half_up(Qd * CLEAN_CAP_FRAC)
    return {"diesel": Qd, "clean": Qc}


try:
    K0 = customers_in(INSTANCE_PATH)
    _fleet = derived_fleet_size(K0)
    N_DIESEL, N_CLEAN = _fleet["diesel"], _fleet["clean"]
except (OSError, ValueError):
    K0, N_DIESEL, N_CLEAN = 0, 0, 0

# ---------------------------------------------------------------
# WEIGHTED SUM
# ---------------------------------------------------------------
N_WEIGHTS = 40                # the budget is split evenly across these
SCALE = 1000                  # integer scaling of the arc costs handed to PyVRP

# ---------------------------------------------------------------
# NSGA-II
# ---------------------------------------------------------------
POP_SIZE = 40                # one generation costs roughly POP_SIZE local searches

# ---------------------------------------------------------------
# OUTPUT (standalone runs only; run_experiments.py writes CSVs instead)
# ---------------------------------------------------------------
FRONT_WS = "front_weighted_sum.json"
