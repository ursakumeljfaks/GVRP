"""
Five-panel trade-off figures: one figure per instance with the NSGA-II
median-run Pareto front in the middle and four route maps around it, each
connected to its solution on the front by a curved arrow.

    +-------------+                         +-------------+
    | min distance| <---.             .---> | trade-off 2 |
    +-------------+      \  Pareto   /      +-------------+
                          ( front   )
    +-------------+      /  NSGA-II  \      +-------------+
    | trade-off 1 | <---'             '---> | min CO2     |
    +-------------+                         +-------------+

Chosen solutions (only from the NSGA-II median-HV run, only solutions that
have routes recorded in routes.csv):
    min-distance : lowest f1
    trade-off 1  : closest to 1/3 of the normalised f1 range of the front
    trade-off 2  : closest to 2/3 of the normalised f1 range of the front
    min-co2      : lowest f2

Run from the `code_gvrp` project root:

    python make_5_panel_trade_off_figures.py

Writes results/trade_off_figures_5panel/<instance>_memetic_NSGA-II.png and
results/trade_off_figures_5panel/chosen_solutions_summary.csv.
"""

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

# reuse the helpers from the existing trade-off script
from make_trade_off_figures import (
    RESULTS_DIR, VRP_DIR, COLOR_CLEAN, COLOR_DIESEL, COLOR_DEPOT,
    nondominated_mask, load_median_seeds, parse_vrp, build_route_xy,
    compute_shared_route_bounds,
)

OUT_DIR = os.path.join(RESULTS_DIR, "trade_off_figures_5panel")
ALGORITHM = "NSGA-II"                 # name in runs.csv
ALGORITHM_LABEL = "Memetic NSGA-II"    # name shown in the figure
FILE_TAG = "memetic_NSGA-II"           # added to the output file names

# --------------------------------------------------------------------------
# Trade-offs: order along the front (low distance -> low CO2), titles,
# colours (marker + arrow + panel title) and panel slot in the layout
# --------------------------------------------------------------------------
TRADE_OFFS = ["min-distance", "trade-off-1", "trade-off-2", "min-co2"]
TRADE_OFF_TITLE = {"min-distance": "Min distance", "trade-off-1": "Trade-off 1",
                   "trade-off-2": "Trade-off 2", "min-co2": "Min CO$_2$"}
TRADE_OFF_COLOR = {"min-distance": "#9467bd", "trade-off-1": "#2ca02c",
                   "trade-off-2": "#00441b", "min-co2": "#08306b"}
# (row, col) in the 2x3 grid; the front spans the whole middle column
SLOT = {"min-distance": (0, 0), "trade-off-1": (1, 0),
        "trade-off-2": (0, 2), "min-co2": (1, 2)}
INTERMEDIATE_TARGETS = {"trade-off-1": 1 / 3, "trade-off-2": 2 / 3}

COLOR_OTHER = "#9ecae1"   # other feasible solutions of the run
COLOR_FRONT = "#3a3a3a"   # Pareto-front points


# ==========================================================================
# Picking the four solutions
# ==========================================================================

def pick_trade_offs(cand):
    """cand: Pareto-front rows that have routes. Returns {label: row}."""
    cand = cand.sort_values("f1").reset_index(drop=True)
    picks = {"min-distance": cand.loc[cand["f1"].idxmin()],
             "min-co2": cand.loc[cand["f2"].idxmin()]}
    used = {picks["min-distance"]["solution_id"], picks["min-co2"]["solution_id"]}

    f1n = (cand["f1"] - cand["f1"].min()) / (cand["f1"].max() - cand["f1"].min() + 1e-12)
    for label, target in INTERMEDIATE_TARGETS.items():
        order = (f1n - target).abs().sort_values().index
        free = [i for i in order if cand.loc[i, "solution_id"] not in used]
        i = free[0] if free else order[0]   # tiny fronts: allow a repeat
        picks[label] = cand.loc[i]
        used.add(cand.loc[i, "solution_id"])
    return picks


# ==========================================================================
# Panels
# ==========================================================================

def draw_front_panel(ax, inst, seed, run_pts, front, picks):
    others = run_pts[~run_pts["solution_id"].isin(front["solution_id"])]
    if len(others):
        ax.scatter(others["f1"], others["f2"], s=12, color=COLOR_OTHER, alpha=0.7,
                   label="Other feasible solutions", zorder=2)
    ax.scatter(front["f1"], front["f2"], s=16, color=COLOR_FRONT,
               label="Pareto front", zorder=3)
    for label in TRADE_OFFS:
        row = picks[label]
        ax.scatter([row["f1"]], [row["f2"]], s=70, color=TRADE_OFF_COLOR[label],
                   edgecolors="black", linewidths=0.7, zorder=5, label=TRADE_OFF_TITLE[label])
    ax.set_title(f"{inst} — {ALGORITHM_LABEL}\nPareto front (seed {seed})", fontsize=11)
    ax.set_xlabel("Total distance $f_1$")
    ax.set_ylabel("CO$_2$ emissions $f_2$ (kg)")
    ax.legend(fontsize=8, loc="lower left", frameon=True)  # always empty for a front
    ax.grid(alpha=0.3, color="0.88", linewidth=0.6)


def draw_route_panel(ax, label, row, routes_sub, vrp_coords, depot_id, xlim, ylim):
    depot_xy = vrp_coords[depot_id]
    ax.scatter([depot_xy[0]], [depot_xy[1]], marker="*", s=180, c=COLOR_DEPOT,
               edgecolors="black", linewidths=0.6, zorder=6)

    for _, r in routes_sub.iterrows():
        pts = build_route_xy(r["sequence"], vrp_coords, depot_id)
        vtype = r["vehicle_type"]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-",
                color=COLOR_CLEAN if vtype == "clean" else COLOR_DIESEL,
                linewidth=1.0, alpha=0.9, marker="o", markersize=1.8, zorder=3)

    n_clean = int((routes_sub["vehicle_type"] == "clean").sum())
    n_diesel = int((routes_sub["vehicle_type"] == "diesel").sum())
    ax.set_title(f"{TRADE_OFF_TITLE[label]}\n"
                 f"dist = {row['f1']:.0f}, CO$_2$ = {row['f2']:.0f} | "
                 f"{n_clean} clean, {n_diesel} diesel",
                 fontsize=9.5, color=TRADE_OFF_COLOR[label])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(TRADE_OFF_COLOR[label])
        spine.set_linewidth(1.6)


def add_route_legend(fig, ax_front):
    """One shared legend for all route maps, in the free space below the
    Pareto-front panel, so it never covers a route."""
    handles = [
        Line2D([], [], marker="*", linestyle="none", markersize=13, color=COLOR_DEPOT,
               markeredgecolor="black", markeredgewidth=0.6, label="Depot"),
        Line2D([], [], color=COLOR_DIESEL, marker="o", markersize=3, label="Diesel vehicle route"),
        Line2D([], [], color=COLOR_CLEAN, marker="o", markersize=3, label="Clean vehicle route"),
    ]
    pos = ax_front.get_position()
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=((pos.x0 + pos.x1) / 2, pos.y0 - 0.075),
               ncol=1, fontsize=10, frameon=True)


def add_arrow(fig, ax_front, ax_route, row, label, col):
    """Curved arrow from the chosen point on the front to the inner edge of
    its route panel (right edge for left panels, left edge for right panels)."""
    on_left = col == 0
    target = (1.0, 0.5) if on_left else (0.0, 0.5)
    # bend so the arrow arcs outward, like the reference figure
    rad = 0.3 if on_left else -0.3
    arrow = ConnectionPatch(
        xyA=(row["f1"], row["f2"]), coordsA=ax_front.transData,
        xyB=target, coordsB=ax_route.transAxes,
        arrowstyle="-|>", mutation_scale=16, linewidth=1.6,
        color=TRADE_OFF_COLOR[label], shrinkA=5, shrinkB=2,
        connectionstyle=f"arc3,rad={rad}", zorder=10,
    )
    fig.add_artist(arrow)


def make_figure(inst, seed, run_pts, front, picks, routes, vrp_coords, depot_id,
                xlim, ylim, out_path):
    # all five panels are the same size: route maps in the four corners of a
    # 2x3 grid, the Pareto front in the middle column, centred vertically
    fig = plt.figure(figsize=(16, 10.4))
    gs = fig.add_gridspec(2, 3, wspace=0.28, hspace=0.22)
    top, bottom = gs[0, 1].get_position(fig), gs[1, 1].get_position(fig)
    height = top.height
    y0 = (top.y1 + bottom.y0) / 2 - height / 2
    ax_front = fig.add_axes([top.x0, y0, top.width, height])
    ax_front.set_box_aspect(1)
    draw_front_panel(ax_front, inst, seed, run_pts, front, picks)

    for label in TRADE_OFFS:
        r, c = SLOT[label]
        ax = fig.add_subplot(gs[r, c])
        ax.set_box_aspect(1)
        row = picks[label]
        routes_sub = routes[routes["solution_id"] == row["solution_id"]].sort_values("route_index")
        draw_route_panel(ax, label, row, routes_sub, vrp_coords, depot_id, xlim, ylim)
        add_arrow(fig, ax_front, ax, row, label, c)

    add_route_legend(fig, ax_front)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# Main
# ==========================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    runs = pd.read_csv(os.path.join(RESULTS_DIR, "runs.csv"))
    routes = pd.read_csv(os.path.join(RESULTS_DIR, "routes.csv"))
    route_ids = set(routes["solution_id"].unique())

    instances = sorted(runs["instance"].unique())
    median_seeds = load_median_seeds(runs, instances)

    parsed = {inst: parse_vrp(os.path.join(VRP_DIR, f"{inst}.vrp")) for inst in instances}
    xlim, ylim = compute_shared_route_bounds({i: c for i, (c, _) in parsed.items()})

    summary_rows = []
    for inst in instances:
        vrp_coords, depot_id = parsed[inst]
        ms = median_seeds[(median_seeds.instance == inst) & (median_seeds.algorithm == ALGORITHM)]
        seed = int(ms["seed"].iloc[0])

        run_pts = runs[(runs.instance == inst) & (runs.algorithm == ALGORITHM)
                       & (runs.seed == seed)].reset_index(drop=True)
        front = run_pts[nondominated_mask(run_pts[["f1", "f2"]].to_numpy())]
        cand = front[front["solution_id"].isin(route_ids)]
        if len(cand) == 0:
            print(f"skipping {inst}: no front solution of seed {seed} has routes recorded")
            continue

        picks = pick_trade_offs(cand)
        out_path = os.path.join(OUT_DIR, f"{inst}_{FILE_TAG}.png")
        make_figure(inst, seed, run_pts, front, picks, routes, vrp_coords, depot_id,
                    xlim, ylim, out_path)
        print("wrote", out_path)

        for label in TRADE_OFFS:
            row = picks[label]
            summary_rows.append(dict(instance=inst, trade_off=label, solution_id=row["solution_id"],
                                     seed=seed, total_distance=row["f1"], co2=row["f2"]))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(OUT_DIR, "chosen_solutions_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
