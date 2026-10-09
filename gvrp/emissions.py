
import numpy as np


def demand_at_visit_id(vid, clients):
    """Demand of customer `vid`. Client lists are 0-based, visit ids are 1-based."""
    d = clients[vid - 1].delivery
    if isinstance(d, (list, tuple, np.ndarray)):
        return int(d[0]) if len(d) > 0 else 0
    return int(d)


def true_co2(sol_or_assignments, data, type_labels, Q_by_type,
             f0, alpha, beta, gamma, mult):
    """CO2 of a PyVRP Solution, or of a list of (visits, vehicle_type_index)."""
    if hasattr(sol_or_assignments, "routes"):
        routes = [(list(r.visits()), r.vehicle_type())
                  for r in sol_or_assignments.routes()]
    else:
        routes = sol_or_assignments

    clients = data.clients()
    D = data.distance_matrix(0)
    total = 0.0

    for visits_only, vt_idx in routes:
        name = type_labels[vt_idx]
        Qk = float(Q_by_type[name])

        # The vehicle starts loaded with everything the route delivers.
        remaining = float(sum(demand_at_visit_id(i, clients) for i in visits_only))
        seq = [0] + list(visits_only) + [0]

        for a, b in zip(seq[:-1], seq[1:]):
            ratio = 0.0 if Qk <= 0 else remaining / Qk
            factor = f0 * (1.0 + alpha * (ratio ** beta)) * mult[name]
            total += float(D[a][b]) * gamma * factor
            if b != 0:
                remaining -= demand_at_visit_id(b, clients)

    return float(total)
