import networkx as nx
import random
from typing import Tuple, Optional, Dict, Any, List


def aco_tour_through_nodes(
    G: nx.MultiDiGraph,
    start: Any,
    targets: List[Any],
    weight: str = "street_time",
    start_time: int = 10*60,
    time_windows: Optional[Dict[Any, Tuple[int, int]]] = None,
    unload_time: int = 10,
    n_ants: int = 30,
    n_iters: int = 80,
    alpha: float = 1.0,
    beta: float = 2.0,
    gamma: float = 0.7,
    rho: float = 0.5,
    q: float = 100.0,
    tau_bounds: Tuple[float, float] = (1e-4, 1.0),
    seed: Optional[int] = None,
    demands: Optional[Dict[Any, float]] = None,
    default_demand: float = 1.0,
    capacity: Optional[float] = None,
    max_operation_time: Optional[float] = None,
    penalty_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    objective_weights: Tuple[float, float] = (1.0, 1.0),
) -> Tuple[
    Optional[List[Any]],
    float,
    List[float],
    List[Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    if start not in G:
        raise ValueError("El nodo de inicio no existe en el grafo")
    unique_targets = [t for t in dict.fromkeys(targets) if t != start]
    if not unique_targets:
        raise ValueError("Se necesita al menos una heladería destino")
    nodes = [start] + unique_targets

    tau_min, tau_max = tau_bounds
    rng = random.Random(seed)

    dist: Dict[Tuple[Any, Any], float] = {}
    spaths: Dict[Tuple[Any, Any], List[Any]] = {}
    for i in nodes:
        lengths, paths = nx.single_source_dijkstra(G, i, weight=weight)
        for j in nodes:
            if i == j:
                continue
            if j not in lengths:
                raise ValueError(f"No hay camino entre {i} y {j}")
            dist[(i, j)] = lengths[j]
            spaths[(i, j)] = paths[j]

    tau = {(i, j): max(tau_min, 1e-3) for i in nodes for j in nodes if i != j}

    best_order: Optional[List[Any]] = None
    best_objective = float("inf")
    best_distance = float("inf")
    best_penalty_total = 0.0
    best_penalty_window = 0.0
    best_penalty_capacity = 0.0
    best_penalty_time = 0.0
    best_total_time = 0.0
    history: List[float] = []

    penalty_alpha, penalty_beta, penalty_gamma = penalty_weights
    lambda_weight, mu_weight = objective_weights
    node_demands = {n: default_demand for n in unique_targets}
    if demands:
        for n, v in demands.items():
            if n in node_demands:
                node_demands[n] = float(v)
    total_demand = sum(node_demands.values())
    penalty_capacity = max(0.0, total_demand - capacity) if capacity is not None else 0.0

    def travel_time_minutes(distance: float) -> float:
        if weight == "street_time":
            return (distance / 60) * 15
        avg_speed_ms = 30 / 3.6
        return (distance / avg_speed_ms) / 60

    def build_tour(start_time) -> Tuple[List[Any], float, float, float, List[Dict[str, Any]]]:
        order = [start]
        unvisited = set(unique_targets)
        total_distance = 0.0
        penalty_window = 0.0
        current_time = start_time
        arrival_log = []
        while unvisited:
            current = order[-1]
            choices = []
            for cand in unvisited:
                d = dist[(current, cand)]

                travel_time_min = travel_time_minutes(d)
                arrival_time = current_time + travel_time_min
                    
                if time_windows and cand in time_windows:
                    earliest, latest = time_windows[cand]    
                    if arrival_time < earliest:
                        urgency_factor = 0.9
                    elif arrival_time <= latest:
                        slack = latest - arrival_time
                        max_slack  = latest - earliest
                        normalized_slack = slack / max_slack if max_slack > 0 else 0
                        urgency_factor = 2.0 - 1.0 * normalized_slack
                    else:
                        delay = arrival_time - latest
                        max_acceptable_delay = 30
                        penalty = min(delay / max_acceptable_delay, 1.0)
                        urgency_factor = 0.5 - 0.3 * penalty
                else:
                    urgency_factor = 1.0
                        
                pher = tau[(current, cand)]
                heuristic = 1.0 / d if d > 0 else 1e6
                score = (pher ** alpha) * (heuristic ** beta) * (urgency_factor ** gamma)
                choices.append((score, cand, d, travel_time_min, arrival_time))
                
            total_score = sum(c[0] for c in choices)
            if total_score <= 0:
                chosen = rng.choice(choices)
            else:
                r = rng.random() * total_score
                accum = 0.0
                chosen = choices[-1]
                for item in choices:
                    accum += item[0]
                    if accum >= r:
                        chosen = item
                        break
            _, nxt, d, travel_time_min, arrival_time = chosen
            order.append(nxt)
            unvisited.remove(nxt)
            total_distance += d

            window_status = "N/A"
            window_info = None
            wait_minutes = 0.0
            if time_windows and nxt in time_windows:
                earliest, latest = time_windows[nxt]
                window_info = (earliest, latest)
                penalty_window += max(0.0, arrival_time - latest) + max(0.0, earliest - arrival_time)
                if arrival_time < earliest:
                    window_status = "TEMPRANO"
                    wait_minutes = earliest - arrival_time
                elif arrival_time <= latest:
                    window_status = "A TIEMPO"
                else:
                    window_status = "TARDÍO"
            service_start = arrival_time + wait_minutes
            depart_time = service_start + unload_time
            current_time = depart_time

            arrival_log.append({
                "node": nxt,
                "arrival_minutes": arrival_time,
                "service_start_minutes": service_start,
                "depart_minutes": depart_time,
                "wait_minutes": wait_minutes,
                "window": window_info,
                "status": window_status,
            })
        last_node = order[-1]
        order.append(start)
        total_distance += dist[(last_node, start)]
        current_time += travel_time_minutes(dist[(last_node, start)])
        total_time = current_time - start_time
        return order, total_distance, penalty_window, total_time, arrival_log

    best_arrival_log = []
    
    for _ in range(n_iters):
        iter_best_order = None
        iter_best_objective = float("inf")
        iter_best_distance = float("inf")
        iter_best_penalty_total = 0.0
        iter_best_penalty_window = 0.0
        iter_best_penalty_time = 0.0
        iter_best_total_time = 0.0
        iter_best_log = []

        for _ in range(n_ants):
            order, distance_cost, penalty_window, total_time, log = build_tour(start_time=start_time)
            penalty_time = max(0.0, total_time - max_operation_time) if max_operation_time is not None else 0.0
            penalty_total = (
                penalty_alpha * penalty_window
                + penalty_beta * penalty_capacity
                + penalty_gamma * penalty_time
            )
            objective_cost = lambda_weight * distance_cost + mu_weight * penalty_total
            if objective_cost < iter_best_objective:
                iter_best_objective = objective_cost
                iter_best_order = order
                iter_best_log = log
                iter_best_distance = distance_cost
                iter_best_penalty_total = penalty_total
                iter_best_penalty_window = penalty_window
                iter_best_penalty_time = penalty_time
                iter_best_total_time = total_time

        for key in tau:
            tau[key] = max(tau_min, (1 - rho) * tau[key])

        if iter_best_order is not None:
            delta = q / max(iter_best_objective, 1e-6)
            for a, b in zip(iter_best_order, iter_best_order[1:]):
                tau[(a, b)] = min(tau_max, tau[(a, b)] + delta)

            if iter_best_objective < best_objective:
                best_objective = iter_best_objective
                best_order = iter_best_order
                best_arrival_log = iter_best_log
                best_distance = iter_best_distance
                best_penalty_total = iter_best_penalty_total
                best_penalty_window = iter_best_penalty_window
                best_penalty_capacity = penalty_capacity
                best_penalty_time = iter_best_penalty_time
                best_total_time = iter_best_total_time

        history.append(best_objective)

    if best_order is None:
        return None, float("inf"), history, [], [], [], [], {}

    full_path: List[Any] = []
    tour_legs: List[Dict[str, Any]] = []
    for a, b in zip(best_order, best_order[1:]):
        leg = spaths[(a, b)]
        if full_path:
            leg = leg[1:]
        full_path.extend(leg)
        tour_legs.append({"from": a, "to": b, "distance": dist[(a, b)], "path": spaths[(a, b)]})

    pairwise = [{"from": a, "to": b, "distance": d} for (a, b), d in dist.items()]

    metrics = {
        "distance_total": best_distance,
        "penalty_total": best_penalty_total,
        "penalty_window": best_penalty_window,
        "penalty_capacity": best_penalty_capacity,
        "penalty_time": best_penalty_time,
        "objective_cost": best_objective,
        "total_time": best_total_time,
        "total_demand": total_demand,
        "capacity": capacity,
        "max_operation_time": max_operation_time,
        "penalty_weights": {
            "alpha": penalty_alpha,
            "beta": penalty_beta,
            "gamma": penalty_gamma,
        },
        "objective_weights": {
            "lambda": lambda_weight,
            "mu": mu_weight,
        },
    }

    return best_order, best_objective, history, full_path, pairwise, tour_legs, best_arrival_log, metrics
