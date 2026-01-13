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
) -> Tuple[Optional[List[Any]], float, List[float], List[Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
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
    best_cost = float("inf")
    history: List[float] = []

    def build_tour(start_time) -> Tuple[List[Any], float, List[Dict]]:
        order = [start]
        unvisited = set(unique_targets)
        total_cost = 0.0
        current_time = start_time
        arrival_log = []
        while unvisited:
            current = order[-1]
            choices = []
            for cand in unvisited:
                d = dist[(current, cand)]
                
                if weight == "street_time": 
                    travel_time_min = (d / 60) * 15
                else: 
                    AVG_SPEED_MS = 30 / 3.6
                    travel_time_min = (d / AVG_SPEED_MS) / 60
                
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
            total_cost += d
            
            current_time = arrival_time + unload_time
            
            window_status = "N/A"
            window_info = None
            if time_windows and nxt in time_windows:
                earliest, latest = time_windows[nxt]
                window_info = (earliest, latest)
                if current_time < earliest:
                    window_status = "TEMPRANO"
                elif current_time <= latest:
                    window_status = "A TIEMPO"
                else:
                    window_status = "TARDÍO"
            
            arrival_log.append({
                "node": nxt,
                "arrival_minutes": current_time,
                "window": window_info,
                "status": window_status,
            })
        order.append(start)
        total_cost += dist[(order[-2], start)]
        return order, total_cost, arrival_log

    best_arrival_log = []
    
    for _ in range(n_iters):
        iter_best_order = None
        iter_best_cost = float("inf")
        iter_best_log = []

        for _ in range(n_ants):
            order, cost, log = build_tour(start_time=start_time)
            if cost < iter_best_cost:
                iter_best_cost = cost
                iter_best_order = order
                iter_best_log = log

        for key in tau:
            tau[key] = max(tau_min, (1 - rho) * tau[key])

        if iter_best_order is not None:
            delta = q / iter_best_cost
            for a, b in zip(iter_best_order, iter_best_order[1:]):
                tau[(a, b)] = min(tau_max, tau[(a, b)] + delta)

            if iter_best_cost < best_cost:
                best_cost = iter_best_cost
                best_order = iter_best_order
                best_arrival_log = iter_best_log

        history.append(best_cost)

    if best_order is None:
        return None, float("inf"), history, [], [], [], []

    full_path: List[Any] = []
    tour_legs: List[Dict[str, Any]] = []
    for a, b in zip(best_order, best_order[1:]):
        leg = spaths[(a, b)]
        if full_path:
            leg = leg[1:]
        full_path.extend(leg)
        tour_legs.append({"from": a, "to": b, "distance": dist[(a, b)], "path": spaths[(a, b)]})

    pairwise = [{"from": a, "to": b, "distance": d} for (a, b), d in dist.items()]

    return best_order, best_cost, history, full_path, pairwise, tour_legs, best_arrival_log
