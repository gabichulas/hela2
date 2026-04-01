from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import networkx as nx
import osmnx as ox
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.graph import get_graph_by_city, get_graph_by_point
from src.core.osm import get_ice_cream_places
from src.core.algorithms import aco_tour_through_nodes


@dataclass(frozen=True)
class Scenario:
    name: str
    mode: str  # city | coords
    city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    radius: int = 1000
    network_type: str = "drive"
    max_targets: int = 12


@dataclass(frozen=True)
class Config:
    name: str
    weight: str
    n_ants: int
    n_iters: int
    alpha: float
    beta: float
    gamma: float
    rho: float
    q: float
    unload_time: int
    default_demand: float
    capacity: Optional[float]
    max_operation_time: Optional[float]
    penalty_alpha: float
    penalty_beta: float
    penalty_gamma: float
    lambda_weight: float
    mu_weight: float
    time_factor: float
    min_travel_minutes: float
    length_speed_kph: float


def project_and_nearest_node(G: nx.MultiDiGraph, lon: float, lat: float) -> int:
    G_proj = ox.project_graph(G)
    pt_proj, _ = ox.projection.project_geometry(Point(lon, lat), to_crs=G_proj.graph["crs"])
    x, y = pt_proj.x, pt_proj.y

    closest = None
    best_dist = float("inf")
    for node, data in G_proj.nodes(data=True):
        nx_coord = data.get("x")
        ny_coord = data.get("y")
        if nx_coord is None or ny_coord is None:
            continue
        d = math.hypot(nx_coord - x, ny_coord - y)
        if d < best_dist:
            best_dist = d
            closest = node

    if closest is None:
        raise RuntimeError("No se encontro nodo cercano en grafo proyectado")
    return closest


def build_graph(s: Scenario) -> nx.MultiDiGraph:
    if s.mode == "city":
        if not s.city:
            raise ValueError("Scenario city sin city")
        return get_graph_by_city(s.city, s.radius, network_type=s.network_type)
    if s.mode == "coords":
        if s.latitude is None or s.longitude is None:
            raise ValueError("Scenario coords sin lat/lon")
        return get_graph_by_point(s.latitude, s.longitude, s.radius, network_type=s.network_type)
    raise ValueError(f"Modo de scenario invalido: {s.mode}")


def get_origin_latlon(s: Scenario) -> Tuple[float, float]:
    if s.mode == "city":
        if not s.city:
            raise ValueError("Scenario city sin city")
        lat, lon = ox.geocode(s.city)
        return float(lat), float(lon)
    if s.latitude is None or s.longitude is None:
        raise ValueError("Scenario coords sin lat/lon")
    return float(s.latitude), float(s.longitude)


def map_targets(
    G: nx.MultiDiGraph,
    s: Scenario,
    origin_node: int,
) -> List[int]:
    if s.mode == "city":
        df = get_ice_cream_places(city=s.city, radius=s.radius)
    else:
        df = get_ice_cream_places(latitude=s.latitude, longitude=s.longitude, radius=s.radius)

    if df.empty:
        raise RuntimeError(f"Sin heladerias para escenario {s.name}")

    mapped = []
    for _, row in df.iterrows():
        try:
            n = project_and_nearest_node(G, float(row["lon"]), float(row["lat"]))
            mapped.append(n)
        except Exception:
            continue

    mapped = list(dict.fromkeys(mapped))
    connected = set(nx.node_connected_component(G.to_undirected(), origin_node))
    mapped = [n for n in mapped if n in connected and n != origin_node]

    if len(mapped) < 4:
        raise RuntimeError(f"Escenario {s.name}: targets alcanzables insuficientes ({len(mapped)})")

    return mapped[: s.max_targets]


def build_time_windows(targets: List[int], run_seed: int) -> Dict[int, Tuple[int, int]]:
    rng = random.Random(run_seed)
    tw = {}
    for node in targets:
        earliest = rng.randint(8 * 60, 11 * 60)
        latest = earliest + rng.randint(240, 420)
        tw[node] = (earliest, latest)
    return tw


def build_configs() -> List[Config]:
    base = Config(
        name="length_base",
        weight="length",
        n_ants=30,
        n_iters=80,
        alpha=1.0,
        beta=2.0,
        gamma=1.0,
        rho=0.5,
        q=100.0,
        unload_time=10,
        default_demand=1.0,
        capacity=20.0,
        max_operation_time=600.0,
        penalty_alpha=1.0,
        penalty_beta=1.0,
        penalty_gamma=1.0,
        lambda_weight=1.0,
        mu_weight=1.0,
        time_factor=5.0,
        min_travel_minutes=1.5,
        length_speed_kph=18.0,
    )

    length_configs = [
        base,
        Config(**{**base.__dict__, "name": "length_ants20", "n_ants": 20}),
        Config(**{**base.__dict__, "name": "length_ants40", "n_ants": 40}),
        Config(**{**base.__dict__, "name": "length_iters50", "n_iters": 50}),
        Config(**{**base.__dict__, "name": "length_iters120", "n_iters": 120}),
        Config(**{**base.__dict__, "name": "length_alpha0.8", "alpha": 0.8}),
        Config(**{**base.__dict__, "name": "length_alpha1.2", "alpha": 1.2}),
        Config(**{**base.__dict__, "name": "length_beta1.5", "beta": 1.5}),
        Config(**{**base.__dict__, "name": "length_beta2.5", "beta": 2.5}),
        Config(**{**base.__dict__, "name": "length_gamma0.7", "gamma": 0.7}),
        Config(**{**base.__dict__, "name": "length_gamma1.3", "gamma": 1.3}),
        Config(**{**base.__dict__, "name": "length_rho0.3", "rho": 0.3}),
        Config(**{**base.__dict__, "name": "length_rho0.7", "rho": 0.7}),
    ]

    time_base = Config(**{**base.__dict__, "name": "street_time_base", "weight": "street_time"})
    street_time_configs = [
        time_base,
        Config(**{**time_base.__dict__, "name": "street_time_ants20", "n_ants": 20}),
        Config(**{**time_base.__dict__, "name": "street_time_ants40", "n_ants": 40}),
        Config(**{**time_base.__dict__, "name": "street_time_iters50", "n_iters": 50}),
        Config(**{**time_base.__dict__, "name": "street_time_iters120", "n_iters": 120}),
        Config(**{**time_base.__dict__, "name": "street_time_alpha0.8", "alpha": 0.8}),
        Config(**{**time_base.__dict__, "name": "street_time_alpha1.2", "alpha": 1.2}),
        Config(**{**time_base.__dict__, "name": "street_time_beta1.5", "beta": 1.5}),
        Config(**{**time_base.__dict__, "name": "street_time_beta2.5", "beta": 2.5}),
        Config(**{**time_base.__dict__, "name": "street_time_gamma0.7", "gamma": 0.7}),
        Config(**{**time_base.__dict__, "name": "street_time_gamma1.3", "gamma": 1.3}),
        Config(**{**time_base.__dict__, "name": "street_time_rho0.3", "rho": 0.3}),
        Config(**{**time_base.__dict__, "name": "street_time_rho0.7", "rho": 0.7}),
    ]

    configs = length_configs + street_time_configs

    return configs


def summarize_arrival(arrival_log: List[Dict[str, Any]]) -> Tuple[float, int, int, int]:
    if not arrival_log:
        return 0.0, 0, 0, 0

    on_time = sum(1 for e in arrival_log if e.get("status") == "A TIEMPO")
    early = sum(1 for e in arrival_log if e.get("status") == "TEMPRANO")
    late = sum(1 for e in arrival_log if e.get("status") == "TARDÍO")
    pct_on_time = 100.0 * on_time / len(arrival_log)
    return pct_on_time, on_time, early, late


def run_experiments(
    scenarios: List[Scenario],
    configs: List[Config],
    runs: int,
    start_time: int,
    enable_time_windows: bool,
    out_csv: Path,
    global_seed: int,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for s_idx, s in enumerate(scenarios):
        print(f"[Scenario] {s.name}")
        G = build_graph(s)

        origin_lat, origin_lon = get_origin_latlon(s)
        origin_node = project_and_nearest_node(G, origin_lon, origin_lat)
        targets = map_targets(G, s, origin_node)

        for run_id in range(1, runs + 1):
            run_seed = global_seed + (s_idx * 100000) + run_id
            tw = build_time_windows(targets, run_seed) if enable_time_windows else None

            for c in configs:
                algo_seed = run_seed + (abs(hash(c.name)) % 10000)
                t0 = time.perf_counter()

                try:
                    (
                        cost,
                        history,
                        arrival_log,
                        metrics,
                    ) = aco_tour_through_nodes(
                        G,
                        start=origin_node,
                        targets=targets,
                        weight=c.weight,
                        start_time=start_time,
                        time_windows=tw,
                        unload_time=c.unload_time,
                        n_ants=c.n_ants,
                        n_iters=c.n_iters,
                        alpha=c.alpha,
                        beta=c.beta,
                        gamma=c.gamma,
                        rho=c.rho,
                        q=c.q,
                        seed=algo_seed,
                        default_demand=c.default_demand,
                        capacity=c.capacity,
                        max_operation_time=c.max_operation_time,
                        penalty_weights=(c.penalty_alpha, c.penalty_beta, c.penalty_gamma),
                        objective_weights=(c.lambda_weight, c.mu_weight),
                        time_factor=c.time_factor,
                        min_travel_minutes=c.min_travel_minutes,
                        length_speed_kph=c.length_speed_kph,
                    )

                    elapsed = time.perf_counter() - t0
                    pct_on_time, on_time, early, late = summarize_arrival(arrival_log)

                    rows.append(
                        {
                            "scenario": s.name,
                            "run": run_id,
                            "config": c.name,
                            "status": "ok",
                            "exec_s": elapsed,
                            "targets": len(targets),
                            "weight": c.weight,
                            "n_ants": c.n_ants,
                            "n_iters": c.n_iters,
                            "alpha": c.alpha,
                            "beta": c.beta,
                            "gamma": c.gamma,
                            "rho": c.rho,
                            "q": c.q,
                            "unload_time": c.unload_time,
                            "capacity": c.capacity,
                            "max_operation_time": c.max_operation_time,
                            "penalty_alpha": c.penalty_alpha,
                            "penalty_beta": c.penalty_beta,
                            "penalty_gamma": c.penalty_gamma,
                            "lambda_weight": c.lambda_weight,
                            "mu_weight": c.mu_weight,
                            "time_factor": c.time_factor,
                            "min_travel_minutes": c.min_travel_minutes,
                            "length_speed_kph": c.length_speed_kph,
                            "objective_cost": float(metrics.get("objective_cost", cost)),
                            "distance_total": float(metrics.get("distance_total", 0.0)),
                            "penalty_total": float(metrics.get("penalty_total", 0.0)),
                            "penalty_window": float(metrics.get("penalty_window", 0.0)),
                            "penalty_capacity": float(metrics.get("penalty_capacity", 0.0)),
                            "penalty_time": float(metrics.get("penalty_time", 0.0)),
                            "total_time": float(metrics.get("total_time", 0.0)),
                            "pct_on_time": pct_on_time,
                            "arrivals_on_time": on_time,
                            "arrivals_early": early,
                            "arrivals_late": late,
                            "history_last": float(history[-1]) if history else float("nan"),
                        }
                    )
                except Exception as exc:
                    elapsed = time.perf_counter() - t0
                    rows.append(
                        {
                            "scenario": s.name,
                            "run": run_id,
                            "config": c.name,
                            "status": "fail",
                            "error": str(exc),
                            "exec_s": elapsed,
                            "weight": c.weight,
                            "gamma": c.gamma,
                        }
                    )

    fields = sorted({k for r in rows for k in r.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Resultado guardado en: {out_csv}")
    print(f"Filas: {len(rows)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulador de experimentos ACO para HELA2")
    p.add_argument("--runs", type=int, default=30, help="Corridas por configuracion y escenario")
    p.add_argument("--start-time", type=int, default=8 * 60, help="Minutos desde medianoche")
    p.add_argument("--seed", type=int, default=20260319, help="Semilla global")
    p.add_argument("--no-time-windows", action="store_true", help="Desactiva ventanas temporales")
    p.add_argument("--out", type=Path, default=Path("data/sim_results/aco_experiments_full.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    scenarios = [
        Scenario(name="E1_small", mode="city", city="Plaza Independencia, Mendoza", radius=900, max_targets=10),
        Scenario(name="E2_medium", mode="city", city="Plaza Independencia, Mendoza", radius=1200, max_targets=14),
        Scenario(name="E3_large", mode="city", city="Plaza Independencia, Mendoza", radius=1800, max_targets=20),
    ]

    configs = build_configs()

    run_experiments(
        scenarios=scenarios,
        configs=configs,
        runs=args.runs,
        start_time=args.start_time,
        enable_time_windows=not args.no_time_windows,
        out_csv=args.out,
        global_seed=args.seed,
    )


if __name__ == "__main__":
    main()
