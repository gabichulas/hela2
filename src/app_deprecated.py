import base64
import math
import os 
from pathlib import Path
from io import BytesIO
from pathlib import Path
from typing import Optional, List

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.templating import Jinja2Templates
from shapely.geometry import Point
from sqlmodel import Session, create_engine, select, SQLModel
from models import DistributionCenter

from utils.graph_utils import (
    get_graph_by_city,
    get_graph_by_point,
    get_graph_stats,
    get_ice_cream_places,
    aco_tour_through_nodes,
)


app = FastAPI(
    title="Hela2 OSM",
    description="Frontend minimo para generar mapas y estadisticas con OSMnx",
)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

data_dir = Path(__file__).parent.parent / "data"
data_dir.mkdir(exist_ok=True)

db_file = data_dir / "centros.db"
engine = create_engine(f"sqlite:///{db_file}")
SQLModel.metadata.create_all(engine)

def render_graph_image(G) -> str:
    fig, ax = ox.plot_graph(
        G,
        node_size=5,
        edge_linewidth=0.5,
        figsize=(10, 10),
        show=False,
        close=False,
    )

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    plt.close(fig)

    encoded = base64.b64encode(buffer.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_graph_with_poi(G, poi_nodes: Optional[list] = None) -> str:
    fig, ax = ox.plot_graph(
        G,
        node_size=4,
        node_color="#e2e8f0",
        edge_color="#475569",
        edge_linewidth=0.7,
        bgcolor="#0b1220",
        figsize=(10, 10),
        show=False,
        close=False,
    )
    if poi_nodes:
        poi_nodes = list(dict.fromkeys(poi_nodes))
        xs, ys = [], []
        for n in poi_nodes:
            data = G.nodes.get(n, {})
            x = data.get("x")
            y = data.get("y")
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)
        if xs:
            ax.scatter(xs, ys, c="#ef4444", s=60, zorder=8, edgecolors="white", linewidths=1.5, alpha=0.95)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    plt.close(fig)
    encoded = base64.b64encode(buffer.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_graph_route_image(G, route_nodes, poi_nodes: Optional[list] = None) -> str:
    fig, ax = ox.plot_graph(
        G,
        node_size=4,
        node_color="#cbd5e1",
        edge_color="#334155",
        edge_linewidth=0.7,
        bgcolor="#0b1220",
        figsize=(10, 10),
        show=False,
        close=False,
    )

    if route_nodes:
        coords = [(G.nodes[n].get("x"), G.nodes[n].get("y")) for n in route_nodes if "x" in G.nodes[n] and "y" in G.nodes[n]]
        if len(coords) >= 2:
            xs, ys = zip(*coords)
            ax.plot(xs, ys, color="#f97316", linewidth=3, zorder=6, alpha=0.95)
            ax.scatter([xs[0]], [ys[0]], c="#22c55e", s=80, zorder=8, edgecolors="white", linewidths=1.5)
            ax.scatter([xs[-1]], [ys[-1]], c="#ae00ff", s=80, zorder=8, edgecolors="white", linewidths=1.5)
            ax.scatter(xs, ys, c="#fef08a", s=8, zorder=6, edgecolors="none", alpha=0.8)

    if poi_nodes:
        poi_nodes = list(dict.fromkeys(poi_nodes))
        xs = []
        ys = []
        for n in poi_nodes:
            data = G.nodes.get(n, {})
            x = data.get("x")
            y = data.get("y")
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)
        if xs:
            ax.scatter(xs, ys, c="#ef4444", s=60, zorder=8, edgecolors="white", linewidths=1.5, alpha=0.95)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    plt.close(fig)

    encoded = base64.b64encode(buffer.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def project_and_nearest_node(G, lon: float, lat: float) -> int:
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
        raise ValueError("No se encontró un nodo cercano en el grafo proyectado")
    return closest


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def build_graph(
    mode: str,
    city: Optional[str],
    latitude: Optional[float],
    longitude: Optional[float],
    radius: int,
    network_type: str,
):
    if mode == "coords":
        if latitude is None or longitude is None:
            raise HTTPException(status_code=400, detail="Faltan coordenadas (latitud y longitud).")
        return get_graph_by_point(latitude, longitude, radius, network_type=network_type)

    if not city:
        raise HTTPException(status_code=400, detail="Falta la ciudad o punto de interes.")
    return get_graph_by_city(city, radius, network_type=network_type)


@app.post("/graph")
async def graph(
    mode: str = Form("city"),
    city: Optional[str] = Form(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    radius: int = Form(1000),
    network_type: str = Form("drive"),
):
    G = build_graph(mode, city, latitude, longitude, radius, network_type)
    stats = get_graph_stats(G)
    image = render_graph_image(G)

    return {"stats": stats, "image": image}


@app.post("/centros")
async def crear_centro(centro: DistributionCenter):
    with Session(engine) as session:
        session.add(centro)
        session.commit()
        session.refresh(centro)
        return centro
    
@app.get("/centros")
async def listar_centros():
    with Session(engine) as session:
        return session.exec(select(DistributionCenter)).all()

@app.post("/heladerias")
async def heladerias(
    mode: str = Form("city"),
    city: Optional[str] = Form(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    radius: int = Form(1000),
):
    """
    Devuelve heladerías (amenity=ice_cream) dentro del radio dado.
    Reutiliza los mismos parámetros de ubicación que /graph.
    """
    try:
        df = get_ice_cream_places(city, latitude, longitude, radius)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al consultar heladerías: {exc}")

    records = df.to_dict(orient="records")
    return {"count": len(records), "places": records}


@app.post("/aco-heladerias")
async def aco_heladerias(
    mode: str = Form("city"),
    city: Optional[str] = Form(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    radius: int = Form(1000),
    network_type: str = Form("drive"),
    origin_latitude: float = Form(...),
    origin_longitude: float = Form(...),
    weight: str = Form("length"),
    n_ants: int = Form(30),
    n_iters: int = Form(80),
    alpha: float = Form(1.0),
    beta: float = Form(2.0),
    rho: float = Form(0.5),
    q: float = Form(100.0),
    start_time: Optional[float] = Form(None),
    gamma: float = Form(0.7),
    unload_time: int = Form(10),  # Tiempo de descarga en minutos
    generate_windows: str = Form("false"),
):
    """
    Ejecuta ACO tipo TSP para visitar todas las heladerías en el radio indicado.
    Usa origen (lat/lon) y las heladerías se mapean al nodo más cercano del grafo.
    """
    G = build_graph(mode, city, latitude, longitude, radius, network_type)

    try:
        origin_node = project_and_nearest_node(G, origin_longitude, origin_latitude)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"No se pudo mapear origen a nodo: {exc}")

    try:
        df = get_ice_cream_places(city, latitude, longitude, radius)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al consultar heladerías: {exc}")

    if df.empty:
        raise HTTPException(status_code=404, detail="No se encontraron heladerías en el radio indicado")

    target_nodes: List[int] = []
    node_labels: Dict[int, str] = {}
    for _, row in df.iterrows():
        node = project_and_nearest_node(G, row["lon"], row["lat"])
        target_nodes.append(node)
        label_parts = []
        if isinstance(row.get("name"), str) and row["name"].strip():
            label_parts.append(row["name"].strip())
        street = row.get("addr:street")
        housenumber = row.get("addr:housenumber")
        addr = " ".join([str(street or ""), str(housenumber or "")]).strip()
        if addr:
            label_parts.append(addr)
        label = " - ".join(label_parts) if label_parts else f"Heladería {row.get('osmid', node)}"
        node_labels[node] = node_labels.get(node, label)  # conserva primer nombre útil

    # Filtrar solo la componente conectada del nodo origen
    G_ud = G.to_undirected()
    connected = set(nx.node_connected_component(G_ud, origin_node))
    reachable_targets = [n for n in target_nodes if n in connected and n != origin_node]
    dropped = len(target_nodes) - len(reachable_targets)
    if not reachable_targets:
        raise HTTPException(
            status_code=404,
            detail="Las heladerías están fuera de la componente conectada del origen. Aumenta el radio o ajusta la ubicación.",
        )
    # Mantener labels solo para los alcanzables
    node_labels = {n: node_labels.get(n, f"Heladería {n}") for n in reachable_targets}

    # Generar ventanas de tiempo aleatorias si está habilitado
    time_windows = None
    if generate_windows.lower() == "true":
        import random
        time_windows = {}
        for node in reachable_targets:
            # Generar ventanas de 2 horas entre las 8:00 y 18:00
            earliest = random.randint(8*60, 16*60)  # 8:00 a 16:00
            latest = earliest + random.randint(120, 240)  # +2 a +4 horas
            time_windows[node] = (earliest, latest)

    try:
        order, cost, history, full_path, pairwise, tour_legs, arrival_log = aco_tour_through_nodes(
            G,
            start=origin_node,
            targets=reachable_targets,
            weight=weight,
            n_ants=n_ants,
            n_iters=n_iters,
            alpha=alpha,
            beta=beta,
            rho=rho,
            q=q,
            start_time=start_time if start_time is not None else 0,
            gamma=gamma,
            time_windows=time_windows,
            unload_time=unload_time,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error durante ACO: {exc}")

    if order is None or not full_path:
        raise HTTPException(status_code=500, detail="ACO no encontró una ruta válida")

    origin_label = "Centro de distribución"
    order_labels = [origin_label] + [node_labels.get(n, str(n)) for n in order[1:-1]] + [origin_label]

    # Mapa de labels para todos los nodos relevantes
    label_map = {origin_node: origin_label}
    for n in reachable_targets:
        label_map[n] = node_labels.get(n, f"Heladería {n}")

    pairwise_labeled = [
        {
            **p,
            "from_label": label_map.get(p["from"], str(p["from"])),
            "to_label": label_map.get(p["to"], str(p["to"])),
        }
        for p in pairwise
    ]
    tour_legs_labeled = [
        {
            **l,
            "from_label": label_map.get(l["from"], str(l["from"])),
            "to_label": label_map.get(l["to"], str(l["to"])),
        }
        for l in tour_legs
    ]

    base_image = render_graph_with_poi(G, poi_nodes=reachable_targets)
    image = render_graph_route_image(G, full_path, poi_nodes=reachable_targets)

    # Formatear arrival_log con labels y tiempo legible
    formatted_log = []
    for i, entry in enumerate(arrival_log, start=1):
        node = entry["node"]
        arrival_minutes = entry["arrival_minutes"]
        
        # Formatear tiempo como HH:MM
        hours = int(arrival_minutes // 60)
        minutes = int(arrival_minutes % 60)
        arrival_time_str = f"{hours:02d}:{minutes:02d}"
        
        # Formatear ventana si existe
        window_str = "N/A"
        if entry["window"] is not None:
            earliest, latest = entry["window"]
            e_hours, e_mins = int(earliest // 60), int(earliest % 60)
            l_hours, l_mins = int(latest // 60), int(latest % 60)
            window_str = f"{e_hours:02d}:{e_mins:02d} - {l_hours:02d}:{l_mins:02d}"
        
        formatted_log.append({
            "sequence": i,
            "node": node,
            "label": node_labels.get(node, f"Heladería {node}"),
            "arrival_time": arrival_time_str,
            "arrival_minutes": round(arrival_minutes, 2),
            "window": window_str,
            "status": entry["status"],
        })

    return {
        "origin_node": origin_node,
        "target_nodes": list(dict.fromkeys(reachable_targets)),
        "order": order,  # ya incluye el origen al final
        "order_labels": order_labels,
        "path": full_path,
        "cost": cost,
        "history": history,
        "count_targets": len(set(reachable_targets)),
        "dropped_targets": dropped,
        "pairwise": pairwise_labeled,
        "tour_legs": tour_legs_labeled,
        "label_map": label_map,
        "base_image": base_image,
        "image": image,
        "arrival_log": formatted_log,
        "time_windows_enabled": generate_windows.lower() == "true",
    }
