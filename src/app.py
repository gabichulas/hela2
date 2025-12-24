import base64
from io import BytesIO
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import osmnx as ox
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.templating import Jinja2Templates

from utils.graph_utils import (
    get_graph_by_city,
    get_graph_by_point,
    get_graph_stats,
)


app = FastAPI(
    title="Hela2 OSM",
    description="Frontend minimo para generar mapas y estadisticas con OSMnx",
)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def render_graph_image(G) -> str:
    """
    Renderiza el grafo en memoria y devuelve una data URI base64 (PNG).
    """
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
