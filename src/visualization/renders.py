import base64
from io import BytesIO
from typing import Optional, List
import matplotlib.pyplot as plt
import osmnx as ox


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
