import osmnx as ox
import networkx as nx
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
import pickle
import os
import random
from math import radians, sin, cos, sqrt, atan2


SPEED_DEFAULTS = {
    'motorway': 120,
    'trunk': 100,
    'primary': 80,
    'secondary': 60,
    'tertiary': 50,
    'residential': 30,
    'service': 20,
    'living_street': 20,
    'unclassified': 40,
}

def get_graph_by_city(
    city: str,
    radius: int,
    network_type: str = "drive",
    simplify: bool = True,
    use_cache: bool = True,
    cache_dir: str = "cache"
) -> nx.MultiDiGraph:
    
    cache_name = f"{city.replace(',', '').replace(' ', '_')}_{radius}m_{network_type}.pkl"
    cache_path = os.path.join(cache_dir, cache_name)
    
    if os.path.exists(cache_path):
        print(f"Loading graph from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print(f"Downloading graph for {city} with radius {radius}m...")
    
    location = ox.geocode(city)
    G = ox.graph_from_point(
        location,
        dist=radius,
        network_type=network_type,
        simplify=simplify
    )
    
    for u, v, k, data in G.edges(keys=True, data=True):
        length = data.get('length', 0)
        maxspeed = data.get('maxspeed')
        
        if isinstance(maxspeed, list): maxspeed = maxspeed[0]
        if maxspeed is None:
            highway_type = data.get('highway', 'unclassified')
            if isinstance(highway_type, list):
                highway_type = highway_type[0]
            maxspeed = SPEED_DEFAULTS.get(highway_type, 50)

        street_time = length / (maxspeed / 3.6)
        G[u][v][k]['street_time'] = street_time
    
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(G, f)
        print(f"Graph saved to cache: {cache_path}")
    
    return G


def get_graph_by_point(
    latitude: float,
    longitude: float,
    radius: int,
    network_type: str = "drive",
    simplify: bool = True,
    use_cache: bool = True,
    cache_dir: str = "cache"
) -> nx.MultiDiGraph:

    cache_name = f"point_{latitude}_{longitude}_{radius}m_{network_type}.pkl"
    cache_path = os.path.join(cache_dir, cache_name)
    
    if os.path.exists(cache_path):
        print(f"Loading graph from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print(f"Downloading graph for point ({latitude}, {longitude}) with radius {radius}m...")
    
    G = ox.graph_from_point(
        (latitude, longitude),
        dist=radius,
        network_type=network_type,
        simplify=simplify
    )
    
    for u, v, k, data in G.edges(keys=True, data=True):
        length = data.get('length', 0)
        maxspeed = data.get('maxspeed')
        
        if isinstance(maxspeed, list): maxspeed = maxspeed[0]
        if maxspeed is None:
            highway_type = data.get('highway', 'unclassified')
            if isinstance(highway_type, list):
                highway_type = highway_type[0]
            maxspeed = SPEED_DEFAULTS.get(highway_type, 50)

        street_time = length / (maxspeed / 3.6)
        G[u][v][k]['street_time'] = street_time
        
        
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(G, f)
        print(f"Graph saved to cache: {cache_path}")
    
    return G


def get_graph_stats(G: nx.MultiDiGraph) -> Dict[str, Any]:
    
    stats = ox.basic_stats(G)
    
    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "total_length_km": stats.get('edge_length_total', 0) / 1000,
        "avg_length_m": stats.get('edge_length_avg', 0),
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
        "density": nx.density(G),
        "nodes_with_coords": all('x' in data and 'y' in data for _, data in G.nodes(data=True))
    }


def graph_to_csv(
    G: nx.MultiDiGraph,
    nodes_path: str = "data/nodes.csv",
    edges_path: str = "data/edges.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    nodes_data = []
    for node_id, data in G.nodes(data=True):
        node = {'node_id': node_id}
        node.update(data)
        nodes_data.append(node)
    
    df_nodes = pd.DataFrame(nodes_data)
    
    edges_data = []
    for u, v, k, data in G.edges(keys=True, data=True):
        edge = {
            'source': u,
            'target': v,
            'key': k
        }
        edge.update(data)
        edges_data.append(edge)
    
    df_edges = pd.DataFrame(edges_data)
    
    os.makedirs(os.path.dirname(nodes_path), exist_ok=True)
    os.makedirs(os.path.dirname(edges_path), exist_ok=True)
    
    df_nodes.to_csv(nodes_path, index=False)
    df_edges.to_csv(edges_path, index=False)
    
    print(f"Nodes saved to: {nodes_path}")
    print(f"Edges saved to: {edges_path}")
    
    return df_nodes, df_edges


def plot_graph(G: nx.MultiDiGraph, save_path: Optional[str] = None):

    fig, ax = ox.plot_graph(
        G,
        node_size=5,
        edge_linewidth=0.5,
        figsize=(12, 12),
        save=save_path is not None,
        filepath=save_path,
        show=save_path is None
    )
    
    return fig, ax


def get_ice_cream_places(
    city: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    radius: int = 1000,
    cache_dir: str = "cache",
) -> pd.DataFrame:
    """
    Devuelve un DataFrame con heladerías (amenity=ice_cream) en un radio dado.
    Si se indica city, se geocodifica; si no, usa lat/lon.
    """
    if city:
        lat, lon = ox.geocode(city)
    else:
        if latitude is None or longitude is None:
            raise ValueError("Debes indicar una ciudad o coordenadas (latitud y longitud).")
        lat, lon = latitude, longitude

    ox.settings.use_cache = True
    ox.settings.cache_folder = cache_dir

    gdf = ox.features_from_point((lat, lon), tags={"amenity": "ice_cream"}, dist=radius)
    if gdf.empty:
        return pd.DataFrame(columns=["osmid", "name", "lat", "lon", "addr:street", "addr:housenumber", "amenity"])

    df = gdf.reset_index()
    df["lat"] = df.geometry.y
    df["lon"] = df.geometry.x

    cols = ["osmid", "name", "lat", "lon", "addr:street", "addr:housenumber", "amenity"]
    for col in cols:
        if col not in df.columns:
            df[col] = None
    df = df[cols]
    df = df.where(pd.notnull(df), None)  # Reemplaza NaN por None para JSON
    return df


def aco_tour_through_nodes(
    G: nx.MultiDiGraph,
    start: Any,
    targets: List[Any],
    weight: str = "street_time",
    start_time: int = 10*60, # 10 AM
    time_windows: Optional[Dict[Any, Tuple[int, int]]] = None,
    n_ants: int = 30,
    n_iters: int = 80,
    alpha: float = 1.0,
    beta: float = 2.0,
    gamma: float = 0.7, # Urgency factor weight
    rho: float = 0.5,
    q: float = 100.0,
    tau_bounds: Tuple[float, float] = (1e-4, 1.0),
    seed: Optional[int] = None,
) -> Tuple[Optional[List[Any]], float, List[float], List[Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    ACO para recorrer todos los nodos objetivo (heladerías) en una sola pasada.
    Se modela como TSP sobre el conjunto (start + targets) usando distancias de camino más corto.

    Returns:
        tour_order: orden de nodos visitados (start + heladerías) en grafo original.
        tour_cost: costo total (suma de distancias entre nodos consecutivos).
        history: mejor costo por iteración.
        full_path: secuencia expandida de nodos del grafo para dibujar la ruta completa.
        pairwise: lista de distancias precomputadas entre pares.
        tour_legs: lista de segmentos usados por el tour con su camino detallado.
    """
    if start not in G:
        raise ValueError("El nodo de inicio no existe en el grafo")
    unique_targets = [t for t in dict.fromkeys(targets) if t != start]
    if not unique_targets:
        raise ValueError("Se necesita al menos una heladería destino")
    nodes = [start] + unique_targets

    tau_min, tau_max = tau_bounds
    rng = random.Random(seed)

    # Precalcula las rutas más cortas y las distancias entre todos los pares de nodos en los nodos
    dist: Dict[Tuple[Any, Any], float] = {}
    spaths: Dict[Tuple[Any, Any], List[Any]] = {}
    for i in nodes:
        lengths, paths = nx.single_source_dijkstra(G, i, weight=weight) #TODO: Add weight to streets
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

    def build_tour(start_time) -> Tuple[List[Any], float]: # TODO: Implement time windows on frontend
        order = [start]
        unvisited = set(unique_targets)
        total_cost = 0.0
        current_time = start_time
        while unvisited:
            current = order[-1]
            choices = []
            for cand in unvisited:
                d = dist[(current, cand)]
                
                if weight == "street_time": arrival_time = current_time + ((d/60)*2) # dist = seconds
                else: 
                    AVG_SPEED_MS = 30 / 3.6
                    arrival_time = current_time + (d / AVG_SPEED_MS) / 60
                    
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
                else: urgency_factor = 1.0
                        
                pher = tau[(current, cand)]
                heuristic = 1.0 / d if d > 0 else 1e6
                score = (pher ** alpha) * (heuristic ** beta) * (urgency_factor ** gamma)
                choices.append((score, cand, d))
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
            _, nxt, d = chosen
            order.append(nxt)
            unvisited.remove(nxt)
            total_cost += d
            
            if weight == "street_time": current_time += (d / 60) * 2
            else: current_time += (d / (30 / 3.6)) / 60
        # Cierre del circuito explícito: volver al inicio
        order.append(start)
        total_cost += dist[(order[-2], start)]
        return order, total_cost

    for _ in range(n_iters):
        iter_best_order = None
        iter_best_cost = float("inf")

        for _ in range(n_ants):
            order, cost = build_tour(start_time=start_time)
            if cost < iter_best_cost:
                iter_best_cost = cost
                iter_best_order = order

        # Evaporación
        for key in tau:
            tau[key] = max(tau_min, (1 - rho) * tau[key])

        if iter_best_order is not None:
            delta = q / iter_best_cost
            for a, b in zip(iter_best_order, iter_best_order[1:]):
                tau[(a, b)] = min(tau_max, tau[(a, b)] + delta)

            if iter_best_cost < best_cost:
                best_cost = iter_best_cost
                best_order = iter_best_order

        history.append(best_cost)

    if best_order is None:
        return None, float("inf"), history, [], [], []

    # Reconstruir path completo en el grafo original concatenando caminos más cortos
    full_path: List[Any] = []
    tour_legs: List[Dict[str, Any]] = []
    for a, b in zip(best_order, best_order[1:]):
        leg = spaths[(a, b)]
        if full_path:
            leg = leg[1:]  # evitar repetir nodo de unión
        full_path.extend(leg)
        tour_legs.append({"from": a, "to": b, "distance": dist[(a, b)], "path": spaths[(a, b)]})

    pairwise = [{"from": a, "to": b, "distance": d} for (a, b), d in dist.items()]

    return best_order, best_cost, history, full_path, pairwise, tour_legs
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    
    R = 6371000  # Earth radius in meters
    
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c


def insert_point_between_nodes(
    G: nx.MultiDiGraph,
    from_id: int,
    to_id: int,
    new_point_id: int,
    new_lat: float,
    new_lon: float,
    new_name: str = None,
    new_type: str = "local"
) -> Dict[str, Any]:

    if not G.has_node(from_id):
        raise ValueError(f"Node {from_id} does not exist in graph")
    if not G.has_node(to_id):
        raise ValueError(f"Node {to_id} does not exist in graph")
    
    if not G.has_edge(from_id, to_id):
        raise ValueError(f"No edge exists from {from_id} to {to_id}")
    
    edge_data = G.get_edge_data(from_id, to_id)
    
    from_node = G.nodes[from_id]
    to_node = G.nodes[to_id]
    
    from_lat = from_node.get('y', from_node.get('lat'))
    from_lon = from_node.get('x', from_node.get('lon'))
    to_lat = to_node.get('y', to_node.get('lat'))
    to_lon = to_node.get('x', to_node.get('lon'))
    
    if any(coord is None for coord in [from_lat, from_lon, to_lat, to_lon]):
        raise ValueError("Nodes must have lat/lon or y/x coordinates")
    
    dist_from_to_new = haversine_distance(from_lat, from_lon, new_lat, new_lon)
    dist_new_to_to = haversine_distance(new_lat, new_lon, to_lat, to_lon)
    total_dist = dist_from_to_new + dist_new_to_to
    
    if total_dist == 0:
        raise ValueError("New point coincides with existing nodes")
    
    proportion_from = dist_from_to_new / total_dist
    proportion_to = dist_new_to_to / total_dist
    
    new_node_attrs = {
        'y': new_lat, 
        'x': new_lon, 
        'lat': new_lat,
        'lon': new_lon,
    }
    
    if new_name:
        new_node_attrs['name'] = new_name
    if new_type:
        new_node_attrs['type'] = new_type
        
    G.add_node(new_point_id, **new_node_attrs) # TODO: check if new_point_id already exists
    
    # Create first edge: from_id -> new_point_id
    first_edge_attrs = edge_data.copy()
    first_edge_attrs['length'] = dist_from_to_new
    
    if 'weight' in edge_data:
        first_edge_attrs['weight'] = edge_data['weight'] * proportion_from
    
    if 'travel_time' in edge_data:
        first_edge_attrs['travel_time'] = edge_data['travel_time'] * proportion_from
    
    G.add_edge(from_id, new_point_id, **first_edge_attrs)
    
    # Create second edge: new_point_id -> to_id
    second_edge_attrs = edge_data.copy()
    second_edge_attrs['length'] = dist_new_to_to
    
    if 'weight' in edge_data:
        second_edge_attrs['weight'] = edge_data['weight'] * proportion_to
    
    if 'travel_time' in edge_data:
        second_edge_attrs['travel_time'] = edge_data['travel_time'] * proportion_to
    
    G.add_edge(new_point_id, to_id, **second_edge_attrs)
    
    G.remove_edge(from_id, to_id)
    
    return True


def find_nearest_edge(
    G: nx.MultiDiGraph,
    lat: float,
    lon: float,
    return_dist: bool = True
) -> Tuple[int, int, int, Optional[float]]:

    min_dist = float('inf')
    nearest_edge = None
    
    for u, v, k in G.edges(keys=True):
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        
        u_lat = u_data.get('y', u_data.get('lat'))
        u_lon = u_data.get('x', u_data.get('lon'))
        v_lat = v_data.get('y', v_data.get('lat'))
        v_lon = v_data.get('x', v_data.get('lon'))
        
        if any(coord is None for coord in [u_lat, u_lon, v_lat, v_lon]):
            continue
        
        dist_to_u = haversine_distance(lat, lon, u_lat, u_lon)
        dist_to_v = haversine_distance(lat, lon, v_lat, v_lon)
        
        # Approximate distance to edge
        edge_dist = min(dist_to_u, dist_to_v)
        
        if edge_dist < min_dist:
            min_dist = edge_dist
            nearest_edge = (u, v, k)
    
    if nearest_edge is None:
        raise ValueError("No valid edges found in graph")
    
    if return_dist:
        return (*nearest_edge, min_dist)
    else:
        return (*nearest_edge, None)
