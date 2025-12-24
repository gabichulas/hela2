import osmnx as ox
import networkx as nx
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import pickle
import os
from math import radians, sin, cos, sqrt, atan2


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
