import osmnx as ox
import networkx as nx
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import pickle
import os


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