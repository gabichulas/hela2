from math import radians, sin, cos, sqrt, atan2
from typing import Tuple, Optional
import networkx as nx


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000
    
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
) -> bool:
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
        
    G.add_node(new_point_id, **new_node_attrs)
    
    first_edge_attrs = edge_data.copy()
    first_edge_attrs['length'] = dist_from_to_new
    
    if 'weight' in edge_data:
        first_edge_attrs['weight'] = edge_data['weight'] * proportion_from
    
    if 'travel_time' in edge_data:
        first_edge_attrs['travel_time'] = edge_data['travel_time'] * proportion_from
    
    G.add_edge(from_id, new_point_id, **first_edge_attrs)
    
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
