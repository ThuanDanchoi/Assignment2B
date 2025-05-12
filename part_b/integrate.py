"""
OSM subgraph construction and routing utilities for Traffic-Based Route Guidance System (TBRGS).

This module provides functions to:
  1. Ensure the project root is on sys.path for imports
  2. Load SCATS node coordinates from a cached pickle
  3. Compute Haversine distance between two GPS points
  4. Build or load an OSM subgraph around a SCATS origin-destination pair
  5. Assign travel-time weights to OSM graph edges
  6. Find the top-k shortest simple paths based on travel time
"""

import os
import sys
import math
import pickle

# Ensure project root is on PYTHONPATH for relative imports
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import osmnx as ox
import networkx as nx

# Paths for SCATS nodes pickle and OSM subgraph cache directory
SCATS_PKL    = os.path.join(PROJECT_ROOT, "part_b", "weighted_graph.pkl")
SUBGRAPH_DIR = os.path.join(PROJECT_ROOT, "part_b", "data", "osm_subgraphs")
os.makedirs(SUBGRAPH_DIR, exist_ok=True)


def load_scats_nodes() -> dict:
    """
    Load SCATS site coordinate dictionary from pickle.

    The pickle stores a tuple (nodes_dict, edges) where nodes_dict maps
    site_id to {'lat': float, 'lon': float}.

    Returns:
        dict: Mapping of site_id to coordinate dicts.
    """
    with open(SCATS_PKL, "rb") as f:
        nodes, _ = pickle.load(f)
    return nodes


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle distance between two points (lat/lon) in meters.

    Args:
        lat1, lon1 (float): Latitude and longitude of the first point.
        lat2, lon2 (float): Latitude and longitude of the second point.

    Returns:
        float: Distance between points in meters.
    """
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def build_osm_subgraph(
    nodes: dict,
    origin_site: int,
    dest_site: int,
    buffer_m: float = 2000,
    force_download: bool = False
) -> nx.MultiDiGraph:
    """
    Build or load an OSM subgraph around a SCATS origin-destination pair.

    The subgraph is centered at the midpoint between origin and destination,
    with radius = half the direct Haversine distance plus buffer_m.
    Results are cached to graphml files in SUBGRAPH_DIR.

    Args:
        nodes (dict): SCATS site coordinate mapping from load_scats_nodes().
        origin_site (int): SCATS site ID for origin.
        dest_site (int): SCATS site ID for destination.
        buffer_m (float): Additional buffer radius in meters.
        force_download (bool): If True, ignore cached file and re-download.

    Returns:
        networkx.MultiDiGraph: OSM street network around the OD pair.
    """
    filename = f"osm_{origin_site}_{dest_site}.graphml"
    filepath = os.path.join(SUBGRAPH_DIR, filename)
    if not force_download and os.path.exists(filepath):
        # Load cached graphml
        return ox.load_graphml(filepath)

    # Compute center point and radius
    o_coord = nodes[origin_site]
    d_coord = nodes[dest_site]
    mid_lat = (o_coord["lat"] + d_coord["lat"]) / 2
    mid_lon = (o_coord["lon"] + d_coord["lon"]) / 2
    distance = haversine_m(o_coord["lat"], o_coord["lon"],
                          d_coord["lat"], d_coord["lon"])
    radius = distance / 2 + buffer_m

    # Download OSM graph
    G = ox.graph_from_point((mid_lat, mid_lon), dist=radius, network_type="drive")
    ox.save_graphml(G, filepath)
    return G


def weight_osm_graph(
    G: nx.MultiDiGraph,
    default_speed_kmh: float = 60.0,
    delay_s: float = 30.0
) -> nx.MultiDiGraph:
    """
    Assign travel time weight to each edge in the OSM graph.

    travel_time = (length_km / default_speed_kmh) * 3600 + delay_s

    Args:
        G (MultiDiGraph): OSM graph with 'length' attribute (meters).
        default_speed_kmh (float): Free-flow speed limit in km/h.
        delay_s (float): Fixed delay per edge in seconds.

    Returns:
        MultiDiGraph: Graph with 'travel_time' attribute on each edge.
    """
    for u, v, key, data in G.edges(keys=True, data=True):
        length_km = data.get("length", 0.0) / 1000.0
        base_time = (length_km / default_speed_kmh) * 3600.0
        data["travel_time"] = base_time + delay_s
    return G


def find_k_paths(
    nodes: dict,
    origin_site: int,
    dest_site: int,
    k: int = 3
) -> list:
    """
    Find top-k shortest simple paths by travel time between two SCATS sites.

    Steps:
      1. Build or load the OSM subgraph.
      2. Map SCATS site IDs to nearest OSM nodes.
      3. Apply travel-time weights to the OSM graph.
      4. Convert MultiDiGraph to DiGraph, keeping minimal travel_time per edge.
      5. Generate up to k shortest simple paths weighted by 'travel_time'.

    Args:
        nodes (dict): SCATS site coordinate mapping.
        origin_site (int): SCATS origin site ID.
        dest_site (int): SCATS destination site ID.
        k (int): Number of top paths to return.

    Returns:
        list of tuples: Each tuple is (path_nodes_list, total_travel_time_s).
    """
    # 1) Build or load subgraph
    MG = build_osm_subgraph(nodes, origin_site, dest_site)

    # 2) Find nearest nodes in OSM graph
    o_node = ox.distance.nearest_nodes(
        MG, nodes[origin_site]["lon"], nodes[origin_site]["lat"]
    )
    d_node = ox.distance.nearest_nodes(
        MG, nodes[dest_site]["lon"], nodes[dest_site]["lat"]
    )

    # 3) Assign travel-time weights
    MG = weight_osm_graph(MG)

    # 4) Collapse MultiDiGraph into DiGraph using minimal travel_time per edge
    DG = nx.DiGraph()
    for u, v, key, data in MG.edges(keys=True, data=True):
        tt = data["travel_time"]
        if DG.has_edge(u, v):
            # keep the smaller travel time if multiple edges exist
            if tt < DG[u][v]["travel_time"]:
                DG[u][v]["travel_time"] = tt
        else:
            DG.add_edge(u, v, travel_time=tt)

    # 5) Generate top-k shortest simple paths
    results = []
    try:
        paths_gen = nx.shortest_simple_paths(DG, o_node, d_node, weight="travel_time")
        for idx, path in enumerate(paths_gen, start=1):
            if idx > k:
                break
            # sum travel_time along the path
            total_tt = sum(
                DG[a][b]["travel_time"] for a, b in zip(path[:-1], path[1:])
            )
            results.append((path, total_tt))
    except nx.NetworkXNoPath:
        # Return empty list if no path exists
        pass

    return results


if __name__ == "__main__":
    # Quick demonstration
    nodes = load_scats_nodes()
    print(f"SCATS sites loaded: {len(nodes)}")
    origin, dest = list(nodes)[:2]
    print(f"Building OSM subgraph for {origin} → {dest}…")
    G = build_osm_subgraph(nodes, origin, dest)
    print("Subgraph nodes/edges:", G.number_of_nodes(), G.number_of_edges())
    G = weight_osm_graph(G)
    print("Top-2 routes:")
    for path, tt in find_k_paths(nodes, origin, dest, k=2):
        print(path, f"{tt:.1f}s")