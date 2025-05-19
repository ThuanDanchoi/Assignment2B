"""
Graph loader utility

This module provides functionality to load a graph structure from CSV files
containing edge and optional coordinate (latitude, longitude) data.
"""

import pandas as pd
from part_a.graph import Graph

def load_graph_from_csv(csv_path, latlon_path=None, origin=None, destinations=None):
    """
    Load a Graph object from edge list CSV and optional coordinates CSV.

    Args:
        csv_path (str): Path to the CSV file containing 'from', 'to', and 'distance_km'.
        latlon_path (str, optional): Path to CSV with 'location', 'longitude', 'latitude'.
        origin (str, optional): Starting node ID.
        destinations (list of str, optional): List of destination node IDs.

    Returns:
        Graph: An instance of the Graph class populated with nodes, edges, origin, and destinations.
    """
    df = pd.read_csv(csv_path)

    # Normalize node names (strip whitespace, uppercase)
    df['from'] = df['from'].str.strip().str.upper()
    df['to'] = df['to'].str.strip().str.upper()

    # Create edge dictionary: {(from, to): distance}
    edges = {
        (row['from'], row['to']): row['distance_km']
        for _, row in df.iterrows()
    }

    # Extract unique nodes from both 'from' and 'to' columns
    unique_nodes = set(df['from']).union(set(df['to']))

    # Load coordinates if provided, else default to (0, 0)
    if latlon_path:
        latlon_df = pd.read_csv(latlon_path)
        latlon_df['location'] = latlon_df['location'].str.strip().str.upper()

        # Map location name to (longitude, latitude)
        coord_map = {
            row['location']: (row['longitude'], row['latitude'])
            for _, row in latlon_df.iterrows()
        }

        nodes = {
            node: coord_map.get(node, (0, 0))  # fallback if coordinate not found
            for node in unique_nodes
        }
    else:
        nodes = {node: (0, 0) for node in unique_nodes}

    # Normalize and validate origin
    if origin:
        origin = origin.strip().upper()
        if origin not in nodes:
            print(f"[WARNING] Origin node '{origin}' not found in graph.")
            origin = None

    # Normalize and validate destination nodes
    if destinations:
        destinations = [d.strip().upper() for d in destinations]
        for dest in destinations:
            if dest not in nodes:
                print(f"[WARNING] Destination node '{dest}' not found in graph.")
        destinations = [d for d in destinations if d in nodes]

    # Return the constructed Graph object
    return Graph(nodes, edges, origin, destinations)
