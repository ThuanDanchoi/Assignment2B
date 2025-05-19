"""
Graph generator script

This script creates a graph edge list based on geographic proximity using the
Haversine formula between all pairs of known locations. Result is saved as a CSV file.
"""

import pandas as pd
import math
from itertools import combinations

# ==== Load location data with coordinates ====
df = pd.read_csv("../data/processed/locations_with_latlon.csv")

# ==== Haversine distance function ====
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two latitude-longitude points.

    Args:
        lat1, lon1 (float): Latitude and longitude of point 1
        lat2, lon2 (float): Latitude and longitude of point 2

    Returns:
        float: Distance in kilometers between the two points
    """
    R = 6371  # Earth's radius in kilometers
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ==== Prepare nodes and edges ====
nodes = df[["location", "latitude", "longitude"]].dropna().reset_index(drop=True)
edges = []

# Threshold for maximum connection distance
threshold_km = 100  # only connect locations within 100 km

# Generate edges for every valid node pair within threshold
for i, j in combinations(nodes.index, 2):
    lat1 = nodes.loc[i, "latitude"]
    lon1 = nodes.loc[i, "longitude"]
    lat2 = nodes.loc[j, "latitude"]
    lon2 = nodes.loc[j, "longitude"]

    # Normalize node names (trim + uppercase)
    loc1 = nodes.loc[i, "location"].strip().upper()
    loc2 = nodes.loc[j, "location"].strip().upper()

    dist = haversine(lat1, lon1, lat2, lon2)

    if 0 < dist <= threshold_km:
        # Add bidirectional edge
        edges.append({"from": loc1, "to": loc2, "distance_km": dist})
        edges.append({"from": loc2, "to": loc1, "distance_km": dist})

# ==== Export edge list to CSV ====
edges_df = pd.DataFrame(edges)
edges_df.to_csv("data/processed/graph_edges.csv", index=False)

print(f"[INFO] Created graph_edges.csv with {len(edges_df)} valid edges.")
