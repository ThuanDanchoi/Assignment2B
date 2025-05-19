"""
Route tester script

This script loads a traffic graph, updates edge costs using a specified ML model,
and runs a selected pathfinding algorithm to print the predicted route and travel time.
"""

import sys
import os
import joblib
from tensorflow.keras.models import load_model

# Add project root to system path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from part_a.graph import Graph
from part_a.algorithms import astar, bfs, dfs, gbfs, cus1, cus2
from part_b.graph_loader import load_graph_from_csv
from part_b.graph_updater import update_graph_with_ml

# ==== Configurations ====
GRAPH_CSV_PATH = "part_b/data/processed/graph_edges.csv"
LATLON_PATH    = "part_b/data/processed/locations_with_latlon.csv"
VOLUME_PATH    = "part_b/data/processed/cleaned_scats_data.csv"

ALGORITHM  = "gbfs"     # Choose from: astar, bfs, dfs, gbfs, cus1, cus2
MODEL_NAME = "lstm"     # Choose from: lstm, gru, xgb
MODEL_PATH = f"part_b/models/{MODEL_NAME}/{MODEL_NAME}.h5"

# Start and destination nodes for test
start_node = "CAMBERWELL_RD NW OF TRAFALGAR_RD"
end_node   = "WARRIGAL_RD N OF HIGH STREET_RD"

# ==== Load Graph and Coordinates ====
print("\n[INFO] Loading graph and updating with ML predictions...")
graph = load_graph_from_csv(
    csv_path=GRAPH_CSV_PATH,
    latlon_path=LATLON_PATH,
    origin=start_node,
    destinations=[end_node]
)

# ==== Load Machine Learning Model ====
model_ext = os.path.splitext(MODEL_PATH)[1]
if model_ext == ".h5":
    model = load_model(MODEL_PATH, compile=False)
else:
    model = joblib.load(MODEL_PATH)

# ==== Update Edge Weights ====
graph = update_graph_with_ml(
    graph=graph,
    graph_csv_path=GRAPH_CSV_PATH,
    volume_path=VOLUME_PATH,
    model_path=None,
    model=model
)

# ==== Execute Search Algorithm ====
ALGO_MAP = {
    "astar": astar.search,
    "bfs": bfs.search,
    "dfs": dfs.search,
    "gbfs": gbfs.search,
    "cus1": cus1.search,
    "cus2": cus2.search
}

if ALGORITHM not in ALGO_MAP:
    raise ValueError(f"[ERROR] Algorithm '{ALGORITHM}' is not supported.")

# Perform search
search_fn = ALGO_MAP[ALGORITHM]
result_node, result_cost, result_path = search_fn(graph)

# ==== Output Results ====
print("\n[INFO] Updated {}/{} edges using ML model.".format(
    len(graph.edges), len(graph.edges)))

if result_path and len(result_path) > 1:
    print(f"\n[RESULT] Predicted path using ML + {ALGORITHM.upper()}:")
    print(" → ".join(result_path))
    print(f"Predicted travel time: {round(result_cost, 2)} minutes")
else:
    print("\n[WARNING] Only found the starting node — no complete path exists.")
