"""
Route generation pipeline

This script loads a traffic graph, updates edge costs using a machine learning model,
and runs a specified search algorithm to find the optimal route.
"""

import os
import joblib
import pandas as pd
from tensorflow.keras.models import load_model

from part_a.graph import Graph
from part_a.algorithms import astar, bfs, dfs, gbfs, cus1, cus2
from part_b.graph_loader import load_graph_from_csv
from part_b.graph_updater import update_graph_with_ml

def run_pipeline(start_node, end_node, algorithm="astar", model_name="xgb"):
    """
    Run the complete route generation pipeline:
    1. Load graph
    2. Load ML model
    3. Update edge costs using traffic data
    4. Execute selected pathfinding algorithm

    Args:
        start_node (str): The starting location ID.
        end_node (str): The destination location ID.
        algorithm (str): Algorithm to use for route finding (e.g., 'astar', 'dfs', etc.).
        model_name (str): Machine learning model name to load for prediction (e.g., 'xgb', 'lstm').

    Returns:
        tuple: (result_path, result_cost) where
            - result_path (list): List of node IDs forming the shortest path.
            - result_cost (float): Total cost/distance/time of the path.
    """
    # ==== Paths configuration ====
    GRAPH_CSV_PATH = "part_b/data/processed/graph_edges.csv"
    LATLON_PATH    = "part_b/data/processed/locations_with_latlon.csv"
    VOLUME_PATH    = "part_b/data/processed/cleaned_scats_data.csv"
    MODEL_PATH     = (
        f"part_b/models/{model_name}/{model_name}.h5"
        if model_name in ["lstm", "gru"]
        else f"part_b/models/{model_name}/{model_name}.joblib"
    )

    # ==== Load initial graph ====
    graph = load_graph_from_csv(
        csv_path=GRAPH_CSV_PATH,
        latlon_path=LATLON_PATH,
        origin=start_node,
        destinations=[end_node]
    )

    # Check if start and end nodes exist in graph
    if start_node not in graph.nodes:
        print(f"[ERROR] Start node '{start_node}' not found in graph.")
        return None, None
    if end_node not in graph.nodes:
        print(f"[ERROR] End node '{end_node}' not found in graph.")
        return None, None

    # ==== Load pre-trained ML model ====
    model_ext = os.path.splitext(MODEL_PATH)[1]
    if model_ext == ".h5":
        model = load_model(MODEL_PATH, compile=False)
    else:
        model = joblib.load(MODEL_PATH)

    # ==== Update graph edges with predicted travel times ====
    graph = update_graph_with_ml(
        graph=graph,
        graph_csv_path=GRAPH_CSV_PATH,
        volume_path=VOLUME_PATH,
        model_path=None,
        model=model
    )

    # ==== Execute selected search algorithm ====
    ALGO_MAP = {
        "astar": astar.search,
        "bfs": bfs.search,
        "dfs": dfs.search,
        "gbfs": gbfs.search,
        "cus1": cus1.search,
        "cus2": cus2.search
    }

    if algorithm not in ALGO_MAP:
        raise ValueError(f"Unknown algorithm: '{algorithm}'")

    # Run selected algorithm and return results
    result_node, result_cost, result_path = ALGO_MAP[algorithm](graph)
    return result_path, result_cost
