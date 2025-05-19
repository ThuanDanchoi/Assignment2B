"""
Graph updater with ML prediction

This module updates edge costs in a graph using a machine learning model
(e.g., LSTM or traditional regressors) based on recent traffic volume data.
"""

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model


def update_graph_with_ml(graph, graph_csv_path, volume_path, model_path=None, model=None):
    """
    Update edge costs in the Graph using predicted travel times from a ML model.

    Args:
        graph (Graph): The Graph object whose edges will be updated.
        graph_csv_path (str): Path to the CSV file with original 'from', 'to', and 'distance_km' edges.
        volume_path (str): Path to CSV containing traffic volume data with 'Location', 'SiteID', 'Volume', 'Datetime'.
        model_path (str, optional): File path to a trained ML model (.joblib or .h5). Used if `model` is not provided.
        model (sklearn or keras model, optional): Pre-loaded ML model. If not provided, `model_path` is used.

    Returns:
        Graph: The same Graph object with updated edge costs.
    """
    # Load model from path if not already provided
    if model is None:
        if model_path.endswith(".h5"):
            model = load_model(model_path, compile=False)
        else:
            model = joblib.load(model_path)

    # Load graph structure and traffic volume data
    df_edges = pd.read_csv(graph_csv_path)
    df_edges['from'] = df_edges['from'].str.strip().str.upper()
    df_edges['to'] = df_edges['to'].str.strip().str.upper()

    volume_df = pd.read_csv(volume_path, parse_dates=['Datetime'])
    volume_df = volume_df.sort_values(by=['SiteID', 'Datetime'])
    volume_df['Location'] = volume_df['Location'].str.strip().str.upper()

    # Configuration
    LOOKBACK = 4  # Number of time steps required for prediction
    updated_edges = 0
    fallback_edges = 0
    total_edges = len(df_edges)

    # Loop through all edges to update travel time
    for _, row in df_edges.iterrows():
        from_loc = row['from']
        to_loc = row['to']
        default_cost = row['distance_km']

        # Match volume data using Location → SiteID
        site_ids = volume_df[volume_df['Location'] == from_loc]['SiteID'].unique()
        if len(site_ids) == 0:
            # No SiteID found for this location
            fallback_edges += 1
            graph.edges[(from_loc, to_loc)] = default_cost
            continue

        site_id = site_ids[0]
        site_data = volume_df[volume_df['SiteID'] == site_id]

        # Ensure enough recent volume records are available
        if len(site_data) < LOOKBACK:
            fallback_edges += 1
            graph.edges[(from_loc, to_loc)] = default_cost
            continue

        latest_vols = site_data['Volume'].values[-LOOKBACK:]
        if len(latest_vols) != LOOKBACK:
            fallback_edges += 1
            graph.edges[(from_loc, to_loc)] = default_cost
            continue

        # Predict travel time using ML model
        X_input = np.array(latest_vols).reshape(1, -1)
        try:
            predicted_time = model.predict(X_input)[0]
            if isinstance(predicted_time, (np.ndarray, list)):
                predicted_time = predicted_time[0]

            graph.edges[(from_loc, to_loc)] = predicted_time
            updated_edges += 1
        except Exception as e:
            print(f"[ERROR] Prediction failed at ({from_loc}, {to_loc}): {e}")
            graph.edges[(from_loc, to_loc)] = default_cost
            fallback_edges += 1

    # Logging summary
    print(f"[INFO] ML-updated edges: {updated_edges}/{total_edges}")
    print(f"[INFO] Fallback to original distance: {fallback_edges}")
    print(f"[INFO] Skipped or missing: {total_edges - updated_edges - fallback_edges}")

    return graph
