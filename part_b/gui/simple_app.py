"""
Simple Route Visualization App

This minimal Streamlit app runs the ML + pathfinding pipeline and visualizes
the predicted route by directly connecting waypoints with straight lines.
Intended for comparison with research_app.py which uses OSRM for realistic routing.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import folium
from streamlit_folium import st_folium
from part_b.pipeline import run_pipeline
import pandas as pd

# UI Title
st.title("🚦 Traffic-Based Route Guidance System")
st.markdown("This app predicts optimal routes using ML and search algorithms in the Boroondara region.")

# Load and clean coordinates (filter only valid Melbourne-area locations)
df = pd.read_csv("part_b/data/processed/locations_with_latlon.csv")
df = df[(df["latitude"] < -30) & (df["latitude"] > -40) & (df["longitude"] > 140) & (df["longitude"] < 150)]

loc_coords = {
    row["location"].strip().upper(): (row["latitude"], row["longitude"])
    for _, row in df.iterrows()
}
location_names = sorted(loc_coords.keys())

# Input selections
start_node = st.selectbox("🛫 Select Start Location", location_names)
end_node = st.selectbox("🏁 Select End Location", location_names)
model_name = st.selectbox("🧠 Select ML Model", ["xgb", "lstm", "gru"])
algorithm = st.selectbox("🧭 Select Pathfinding Algorithm", ["astar", "bfs", "dfs", "gbfs", "cus1", "cus2"])

# Store results in session
if "result_path" not in st.session_state:
    st.session_state.result_path = None
    st.session_state.result_cost = None

# Trigger pipeline
if st.button("🚀 Run Route Prediction"):
    with st.spinner("Running route prediction..."):
        try:
            result_path, result_cost = run_pipeline(
                start_node=start_node,
                end_node=end_node,
                algorithm=algorithm,
                model_name=model_name
            )
            st.session_state.result_path = result_path
            st.session_state.result_cost = result_cost
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

# Display results
if st.session_state.result_path and len(st.session_state.result_path) > 1:
    st.success(f"Predicted Travel Time: {round(st.session_state.result_cost, 2)} minutes")
    st.markdown("### 🗺️ Route Path:")
    st.code(" → ".join(st.session_state.result_path))

    # Draw map
    start_coords = loc_coords.get(st.session_state.result_path[0], (-37.82, 145.0))
    m = folium.Map(location=start_coords, zoom_start=14)

    # Add start marker (green)
    folium.Marker(
        location=start_coords,
        tooltip="Start: " + st.session_state.result_path[0],
        icon=folium.Icon(color="green", icon="play", prefix="fa")
    ).add_to(m)

    # Add end marker (red)
    folium.Marker(
        location=loc_coords[st.session_state.result_path[-1]],
        tooltip="End: " + st.session_state.result_path[-1],
        icon=folium.Icon(color="red", icon="flag", prefix="fa")
    ).add_to(m)

    # Draw full route as single connected path
    path_coords = []
    for node in st.session_state.result_path:
        if node in loc_coords:
            path_coords.append(loc_coords[node])
        else:
            st.warning(f"⚠️ Missing coordinate for: {node}")

    if len(path_coords) >= 2:
        folium.PolyLine(locations=path_coords, color="blue", weight=4).add_to(m)
        m.fit_bounds(path_coords)

    st.markdown("### 🌍 Route Map:")
    st_folium(m, width=700, height=500)
elif st.session_state.result_path == []:
    st.warning("Only the start node was found. No full path available.")
