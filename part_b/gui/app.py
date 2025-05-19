"""
Main Streamlit UI for Route Prediction

This Streamlit app allows users to select a start and end location in Boroondara,
run a machine learning-enhanced pathfinding algorithm, and visualize the route
on an interactive Folium map.
"""

import sys
import os
import requests
import streamlit as st
import folium
import pandas as pd
from streamlit_folium import st_folium

# Add root path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from part_b.pipeline import run_pipeline

# UI Title
st.title("🚦 Traffic-Based Route Guidance System")
st.markdown("This app predicts optimal routes using ML and search algorithms in the Boroondara region.")

# Load and clean coordinates
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

# Helper: Get OSRM-based real-world path
@st.cache_data(show_spinner=False)
def get_osrm_path(from_coord, to_coord):
    lon1, lat1 = from_coord[1], from_coord[0]
    lon2, lat2 = to_coord[1], to_coord[0]
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
        r = requests.get(url)
        data = r.json()
        return [(pt[1], pt[0]) for pt in data['routes'][0]['geometry']['coordinates']]
    except:
        return [from_coord, to_coord]

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

    # Add markers
    folium.Marker(
        location=start_coords,
        tooltip="Start: " + st.session_state.result_path[0],
        icon=folium.Icon(color="green", icon="play", prefix="fa")
    ).add_to(m)

    folium.Marker(
        location=loc_coords[st.session_state.result_path[-1]],
        tooltip="End: " + st.session_state.result_path[-1],
        icon=folium.Icon(color="red", icon="flag", prefix="fa")
    ).add_to(m)

    # Draw realistic paths between points using OSRM
    for i in range(len(st.session_state.result_path) - 1):
        from_node = st.session_state.result_path[i]
        to_node = st.session_state.result_path[i + 1]

        if from_node in loc_coords and to_node in loc_coords:
            route_segment = get_osrm_path(loc_coords[from_node], loc_coords[to_node])
            folium.PolyLine(locations=route_segment, color="blue", weight=4).add_to(m)

    st.markdown("### 🌍 Route Map:")
    st_folium(m, width=700, height=500)

elif st.session_state.result_path == []:
    st.warning("Only the start node was found. No full path available.")
