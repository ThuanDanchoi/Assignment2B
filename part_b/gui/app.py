"""
Streamlit application for Traffic-Based Route Guidance System (TBRGS).

This app allows users to:
  - Select SCATS origin and destination sites from a sidebar
  - Specify the number of routes (k) to find
  - Display top-k routes with travel times in both table and map views
  - Visualize routes on a PyDeck map with origin/destination markers
"""

import os
import sys
import streamlit as st
import pandas as pd
import pydeck as pdk
import osmnx as ox

# Ensure project root is on PYTHONPATH for local imports
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from part_b.integrate import (
    load_scats_nodes,
    build_osm_subgraph,
    weight_osm_graph,
    find_k_paths,
)


def fmt_time(seconds: float) -> str:
    """
    Convert seconds to a human-readable string (e.g., '1h 2m', '5m 30s').

    Args:
        seconds (float): Duration in seconds.

    Returns:
        str: Formatted time string.
    """
    sec = int(round(seconds))
    if sec >= 3600:
        hours = sec // 3600
        minutes = (sec % 3600) // 60
        return f"{hours}h {minutes}m"
    elif sec >= 60:
        minutes = sec // 60
        secs = sec % 60
        return f"{minutes}m {secs}s"
    else:
        return f"{sec}s"


def main():
    """
    Main Streamlit UI: sidebar controls for inputs and route display.
    """
    st.title("🚦 TBRGS - Traffic Prediction & Route Guidance")

    # Load SCATS site coordinates
    nodes = load_scats_nodes()

    # Sidebar inputs
    origin = st.sidebar.selectbox(
        "Origin SCATS site", sorted(nodes.keys())
    )
    destination = st.sidebar.selectbox(
        "Destination SCATS site", sorted(nodes.keys())
    )
    k = st.sidebar.slider(
        "Number of routes", min_value=1, max_value=5, value=3
    )

    if st.sidebar.button("Find Routes"):
        # Build and weight the OSM subgraph
        MG = build_osm_subgraph(nodes, origin, destination)
        MG = weight_osm_graph(MG)

        # Compute top-k routes and travel times
        routes = find_k_paths(nodes, origin, destination, k)

        # Compute map center for initial zoom
        lats = [MG.nodes[n]["y"] for path, _ in routes for n in path]
        lons = [MG.nodes[n]["x"] for path, _ in routes for n in path]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

        # Display each route
        for idx, (path, travel_time) in enumerate(routes, start=1):
            st.subheader(f"Route {idx} — {fmt_time(travel_time)}")

            # Show node table with coordinates
            df = pd.DataFrame({
                "node": path,
                "lat": [MG.nodes[n]["y"] for n in path],
                "lon": [MG.nodes[n]["x"] for n in path],
            }).set_index("node")
            st.dataframe(df)

            # Prepare path coordinates for PyDeck
            coords = [[MG.nodes[n]["x"], MG.nodes[n]["y"]] for n in path]
            df_path = pd.DataFrame({"path": [coords]})
            path_layer = pdk.Layer(
                "PathLayer",
                data=df_path,
                get_path="path",
                get_color=[255, 140, 0],
                get_width=4
            )

            # Mark origin and destination
            o_node = ox.distance.nearest_nodes(
                MG,
                nodes[origin]["lon"], nodes[origin]["lat"]
            )
            d_node = ox.distance.nearest_nodes(
                MG,
                nodes[destination]["lon"], nodes[destination]["lat"]
            )
            df_mark = pd.DataFrame([
                {"lon": MG.nodes[o_node]["x"], "lat": MG.nodes[o_node]["y"], "color": [0, 255, 0]},
                {"lon": MG.nodes[d_node]["x"], "lat": MG.nodes[d_node]["y"], "color": [255, 0, 0]},
            ])
            marker_layer = pdk.Layer(
                "ScatterplotLayer",
                data=df_mark,
                get_position=["lon", "lat"],
                get_color="color",
                get_radius=60
            )

            # Configure the map view
            view = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=14,
                pitch=0,
            )
            deck = pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=view,
                layers=[path_layer, marker_layer],
            )
            st.pydeck_chart(deck)
            st.markdown("---")
            st.markdown(
                "### 🚦Traffic-Based Route Guidance System (TBRGS) - COS30019 Introduction to AI"
            )
            st.markdown(
                "Traffic volume prediction and optimal routing system for Boroondara area, Melbourne"
            )


if __name__ == "__main__":
    main()
