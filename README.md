# Traffic-Based Route Guidance System (TBRGS)

## Introduction

This project implements an intelligent routing system that integrates traffic-volume prediction with traditional pathfinding algorithms to guide vehicles through the Boroondara region. Built for COS30019 Assignment 2 (Part B), the system combines real traffic data, machine learning models, and routing logic for smart urban navigation.

The solution includes:

1. **Traffic Data Preprocessing**  
   Raw SCATS traffic flow data is cleaned, aligned by site and timestamp, and transformed into supervised learning format for forecasting.

2. **ML-Based Volume Prediction**  
   Three ML models are trained to predict traffic volume at 15-minute intervals:
   - `LSTM`
   - `GRU`
   - `XGBoost` (custom non-sequential baseline)

3. **Travel-Time Estimation**  
   Using the PDF v1.0 traffic model:
   - Free-flow travel time is computed via constant speed + delay.
   - Congested scenarios apply nonlinear conversion from flow to estimated speed.

4. **Dynamic Graph Construction**  
   A road network graph is built using coordinates and realistic geographic thresholds. Each edge reflects the predicted travel time from ML output.

5. **Search Algorithms for Routing**  
   Multiple routing strategies are implemented:
   - A*, BFS, DFS, GBFS
   - Custom strategies (CUS1, CUS2)
   Each algorithm searches the updated ML-weighted graph to find optimal paths.

6. **Web-Based Visualization Interface**  
   A Streamlit app allows users to:
   - Select origin/destination points
   - Choose ML models and algorithms
   - Visualize routes with realistic road geometry using OpenStreetMap + OSRM

## Project Structure

```
TBRGS/
├── part_a/
│   ├── algorithms/                   # Search algorithms for route finding
│   │   ├── astar.py                  # A* (A-star) search
│   │   ├── bfs.py                    # Breadth-First Search
│   │   ├── dfs.py                    # Depth-First Search
│   │   ├── gbfs.py                   # Greedy Best-First Search
│   │   ├── cus1.py                   # Custom algorithm 1
│   │   └── cus2.py                   # Custom algorithm 2
│   ├── utils/
│   │   ├── graph.py                  # Graph data structure and helper methods
│   │   └── search.py                 # Unified interface to run different algorithms
│   └── __init__.py

├── part_b/
│   ├── data/
│   │   ├── raw/                      # Raw input datasets
│   │   │   ├── Scats Data October 2006.csv
│   │   │   └── Traffic_Count_Locations_with_LONG_LAT.csv
│   │   └── processed/                # Preprocessed and enriched datasets
│   │       ├── graph_edges.csv           # Graph edge list with distances
│   │       ├── locations_with_latlon.csv # Final location names with coordinates
│   │       └── cleaned_scats_data.csv    # Cleaned and structured SCATS data
│   ├── geo/                          # Scripts for graph generation and geolocation
│   │   ├── generate_graph.py         # Create graph edges based on lat/lon distance
│   │   ├── geocode.py                # Auto-geocode missing locations to lat/lon
│   │   └── prep_geocode.py           # Extract unique location names to geocode
│   ├── gui/
│   │   └── app.py                    # Streamlit UI for interactive route prediction
│   ├── model_training/              # ML model training scripts
│   │   ├── lstm.py                   # Train LSTM model on traffic volume
│   │   ├── gru.py                    # Train GRU model on traffic volume
│   │   ├── xgb.py                    # Train XGBoost regression model
│   │   └── compare_models.py         # Evaluate and compare performance of all models
│   ├── models/                       # Saved trained models
│   │   ├── lstm/lstm.h5
│   │   ├── gru/gru.h5
│   │   └── xgb/xgb.joblib
│   ├── test_cases/
│   │   ├── test_cases.py             # Batch test pipeline over multiple routes
│   │   ├── test_results_*.log        # Saved logs of test runs
│   │   └── __init__.py
│   ├── data_processing.ipynb         # Jupyter notebook for cleaning raw SCATS data
│   ├── graph_loader.py               # Function to load graph from CSV
│   ├── graph_updater.py              # Update graph edge weights using ML predictions
│   ├── pipeline.py                   # Core routing pipeline (used in both CLI & UI)
│   ├── run_route.py                  # Minimal CLI interface for single route testing
│   ├── tbrgs                         # Conda virtual environment name
│   └── requirements.txt              # All Python dependencies (e.g. pandas, folium)
```

## Installation

### Prerequisites
- Python 3.9+
- Git
- Streamlit
- Conda or virtualenv (recommended)

### Setup

```bash
git clone https://github.com/ThuanDanchoi/TBRGS.git
cd TBRGS
python -m venv tbrgs
source tbrgs/bin/activate  # Or: .\tbrgs\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

### CLI Pipeline

```bash
python part_b/run_route.py
```

You can configure:
- `start_node`
- `end_node`
- `model_name` (`xgb`, `lstm`, `gru`)
- `algorithm` (`astar`, `bfs`, `dfs`, `gbfs`, `cus1`, `cus2`)

### Web Interface (Streamlit)

```bash
streamlit run part_b/gui/app.py
```

Features:
- Interactive dropdowns for location, model, and algorithm.
- Travel-time prediction results.
- OSRM-enhanced map path display with proper road-following logic.

## Testing

```bash
python part_b/test_cases/test_cases.py
```

This logs results for 10 test cases with combinations of start-end nodes, ML models, and search algorithms.

## Features

- 📊 **ML Forecasting** for traffic volume using time series inputs
- 🛣️ **Graph Routing** with realistic ML travel-time weights
- 🗺️ **OpenStreetMap Integration** for geolocation fidelity
- 🧪 **Batch Testing Framework** for performance comparison
- 🌐 **User-Friendly Web UI** powered by Streamlit + Folium
- 🔧 **Modular Architecture** for rapid experimentation

## Notes

- Edge generation threshold is configurable in `generate_graph.py`
- OSRM API is used for realistic routing on the Streamlit map
- Models are saved after training in `part_b/models/`

## Development Team

- **Duc Thuan Tran** (104330455)
- **Vu Anh Le** (104653505)
- **Harrish** (104333333)

## License

This project is for educational use in COS30019 – *Introduction to Artificial Intelligence*, Swinburne University of Technology.
