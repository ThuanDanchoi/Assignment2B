# Traffic-Based Route Guidance System (TBRGS)

## Introduction

This project implements Part B of the COS30019 Assignment 2. It builds upon the route-finding algorithms developed in Part A by adding traffic-volume prediction and end-to-end integration to guide drivers along the fastest paths based on predicted travel times. The pipeline covers:

1. **Data Processing**: Ingest and clean SCATS traffic-flow data from VicRoads.
2. **Machine Learning Models**: Train and evaluate three different models for 15-minute-interval traffic-volume forecasting:

   * **LSTM**
   * **GRU**
   * **Custom Model(XgBoost)** 
3. **Flow-to-Time Conversion**: Convert predicted traffic flows to travel-time weights using the PDF v1.0 specification:

   * Free-flow: constant speed limit (60 km/h) + fixed delay (30 s).
   * Congested: solve quadratic relation to derive speed and add delay.
4. **OSM Subgraph & Routing Integration**:

   * Extract an OpenStreetMap subgraph around SCATS origin-destination pairs.
   * Assign computed travel-time weights to graph edges.
   * Compute the top-*k* shortest simple paths by total travel time.
5. **User Interface**: Provide both a CLI and a Streamlit web app for selecting origin, destination, ML model, and number of routes (*k*).

## Project Structure

```
TBRGS/
├── cache/                  # Cached files (pickles, graphml…)
├── data/                   # Shared data assets
├── part_a/                 # Part A: route-finding algorithms
│   ├── algorithms/         # DFS, BFS, GBFS, A*, …
│   ├── Docs/               # Problem specs, input formats
│   ├── test_cases/         # Input/output test files
│   ├── utils/              # Helpers (parsers, formatters)
│   ├── __init__.py
│   ├── graph.py
│   ├── run_all_test.py
│   └── search.py
├── part_b/                 # Part B: ML + integration + UI
│   ├── data/               # Raw & processed CSVs, subgraph cache
│   ├── gui/                # Streamlit dashboard & assets
│   ├── models/             # Saved ML model weights
│   ├── __init__.py
│   ├── data_processing.py
│   ├── evaluate.py
│   ├── integrate.py
│   ├── metrics_summary.csv
│   ├── test.py
│   ├── train.py
│   ├── travel_time.py
│   └── weighted_graph.pkl
├── tbrgs/                  # 🐍 Python virtual environment
├── requirements.txt        # Project dependencies
└── README.md             
```

## Installation

### Prerequisites

* Python 3.9 or above
* Git

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/ThuanDanchoi/TBRGS.git
   cd TBRGS
   ```
2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command-Line Interface (CLI)

Run integration to compute top-*k* routes:

```bash
python part_b/integrate.py \
    --origin 2000 \
    --dest 3002 \
    --model lstm \
    --k 3
```

* `--origin` / `--dest`: SCATS site IDs for origin and destination
* `--model`: ML predictor to use (`lstm`, `gru`, or `custom`)
* `--k`: number of shortest paths to return

### Web Dashboard (Streamlit)

Launch the interactive app:

```bash
streamlit run part_b/app.py
```

Use the sidebar to select origin, destination, ML model, and routes to visualize on an embedded map.

## Testing

Execute the full pytest suite (Part A + Part B):

```bash
pytest
```

By default verbose output is enabled via `pytest.ini`.

## Features

* **Data Processing** pipeline for traffic-flow CSVs
* **ML Forecasting** with LSTM, GRU, and custom architectures
* **Travel-Time Conversion** implementing PDF v1.0 rules
* **OSM Subgraph Extraction** around SCATS OD pairs
* **Top-*k* Routing** based on travel-time weights
* **CLI & Web UI** for flexible user interaction

## Notes

* OSM subgraphs are cached under `part_b/data/osm_subgraphs` to avoid repeated downloads.
* Model weights are saved in `part_b/models` after training.
* If no path exists between two sites, the CLI returns an empty list gracefully.

## Development Team

* **Duc Thuan Tran** (104330455)
* **Vu Anh Le** (104653505)
* **harrish** (104333333)
 
## License

This project is developed for educational purposes as part of the COS30019 – Introduction to Artificial Intelligence course at Swinburne University of Technology.

