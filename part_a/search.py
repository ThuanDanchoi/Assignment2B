"""
Search program for Traffic-based Route Guidance System – Part A.

Usage: python -m part_a.search <problem_file> <method>
"""

import sys
import os

# --- ensure project root so that part_a.utils is available ---
SCRIPT_DIR = os.path.dirname(__file__)           # .../part_a
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from part_a.utils.file_parser import parse_problem_file
from part_a.utils.output      import format_output
from part_a.graph            import Graph

# Import search algorithms
from part_a.algorithms.dfs    import search as dfs_search
from part_a.algorithms.bfs    import search as bfs_search
from part_a.algorithms.gbfs   import search as gbfs_search
from part_a.algorithms.astar  import search as astar_search
from part_a.algorithms.cus1   import search as cus1_search
from part_a.algorithms.cus2   import search as cus2_search

def main():
    if len(sys.argv) != 3:
        print("Usage: python -m part_a.search <problem_file> <method>")
        sys.exit(1)

    filename = sys.argv[1]
    method   = sys.argv[2].upper()

    if not os.path.isfile(filename):
        print(f"Error: File '{filename}' not found")
        sys.exit(1)

    valid = ['DFS','BFS','GBFS','AS','CUS1','CUS2']
    if method not in valid:
        print(f"Error: Method '{method}' not recognized. Choose from: {', '.join(valid)}")
        sys.exit(1)

    nodes, edges, origin, destinations = parse_problem_file(filename)
    graph = Graph(nodes, edges, origin, destinations)

    if method == 'DFS':
        goal, created, path = dfs_search(graph)
    elif method == 'BFS':
        goal, created, path = bfs_search(graph)
    elif method == 'GBFS':
        goal, created, path = gbfs_search(graph)
    elif method == 'AS':
        goal, created, path = astar_search(graph)
    elif method == 'CUS1':
        goal, created, path = cus1_search(graph)
    else:  # CUS2
        goal, created, path = cus2_search(graph)

    if goal is not None:
        print(format_output(filename, method, goal, created, path))
    else:
        print(f"{filename} {method}\nNo solution found.")

if __name__ == "__main__":
    main()
