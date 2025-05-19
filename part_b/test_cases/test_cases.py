import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from part_b.pipeline import run_pipeline
import datetime

# Create log file in test_cases directory
log_dir = "part_b/test_cases"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"test_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
log_file = open(log_filename, "w")

test_cases = [
    {"start": "WARRIGAL_RD N OF HIGH STREET_RD", "end": "HIGHBURY_RD E OF WARRIGAL_RD", "algorithm": "astar", "model": "xgb"},
    {"start": "GLENFERRIE_RD S OF COTHAM_RD", "end": "COTHAM_RD W OF GLENFERRIE_RD", "algorithm": "bfs", "model": "gru"},
    {"start": "HIGH_ST NE OF BARKERS_RD", "end": "BURWOOD_RD E OF GLENFERRIE_RD", "algorithm": "gbfs", "model": "lstm"},
    {"start": "COTHAM_RD W OF BURKE_RD", "end": "CANTERBURY_RD W OF BALWYN_RD", "algorithm": "dfs", "model": "gru"},
    {"start": "RIVERSDALE_RD W OF BURKE_RD", "end": "RIVERSDALE_RD W OF GLENFERRIE_RD", "algorithm": "cus1", "model": "xgb"},
    {"start": "POWER_ST S OF BARKERS_RD", "end": "WARRIGAL_RD S OF HIGH STREET_RD", "algorithm": "cus2", "model": "xgb"},
    {"start": "WARRIGAL_RD N OF HIGH STREET_RD", "end": "WARRIGAL_RD S OF RIVERSDALE_RD", "algorithm": "astar", "model": "gru"},
    {"start": "BALWYN_RD S OF CANTERBURY_RD", "end": "HIGHBURY_RD W OF MIDDLEBOROUGH_RD", "algorithm": "dfs", "model": "lstm"},
    {"start": "TOORONGA_RD N OF RIVERSDALE_RD", "end": "GLENFERRIE_RD S OF BURWOOD_RD", "algorithm": "gbfs", "model": "gru"},
    {"start": "BULLEEN_RD N OF THOMPSONS_RD", "end": "BARKERS_RD E OF HIGH_ST", "algorithm": "bfs", "model": "xgb"}
]

def run_all_tests():
    for idx, case in enumerate(test_cases):
        header = f"===== Test Case {idx + 1} ====="
        print(f"\n{header}")
        log_file.write(f"\n{header}\n")

        result_path, result_cost = run_pipeline(
            start_node=case["start"],
            end_node=case["end"],
            algorithm=case.get("algorithm", "astar"),
            model_name=case.get("model", "xgb")
        )

        if result_path and len(result_path) > 1:
            path_str = " → ".join(result_path)
            cost_str = f"{round(result_cost, 2)} minutes"
            print("Path:", path_str)
            print("Time:", cost_str)
            log_file.write(f"Path: {path_str}\n")
            log_file.write(f"Time: {cost_str}\n")
        else:
            print("No valid path found.")
            log_file.write("No valid path found.\n")

if __name__ == "__main__":
    run_all_tests()
    log_file.close()
    print(f"\nLog saved to: {log_filename}")
