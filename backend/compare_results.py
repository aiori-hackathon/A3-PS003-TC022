
"""Compare NoCivicSense synthetic experiment metrics."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

baseline_path = RESULTS / "baseline_metrics.json"
federated_path = RESULTS / "federated_metrics.json"
output_path = RESULTS / "comparison_report.csv"

with baseline_path.open("r", encoding="utf-8") as file:
    baseline = json.load(file)

with federated_path.open("r", encoding="utf-8") as file:
    federated_runs = json.load(file)

rows = [
    {
        "experiment": baseline["experiment"],
        "data_type": baseline["data_type"],
        "clients": 1,
        "rounds": 0,
        "mae": baseline["mae"],
        "rmse": baseline["rmse"],
        "communication_bytes": 0,
    }
]

for run in federated_runs:
    rows.append(
        {
            "experiment": run["experiment"],
            "data_type": run["data_type"],
            "clients": run["num_clients"],
            "rounds": run["federated_rounds"],
            "mae": run["mae"],
            "rmse": run["rmse"],
            "communication_bytes": run[
                "estimated_total_communication_bytes"
            ],
        }
    )

columns = [
    "experiment",
    "data_type",
    "clients",
    "rounds",
    "mae",
    "rmse",
    "communication_bytes",
]

with output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

print("NoCivicSense comparison report created.")
print(f"Saved to: {output_path}")
print("\nResults (synthetic data only):")

for row in rows:
    print(
        f'{row["experiment"]:26} '
        f'MAE={row["mae"]:.6f}  '
        f'RMSE={row["rmse"]:.6f}  '
        f'Communication={row["communication_bytes"]:,} bytes'
    )
