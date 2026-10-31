
"""Compare centralized, federated, and persistence models on PeMS04."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def read_json(filename):
    path = RESULTS / filename
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    baseline = read_json("baseline_real_metrics.json")
    federated = read_json("federated_real_metrics.json")
    persistence = read_json("persistence_real_metrics.json")

    rows = [
        {
            "experiment": "Centralized GRU",
            "mae": baseline["mae"],
            "rmse": baseline["rmse"],
            "communication_bytes": None,
        },
        {
            "experiment": "FedAvg IID",
            "mae": federated["iid"]["mae"],
            "rmse": federated["iid"]["rmse"],
            "communication_bytes":
                federated["iid"]["total_communication_bytes"],
        },
        {
            "experiment": "FedAvg non-IID",
            "mae": federated["non_iid"]["mae"],
            "rmse": federated["non_iid"]["rmse"],
            "communication_bytes":
                federated["non_iid"]["total_communication_bytes"],
        },
        {
            "experiment": "Persistence baseline",
            "mae": persistence["mae"],
            "rmse": persistence["rmse"],
            "communication_bytes": None,
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "real_comparison_report.csv", index=False)

    with open(
        RESULTS / "real_comparison_report.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("NoCivicSense — Real PeMS04 Results\n")
        file.write("=" * 55 + "\n")
        file.write("Dataset: public PeMS04, 32 selected sensors\n")
        file.write("Errors: normalized traffic-flow units\n")
        file.write("Federated communication: estimated model payload only\n\n")
        file.write(df.to_string(index=False))
        file.write("\n\nLimitations:\n")
        file.write("- PeMS04 is California traffic data, not Indian traffic data.\n")
        file.write("- IID and non-IID clients are simulated.\n")
        file.write("- Communication excludes protocol overhead.\n")
        file.write("- Centralized communication was not measured.\n")
        file.write("- Persistence predicts the next value using the last observation.\n")

    # MAE and RMSE comparison.
    ax = df.plot(
        x="experiment",
        y=["mae", "rmse"],
        kind="bar",
        figsize=(10, 5),
        rot=15,
    )
    ax.set_title("NoCivicSense: Prediction Error Comparison")
    ax.set_ylabel("Error (normalized flow units)")
    ax.set_xlabel("Model")
    ax.legend(["MAE", "RMSE"])
    plt.tight_layout()
    plt.savefig(RESULTS / "real_accuracy_comparison.png", dpi=200)
    plt.close()

    # Communication cost for federated experiments only.
    fed_df = df[df["communication_bytes"].notna()]
    ax = fed_df.plot(
        x="experiment",
        y="communication_bytes",
        kind="bar",
        figsize=(8, 5),
        legend=False,
        rot=0,
    )
    ax.set_title("Estimated Federated Communication Cost")
    ax.set_ylabel("Total model payload (bytes)")
    ax.set_xlabel("Federated experiment")
    plt.tight_layout()
    plt.savefig(RESULTS / "real_communication_comparison.png", dpi=200)
    plt.close()

    print("\nNoCivicSense — Updated Real PeMS04 Comparison")
    print("=" * 65)
    print(df.to_string(index=False))
    print("\nUpdated report and graphs saved in:", RESULTS)


if __name__ == "__main__":
    main()
