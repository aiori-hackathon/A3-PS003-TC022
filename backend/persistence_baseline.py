
"""NoCivicSense: Persistence baseline on real PeMS04 traffic data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from backend.data_loader import load_pems04


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def main():
    RESULTS.mkdir(exist_ok=True)

    data = load_pems04()

    # Persistence prediction: the next value equals the last
    # observed value in each 24-step input sequence.
    x_test = data["x_test"].numpy()
    y_test = data["y_test"].numpy().reshape(-1)

    predictions = x_test[:, -1, 0]

    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    metrics = {
        "experiment": "Persistence baseline",
        "data_type": "real_public_pems04",
        "mae": mae,
        "rmse": rmse,
        "prediction_rule": "Next value equals last observed value",
        "sequence_length": 24,
        "selected_sensors": len(data["sensor_indices"]),
        "note": "Errors are in normalized traffic-flow units.",
    }

    metrics_path = RESULTS / "persistence_real_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    predictions_path = RESULTS / "persistence_real_predictions.csv"
    pd.DataFrame({
        "actual_normalized_flow": y_test,
        "predicted_normalized_flow": predictions,
    }).to_csv(predictions_path, index=False)

    print("\nNoCivicSense — Persistence Baseline")
    print("=" * 50)
    print("DATA MODE: REAL PUBLIC PeMS04")
    print("Prediction rule: next flow = last observed flow")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"\nMetrics saved: {metrics_path}")
    print(f"Predictions saved: {predictions_path}")


if __name__ == "__main__":
    main()
