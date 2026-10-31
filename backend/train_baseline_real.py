
"""Train a centralized GRU on real PeMS04 traffic data."""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from backend.data_loader import load_pems04
from backend.model import TrafficGRU


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

SEED = 42
EPOCHS = 5
BATCH_SIZE = 1024
LEARNING_RATE = 0.001


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    torch.set_num_threads(4)
    device = torch.device("cpu")

    print("\nLoading real PeMS04 data...")
    data = load_pems04()

    train_loader = DataLoader(
        TensorDataset(data["x_train"], data["y_train"]),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
    )

    test_loader = DataLoader(
        TensorDataset(data["x_test"], data["y_test"]),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = TrafficGRU().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )
    loss_function = nn.MSELoss()

    print("\n" + "=" * 55)
    print("NoCivicSense | Centralized GRU Training")
    print("=" * 55)
    print("Dataset: Real public PeMS04")
    print(f"Device: {device}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        sample_count = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = loss_function(predictions, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(x_batch)
            sample_count += len(x_batch)

        average_loss = total_loss / sample_count
        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| Training MSE: {average_loss:.6f}"
        )

    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            predictions = model(x_batch.to(device))
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y_batch.numpy())

    predictions = np.concatenate(all_predictions).reshape(-1)
    targets = np.concatenate(all_targets).reshape(-1)

    mae = float(np.mean(np.abs(predictions - targets)))
    rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))

    # These metrics are in normalized units, not vehicles per interval.
    metrics = {
        "experiment": "centralized_gru_pems04",
        "data_type": "real_public_pems04",
        "real_world_results": True,
        "dataset": "PeMS04",
        "selected_sensor_count": len(data["sensor_indices"]),
        "selected_sensor_indices": data["sensor_indices"].tolist(),
        "sequence_length": 24,
        "prediction_horizon": "next 5-minute time step",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "train_sequences": len(data["x_train"]),
        "test_sequences": len(data["x_test"]),
        "mae": mae,
        "rmse": rmse,
        "metric_units": "per-sensor MinMax-normalized traffic-flow units",
        "device": str(device),
    }

    metrics_path = RESULTS / "baseline_real_metrics.json"
    predictions_path = RESULTS / "baseline_real_predictions.csv"
    model_path = RESULTS / "centralized_gru_real.pt"

    pd.DataFrame({
        "actual_normalized_flow": targets,
        "predicted_normalized_flow": predictions,
    }).to_csv(predictions_path, index=False)

    torch.save(model.state_dict(), model_path)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\n" + "=" * 55)
    print("CENTRALIZED MODEL RESULTS — REAL PeMS04 DATA")
    print("=" * 55)
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Metrics saved: {metrics_path}")
    print(f"Predictions saved: {predictions_path}")
    print(f"Model saved: {model_path}")
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
