
"""Train and evaluate the first NoCivicSense GRU baseline.

This first experiment uses SYNTHETIC traffic data to verify the
training pipeline. Its results are not real-world traffic results.
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import TrafficGRU


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
SEQUENCE_LENGTH = 24
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")


def create_synthetic_traffic(n_points: int = 1800) -> np.ndarray:
    """Generate a reproducible hourly traffic-like time series."""

    t = np.arange(n_points, dtype=np.float32)
    daily_pattern = 0.22 * np.sin(2 * np.pi * t / 24 - 1.0)
    weekly_pattern = 0.08 * np.sin(2 * np.pi * t / (24 * 7))
    rush_hours = (
        0.16 * np.exp(-((t % 24 - 8) ** 2) / 8)
        + 0.20 * np.exp(-((t % 24 - 17) ** 2) / 10)
    )
    trend = 0.0001 * t
    noise = np.random.normal(0, 0.035, n_points)

    traffic = 0.5 + daily_pattern + weekly_pattern + rush_hours + trend + noise
    traffic = np.clip(traffic, 0.05, 1.0)

    return traffic.astype(np.float32)


def make_sequences(
    values: np.ndarray, sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Use past observations to predict the next observation."""

    x, y = [], []

    for i in range(sequence_length, len(values)):
        x.append(values[i - sequence_length:i])
        y.append(values[i])

    x_array = np.asarray(x, dtype=np.float32)[..., np.newaxis]
    y_array = np.asarray(y, dtype=np.float32)[:, np.newaxis]

    return x_array, y_array


def main() -> None:
    print("=" * 55)
    print("NoCivicSense | Centralized GRU baseline")
    print("DATA MODE: SYNTHETIC — NOT REAL TRAFFIC DATA")
    print("=" * 55)

    traffic = create_synthetic_traffic()

    # Chronological split prevents future observations leaking
    # into the training set.
    split = int(len(traffic) * 0.8)
    train_values = traffic[:split]
    test_values = traffic[split:]

    # Normalize using training data only.
    minimum = float(train_values.min())
    maximum = float(train_values.max())
    scale = max(maximum - minimum, 1e-8)

    train_scaled = (train_values - minimum) / scale

    # Include the preceding sequence as context for the test set.
    test_context = np.concatenate(
        [train_values[-SEQUENCE_LENGTH:], test_values]
    )
    test_scaled = (test_context - minimum) / scale

    x_train, y_train = make_sequences(
        train_scaled, SEQUENCE_LENGTH
    )
    x_test, y_test = make_sequences(
        test_scaled, SEQUENCE_LENGTH
    )

    train_dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = TrafficGRU().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )
    loss_function = nn.MSELoss()

    x_test_tensor = torch.from_numpy(x_test).to(DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(xb)
            loss = loss_function(predictions, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)

        average_loss = total_loss / len(train_dataset)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"| Training MSE: {average_loss:.6f}"
        )

    model.eval()
    with torch.no_grad():
        predictions_scaled = (
            model(x_test_tensor).cpu().numpy().reshape(-1)
        )

    actual_scaled = y_test.reshape(-1)

    # Convert normalized predictions back to original units.
    predictions = predictions_scaled * scale + minimum
    actual = actual_scaled * scale + minimum

    mae = float(np.mean(np.abs(actual - predictions)))
    rmse = float(np.sqrt(np.mean((actual - predictions) ** 2)))

    output_csv = RESULTS_DIR / "baseline_predictions.csv"
    output_json = RESULTS_DIR / "baseline_metrics.json"
    model_path = RESULTS_DIR / "centralized_gru.pt"

    pd.DataFrame({
        "actual_traffic": actual,
        "predicted_traffic": predictions,
    }).to_csv(output_csv, index=False)

    torch.save(model.state_dict(), model_path)

    metrics = {
        "experiment": "centralized_gru_baseline",
        "data_type": "synthetic",
        "real_world_results": False,
        "seed": SEED,
        "sequence_length": SEQUENCE_LENGTH,
        "epochs": EPOCHS,
        "train_observations": int(len(train_values)),
        "test_observations": int(len(test_values)),
        "mae": mae,
        "rmse": rmse,
        "units": "synthetic traffic-flow units",
    }

    output_json.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nEvaluation on held-out synthetic observations")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print("\nSaved files:")
    print(f"  Predictions: {output_csv}")
    print(f"  Metrics:     {output_json}")
    print(f"  Model:       {model_path}")
    print("\nBaseline training completed successfully.")


if __name__ == "__main__":
    main()
