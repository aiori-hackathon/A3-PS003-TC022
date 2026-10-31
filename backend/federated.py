
"""NoCivicSense: Federated traffic prediction with FedAvg.

Uses synthetic traffic data for pipeline testing.
Communication costs are estimated from model tensor sizes,
not measured over a real network.
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
NUM_CLIENTS = 4
FEDERATED_ROUNDS = 5
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 24
DEVICE = torch.device("cpu")


def create_synthetic_traffic(n_points=1800):
    """Generate traffic-like hourly observations for testing."""
    t = np.arange(n_points, dtype=np.float32)

    daily = 0.22 * np.sin(2 * np.pi * t / 24 - 1.0)
    weekly = 0.08 * np.sin(2 * np.pi * t / (24 * 7))
    rush = (
        0.16 * np.exp(-((t % 24 - 8) ** 2) / 8)
        + 0.20 * np.exp(-((t % 24 - 17) ** 2) / 10)
    )
    trend = 0.0001 * t
    noise = np.random.normal(0, 0.035, n_points)

    return np.clip(
        0.5 + daily + weekly + rush + trend + noise,
        0.05,
        1.0,
    ).astype(np.float32)


def make_sequences(values):
    x, y = [], []

    for i in range(SEQUENCE_LENGTH, len(values)):
        x.append(values[i - SEQUENCE_LENGTH:i])
        y.append(values[i])

    return (
        np.asarray(x, dtype=np.float32)[..., np.newaxis],
        np.asarray(y, dtype=np.float32)[:, np.newaxis],
    )


def make_clients(x, y, mode):
    """Split training sequences between simulated clients."""
    indices = np.arange(len(x))

    if mode == "iid":
        rng = np.random.default_rng(SEED)
        rng.shuffle(indices)
        partitions = np.array_split(indices, NUM_CLIENTS)
    else:
        # Chronological partitions approximate temporal
        # distribution differences between traffic clients.
        partitions = np.array_split(indices, NUM_CLIENTS)

    clients = []

    for partition in partitions:
        clients.append(
            TensorDataset(
                torch.from_numpy(x[partition]),
                torch.from_numpy(y[partition]),
            )
        )

    return clients


def model_state_bytes(state):
    """Size of a serialized tensor payload, excluding protocol overhead."""
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in state.values()
    )


def train_client(global_state, dataset, seed):
    """Train one local copy of the global model."""
    torch.manual_seed(seed)

    local_model = TrafficGRU().to(DEVICE)
    local_model.load_state_dict(global_state)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(
        local_model.parameters(),
        lr=LEARNING_RATE,
    )
    loss_function = nn.MSELoss()

    local_model.train()

    for _ in range(LOCAL_EPOCHS):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_function(local_model(xb), yb)
            loss.backward()
            optimizer.step()

    return (
        {
            key: value.detach().cpu().clone()
            for key, value in local_model.state_dict().items()
        },
        len(dataset),
    )


def fedavg(client_states):
    """Weighted average of client parameters by sample count."""
    total_samples = sum(count for _, count in client_states)
    aggregated = {}

    for key in client_states[0][0]:
        first = client_states[0][0][key]

        if torch.is_floating_point(first):
            result = torch.zeros_like(first)

            for state, count in client_states:
                result += state[key] * (count / total_samples)

            aggregated[key] = result
        else:
            # Preserve non-floating state if one is introduced.
            aggregated[key] = first.clone()

    return aggregated


def evaluate(model, x_test, y_test, minimum, scale):
    model.eval()

    with torch.no_grad():
        predicted_scaled = model(
            torch.from_numpy(x_test).to(DEVICE)
        ).cpu().numpy().reshape(-1)

    actual = y_test.reshape(-1) * scale + minimum
    predicted = predicted_scaled * scale + minimum

    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

    return mae, rmse, actual, predicted


def run_experiment(mode, x_train, y_train, x_test, y_test, minimum, scale):
    """Run one complete federated experiment."""
    clients = make_clients(x_train, y_train, mode)

    model = TrafficGRU().to(DEVICE)
    global_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

    model_bytes = model_state_bytes(global_state)
    total_upload_bytes = 0
    total_download_bytes = 0
    history = []

    print(f"\n--- {mode.upper()} federated experiment ---")

    for round_number in range(1, FEDERATED_ROUNDS + 1):
        client_states = []

        for client_id, dataset in enumerate(clients):
            # A model download and an updated model upload per client.
            total_download_bytes += model_bytes
            local_state, sample_count = train_client(
                global_state,
                dataset,
                seed=SEED + round_number * 100 + client_id,
            )
            total_upload_bytes += model_bytes

            client_states.append((local_state, sample_count))

        global_state = fedavg(client_states)
        model.load_state_dict(global_state)

        mae, rmse, _, _ = evaluate(
            model, x_test, y_test, minimum, scale
        )

        history.append({
            "round": round_number,
            "mae": mae,
            "rmse": rmse,
        })

        print(
            f"Round {round_number}/{FEDERATED_ROUNDS} "
            f"| MAE: {mae:.6f} | RMSE: {rmse:.6f}"
        )

    mae, rmse, actual, predicted = evaluate(
        model, x_test, y_test, minimum, scale
    )

    torch.save(
        model.state_dict(),
        RESULTS_DIR / f"federated_gru_{mode}.pt",
    )

    pd.DataFrame({
        "actual_traffic": actual,
        "predicted_traffic": predicted,
    }).to_csv(
        RESULTS_DIR / f"federated_predictions_{mode}.csv",
        index=False,
    )

    return {
        "experiment": f"fedavg_{mode}",
        "data_type": "synthetic",
        "real_world_results": False,
        "split_method": mode,
        "num_clients": NUM_CLIENTS,
        "federated_rounds": FEDERATED_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "mae": mae,
        "rmse": rmse,
        "model_payload_bytes_per_client_transfer": model_bytes,
        "estimated_download_bytes": total_download_bytes,
        "estimated_upload_bytes": total_upload_bytes,
        "estimated_total_communication_bytes": (
            total_download_bytes + total_upload_bytes
        ),
        "communication_estimate_includes_protocol_overhead": False,
        "round_history": history,
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 60)
    print("NoCivicSense | Federated GRU + FedAvg")
    print("DATA MODE: SYNTHETIC — NOT REAL TRAFFIC DATA")
    print("=" * 60)

    traffic = create_synthetic_traffic()
    split = int(len(traffic) * 0.8)

    train_values = traffic[:split]
    test_values = np.concatenate([
        train_values[-SEQUENCE_LENGTH:],
        traffic[split:],
    ])

    # Fit normalization on training observations only.
    minimum = float(train_values.min())
    maximum = float(train_values.max())
    scale = max(maximum - minimum, 1e-8)

    train_scaled = (train_values - minimum) / scale
    test_scaled = (test_values - minimum) / scale

    x_train, y_train = make_sequences(train_scaled)
    x_test, y_test = make_sequences(test_scaled)

    results = []

    for mode in ("iid", "non_iid"):
        result = run_experiment(
            mode,
            x_train,
            y_train,
            x_test,
            y_test,
            minimum,
            scale,
        )
        results.append(result)

    output_path = RESULTS_DIR / "federated_metrics.json"
    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n=== FINAL FEDERATED RESULTS ===")

    for result in results:
        print(f"\n{result['experiment']}")
        print(f"MAE: {result['mae']:.6f}")
        print(f"RMSE: {result['rmse']:.6f}")
        print(
            "Estimated total communication: "
            f"{result['estimated_total_communication_bytes']:,} bytes"
        )

    print(f"\nSaved metrics: {output_path}")
    print("Federated experiments completed successfully.")


if __name__ == "__main__":
    main()
