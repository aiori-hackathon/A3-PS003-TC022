
"""NoCivicSense: FedAvg experiments on real PeMS04 traffic data."""

import copy
import argparse
import json
import random
import uuid
from datetime import datetime, timezone
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
NUM_CLIENTS = 4
ROUNDS = 3
LOCAL_EPOCHS = 1
BATCH_SIZE = 2048
LEARNING_RATE = 0.001


def set_seed(seed):
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_client_indices(num_samples, mode, sensor_indices, num_clients=NUM_CLIENTS):
    """Partition training examples into IID or sensor-grouped clients."""

    indices = np.arange(num_samples)
    num_sensors = len(sensor_indices)

    if num_clients < 1:
        raise ValueError("Client count must be at least one.")
    if num_sensors == 0 or num_samples % num_sensors != 0:
        raise ValueError(
            "Training sample count must be divisible by sensor count."
        )

    samples_per_sensor = num_samples // num_sensors

    if mode == "iid":
        # IID simulation: randomly distribute examples among clients.
        rng = np.random.default_rng(SEED)
        rng.shuffle(indices)

        clients = [
            chunk.astype(np.int64)
            for chunk in np.array_split(indices, num_clients)
        ]

    elif mode == "non_iid":
        # Non-IID simulation: each client receives a distinct group
        # of sensors. The data loader uses sensor-major ordering.
        if num_sensors < num_clients:
            raise ValueError(
                "The number of sensors must be at least the client count."
            )

        sensor_groups = np.array_split(
            np.arange(num_sensors), num_clients
        )

        clients = []

        for sensor_group in sensor_groups:
            client_indices = []

            for sensor_id in sensor_group:
                start = sensor_id * samples_per_sensor
                end = start + samples_per_sensor

                client_indices.extend(range(start, end))

            clients.append(
                np.asarray(client_indices, dtype=np.int64)
            )

    else:
        raise ValueError(f"Unknown client distribution: {mode}")

    # Verify every training example is assigned exactly once.
    combined = np.concatenate(clients)

    if (
        len(combined) != num_samples
        or len(np.unique(combined)) != num_samples
        or combined.min() < 0
        or combined.max() >= num_samples
    ):
        raise RuntimeError(
            f"Invalid {mode} partition: samples are missing or duplicated."
        )

    print("\nClient partition:")
    for client_id, client_indices in enumerate(clients, start=1):
        print(
            f"  Client {client_id}: "
            f"{len(client_indices):,} training sequences"
        )

    if mode == "non_iid":
        print("  Distribution: separate sensor groups per client")
        for client_id, sensor_group in enumerate(
            np.array_split(np.arange(num_sensors), num_clients),
            start=1,
        ):
            actual_sensor_ids = [
                int(sensor_indices[sensor_id])
                for sensor_id in sensor_group
            ]
            print(
                f"  Client {client_id} sensor IDs: {actual_sensor_ids}"
            )
    else:
        print("  Distribution: randomly shuffled training sequences")

    return clients


def get_model_payload_bytes(model):
    """Count model tensor payload bytes, excluding protocol overhead."""
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in model.state_dict().values()
    )


def train_local_model(
    global_state,
    x_train,
    y_train,
    indices,
    client_id,
    round_id,
    batch_size,
    learning_rate,
    local_epochs,
):
    """Train one client locally and return its model state and sample count."""

    local_model = TrafficGRU()
    local_model.load_state_dict(copy.deepcopy(global_state))
    local_model.train()

    client_x = x_train[indices]
    client_y = y_train[indices]

    loader_generator = torch.Generator()
    loader_generator.manual_seed(
        SEED + round_id * 100 + client_id
    )

    loader = DataLoader(
        TensorDataset(client_x, client_y),
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
    )

    optimizer = torch.optim.Adam(
        local_model.parameters(),
        lr=learning_rate,
    )
    loss_function = nn.MSELoss()

    for _ in range(local_epochs):
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            prediction = local_model(x_batch)
            loss = loss_function(prediction, y_batch)
            loss.backward()
            optimizer.step()

    return copy.deepcopy(local_model.state_dict()), len(indices)


def fedavg(states_and_sizes):
    """Average model states weighted by client training sample counts."""

    total_samples = sum(
        sample_count for _, sample_count in states_and_sizes
    )
    averaged_state = {}

    for key in states_and_sizes[0][0]:
        averaged_state[key] = torch.zeros_like(
            states_and_sizes[0][0][key]
        )

        for client_state, sample_count in states_and_sizes:
            averaged_state[key] += (
                client_state[key] * (sample_count / total_samples)
            )

    return averaged_state


def evaluate(model, x_test, y_test, batch_size):
    """Evaluate a model on the shared held-out test set."""

    model.eval()
    predictions = []

    loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    with torch.no_grad():
        for x_batch, _ in loader:
            predictions.append(model(x_batch).numpy())

    predicted = np.concatenate(predictions).reshape(-1)
    actual = y_test.numpy().reshape(-1)

    mae = float(np.mean(np.abs(predicted - actual)))
    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))

    return mae, rmse, predicted, actual


def run_federated_experiment(data, mode, num_clients=NUM_CLIENTS, rounds=ROUNDS, local_epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE):
    """Run one federated experiment and return its metrics and model."""

    if not 2 <= num_clients <= 16:
        raise ValueError("Client count must be between 2 and 16.")
    if not 1 <= rounds <= 20:
        raise ValueError("Communication rounds must be between 1 and 20.")
    if not 1 <= local_epochs <= 10:
        raise ValueError("Local epochs must be between 1 and 10.")
    if not 1 <= batch_size <= 65536:
        raise ValueError("Batch size must be between 1 and 65536.")
    if not 0 < learning_rate <= 1:
        raise ValueError("Learning rate must be greater than 0 and at most 1.")

    print(f"\n{'=' * 60}")
    print(f"Federated experiment: {mode.upper()}")
    print(f"{'=' * 60}")

    set_seed(SEED)

    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    client_indices = make_client_indices(
        len(x_train),
        mode,
        data["sensor_indices"],
        num_clients,
    )

    print(f"\nClients: {num_clients}")
    print(f"Rounds: {rounds}")
    print(f"Local epochs per round: {local_epochs}")
    print("Training device: CPU")
    print("Evaluation: same held-out test set for both experiments")

    # Reset initialization so IID and non-IID begin with the same weights.
    set_seed(SEED)
    global_model = TrafficGRU()
    global_state = copy.deepcopy(global_model.state_dict())

    payload_bytes = get_model_payload_bytes(global_model)
    total_download_bytes = 0
    total_upload_bytes = 0
    communication_history = []

    for round_id in range(1, rounds + 1):
        print(f"\nRound {round_id}/{rounds}")
        client_results = []

        for client_id, indices in enumerate(client_indices):
            local_state, sample_count = train_local_model(
                global_state,
                x_train,
                y_train,
                indices,
                client_id,
                round_id, batch_size, learning_rate, local_epochs,
            )

            client_results.append((local_state, sample_count))

            print(
                f"  Client {client_id + 1}/{num_clients} finished "
                f"| training samples: {sample_count:,}"
            )

        # Estimate each client's global-model download and updated-model upload.
        round_download = num_clients * payload_bytes
        round_upload = num_clients * payload_bytes

        total_download_bytes += round_download
        total_upload_bytes += round_upload

        global_state = fedavg(client_results)
        global_model.load_state_dict(global_state)

        mae, rmse, _, _ = evaluate(
            global_model,
            x_test,
            y_test, batch_size,
        )

        cumulative_bytes = total_download_bytes + total_upload_bytes

        communication_history.append({
            "round": round_id,
            "test_mae": mae,
            "test_rmse": rmse,
            "download_bytes": round_download,
            "upload_bytes": round_upload,
            "total_round_bytes": round_download + round_upload,
            "cumulative_bytes": cumulative_bytes,
        })

        print(f"  Test MAE: {mae:.6f}")
        print(f"  Test RMSE: {rmse:.6f}")
        print(
            f"  Estimated cumulative communication: "
            f"{cumulative_bytes:,} bytes"
        )

    mae, rmse, predictions, actual = evaluate(
        global_model,
        x_test,
        y_test, batch_size,
    )

    if mode == "iid":
        split_description = (
            "Randomly shuffled training sequences distributed across clients"
        )
    else:
        split_description = (
            "Training sequences grouped by sensor; each client receives "
            "a distinct group of sensors"
        )

    return {
        "experiment": f"fedavg_{mode}_pems04",
        "data_type": "real_public_pems04",
        "real_world_results": True,
        "dataset": "PeMS04",
        "selected_sensor_count": len(data["sensor_indices"]),
        "selected_sensor_indices": data["sensor_indices"].tolist(),
        "client_distribution": mode,
        "client_split_description": split_description,
        "num_clients": num_clients,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": SEED,
        "sequence_length": 24,
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "mae": mae,
        "rmse": rmse,
        "metric_units": (
            "per-sensor MinMax-normalized traffic-flow units"
        ),
        "model_payload_bytes": payload_bytes,
        "download_bytes": total_download_bytes,
        "upload_bytes": total_upload_bytes,
        "total_communication_bytes": (
            total_download_bytes + total_upload_bytes
        ),
        "communication_estimate_excludes": [
            "network protocol overhead",
            "encryption and transport overhead",
            "retries and acknowledgements",
        ],
        "communication_history": communication_history,
        "predictions": predictions,
        "actual": actual,
        "model_state": copy.deepcopy(global_model.state_dict()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clients", type=int, default=NUM_CLIENTS)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--local-epochs", type=int, default=LOCAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--mode", choices=("iid", "non_iid", "both"), default="both")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_id = args.output_dir.name if args.output_dir else (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + "-" + uuid.uuid4().hex[:8]
    )
    output_dir = args.output_dir.resolve() if args.output_dir else RESULTS / "runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    print("Loading real PeMS04 data...")
    data = load_pems04()

    results = {}

    modes = ("iid", "non_iid") if args.mode == "both" else (args.mode,)
    for mode in modes:
        results[mode] = run_federated_experiment(data, mode, args.num_clients,
            args.rounds, args.local_epochs, args.batch_size, args.learning_rate)

    # Save metrics without prediction arrays or model tensors.
    metrics_output = {}

    for mode, result in results.items():
        metrics_output[mode] = {
            key: value
            for key, value in result.items()
            if key not in ("predictions", "actual", "model_state")
        }

        torch.save(
            result["model_state"],
            output_dir / f"federated_gru_real_{mode}.pt",
        )

        pd.DataFrame({
            "actual_normalized_flow": result["actual"],
            "predicted_normalized_flow": result["predictions"],
        }).to_csv(
            output_dir / f"federated_real_predictions_{mode}.csv",
            index=False,
        )

        pd.DataFrame(
            result["communication_history"]
        ).to_csv(
            output_dir / f"federated_real_communication_{mode}.csv",
            index=False,
        )

    metrics_path = output_dir / "federated_real_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics_output, file, indent=2)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "experiment": "federated",
            "num_clients": args.num_clients,
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "mode": args.mode,
        },
    }
    (output_dir / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print("FEDERATED RESULTS — REAL PeMS04")
    print(f"{'=' * 60}")

    for mode, result in results.items():
        print(f"\n{mode.upper()}")
        print(f"MAE: {result['mae']:.6f}")
        print(f"RMSE: {result['rmse']:.6f}")
        print(
            "Total estimated communication: "
            f"{result['total_communication_bytes']:,} bytes"
        )

    print(f"\nMetrics saved: {metrics_path}")
    print("Federated experiments completed successfully.")


if __name__ == "__main__":
    main()
