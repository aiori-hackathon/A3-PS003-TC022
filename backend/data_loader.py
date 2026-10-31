
"""Load and preprocess the public PeMS04 traffic dataset for NoCivicSense."""

from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from numpy.lib.stride_tricks import sliding_window_view


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "PEMS04.npz"

SEQUENCE_LENGTH = 24
TRAIN_RATIO = 0.8

# Use 32 evenly spaced sensors initially to keep CPU training manageable.
# The dataset itself contains 307 sensors.
NUM_SENSORS = 32


def load_pems04():
    """Load traffic flow and prepare chronological train/test sequences."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Place PEMS04.npz inside the project's data folder."
        )

    with np.load(DATA_PATH) as dataset:
        if "data" not in dataset.files:
            raise ValueError(
                f"Expected an array named 'data'. Found: {dataset.files}"
            )
        raw_data = dataset["data"]

    if raw_data.ndim != 3 or raw_data.shape[2] < 1:
        raise ValueError(
            f"Unexpected dataset shape: {raw_data.shape}"
        )

    # PeMS04 feature order: flow, occupancy, speed.
    # Traffic-flow prediction uses feature index 0.
    flow = raw_data[:, :, 0].astype(np.float32)

    # Select evenly spaced sensors deterministically.
    sensor_indices = np.linspace(
        0, flow.shape[1] - 1,
        num=min(NUM_SENSORS, flow.shape[1]),
        dtype=int,
    )
    flow = flow[:, sensor_indices]

    # Replace non-finite observations using each sensor's finite median.
    for sensor in range(flow.shape[1]):
        series = flow[:, sensor]
        valid = np.isfinite(series)

        if not valid.any():
            raise ValueError(
                f"Sensor {sensor_indices[sensor]} contains no finite values."
            )

        median_value = np.median(series[valid])
        series[~valid] = median_value

    # Split by time BEFORE fitting the scaler to prevent test-data leakage.
    split_time = int(len(flow) * TRAIN_RATIO)

    scaler = MinMaxScaler()
    scaler.fit(flow[:split_time])

    scaled_flow = scaler.transform(flow).astype(np.float32)

    def create_sequences(values):
        """Create windows: 24 observations predict the next observation."""
        windows = sliding_window_view(
            values, SEQUENCE_LENGTH + 1, axis=0
        )

        # Shape after windowing: sensors x windows x window_length.
        windows = windows.transpose(1, 0, 2)

        x = windows[:, :, :-1].reshape(
            -1, SEQUENCE_LENGTH, 1
        )
        y = windows[:, :, -1].reshape(-1, 1)

        return x.copy(), y.copy()

    # Training windows never predict beyond the training period.
    x_train, y_train = create_sequences(
        scaled_flow[:split_time]
    )

    # Test targets begin at split_time; their input histories may
    # include the preceding 24 observations from the training period.
    test_values = scaled_flow[
        split_time - SEQUENCE_LENGTH:
    ]
    x_test, y_test = create_sequences(test_values)

    # Verify that test windows correspond only to test-period targets.
    expected_test_samples = (
        len(scaled_flow) - split_time
    ) * flow.shape[1]

    if len(x_test) != expected_test_samples:
        raise RuntimeError(
            "Unexpected test sequence count. Check preprocessing."
        )

    print("=" * 55)
    print("NoCivicSense | PeMS04 Data Loader")
    print("=" * 55)
    print("DATA MODE: REAL PUBLIC DATASET")
    print(f"Dataset shape: {raw_data.shape}")
    print(f"Selected sensors: {len(sensor_indices)}")
    print(f"Sequence length: {SEQUENCE_LENGTH} time steps (2 hours)")
    print(f"Training time steps: {split_time}")
    print(f"Testing time steps: {len(flow) - split_time}")
    print(f"Training sequences: {len(x_train):,}")
    print(f"Testing sequences: {len(x_test):,}")
    print(f"Input shape: {x_train.shape}")
    print(f"Target shape: {y_train.shape}")
    print("Target: next-step traffic flow")
    print("=" * 55)

    return {
        "x_train": torch.from_numpy(x_train),
        "y_train": torch.from_numpy(y_train),
        "x_test": torch.from_numpy(x_test),
        "y_test": torch.from_numpy(y_test),
        "scaler": scaler,
        "sensor_indices": sensor_indices,
        "split_time": split_time,
        "data_type": "real_public_pems04",
    }


if __name__ == "__main__":
    data = load_pems04()

    print("\nSample input (first sequence):")
    print(data["x_train"][0].squeeze()[:5])

    print("\nSample target (normalized flow):")
    print(data["y_train"][0].item())
