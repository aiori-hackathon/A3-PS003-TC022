"""
NoCivicSense — FastAPI backend.

Serves saved experiment results and provides a validated endpoint for
configurable federated training runs. Every dashboard run is stored separately.
"""

import asyncio
import csv
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Literal
from datetime import datetime, timezone
import uuid
from pydantic import Field

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

METRICS_FILES = {
    "baseline": RESULTS / "baseline_real_metrics.json",
    "federated": RESULTS / "federated_real_metrics.json",
    "persistence": RESULTS / "persistence_real_metrics.json",
}

PREDICTION_FILES = {
    "centralized": RESULTS / "baseline_real_predictions.csv",
    "federated_iid": RESULTS / "federated_real_predictions_iid.csv",
    "federated_non_iid": RESULTS / "federated_real_predictions_non_iid.csv",
    "persistence": RESULTS / "persistence_real_predictions.csv",
}

COMMUNICATION_FILES = {
    "iid": RESULTS / "federated_real_communication_iid.csv",
    "non_iid": RESULTS / "federated_real_communication_non_iid.csv",
}

# Default prediction sample size for charts.
PREDICTION_SAMPLE_ROWS = int(os.environ.get("PREDICTION_SAMPLE_ROWS", "500"))

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

ALLOWED_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")

app = FastAPI(
    title="NoCivicSense API",
    description="Privacy-preserving traffic intelligence dashboard API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Job state (in-process; single-server deployment only)
# ---------------------------------------------------------------------------

_job_lock = asyncio.Lock()

_job_state: dict[str, Any] = {
    "running": False,
    "experiment": None,
    "status": "idle",   # idle | queued | running | completed | error
    "message": "",
    "run_id": None,
    "configuration": None,
    "exit_code": None,
    "exception_message": None,
}
_jobs: dict[str, dict[str, Any]] = {}


def _persist_job(run_id: str, **updates: Any) -> dict[str, Any]:
    """Update in-memory state and atomically persist the run manifest."""
    state = _jobs[run_id]
    state.update(updates)
    _job_state.update(state)
    manifest_path = RESULTS / "runs" / run_id / "run.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary_path.replace(manifest_path)
    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Any:
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Result file not found: {path.name}. "
                   "Run the corresponding experiment first.",
        )
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_csv_sample(path: Path, max_rows: int) -> list[dict[str, str]]:
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Prediction file not found: {path.name}. "
                   "Run the corresponding experiment first.",
        )
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(row)
    return rows


def _read_csv_rows_at(path: Path, row_indices: list[int]) -> dict[int, dict[str, str]]:
    """Read selected zero-based CSV rows while preserving their source positions."""
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Prediction file not found: {path.name}. Run the corresponding experiment first.",
        )

    wanted = set(row_indices)
    selected: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for index, row in enumerate(csv.DictReader(fh)):
            if index in wanted:
                selected[index] = row
                if len(selected) == len(wanted):
                    break
    return selected


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/experiments/runs")
async def list_experiment_runs() -> list[dict[str, Any]]:
    """List immutable configured federated runs, newest first."""
    runs = []
    run_root = RESULTS / "runs"
    if run_root.exists():
        for folder in run_root.iterdir():
            manifest_path = folder / "run.json"
            metrics_path = folder / "federated_real_metrics.json"
            if not manifest_path.exists():
                continue
            manifest = _read_json(manifest_path)
            manifest["results"] = _read_json(metrics_path) if metrics_path.exists() else {}
            # Keep full process output in run.json, but don't send it with every
            # history refresh. The per-run status endpoint returns the message tail.
            manifest.pop("stdout", None)
            manifest.pop("stderr", None)
            manifest.pop("traceback", None)
            if manifest.get("message"):
                manifest["message"] = manifest["message"][-1200:]
            runs.append(manifest)
    return sorted(runs, key=lambda item: item.get("created_at", ""), reverse=True)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/metrics")
async def get_metrics() -> dict[str, Any]:
    """Return all available experiment metrics."""
    result: dict[str, Any] = {}

    for key, path in METRICS_FILES.items():
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                result[key] = json.load(fh)
        else:
            result[key] = None

    return result


@app.get("/api/predictions/{experiment}")
async def get_predictions(
    experiment: Literal["centralized", "federated_iid", "federated_non_iid", "persistence"],
    rows: int = PREDICTION_SAMPLE_ROWS,
) -> dict[str, Any]:
    """
    Return a sample of prediction vs actual values from a result CSV.

    Query parameter `rows` controls the number of rows returned (default 500).
    Values are MinMax-normalized traffic-flow units, not vehicle counts.
    """
    if rows < 1 or rows > 5000:
        raise HTTPException(
            status_code=422,
            detail="`rows` must be between 1 and 5000.",
        )

    path = PREDICTION_FILES[experiment]
    sample = _read_csv_sample(path, rows)

    return {
        "experiment": experiment,
        "metric_units": "MinMax-normalized traffic-flow units",
        "rows_returned": len(sample),
        "rows_requested": rows,
        "disclaimer": (
            "Values are per-sensor MinMax-normalized traffic flow. "
            "They do not represent vehicle counts."
        ),
        "data": sample,
    }


@app.get("/api/predictions/{experiment}/sensor-snapshot")
async def get_sensor_snapshot(
    experiment: Literal["centralized", "federated_iid", "federated_non_iid", "persistence"],
    observation_index: int = 0,
) -> dict[str, Any]:
    """Return one saved prediction row per selected sensor for an observation index.

    Prediction CSVs are flattened in sensor-major order by the existing data loader.
    This additive endpoint makes that existing ordering available to the UI without
    changing the response format of `/api/predictions/{experiment}`.
    """
    if observation_index < 0:
        raise HTTPException(status_code=422, detail="`observation_index` must be non-negative.")

    baseline_path = METRICS_FILES["baseline"]
    if not baseline_path.exists():
        raise HTTPException(status_code=404, detail="Baseline metrics are required to identify selected sensors.")
    baseline = _read_json(baseline_path)
    sensor_indices = baseline.get("selected_sensor_indices", [])
    test_sequences = baseline.get("test_sequences", 0)
    sensor_count = len(sensor_indices)

    if not sensor_count or not test_sequences or test_sequences % sensor_count:
        raise HTTPException(status_code=422, detail="Saved sensor metadata is incomplete or inconsistent.")

    observations_per_sensor = test_sequences // sensor_count
    if observation_index >= observations_per_sensor:
        raise HTTPException(
            status_code=422,
            detail=f"`observation_index` must be less than {observations_per_sensor}.",
        )

    row_indices = [rank * observations_per_sensor + observation_index for rank in range(sensor_count)]
    raw_rows = _read_csv_rows_at(PREDICTION_FILES[experiment], row_indices)
    if len(raw_rows) != sensor_count:
        raise HTTPException(status_code=404, detail="The prediction file does not contain all expected sensor rows.")

    sensors = []
    for sensor_index, row_index in zip(sensor_indices, row_indices):
        row = raw_rows[row_index]
        try:
            actual = float(row["actual_normalized_flow"])
            predicted = float(row["predicted_normalized_flow"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=500, detail="Prediction CSV contains invalid sensor values.") from exc
        sensors.append({
            "sensor_index": sensor_index,
            "actual_normalized_flow": actual,
            "predicted_normalized_flow": predicted,
        })

    return {
        "experiment": experiment,
        "observation_index": observation_index,
        "metric_units": "per-sensor MinMax-normalized traffic-flow units",
        "data": sensors,
    }


@app.get("/api/communication/{mode}")
async def get_communication(
    mode: Literal["iid", "non_iid"],
) -> dict[str, Any]:
    """Return per-round communication history for a federated experiment."""
    path = COMMUNICATION_FILES[mode]
    raw_rows = _read_csv_sample(path, max_rows=1000)

    # CSV DictReader yields all values as strings; coerce to proper types.
    int_fields = {"round", "download_bytes", "upload_bytes", "total_round_bytes", "cumulative_bytes"}
    float_fields = {"test_mae", "test_rmse"}

    typed_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        typed: dict[str, Any] = {}
        for k, v in row.items():
            if k in int_fields:
                typed[k] = int(v) if v else 0
            elif k in float_fields:
                typed[k] = float(v) if v else 0.0
            else:
                typed[k] = v
        typed_rows.append(typed)

    return {
        "mode": mode,
        "rounds": typed_rows,
        "note": (
            "Communication bytes are estimated model-payload sizes only. "
            "Network protocol overhead, encryption, and retransmission "
            "are not included."
        ),
    }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

class ExperimentRequest(BaseModel):
    experiment: Literal["federated"]
    num_clients: int = Field(4, ge=2, le=16)
    rounds: int = Field(3, ge=1, le=20)
    local_epochs: int = Field(1, ge=1, le=10)
    batch_size: int = Field(2048, ge=1, le=65536)
    learning_rate: float = Field(0.001, gt=0, le=1)
    mode: Literal["iid", "non_iid", "both"] = "both"


async def _run_experiment(configuration: dict[str, Any], run_id: str) -> None:
    """Run training and persist process output, exit code, and terminal state."""
    output_dir = RESULTS / "runs" / run_id
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    python = str(Path(sys.executable))
    command = [
        python, "-m", "backend.federated_real",
        "--num-clients", str(configuration["num_clients"]),
        "--rounds", str(configuration["rounds"]),
        "--local-epochs", str(configuration["local_epochs"]),
        "--batch-size", str(configuration["batch_size"]),
        "--learning-rate", str(configuration["learning_rate"]),
        "--mode", configuration["mode"],
        "--output-dir", str(output_dir),
    ]
    process: subprocess.Popen[bytes] | None = None
    stdout = ""
    stderr = ""
    exception_message: str | None = None
    exception_traceback: str | None = None
    exit_code: int | None = None

    try:
        _persist_job(
            run_id,
            status="running",
            running=True,
            message=f"Training run {run_id} is running.",
            started_at=datetime.now(timezone.utc).isoformat(),
            python_executable=python,
            working_directory=str(ROOT),
            command=command,
        )
        # File redirection avoids Windows asyncio pipe restrictions. Popen is
        # started and waited in worker threads so SelectorEventLoop is supported.
        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            process = await asyncio.to_thread(
                subprocess.Popen,
                command,
                cwd=str(ROOT),
                stdout=stdout_file,
                stderr=stderr_file,
            )
            exit_code = await asyncio.to_thread(process.wait)
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")

        if exit_code == 0 and (output_dir / "federated_real_metrics.json").exists():
            status = "completed"
            message = "Training completed successfully."
        else:
            status = "error"
            exception_message = (
                "Training exited with code 0 but did not produce federated_real_metrics.json."
                if exit_code == 0 else f"Training subprocess exited with code {exit_code}."
            )
            message = exception_message

    except Exception as exc:  # noqa: BLE001
        status = "error"
        exception_message = f"{type(exc).__name__}: {exc!r}"
        exception_traceback = traceback.format_exc()
        if process is not None and process.poll() is None:
            process.kill()
            try:
                await asyncio.to_thread(process.wait)
            except Exception:  # noqa: BLE001
                pass
        exit_code = process.returncode if process is not None else None
        if stdout_path.exists():
            stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        if stderr_path.exists():
            stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        message = f"{exception_message}\n{exception_traceback}"
    except asyncio.CancelledError:
        status = "error"
        exception_message = "Training task was cancelled while the API process was shutting down."
        exception_traceback = traceback.format_exc()
        if process is not None and process.poll() is None:
            process.kill()
            try:
                await asyncio.to_thread(process.wait)
            except Exception:  # noqa: BLE001
                pass
        exit_code = process.returncode if process is not None else None
        if stdout_path.exists():
            stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        if stderr_path.exists():
            stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        message = f"{exception_message}\n{exception_traceback}"

    tail = "\n".join(part for part in (stdout[-5000:], stderr[-5000:]) if part)
    if status == "error" and tail:
        message = f"{message}\n\nSubprocess output:\n{tail}"
    elif status == "completed" and tail:
        message = tail[-10000:]

    _persist_job(
        run_id,
        status=status,
        running=False,
        message=message[-15000:],
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        exception_message=exception_message,
        traceback=exception_traceback,
            stdout_log=str(stdout_path),
            stderr_log=str(stderr_path),
        completed_at=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/api/experiments/run")
async def run_experiment(
    body: ExperimentRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Queue one validated federated run in a new timestamped folder."""
    async with _job_lock:
        if _job_state["running"]:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Experiment '{_job_state['experiment']}' is already running. "
                    "Wait for it to finish before starting another."
                ),
            )
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
        configuration = body.model_dump()
        run_dir = RESULTS / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        state = {
            "run_id": run_id,
            "experiment": "federated",
            "configuration": configuration,
            "status": "queued",
            "running": True,
            "message": "Run accepted and queued.",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stdout": "",
            "stderr": "",
            "exit_code": None,
            "exception_message": None,
        }
        _jobs[run_id] = state
        _persist_job(run_id)

    background_tasks.add_task(_run_experiment, configuration, run_id)
    return {
        "accepted": True,
        "experiment": "federated",
        "run_id": run_id,
        "configuration": configuration,
        "status": "queued",
        "message": f"Federated experiment queued as run {run_id}.",
    }


@app.get("/api/experiments/status")
async def experiment_status(run_id: str | None = None) -> dict[str, Any]:
    """Return status for the requested run or the latest saved run."""
    selected_id = run_id or _job_state.get("run_id")
    if selected_id and not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", selected_id):
        raise HTTPException(status_code=422, detail="Invalid run ID.")
    if selected_id and selected_id in _jobs:
        state = dict(_jobs[selected_id])
        state["stdout"] = state.get("stdout", "")[-5000:]
        state["stderr"] = state.get("stderr", "")[-5000:]
        return state
    if selected_id:
        manifest = RESULTS / "runs" / selected_id / "run.json"
        if manifest.exists():
            state = _read_json(manifest)
            if "status" not in state:
                has_metrics = (manifest.parent / "federated_real_metrics.json").exists()
                state["status"] = "completed" if has_metrics else "error"
                state["running"] = False
                state["message"] = "Training completed successfully." if has_metrics else "Run metadata exists but training results are missing."
            return state
    if not selected_id:
        run_root = RESULTS / "runs"
        manifests = sorted(
            run_root.glob("*/run.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ) if run_root.exists() else []
        if manifests:
            state = _read_json(manifests[0])
            if "status" not in state:
                has_metrics = (manifests[0].parent / "federated_real_metrics.json").exists()
                state["status"] = "completed" if has_metrics else "error"
                state["running"] = False
                state["message"] = "Training completed successfully." if has_metrics else "Run metadata exists but training results are missing."
            return state
    return {
        "running": _job_state["running"],
        "experiment": _job_state["experiment"],
        "status": _job_state["status"],
        "message": _job_state["message"],
        "run_id": _job_state["run_id"],
        "configuration": _job_state["configuration"],
        "exit_code": _job_state.get("exit_code"),
        "exception_message": _job_state.get("exception_message"),
    }
