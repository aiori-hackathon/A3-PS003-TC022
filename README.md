# NoCivicSense — Privacy-Preserving Traffic Intelligence

**Hackathon project for PS-003: Federated Traffic-Flow Prediction with Communication-Cost Evaluation**

Predicts short-horizon highway traffic flow using a GRU neural network, evaluates centralized vs. federated (FedAvg IID and non-IID) training, and quantifies the communication overhead of the federated approach — all on the real public **PeMS04** dataset.

---

## Project Structure

```text
aiori/
├── backend/
│   ├── api.py                  ← FastAPI application
│   ├── model.py                ← GRU model definition
│   ├── data_loader.py          ← PeMS04 loader + preprocessing
│   ├── train_baseline_real.py  ← Centralized GRU training
│   ├── federated_real.py       ← FedAvg IID + non-IID
│   ├── persistence_baseline.py ← Persistence rule evaluation
│   └── compare_real_results.py ← Report + charts generator
├── frontend/                   ← Vite + React + TypeScript
├── data/
│   └── PEMS04.npz              ← PeMS04 dataset
├── results/                    ← Saved metrics, predictions, models
└── .venv/                      ← Python virtual environment
```

---

## Quick Start — Windows Command Prompt

### Prerequisites

- Python 3.12 with the existing `.venv` (already set up)
- Node.js 18+ with npm
- Dataset file at `data\PEMS04.npz` (already present)

### 1. Start the Backend API

Open a Command Prompt in the project root:

```cmd
cd /d C:\Users\ronal\Downloads\aiori
.venv\Scripts\uvicorn backend.api:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

Interactive docs: `http://localhost:8000/docs`

### 2. Start the Frontend

Open a **second** Command Prompt in the project root:

```cmd
cd /d C:\Users\ronal\Downloads\aiori\frontend
npm run dev
```

The dashboard will open at `http://localhost:5173`.

---

## Environment Variables

Copy `.env.example` if needed. The frontend dev server proxies `/api` to `http://localhost:8000` automatically — no additional configuration is required for local development.

Optional backend environment variables:

| Variable | Default | Description |
|---|---|---|
| `CORS_ORIGINS` | `http://localhost:5173,...` | Comma-separated allowed origins |
| `PREDICTION_SAMPLE_ROWS` | `500` | Max rows returned by `/api/predictions/{experiment}` |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/metrics` | All experiment metrics (reads JSON files) |
| GET | `/api/predictions/{experiment}` | Prediction sample CSV rows |
| GET | `/api/communication/{mode}` | Per-round communication history |
| POST | `/api/experiments/run` | Launch a configured federated run |
| GET | `/api/experiments/status` | Current/last job status and configuration |
| GET | `/api/experiments/runs` | Saved configured runs, newest first |

The run request accepts:

- `experiment`: `"federated"`
- `num_clients`: 2–16
- `rounds`: 1–20
- `local_epochs`: 1–10
- `batch_size`: 1–65,536
- `learning_rate`: greater than 0 and at most 1
- `mode`: `"iid"`, `"non_iid"`, or `"both"`

Each dashboard run is saved under a new timestamped `results/runs/` directory with a run manifest, metrics, predictions, communication history, and model checkpoint. Existing baseline and experiment artifacts are left in place.

---

## Important Disclosures

- **Federated clients are simulated on one machine.** No real network boundary or privacy sandbox exists between clients.
- **Dataset is PeMS04** — California highway traffic (Caltrans District 4), not Indian or Vellore traffic data.
- **All error metrics are MinMax-normalized** — they are not vehicle counts.
- **Communication figures are model-payload estimates only** — protocol overhead is excluded.

See the **About & Methodology** page in the dashboard for full details.

---

## Frontend Pages

| Page | Description |
|---|---|
| Overview | Dataset summary, all experiment metrics, communication cost |
| Traffic Prediction | Actual vs. predicted chart, experiment selector |
| Federated Learning | Error comparison and round-by-round convergence |
| Communication Cost | IID and non-IID payload totals and per-round upload/download history |
| Experiments | Run training scripts, monitor status, view output |
| About & Methodology | Problem statement, model design, limitations |


Youtube link of the demonstration: https://youtu.be/6q5g5uAnJVg?si=16DScdLUBLH7X7TZ
