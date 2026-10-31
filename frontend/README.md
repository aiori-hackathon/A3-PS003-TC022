# NoCivicSense Dashboard

React, Vite, and TypeScript frontend for the NoCivicSense PeMS04 traffic-flow experiments.

## Start

From this directory, run `npm run dev`. The Vite development server listens on port 5173 and proxies `/api` requests to the FastAPI backend at `http://localhost:8000`.

From the project root, start the API with `.venv\Scripts\uvicorn backend.api:app --reload --port 8000`.

## Pages

- Overview: saved centralized, IID FedAvg, non-IID FedAvg, and persistence metrics.
- Traffic Prediction: actual and predicted values from saved result CSV files.
- Federated Learning: model error comparisons and convergence history.
- Communication Cost: saved IID and non-IID communication totals and round histories.
- Experiments: launch the existing Python scripts and follow their status and output.
- About & Methodology: dataset, model, training, and limitations.

All displayed experimental values are read through the API from the existing `results/` files.
