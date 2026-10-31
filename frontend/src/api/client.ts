// Typed fetch wrappers for NoCivicSense API
import type {
  AllMetrics,
  CommunicationResponse,
  ExperimentName,
  ExperimentStatus,
  PredictionsResponse,
  SensorSnapshotResponse,
  FederatedRun,
  FederatedRunConfig,
} from './types';

const BASE = '/api';

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail?.detail ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail?.detail ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

export const api = {
  getMetrics: (): Promise<AllMetrics> =>
    get('/metrics'),

  getPredictions: (experiment: ExperimentName, rows = 500): Promise<PredictionsResponse> =>
    get(`/predictions/${experiment}?rows=${rows}`),

  getSensorSnapshot: (experiment: ExperimentName, observationIndex: number): Promise<SensorSnapshotResponse> =>
    get(`/predictions/${experiment}/sensor-snapshot?observation_index=${observationIndex}`),

  getCommunication: (mode: 'iid' | 'non_iid'): Promise<CommunicationResponse> =>
    get(`/communication/${mode}`),

  runExperiment: (configuration: FederatedRunConfig): Promise<{ accepted: boolean; message: string; run_id: string }> =>
    post('/experiments/run', configuration),

  getExperimentRuns: (): Promise<FederatedRun[]> => get('/experiments/runs'),

  getStatus: (runId?: string): Promise<ExperimentStatus> =>
    get(`/experiments/status${runId ? `?run_id=${encodeURIComponent(runId)}` : ''}`),

  health: (): Promise<{ status: string }> =>
    get('/health'),
};
