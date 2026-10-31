// API types for NoCivicSense

export interface BaselineMetrics {
  experiment: string;
  data_type: string;
  real_world_results: boolean;
  dataset: string;
  selected_sensor_count: number;
  selected_sensor_indices: number[];
  sequence_length: number;
  prediction_horizon: string;
  epochs: number;
  batch_size: number;
  seed: number;
  train_sequences: number;
  test_sequences: number;
  mae: number;
  rmse: number;
  metric_units: string;
  device: string;
}

export interface CommRound {
  round: number;
  test_mae: number;
  test_rmse: number;
  download_bytes: number;
  upload_bytes: number;
  total_round_bytes: number;
  cumulative_bytes: number;
}

export interface FederatedExperiment {
  experiment: string;
  data_type: string;
  real_world_results: boolean;
  dataset: string;
  selected_sensor_count: number;
  selected_sensor_indices: number[];
  client_distribution: string;
  client_split_description: string;
  num_clients: number;
  rounds: number;
  local_epochs: number;
  batch_size: number;
  learning_rate: number;
  seed: number;
  sequence_length: number;
  train_samples: number;
  test_samples: number;
  mae: number;
  rmse: number;
  metric_units: string;
  model_payload_bytes: number;
  download_bytes: number;
  upload_bytes: number;
  total_communication_bytes: number;
  communication_estimate_excludes: string[];
  communication_history: CommRound[];
}

export interface FederatedMetrics {
  iid: FederatedExperiment;
  non_iid: FederatedExperiment;
}

export interface PersistenceMetrics {
  experiment: string;
  data_type: string;
  mae: number;
  rmse: number;
  prediction_rule: string;
  sequence_length: number;
  selected_sensors: number;
  note: string;
}

export interface AllMetrics {
  baseline: BaselineMetrics | null;
  federated: FederatedMetrics | null;
  persistence: PersistenceMetrics | null;
}

export interface PredictionRow {
  actual_normalized_flow: string;
  predicted_normalized_flow: string;
}

export interface PredictionsResponse {
  experiment: string;
  metric_units: string;
  rows_returned: number;
  rows_requested: number;
  disclaimer: string;
  data: PredictionRow[];
}

export interface SensorSnapshotRow {
  sensor_index: number;
  actual_normalized_flow: number;
  predicted_normalized_flow: number;
}

export interface SensorSnapshotResponse {
  experiment: ExperimentName;
  observation_index: number;
  metric_units: string;
  data: SensorSnapshotRow[];
}

export interface CommunicationResponse {
  mode: string;
  rounds: CommRound[];
  note: string;
}

export type ExperimentName = 'centralized' | 'federated_iid' | 'federated_non_iid' | 'persistence';

export interface ExperimentStatus {
  running: boolean;
  experiment: ExperimentName | 'federated' | null;
  status: 'idle' | 'queued' | 'running' | 'completed' | 'success' | 'error';
  message: string;
  exit_code?: number | null;
  exception_message?: string | null;
  run_id?: string | null;
  configuration?: FederatedRunConfig | null;
}

export interface FederatedRunConfig {
  experiment: 'federated';
  num_clients: number;
  rounds: number;
  local_epochs: number;
  batch_size: number;
  learning_rate: number;
  mode: 'iid' | 'non_iid' | 'both';
}

export interface FederatedRun {
  run_id: string;
  created_at: string;
  configuration: FederatedRunConfig;
  results: Partial<FederatedMetrics>;
  status?: string;
  message?: string;
  exit_code?: number | null;
  exception_message?: string | null;
}
