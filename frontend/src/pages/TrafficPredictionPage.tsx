import { useCallback, useEffect, useMemo, useState } from 'react';
import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../api/client';
import type { AllMetrics, ExperimentName, PredictionRow, PredictionsResponse, SensorSnapshotResponse } from '../api/types';
import { EmptyState, ErrorState, LoadingState } from '../components/States';

const options: { value: ExperimentName; label: string }[] = [
  { value: 'centralized', label: 'Centralized GRU' }, { value: 'federated_iid', label: 'FedAvg · IID' },
  { value: 'federated_non_iid', label: 'FedAvg · non-IID' }, { value: 'persistence', label: 'Persistence baseline' },
];
const rangeOptions = [100, 250, 500, 1000];

export function TrafficPredictionPage() {
  const [experiment, setExperiment] = useState<ExperimentName>('centralized');
  const [range, setRange] = useState(500);
  const [response, setResponse] = useState<PredictionsResponse | null>(null);
  const [metrics, setMetrics] = useState<AllMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [snapshot, setSnapshot] = useState<SensorSnapshotResponse | null>(null);
  const [snapshotError, setSnapshotError] = useState<string | null>(null);
  const [replayIndex, setReplayIndex] = useState(0);
  const [playing, setPlaying] = useState(false);

  const load = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const [predictions, allMetrics] = await Promise.all([api.getPredictions(experiment, range), api.getMetrics()]);
      setResponse(predictions); setMetrics(allMetrics);
    } catch (e) { setError(e instanceof Error ? e.message : 'Unable to load prediction data'); }
    finally { setLoading(false); }
  }, [experiment, range]);
  useEffect(() => { const timer = window.setTimeout(() => { void load(); }, 0); return () => window.clearTimeout(timer); }, [load]);

  const sensorCount = metrics?.baseline?.selected_sensor_indices?.length ?? 0;
  const replayLength = sensorCount && metrics?.baseline?.test_sequences
    ? Math.floor(metrics.baseline.test_sequences / sensorCount) : 0;
  useEffect(() => {
    const timer = window.setTimeout(() => {
      setSnapshotError(null);
      void api.getSensorSnapshot(experiment, replayIndex)
        .then(setSnapshot)
        .catch(e => setSnapshotError(e instanceof Error ? e.message : 'Unable to load this saved sensor observation'));
    }, 0);
    return () => window.clearTimeout(timer);
  }, [experiment, replayIndex]);
  useEffect(() => {
    if (!playing || replayLength < 1) return;
    const timer = window.setInterval(() => setReplayIndex(index => index + 1 >= replayLength ? 0 : index + 1), 1200);
    return () => window.clearInterval(timer);
  }, [playing, replayLength]);

  const chartData = useMemo(() => (response?.data ?? []).map((row: PredictionRow, index) => ({
    index: index + 1, actual: Number(row.actual_normalized_flow), predicted: Number(row.predicted_normalized_flow),
  })), [response]);
  const runMetrics = experiment === 'centralized' ? metrics?.baseline
    : experiment === 'persistence' ? metrics?.persistence
    : experiment === 'federated_iid' ? metrics?.federated?.iid : metrics?.federated?.non_iid;
  const activityBins = chartData.slice(0, 48).map(point => ({ ...point, intensity: Math.max(0, Math.min(100, Math.round(point.actual * 100))) }));

  return (
    <div className="page-body">
      <div className="page-header">
        <div className="page-kicker">PeMS04 · Historical replay</div>
        <h1 className="page-title">Traffic Intelligence</h1>
        <p className="page-description">Explore saved actual-versus-predicted traffic-flow observations for each experiment. This is a replay of historical test-set results, not a live sensor connection.</p>
      </div>

      <div className="info-banner"><span>◷</span><span><strong>Historical data replay.</strong> The charts and activity indicators below are derived only from the selected experiment’s saved prediction CSV. No live traffic feed or sensor location data is connected.</span></div>

      <section className="card section-gap">
        <div className="card-header"><div><div className="card-title">Prediction controls</div><div className="card-subtitle">Choose a saved run and the number of CSV rows to request</div></div><span className="status-tag"><i className="status-dot" />Historical snapshot</span></div>
        <div className="control-row">
          <label className="control-field"><span className="eyebrow">Experiment</span><select id="experiment-select" className="select" value={experiment} onChange={event => { setExperiment(event.target.value as ExperimentName); setReplayIndex(0); setPlaying(false); }}>{options.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label>
          <label className="control-field"><span className="eyebrow">Visible observations</span><select className="select" value={range} onChange={event => setRange(Number(event.target.value))}>{rangeOptions.map(value => <option key={value} value={value}>{value.toLocaleString()} rows</option>)}</select></label>
          <div className="control-summary"><span className="eyebrow">CSV rows returned</span><strong>{response?.rows_returned ?? '—'}</strong></div>
        </div>
      </section>

      <section className="kpi-grid section-gap">
        <div className="kpi-card"><div className="kpi-head">Selected model<span className="kpi-icon">⌁</span></div><div className="kpi-value" style={{ fontSize: 16 }}>{options.find(option => option.value === experiment)?.label}</div><div className="kpi-foot">loaded from saved predictions</div></div>
        <div className="kpi-card"><div className="kpi-head">Test MAE<span className="kpi-icon">Δ</span></div><div className="kpi-value">{runMetrics ? runMetrics.mae.toFixed(5) : '—'}</div><div className="kpi-foot">normalized flow units</div></div>
        <div className="kpi-card"><div className="kpi-head">Test RMSE<span className="kpi-icon">σ</span></div><div className="kpi-value">{runMetrics ? runMetrics.rmse.toFixed(5) : '—'}</div><div className="kpi-foot">normalized flow units</div></div>
        <div className="kpi-card"><div className="kpi-head">Prediction horizon<span className="kpi-icon">↗</span></div><div className="kpi-value" style={{ fontSize: 17 }}>{metrics?.baseline?.prediction_horizon ?? '—'}</div><div className="kpi-foot">context: {metrics?.baseline?.sequence_length ?? '—'} steps</div></div>
      </section>

      <div className="content-grid section-gap">
        <section className="card">
          <div className="card-header"><div><div className="card-title">Actual vs predicted flow</div><div className="card-subtitle">Observation index · MinMax-normalized per-sensor values</div></div><span className="badge badge-accent">{response?.rows_returned ?? 0} points</span></div>
          {loading ? <LoadingState message="Loading saved prediction rows…" /> : error ? <ErrorState message={error} onRetry={load} /> : chartData.length === 0 ? <EmptyState message="No saved predictions are available for this experiment." /> : (
            <ResponsiveContainer width="100%" height={365}><ComposedChart data={chartData} margin={{ top: 8, right: 12, bottom: 14, left: 0 }}><defs><linearGradient id="trafficActualFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="var(--color-accent)" stopOpacity={.18} /><stop offset="100%" stopColor="var(--color-accent)" stopOpacity={0} /></linearGradient></defs><CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-dim)" /><XAxis dataKey="index" tick={{ fill: 'var(--color-text-muted)', fontSize: 9 }} label={{ value: 'CSV observation index', position: 'insideBottom', offset: -7, fill: 'var(--color-text-muted)', fontSize: 9 }} /><YAxis domain={[0, 1]} tick={{ fill: 'var(--color-text-muted)', fontSize: 9 }} label={{ value: 'Normalized flow', angle: -90, position: 'insideLeft', fill: 'var(--color-text-muted)', fontSize: 9 }} /><Tooltip labelFormatter={label => `Observation ${label}`} formatter={value => Number(value).toFixed(6)} contentStyle={{ background: 'var(--color-surface-1)', borderColor: 'var(--color-border)', borderRadius: 12 }} /><Area type="monotone" dataKey="actual" name="Actual flow" stroke="var(--color-accent)" fill="url(#trafficActualFill)" strokeWidth={1.7} dot={false} /><Line type="monotone" dataKey="predicted" name="Predicted flow" stroke="var(--color-info)" strokeWidth={1.8} dot={false} strokeDasharray="5 3" activeDot={{ r: 4 }} /><ReferenceLine y={.5} stroke="var(--color-border)" strokeDasharray="3 5" /></ComposedChart></ResponsiveContainer>
          )}
        </section>
        <section className="card">
          <div className="card-header"><div><div className="card-title">Historical activity strip</div><div className="card-subtitle">First 48 actual values in the selected CSV window</div></div><span className="badge">Replay</span></div>
          {loading ? <div className="skeleton" /> : error ? <p className="state-body">Activity strip unavailable: {error}</p> : activityBins.length === 0 ? <EmptyState message="No actual values available." /> : <div className="activity-grid" role="img" aria-label="Historical flow intensity strip derived from actual normalized flow values"><div className="activity-strip">{activityBins.map(point => <span key={point.index} title={`Observation ${point.index}: actual ${point.actual.toFixed(4)}`} style={{ height: `${Math.max(5, point.intensity)}%` }} />)}</div><div className="activity-scale"><span>Lower normalized flow</span><span>Higher normalized flow</span></div><div className="replay-callout"><span className="replay-pulse" />Intensity is derived from saved actual flow values. It does not represent live conditions.</div></div>}
        </section>
      </div>

      <section className="card section-gap">
        <div className="card-header"><div><div className="card-title">Historical sensor replay</div><div className="card-subtitle">Synchronized observation from the selected saved prediction CSV · normalized flow</div></div><span className="badge badge-yellow">Historical · not live</span></div>
        <div className="replay-controls"><button className="btn btn-primary btn-small" onClick={() => setPlaying(value => !value)} disabled={!replayLength}>{playing ? 'Pause' : 'Play'}</button><button className="btn btn-ghost btn-small" onClick={() => { setPlaying(false); setReplayIndex(0); }}>Reset</button><span className="replay-index">Observation <strong>{replayIndex.toLocaleString()}</strong> / {Math.max(0, replayLength - 1).toLocaleString()}</span><span className="replay-intensity">Mean intensity <strong>{snapshot?.data.length ? (snapshot.data.reduce((sum, row) => sum + row.actual_normalized_flow, 0) / snapshot.data.length).toFixed(3) : '—'}</strong></span></div>
        <label className="replay-progress"><span className="sr-only">Historical observation position</span><input type="range" min="0" max={Math.max(0, replayLength - 1)} value={Math.min(replayIndex, Math.max(0, replayLength - 1))} onChange={event => { setPlaying(false); setReplayIndex(Number(event.target.value)); }} disabled={!replayLength} /><span>{replayLength ? `${Math.round((replayIndex / Math.max(1, replayLength - 1)) * 100)}% through saved test observations` : 'Observation range unavailable'}</span></label>
        <div className="sensor-legend"><span><i className="sensor-low" />Low</span><span><i className="sensor-medium" />Medium</span><span><i className="sensor-high" />High</span><small>Bins reflect normalized actual flow (0–1); they are not congestion labels.</small></div>
        {snapshotError ? <div className="disclaimer" role="alert">Sensor snapshot unavailable: {snapshotError} <button className="btn btn-ghost btn-small" onClick={() => { setSnapshotError(null); void api.getSensorSnapshot(experiment, replayIndex).then(setSnapshot).catch(e => setSnapshotError(e instanceof Error ? e.message : 'Unable to load this saved sensor observation')); }}>Retry</button></div> : snapshot?.data.length ? <div className="sensor-grid" aria-label={`${snapshot.data.length} selected sensors at the current historical observation`}>{snapshot.data.map(row => {
          const value = row.actual_normalized_flow;
          const level = value < 1 / 3 ? 'low' : value < 2 / 3 ? 'medium' : 'high';
          return <div key={row.sensor_index} className={`sensor-cell ${level}`} title={`Sensor ${row.sensor_index} · actual ${value.toFixed(5)} · predicted ${row.predicted_normalized_flow.toFixed(5)}`}><span>Sensor {row.sensor_index}</span><strong>{value.toFixed(3)}</strong></div>;
        })}</div> : <div className="skeleton" />}
      </section>

      <section className="card"><div className="section-heading"><div><h2 className="section-title">Reading this view</h2><p className="section-caption">Context for the observation series</p></div></div><div className="grid-3"><div className="insight-item"><strong>Normalized units</strong><span>Values are MinMax-normalized per-sensor flow values, not vehicle counts.</span></div><div className="insight-item"><strong>Saved file order</strong><span>Rows follow the selected experiment’s CSV order. Sensor identifiers are not present in the prediction response.</span></div><div className="insight-item"><strong>Forecast setup</strong><span>Each prediction uses a 24-observation context window to forecast the next 5-minute step.</span></div></div></section>
    </div>
  );
}
