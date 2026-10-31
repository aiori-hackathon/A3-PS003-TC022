import { useCallback, useEffect, useMemo, useState } from 'react';
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../api/client';
import type { AllMetrics, CommunicationResponse, ExperimentName, PredictionRow } from '../api/types';
import { EmptyState, ErrorState, LoadingState } from '../components/States';
import { MetricCard } from '../components/MetricCard';

const labels: Record<ExperimentName, string> = {
  centralized: 'Centralized GRU', federated_iid: 'FedAvg · IID',
  federated_non_iid: 'FedAvg · non-IID', persistence: 'Persistence',
};
const fmt = (n: number) => n.toFixed(5);
const bytes = (n: number) => n >= 1_000_000 ? `${(n / 1_000_000).toFixed(2)} MB` : `${(n / 1_000).toFixed(1)} KB`;

export function OverviewPage() {
  const [metrics, setMetrics] = useState<AllMetrics | null>(null);
  const [predictions, setPredictions] = useState<PredictionRow[]>([]);
  const [iidComm, setIidComm] = useState<CommunicationResponse | null>(null);
  const [nonIidComm, setNonIidComm] = useState<CommunicationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sampleRange, setSampleRange] = useState(100);
  const [performanceMetric, setPerformanceMetric] = useState<'mae' | 'rmse'>('mae');

  const load = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const [m, p, iid, nonIid] = await Promise.all([
        api.getMetrics(), api.getPredictions('centralized', 250),
        api.getCommunication('iid'), api.getCommunication('non_iid'),
      ]);
      setMetrics(m); setPredictions(p.data); setIidComm(iid); setNonIidComm(nonIid);
    } catch (e) { setError(e instanceof Error ? e.message : 'Unable to load the saved experiment results'); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { const timer = window.setTimeout(() => { void load(); }, 0); return () => window.clearTimeout(timer); }, [load]);

  const b = metrics?.baseline;
  const f = metrics?.federated;
  const p = metrics?.persistence;
  const experiments = useMemo(() => [
    { name: labels.centralized, mae: b?.mae, rmse: b?.rmse },
    { name: labels.federated_iid, mae: f?.iid.mae, rmse: f?.iid.rmse },
    { name: labels.federated_non_iid, mae: f?.non_iid.mae, rmse: f?.non_iid.rmse },
    { name: labels.persistence, mae: p?.mae, rmse: p?.rmse },
  ], [b, f, p]);
  const ranked = experiments.filter(item => item.mae != null).sort((a, c) => a.mae! - c.mae!);
  const performanceRows = experiments.map(row => ({ name: row.name, value: row[performanceMetric] }));
  const bestExperiment = performanceRows.filter(row => row.value != null).sort((a, c) => a.value! - c.value!)[0]?.name;
  const selectedSensors = b?.selected_sensor_count ?? f?.iid.selected_sensor_count;
  const sensorCount = b?.selected_sensor_indices?.length ? Math.max(...b.selected_sensor_indices) + 1 : undefined;
  const iidTotal = f?.iid.total_communication_bytes;
  const nonIidTotal = f?.non_iid.total_communication_bytes;
  const chartRows = predictions.slice(0, sampleRange).map((row, index) => ({
    index: index + 1, actual: Number(row.actual_normalized_flow), predicted: Number(row.predicted_normalized_flow),
  }));
  const communicationRows = (iidComm?.rounds ?? []).map((round, index) => ({
    round: `R${round.round}`, iidDownload: round.download_bytes, iidUpload: round.upload_bytes,
    nonIidDownload: nonIidComm?.rounds[index]?.download_bytes,
    nonIidUpload: nonIidComm?.rounds[index]?.upload_bytes,
  }));
  const relativeGap = f ? Math.abs(f.iid.mae - f.non_iid.mae) / Math.min(f.iid.mae, f.non_iid.mae) * 100 : null;
  const fedVsPersistence = f && p ? ((p.mae - f.iid.mae) / p.mae) * 100 : null;

  if (loading) return <div className="page-body"><LoadingState message="Reading saved PeMS04 results…" /></div>;
  if (error) return <div className="page-body"><ErrorState message={error} onRetry={load} /></div>;
  if (!metrics) return <div className="page-body"><EmptyState message="No metrics JSON is available yet. Run an experiment to create saved results." /></div>;

  return (
    <div className="page-body">
      <section className="hero-panel">
        <div>
          <div className="page-kicker">PS-003 · Privacy-preserving traffic intelligence</div>
          <h1 className="hero-title">Forecast the flow.<br /><span>Keep the data local.</span></h1>
          <p className="hero-copy">A research prototype evaluating GRU traffic-flow prediction with centralized training and simulated FedAvg clients on California’s PeMS04 benchmark.</p>
          <div className="hero-meta"><span className="badge badge-accent"><span className="status-dot" style={{ background: 'var(--color-accent)' }} />PeMS04 · public dataset</span><span className="badge">{selectedSensors ?? '—'} selected detectors</span><span className="badge">{b?.prediction_horizon ?? '—'}</span></div>
        </div>
        <div className="hero-orbit"><div className="orbit-ring" /><div className="orbit-core">Σ</div><i className="orbit-node n1" /><i className="orbit-node n2" /><i className="orbit-node n3" /><i className="orbit-node n4" /><span className="orbit-caption">{f?.iid.num_clients ?? '—'} CLIENT NODES · FEDAVG</span></div>
      </section>

      <div className="section-heading"><div><h2 className="section-title">Experiment snapshot</h2><p className="section-caption">Metrics loaded from saved PeMS04 experiment outputs</p></div><button className="btn btn-ghost btn-small" onClick={() => { void load(); }}>↻ Refresh data</button></div>
      <section className="kpi-grid section-gap">
        <div className="kpi-card"><div className="kpi-head">Dataset sensors<span className="kpi-icon">⌖</span></div><div className="kpi-value">{sensorCount ?? '—'}</div><div className="kpi-foot">source indices in saved run metadata</div></div>
        <div className="kpi-card"><div className="kpi-head">Selected sensors<span className="kpi-icon">▦</span></div><div className="kpi-value">{selectedSensors ?? '—'}</div><div className="kpi-foot">used for current experiments</div></div>
        <div className="kpi-card"><div className="kpi-head">Training sequences<span className="kpi-icon">↗</span></div><div className="kpi-value">{b?.train_sequences?.toLocaleString() ?? '—'}</div><div className="kpi-foot">chronological training windows</div></div>
        <div className="kpi-card"><div className="kpi-head">Testing sequences<span className="kpi-icon">⌁</span></div><div className="kpi-value">{b?.test_sequences?.toLocaleString() ?? '—'}</div><div className="kpi-foot">held-out evaluation windows</div></div>
      </section>
      <section className="kpi-grid section-gap">
        <div className="kpi-card"><div className="kpi-head">Centralized MAE<span className="kpi-icon">C</span></div><div className="kpi-value">{b ? fmt(b.mae) : '—'}</div><div className="kpi-foot">normalized flow units</div></div>
        <div className="kpi-card"><div className="kpi-head">FedAvg IID MAE<span className="kpi-icon">I</span></div><div className="kpi-value">{f ? fmt(f.iid.mae) : '—'}</div><div className="kpi-foot">normalized flow units</div></div>
        <div className="kpi-card"><div className="kpi-head">FedAvg non-IID MAE<span className="kpi-icon">N</span></div><div className="kpi-value">{f ? fmt(f.non_iid.mae) : '—'}</div><div className="kpi-foot">normalized flow units</div></div>
        <div className="kpi-card"><div className="kpi-head">FedAvg communication<span className="kpi-icon">⇅</span></div><div className="kpi-value">{iidTotal != null && nonIidTotal != null ? bytes(iidTotal + nonIidTotal) : '—'}</div><div className="kpi-foot">combined IID + non-IID payload</div></div>
      </section>

      <div className="content-grid section-gap">
        <section className="card">
          <div className="card-header"><div><div className="card-title">Model performance</div><div className="card-subtitle">Actual saved metrics · lower error is better</div></div><div className="segmented" role="group" aria-label="Select performance metric"><button aria-pressed={performanceMetric === 'mae'} onClick={() => setPerformanceMetric('mae')}>MAE</button><button aria-pressed={performanceMetric === 'rmse'} onClick={() => setPerformanceMetric('rmse')}>RMSE</button></div></div>
          <div className="chart-wrap"><ResponsiveContainer width="100%" height={290}><BarChart data={performanceRows} margin={{ top: 8, right: 10, bottom: 8, left: 0 }}><CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-dim)" /><XAxis dataKey="name" tick={{ fill: 'var(--color-text-muted)', fontSize: 9 }} interval={0} angle={-9} textAnchor="end" height={48} /><YAxis tick={{ fill: 'var(--color-text-muted)', fontSize: 9 }} /><Tooltip formatter={value => Number(value).toFixed(6)} contentStyle={{ background: 'var(--color-surface-1)', borderColor: 'var(--color-border)', borderRadius: 12 }} /><Bar key={performanceMetric} dataKey="value" name={performanceMetric.toUpperCase()} radius={[5, 5, 0, 0]} isAnimationActive animationDuration={420}>{performanceRows.map(row => <Cell key={row.name} fill={row.name === bestExperiment ? 'var(--color-accent)' : 'var(--color-info)'} />)}</Bar></BarChart></ResponsiveContainer></div>
          <p className="section-caption">Highlighted bar marks the lowest reported {performanceMetric.toUpperCase()} among available saved results.</p>
        </section>
        <section className="card">
          <div className="card-header"><div><div className="card-title">Reported model results</div><div className="card-subtitle">Centralized, federated, and persistence comparison</div></div></div>
          <div className="data-table-wrap"><table className="data-table"><thead><tr><th>Experiment</th><th>MAE</th><th>RMSE</th></tr></thead><tbody>{experiments.map(row => <tr key={row.name}><td className="primary-cell">{row.name}</td><td>{row.mae == null ? 'N/A' : fmt(row.mae)}</td><td>{row.rmse == null ? 'N/A' : fmt(row.rmse)}</td></tr>)}</tbody></table></div>
          <p className="section-caption" style={{ marginTop: 12 }}>All reported errors are per-sensor MinMax-normalized flow units, not vehicle counts.</p>
        </section>
      </div>

      <div className="content-grid section-gap">
        <section className="card">
          <div className="card-header"><div><div className="card-title">Historical prediction trace</div><div className="card-subtitle">Centralized GRU · actual and predicted values from the saved CSV</div></div><select className="select" aria-label="Prediction chart range" value={sampleRange} onChange={event => setSampleRange(Number(event.target.value))}><option value={50}>50 samples</option><option value={100}>100 samples</option><option value={250}>250 samples</option></select></div>
          {chartRows.length ? <ResponsiveContainer width="100%" height={240}><AreaChart data={chartRows}><defs><linearGradient id="actualFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="var(--color-accent)" stopOpacity={.22} /><stop offset="95%" stopColor="var(--color-accent)" stopOpacity={0} /></linearGradient></defs><CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-dim)" /><XAxis dataKey="index" tick={{ fill: 'var(--color-text-muted)', fontSize: 9 }} /><YAxis domain={[0, 1]} tick={{ fill: 'var(--color-text-muted)', fontSize: 9 }} /><Tooltip formatter={value => Number(value).toFixed(5)} contentStyle={{ background: 'var(--color-surface-1)', borderColor: 'var(--color-border)', borderRadius: 12 }} /><Legend wrapperStyle={{ fontSize: 10 }} /><Area type="monotone" dataKey="actual" name="Actual" stroke="var(--color-accent)" fill="url(#actualFill)" strokeWidth={1.8} dot={false} /><Line type="monotone" dataKey="predicted" name="Predicted" stroke="var(--color-info)" strokeWidth={1.7} dot={false} /></AreaChart></ResponsiveContainer> : <EmptyState message="Centralized predictions are not available yet." />}
        </section>
        <section className="card">
          <div className="card-header"><div><div className="card-title">Communication profile</div><div className="card-subtitle">Estimated model payload by federated round</div></div><span className="badge badge-yellow">Upload + download</span></div>
          {communicationRows.length ? <ResponsiveContainer width="100%" height={240}><BarChart data={communicationRows}><CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-dim)" /><XAxis dataKey="round" tick={{ fill: 'var(--color-text-muted)', fontSize: 9 }} /><YAxis tick={{ fill: 'var(--color-text-muted)', fontSize: 9 }} /><Tooltip formatter={value => bytes(Number(value))} contentStyle={{ background: 'var(--color-surface-1)', borderColor: 'var(--color-border)', borderRadius: 12 }} /><Legend wrapperStyle={{ fontSize: 9 }} /><Bar dataKey="iidDownload" name="IID download" stackId="iid" fill="var(--color-accent)" /><Bar dataKey="iidUpload" name="IID upload" stackId="iid" fill="var(--color-info)" /><Bar dataKey="nonIidDownload" name="non-IID download" stackId="non" fill="#e8a858" /><Bar dataKey="nonIidUpload" name="non-IID upload" stackId="non" fill="#e17783" /></BarChart></ResponsiveContainer> : <EmptyState message="Communication history is unavailable." />}
          <p className="section-caption" style={{ marginTop: 9 }}>Model-payload estimate only; network and protocol overhead are excluded.</p>
        </section>
      </div>

      <div className="grid-2 section-gap">
        <section className="card"><div className="card-header"><div><div className="card-title">Calculated insights</div><div className="card-subtitle">Derived from the currently saved experiment metrics</div></div><span className="badge">Calculated</span></div><div className="insight-list">
          {ranked.length > 0 && <div className="insight-item"><strong>Lowest reported MAE · {ranked[0].name}</strong><span>{fmt(ranked[0].mae!)} normalized flow units, read from the saved metrics JSON.</span></div>}
          {relativeGap != null && <div className="insight-item"><strong>IID vs non-IID MAE gap · {relativeGap.toFixed(2)}%</strong><span>Calculated as the absolute difference divided by the lower of the two reported FedAvg MAEs.</span></div>}
          {b && f && <div className="insight-item"><strong>Centralized vs federated IID</strong><span>MAE difference: {fmt(Math.abs(b.mae - f.iid.mae))} normalized units. This is a direct comparison of the saved evaluations.</span></div>}
          {fedVsPersistence != null && <div className="insight-item"><strong>FedAvg IID vs persistence · {Math.abs(fedVsPersistence).toFixed(2)}% {fedVsPersistence >= 0 ? 'lower' : 'higher'} MAE</strong><span>Calculated relative to the saved persistence-baseline MAE.</span></div>}
          {(!ranked.length || !f) && <div className="empty-state">Insights appear when the corresponding saved metrics are available.</div>}
        </div></section>
        <section className="card"><div className="card-header"><div><div className="card-title">Federated run summary</div><div className="card-subtitle">Configuration and communication from result metadata</div></div></div><div className="metric-grid">
          <MetricCard label="Clients" value={f?.iid.num_clients ?? '—'} unit="simulated on one machine" />
          <MetricCard label="Rounds" value={f?.iid.rounds ?? '—'} />
          <MetricCard label="IID payload" value={iidTotal == null ? 'N/A' : bytes(iidTotal)} unit="download + upload" />
          <MetricCard label="Non-IID payload" value={nonIidTotal == null ? 'N/A' : bytes(nonIidTotal)} unit="download + upload" />
        </div></section>
      </div>
    </div>
  );
}
