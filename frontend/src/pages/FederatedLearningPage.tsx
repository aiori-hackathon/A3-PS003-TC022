import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../api/client';
import type { AllMetrics, CommunicationResponse, FederatedRun } from '../api/types';
import { EmptyState, ErrorState, LoadingState } from '../components/States';

type Mode = 'iid' | 'non_iid';
const number = (n: number) => n.toLocaleString();

export function FederatedLearningPage() {
  const [metrics, setMetrics] = useState<AllMetrics | null>(null);
  const [iidHistory, setIidHistory] = useState<CommunicationResponse | null>(null);
  const [nonIidHistory, setNonIidHistory] = useState<CommunicationResponse | null>(null);
  const [latestRun, setLatestRun] = useState<FederatedRun | null>(null);
  const [mode, setMode] = useState<Mode>('iid');
  const [activeStage, setActiveStage] = useState(-1);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const [m, iid, nonIid, runs] = await Promise.all([api.getMetrics(), api.getCommunication('iid'), api.getCommunication('non_iid'), api.getExperimentRuns()]);
      setMetrics(m); setIidHistory(iid); setNonIidHistory(nonIid); setLatestRun(runs[0] ?? null);
    } catch (e) { setError(e instanceof Error ? e.message : 'Unable to load federated results'); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { const timer = window.setTimeout(() => { void load(); }, 0); return () => window.clearTimeout(timer); }, [load]);
  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => setActiveStage(stage => (stage + 1) % 5), 1700);
    return () => window.clearInterval(timer);
  }, [playing]);

  const selected = mode === 'iid' ? metrics?.federated?.iid : metrics?.federated?.non_iid;
  const clients = selected?.num_clients ?? metrics?.federated?.iid.num_clients ?? 0;
  const convergenceData = useMemo(() => {
    const iidRows = iidHistory?.rounds ?? [];
    const nonRows = nonIidHistory?.rounds ?? [];
    return Array.from({ length: Math.max(iidRows.length, nonRows.length) }, (_, index) => ({
      round: index + 1,
      iidMae: iidRows[index]?.test_mae,
      nonIidMae: nonRows[index]?.test_mae,
    }));
  }, [iidHistory, nonIidHistory]);
  const stages = ['Server broadcasts global model', 'Clients receive model parameters', 'Clients train locally', 'Clients upload model updates', 'Coordinator performs sample-weighted FedAvg'];

  if (loading) return <div className="page-body"><LoadingState message="Loading FedAvg experiment data…" /></div>;
  if (error) return <div className="page-body"><ErrorState message={error} onRetry={load} /></div>;
  if (!metrics) return <div className="page-body"><EmptyState message="Federated metrics are not available." /></div>;

  const iid = metrics.federated?.iid;
  const nonIid = metrics.federated?.non_iid;
  const groups = Array.from({ length: clients }, (_, index) => {
    const label = selected?.client_distribution === 'non_iid' ? 'Sensor-grouped partition' : 'Random sample partition';
    return { index, label };
  });

  return (
    <div className="page-body">
      <div className="page-header"><div className="page-kicker">Collaborative model training</div><h1 className="page-title">Federated Learning</h1><p className="page-description">Inspect the saved FedAvg experiments, client distribution, round convergence, and configuration for IID and sensor-grouped non-IID splits.</p></div>
      <div className="info-banner"><span>ⓘ</span><span><strong>Single-machine simulation.</strong> The {clients || 'configured'} clients are simulated in the Python process. This prototype does not provide physical client separation, differential privacy, encryption, or secure aggregation.</span></div>

      {latestRun && <section className="card section-gap"><div className="card-header"><div><div className="card-title">Latest configurable run</div><div className="card-subtitle">{new Date(latestRun.created_at).toLocaleString()} ? {latestRun.run_id}</div></div><Link className="btn btn-ghost btn-small" to="/experiments">Configure an experiment</Link></div><div className="metric-grid">{(['iid', 'non_iid'] as const).filter(key => latestRun.results[key]).map(key => { const result = latestRun.results[key]!; const c = latestRun.configuration; return <div className="metric-card" key={key}><div className="metric-label">{key === 'iid' ? 'IID' : 'non-IID'}</div><div className="metric-value">MAE {result.mae.toFixed(6)}</div><div className="metric-unit">RMSE {result.rmse.toFixed(6)} ? {result.total_communication_bytes.toLocaleString()} bytes</div><div className="metric-unit">{c.num_clients} clients ? {c.rounds} rounds ? {c.local_epochs} epochs ? batch {c.batch_size} ? LR {c.learning_rate}</div></div>; })}</div></section>}

      <section className="card architecture-card section-gap">
        <div className="card-header"><div><div className="card-title">FedAvg training cycle</div><div className="card-subtitle">Illustrative architecture of the existing experiment, not a live network topology</div></div><span className="badge badge-accent">{clients || '—'} simulated clients</span></div>
        <div className={`federated-flow ${playing ? 'playing' : ''}`}>
          <div className="client-stack">{groups.map(({ index, label }) => <div className={`flow-node ${activeStage === 1 || activeStage === 2 || activeStage === 3 ? 'active' : ''}`} key={index}><span className="flow-node-icon">{String(index + 1).padStart(2, '0')}</span><div><strong>Simulated client {index + 1}</strong><small>{label}</small></div></div>)}</div>
          <div className="flow-arrows" aria-label="Model weights distributed and updated weights returned"><span className="arrow-label">Global model →</span><span className={activeStage === 0 || activeStage === 1 ? 'flow-pulse' : ''}>⇄</span><span className="arrow-label">← Model updates</span></div>
          <div className={`flow-node server-node ${activeStage === 0 || activeStage === 4 ? 'active' : ''}`}><span className="flow-node-icon">Σ</span><div><strong>Coordinator</strong><small>Sample-weighted FedAvg</small></div></div>
        </div>
        <div className="flow-controls"><button className="btn btn-primary btn-small" onClick={() => { setActiveStage(stage => stage < 0 ? 0 : stage); setPlaying(value => !value); }}>{playing ? 'Pause' : 'Play walkthrough'}</button><button className="btn btn-ghost btn-small" onClick={() => { setPlaying(false); setActiveStage(-1); }}>Reset</button><span className="section-caption">{activeStage < 0 ? 'Ready · five stages' : `Stage ${activeStage + 1} of 5 · ${stages[activeStage]}`}</span></div>
        <div className="grid-3" style={{ marginTop: 15 }}>{stages.map((stage, index) => <div className={`step-item flow-stage ${activeStage === index ? 'active' : ''}`} key={stage}><span className="step-number">{String(index + 1).padStart(2, '0')}</span><span>{stage}{index === 4 ? ', then redistributes the updated global model.' : '.'}</span></div>)}</div>
      </section>

      <div className="section-heading"><div><h2 className="section-title">Mode comparison</h2><p className="section-caption">Reported values from federated_real_metrics.json</p></div><div className="segmented" role="group" aria-label="Select client distribution"><button aria-pressed={mode === 'iid'} onClick={() => setMode('iid')}>IID</button><button aria-pressed={mode === 'non_iid'} onClick={() => setMode('non_iid')}>non-IID</button></div></div>
      <section className="kpi-grid section-gap">
        <div className="kpi-card"><div className="kpi-head">Selected distribution<span className="kpi-icon">⇄</span></div><div className="kpi-value" style={{ fontSize: 20 }}>{mode === 'iid' ? 'IID' : 'non-IID'}</div><div className="kpi-foot">{selected?.client_split_description ?? '—'}</div></div>
        <div className="kpi-card"><div className="kpi-head">Clients<span className="kpi-icon">⌘</span></div><div className="kpi-value">{selected?.num_clients ?? '—'}</div><div className="kpi-foot">simulated in one process</div></div>
        <div className="kpi-card"><div className="kpi-head">Global rounds<span className="kpi-icon">↻</span></div><div className="kpi-value">{selected?.rounds ?? '—'}</div><div className="kpi-foot">local epochs each: {selected?.local_epochs ?? '—'}</div></div>
        <div className="kpi-card"><div className="kpi-head">Final MAE<span className="kpi-icon">Δ</span></div><div className="kpi-value">{selected?.mae.toFixed(5) ?? '—'}</div><div className="kpi-foot">RMSE {selected?.rmse.toFixed(5) ?? '—'} · normalized units</div></div>
      </section>

      <div className="content-grid section-gap">
        <section className="card"><div className="card-header"><div><div className="card-title">Round convergence · IID vs non-IID</div><div className="card-subtitle">Saved test MAE for both runs · lower is better</div></div></div>
          {convergenceData.length ? <ResponsiveContainer width="100%" height={290}><LineChart data={convergenceData} margin={{ top: 8, right: 12, bottom: 5, left: 0 }}><CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-dim)" /><XAxis dataKey="round" tickFormatter={round => `Round ${round}`} tick={{ fill: 'var(--color-text-muted)', fontSize: 9 }} /><YAxis tick={{ fill: 'var(--color-text-muted)', fontSize: 9 }} /><Tooltip labelFormatter={label => `Round ${label}`} formatter={value => Number(value).toFixed(6)} contentStyle={{ background: 'var(--color-surface-1)', borderColor: 'var(--color-border)', borderRadius: 12 }} /><Line type="monotone" dataKey="iidMae" name="IID test MAE" stroke="var(--color-accent)" strokeWidth={2.3} dot={{ r: 3 }} activeDot={{ r: 6 }} connectNulls /><Line type="monotone" dataKey="nonIidMae" name="non-IID test MAE" stroke="var(--color-info)" strokeWidth={2.1} dot={{ r: 3 }} activeDot={{ r: 6 }} connectNulls /></LineChart></ResponsiveContainer> : <EmptyState message="Per-round history is not available." />}
          <p className="section-caption">The mode selector above still controls the configuration and distribution detail cards. These lines compare both saved runs.</p>
        </section>
        <section className="card"><div className="card-header"><div><div className="card-title">Client data distribution</div><div className="card-subtitle">Schematic distribution pattern · not client-level sample measurements</div></div><span className="badge">{mode.toUpperCase()}</span></div>
          <div className="distribution-visual"><div className="distribution-label"><span>Total training sequences</span><strong>{selected ? number(selected.train_samples) : '—'}</strong></div><div className={`distribution-band ${mode}`} aria-label={mode === 'iid' ? 'Schematic: shuffled samples spread across clients' : 'Schematic: disjoint sensor groups assigned to clients'}>{groups.map(({ index }) => <div key={index} className="distribution-client"><span>Client {index + 1}</span><div className="distribution-segments">{Array.from({ length: 16 }, (_, segment) => <i key={segment} />)}</div></div>)}</div><p className="section-caption">{mode === 'iid' ? 'Training windows are shuffled and distributed among clients.' : 'Each client receives training windows for a disjoint sensor group.'} Per-client counts are not included in the saved files.</p><div className="client-count-row"><span>Held-out test sequences</span><strong>{selected ? number(selected.test_samples) : '—'}</strong></div><div className="client-count-row"><span>Batch size</span><strong>{selected?.batch_size ?? '—'}</strong></div><div className="client-count-row"><span>Learning rate</span><strong>{selected?.learning_rate ?? '—'}</strong></div></div>
        </section>
      </div>

      <section className="card"><div className="card-header"><div><div className="card-title">IID and non-IID results</div><div className="card-subtitle">Four named experiment outcomes, with communication where measured</div></div></div><div className="data-table-wrap"><table className="data-table"><thead><tr><th>Experiment</th><th>Clients</th><th>Rounds</th><th>Local epochs</th><th>MAE</th><th>RMSE</th><th>Total payload</th></tr></thead><tbody><tr><td className="primary-cell">Centralized GRU</td><td>N/A</td><td>N/A</td><td>N/A</td><td>{metrics.baseline?.mae.toFixed(6) ?? 'N/A'}</td><td>{metrics.baseline?.rmse.toFixed(6) ?? 'N/A'}</td><td>N/A · not measured</td></tr><tr><td className="primary-cell">FedAvg IID</td><td>{iid?.num_clients ?? 'N/A'}</td><td>{iid?.rounds ?? 'N/A'}</td><td>{iid?.local_epochs ?? 'N/A'}</td><td>{iid?.mae.toFixed(6) ?? 'N/A'}</td><td>{iid?.rmse.toFixed(6) ?? 'N/A'}</td><td>{iid ? number(iid.total_communication_bytes) : 'N/A'} B</td></tr><tr><td className="primary-cell">FedAvg non-IID</td><td>{nonIid?.num_clients ?? 'N/A'}</td><td>{nonIid?.rounds ?? 'N/A'}</td><td>{nonIid?.local_epochs ?? 'N/A'}</td><td>{nonIid?.mae.toFixed(6) ?? 'N/A'}</td><td>{nonIid?.rmse.toFixed(6) ?? 'N/A'}</td><td>{nonIid ? number(nonIid.total_communication_bytes) : 'N/A'} B</td></tr><tr><td className="primary-cell">Persistence baseline</td><td>N/A</td><td>N/A</td><td>N/A</td><td>{metrics.persistence?.mae.toFixed(6) ?? 'N/A'}</td><td>{metrics.persistence?.rmse.toFixed(6) ?? 'N/A'}</td><td>N/A · not measured</td></tr></tbody></table></div></section>
    </div>
  );
}
