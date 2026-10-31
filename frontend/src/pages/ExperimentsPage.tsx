import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api/client';
import type { AllMetrics, ExperimentStatus, FederatedRun, FederatedRunConfig } from '../api/types';
import { LoadingState } from '../components/States';

const initial: FederatedRunConfig = { experiment: 'federated', num_clients: 4, rounds: 3, local_epochs: 1, batch_size: 2048, learning_rate: 0.001, mode: 'both' };
const fmt = (value?: number) => value == null ? '—' : value.toFixed(6);
const modes = [{ key: 'iid', label: 'IID' }, { key: 'non_iid', label: 'non-IID' }] as const;

export function ExperimentsPage() {
  const [config, setConfig] = useState(initial);
  const [status, setStatus] = useState<ExperimentStatus>({ running: false, experiment: null, status: 'idle', message: '' });
  const [metrics, setMetrics] = useState<AllMetrics | null>(null);
  const [runs, setRuns] = useState<FederatedRun[]>([]);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const poll = useRef<ReturnType<typeof setInterval> | null>(null);

  const refresh = useCallback(async (requestedRunId?: string) => {
    try {
      const runId = requestedRunId ?? activeRunId ?? undefined;
      const [m, r, s] = await Promise.all([api.getMetrics(), api.getExperimentRuns(), api.getStatus(runId)]);
      setMetrics(m); setRuns(r); setStatus(s); setError(null);
      if (!s.running && poll.current) { clearInterval(poll.current); poll.current = null; }
    } catch (e) { setError(e instanceof Error ? e.message : 'Unable to load experiment data'); }
    finally { setLoading(false); }
  }, [activeRunId]);
  useEffect(() => { const id = window.setTimeout(() => { void refresh(); }, 0); return () => { clearTimeout(id); if (poll.current) clearInterval(poll.current); }; }, [refresh]);
  useEffect(() => { if ((status.status === 'queued' || status.status === 'running') && !poll.current) poll.current = setInterval(() => { void refresh(activeRunId ?? undefined); }, 2000); }, [status.status, activeRunId, refresh]);

  const update = (key: keyof FederatedRunConfig, value: string) => setConfig(current => ({ ...current, [key]: key === 'mode' ? value : key === 'learning_rate' ? Number(value) : Number(value) }));
  const valid = Number.isInteger(config.num_clients) && config.num_clients >= 2 && config.num_clients <= 16 && Number.isInteger(config.rounds) && config.rounds >= 1 && config.rounds <= 20 && Number.isInteger(config.local_epochs) && config.local_epochs >= 1 && config.local_epochs <= 10 && Number.isInteger(config.batch_size) && config.batch_size >= 1 && config.batch_size <= 65536 && Number.isFinite(config.learning_rate) && config.learning_rate > 0 && config.learning_rate <= 1;
  const start = async () => {
    setPending(false); setError(null);
    try { const accepted = await api.runExperiment(config); setActiveRunId(accepted.run_id); await refresh(accepted.run_id); }
    catch (e) { setError(e instanceof Error ? e.message : 'Failed to start experiment'); }
  };
  const legacyRows = [
    { name: 'Centralized GRU baseline', item: metrics?.baseline },
    { name: 'Persistence baseline', item: metrics?.persistence },
  ];

  return <div className="page-body">
    <div className="page-header"><div className="page-kicker">Run control · saved outputs</div><h1 className="page-title">Experiments</h1><p className="page-description">Configure and launch real PeMS04 federated experiments. Loading this page never starts training.</p></div>
    <section className="card section-gap status-card"><div className="status-main"><div className={`status-orb ${status.status === 'completed' ? 'success' : status.status === 'queued' ? 'running' : status.status}`}><span /></div><div><div className="status-title">{status.running ? 'Experiment in progress' : `Last run · ${status.status}`}</div><div className="status-description">{status.run_id ? `Run ${status.run_id}` : 'No configured run has started in this session.'}</div></div><button className="btn btn-ghost btn-small status-refresh" onClick={() => { void refresh(); }}>↻ Refresh</button></div>{status.message && <div className="log-panel status-output">{status.message}</div>}</section>
    {error && <div className="disclaimer" role="alert">{error}</div>}

    <section className="card section-gap">
      <div className="card-header"><div><div className="card-title">Federated training configuration</div><div className="card-subtitle">These values are passed to the Python training process.</div></div><span className="badge badge-accent">PeMS04 · 32 sensors</span></div>
      <div className="form-grid">
        <label className="config-field">Number of clients <small>2–16</small><input type="number" min="2" max="16" step="1" value={config.num_clients} onChange={e => update('num_clients', e.target.value)} /></label>
        <label className="config-field">Communication rounds <small>1–20</small><input type="number" min="1" max="20" step="1" value={config.rounds} onChange={e => update('rounds', e.target.value)} /></label>
        <label className="config-field">Local epochs <small>1–10</small><input type="number" min="1" max="10" step="1" value={config.local_epochs} onChange={e => update('local_epochs', e.target.value)} /></label>
        <label className="config-field">Batch size <small>1–65,536</small><input type="number" min="1" max="65536" step="1" value={config.batch_size} onChange={e => update('batch_size', e.target.value)} /></label>
        <label className="config-field">Learning rate <small>greater than 0, up to 1</small><input type="number" min="0.000001" max="1" step="any" value={config.learning_rate} onChange={e => update('learning_rate', e.target.value)} /></label>
        <label className="config-field">Experiment mode<select value={config.mode} onChange={e => update('mode', e.target.value)}><option value="iid">IID</option><option value="non_iid">non-IID</option><option value="both">Both</option></select></label>
      </div>
      {!valid && <p className="field-error" role="alert">Check the values: clients must be 2–16, rounds 1–20, epochs 1–10, batch size 1–65,536, and learning rate greater than 0 and at most 1.</p>}
      <div className="form-actions"><span className="section-caption">Each confirmed run saves to a new timestamped folder.</span><button className="btn btn-primary" disabled={!valid || status.running} onClick={() => setPending(true)}>{status.running ? 'Training in progress…' : 'Review and run'}</button></div>
    </section>

    <section className="card section-gap"><div className="card-header"><div><div className="card-title">Saved experiment runs</div><div className="card-subtitle">Each result is shown with the configuration that produced it.</div></div><button className="btn btn-ghost btn-small" onClick={() => { void refresh(); }}>Refresh results</button></div>
      {loading ? <LoadingState message="Loading saved results…" /> : metrics ? <div className="data-table-wrap"><table className="data-table"><thead><tr><th>Run / experiment</th><th>Configuration</th><th>MAE</th><th>RMSE</th><th>Communication</th><th>Saved</th></tr></thead><tbody>
        {legacyRows.map(({ name, item }) => <tr key={name}><td className="primary-cell">{name}</td><td>Existing saved baseline</td><td>{fmt(item?.mae)}</td><td>{fmt(item?.rmse)}</td><td>Not measured</td><td>Existing artifact</td></tr>)}
        {runs.flatMap(run => {
          const c = run.configuration;
          const results = modes.filter(({ key }) => run.results[key]).map(({ key, label }) => { const result = run.results[key]!; return <tr key={`${run.run_id}-${key}`}><td className="primary-cell">{label} · {run.run_id}</td><td>{c.num_clients} clients · {c.rounds} rounds · {c.local_epochs} epochs · batch {c.batch_size} · LR {c.learning_rate}</td><td>{fmt(result.mae)}</td><td>{fmt(result.rmse)}</td><td>{result.total_communication_bytes?.toLocaleString() ?? '—'} B</td><td>{new Date(run.created_at).toLocaleString()}</td></tr>; });
          if (results.length) return results;
          return [<tr key={run.run_id}><td className="primary-cell">{run.run_id} · {run.status ?? 'unknown'}</td><td>{c ? `${c.num_clients} clients · ${c.rounds} rounds · ${c.local_epochs} epochs · batch ${c.batch_size} · LR ${c.learning_rate}` : 'Configuration unavailable'}</td><td>—</td><td>—</td><td>{run.exception_message ?? run.message ?? 'No result files were produced.'}</td><td>{new Date(run.created_at).toLocaleString()}</td></tr>];
        })}
      </tbody></table></div> : <LoadingState message="No metrics available." />}
    </section>
    <p className="section-caption">The centralized and persistence baselines remain available. See <Link to="/federated">Federated Learning</Link> for client distribution and round history.</p>
    {pending && <div className="modal-scrim" role="presentation" onMouseDown={e => { if (e.target === e.currentTarget) setPending(false); }}><div className="modal-panel" role="dialog" aria-modal="true" aria-labelledby="run-modal-title"><h2 className="modal-title" id="run-modal-title">Start federated training?</h2><p className="modal-copy">This will train {config.mode === 'both' ? 'IID and non-IID' : config.mode === 'iid' ? 'IID' : 'non-IID'} models with {config.num_clients} clients, {config.rounds} rounds, {config.local_epochs} local epochs, batch size {config.batch_size}, and learning rate {config.learning_rate}. Training may take time and creates new result files in a separate folder. Existing runs and baselines are preserved.</p><div className="modal-actions"><button className="btn btn-ghost" onClick={() => setPending(false)}>Cancel</button><button className="btn btn-primary" onClick={() => { void start(); }}>Start experiment</button></div></div></div>}
  </div>;
}
