import { useCallback, useEffect, useMemo, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../api/client';
import type { AllMetrics, CommunicationResponse } from '../api/types';
import { EmptyState, ErrorState, LoadingState } from '../components/States';

const formatBytes = (value: number) => value >= 1_000_000 ? `${(value / 1_000_000).toFixed(2)} MB` : value >= 1_000 ? `${(value / 1_000).toFixed(1)} KB` : `${value} B`;
type Mode = 'iid' | 'non_iid';

export function CommunicationCostPage() {
  const [metrics, setMetrics] = useState<AllMetrics | null>(null);
  const [iid, setIid] = useState<CommunicationResponse | null>(null);
  const [nonIid, setNonIid] = useState<CommunicationResponse | null>(null);
  const [mode, setMode] = useState<Mode>('iid');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const [allMetrics, iidRows, nonIidRows] = await Promise.all([api.getMetrics(), api.getCommunication('iid'), api.getCommunication('non_iid')]);
      setMetrics(allMetrics); setIid(iidRows); setNonIid(nonIidRows);
    } catch (e) { setError(e instanceof Error ? e.message : 'Unable to load communication results'); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { const timer = window.setTimeout(() => { void load(); }, 0); return () => window.clearTimeout(timer); }, [load]);

  const iidMetric = metrics?.federated?.iid;
  const nonIidMetric = metrics?.federated?.non_iid;
  const selectedMetric = mode === 'iid' ? iidMetric : nonIidMetric;
  const selectedHistory = mode === 'iid' ? iid?.rounds ?? [] : nonIid?.rounds ?? [];
  const comparison = useMemo(() => [
    { mode: 'IID', upload: iidMetric?.upload_bytes, download: iidMetric?.download_bytes },
    { mode: 'non-IID', upload: nonIidMetric?.upload_bytes, download: nonIidMetric?.download_bytes },
  ], [iidMetric, nonIidMetric]);

  if (loading) return <div className="page-body"><LoadingState message="Reading communication estimates…" /></div>;
  if (error) return <div className="page-body"><ErrorState message={error} onRetry={load} /></div>;

  return (
    <div className="page-body">
      <div className="page-header"><div className="page-kicker">FedAvg · payload accounting</div><h1 className="page-title">Communication Cost</h1><p className="page-description">Compare estimated model-payload transfers across IID and non-IID federated runs. Totals and round histories are read from the saved metrics JSON and communication CSVs.</p></div>
      <div className="info-banner"><span>ⓘ</span><span>Bytes represent estimated model tensor payload only. Protocol/network overhead, encryption, retries, and acknowledgements are excluded. Centralized and persistence communication were not measured and are shown as N/A.</span></div>

      <div className="section-heading"><div><h2 className="section-title">Transfer summary</h2><p className="section-caption">Aggregate totals are reported in federated_real_metrics.json</p></div><div className="segmented" role="group" aria-label="Select communication mode"><button aria-pressed={mode === 'iid'} onClick={() => setMode('iid')}>IID</button><button aria-pressed={mode === 'non_iid'} onClick={() => setMode('non_iid')}>non-IID</button></div></div>
      <section className="kpi-grid section-gap">
        <div className="kpi-card"><div className="kpi-head">Total transferred<span className="kpi-icon">⇅</span></div><div className="kpi-value">{selectedMetric ? formatBytes(selectedMetric.total_communication_bytes) : 'N/A'}</div><div className="kpi-foot">{mode.toUpperCase()} · upload + download</div></div>
        <div className="kpi-card"><div className="kpi-head">Upload bytes<span className="kpi-icon">↑</span></div><div className="kpi-value">{selectedMetric ? formatBytes(selectedMetric.upload_bytes) : 'N/A'}</div><div className="kpi-foot">reported aggregate</div></div>
        <div className="kpi-card"><div className="kpi-head">Download bytes<span className="kpi-icon">↓</span></div><div className="kpi-value">{selectedMetric ? formatBytes(selectedMetric.download_bytes) : 'N/A'}</div><div className="kpi-foot">reported aggregate</div></div>
        <div className="kpi-card"><div className="kpi-head">Average per client<span className="kpi-icon">⌘</span></div><div className="kpi-value">{selectedMetric?.num_clients ? formatBytes(selectedMetric.total_communication_bytes / selectedMetric.num_clients) : 'N/A'}</div><div className="kpi-foot">derived aggregate ÷ client count; not measured per client</div></div>
      </section>

      <div className="content-grid section-gap">
        <section className="card"><div className="card-header"><div><div className="card-title">Upload vs download</div><div className="card-subtitle">Reported aggregate bytes for each saved federated mode</div></div><span className="badge badge-yellow">Payload estimate</span></div>
          {comparison.some(item => item.upload != null || item.download != null) ? <ResponsiveContainer width="100%" height={285}><BarChart data={comparison} margin={{ top: 8, right: 12, bottom: 5, left: 0 }}><CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-dim)" /><XAxis dataKey="mode" tick={{ fill: 'var(--color-text-muted)', fontSize: 10 }} /><YAxis tickFormatter={formatBytes} tick={{ fill: 'var(--color-text-muted)', fontSize: 9 }} /><Tooltip formatter={value => formatBytes(Number(value))} contentStyle={{ background: 'var(--color-surface-1)', borderColor: 'var(--color-border)', borderRadius: 12 }} /><Legend wrapperStyle={{ fontSize: 10 }} /><Bar dataKey="upload" name="Upload" fill="var(--color-accent)" radius={[5, 5, 0, 0]} /><Bar dataKey="download" name="Download" fill="var(--color-info)" radius={[5, 5, 0, 0]} /></BarChart></ResponsiveContainer> : <EmptyState message="No aggregate communication totals are available." />}
        </section>
        <section className="card"><div className="card-header"><div><div className="card-title">Per-round transfer</div><div className="card-subtitle">{mode.toUpperCase()} run · bytes recorded by communication CSV</div></div></div>
          {selectedHistory.length ? <ResponsiveContainer width="100%" height={285}><BarChart data={selectedHistory.map(row => ({ ...row, roundLabel: `Round ${row.round}` }))} margin={{ top: 8, right: 12, bottom: 5, left: 0 }}><CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-dim)" /><XAxis dataKey="roundLabel" tick={{ fill: 'var(--color-text-muted)', fontSize: 9 }} /><YAxis tickFormatter={formatBytes} tick={{ fill: 'var(--color-text-muted)', fontSize: 9 }} /><Tooltip formatter={value => formatBytes(Number(value))} contentStyle={{ background: 'var(--color-surface-1)', borderColor: 'var(--color-border)', borderRadius: 12 }} /><Legend wrapperStyle={{ fontSize: 10 }} /><Bar dataKey="download_bytes" name="Download" stackId="round" fill="var(--color-accent)" /><Bar dataKey="upload_bytes" name="Upload" stackId="round" fill="var(--color-info)" radius={[5, 5, 0, 0]} /></BarChart></ResponsiveContainer> : <EmptyState message="No saved per-round communication rows for this mode." />}
        </section>
      </div>

      <section className="card section-gap"><div className="card-header"><div><div className="card-title">Federated communication comparison</div><div className="card-subtitle">Reported totals and number of rounds from experiment metadata</div></div></div><div className="data-table-wrap"><table className="data-table"><thead><tr><th>Experiment</th><th>Clients</th><th>Rounds</th><th>Model payload</th><th>Upload</th><th>Download</th><th>Total</th><th>Accuracy</th></tr></thead><tbody><tr><td className="primary-cell">FedAvg IID</td><td>{iidMetric?.num_clients ?? 'N/A'}</td><td>{iidMetric?.rounds ?? 'N/A'}</td><td>{iidMetric ? formatBytes(iidMetric.model_payload_bytes) : 'N/A'}</td><td>{iidMetric ? formatBytes(iidMetric.upload_bytes) : 'N/A'}</td><td>{iidMetric ? formatBytes(iidMetric.download_bytes) : 'N/A'}</td><td>{iidMetric ? formatBytes(iidMetric.total_communication_bytes) : 'N/A'}</td><td>{iidMetric?.mae.toFixed(5) ?? 'N/A'} MAE</td></tr><tr><td className="primary-cell">FedAvg non-IID</td><td>{nonIidMetric?.num_clients ?? 'N/A'}</td><td>{nonIidMetric?.rounds ?? 'N/A'}</td><td>{nonIidMetric ? formatBytes(nonIidMetric.model_payload_bytes) : 'N/A'}</td><td>{nonIidMetric ? formatBytes(nonIidMetric.upload_bytes) : 'N/A'}</td><td>{nonIidMetric ? formatBytes(nonIidMetric.download_bytes) : 'N/A'}</td><td>{nonIidMetric ? formatBytes(nonIidMetric.total_communication_bytes) : 'N/A'}</td><td>{nonIidMetric?.mae.toFixed(5) ?? 'N/A'} MAE</td></tr><tr><td className="primary-cell">Centralized GRU</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A · not measured</td><td>{metrics?.baseline?.mae.toFixed(5) ?? 'N/A'} MAE</td></tr><tr><td className="primary-cell">Persistence baseline</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A · not measured</td><td>{metrics?.persistence?.mae.toFixed(5) ?? 'N/A'} MAE</td></tr></tbody></table></div></section>

      <section className="card"><div className="card-header"><div><div className="card-title">Round detail · {mode === 'iid' ? 'IID' : 'non-IID'}</div><div className="card-subtitle">Round-level estimates and test performance loaded from CSV</div></div></div><div className="data-table-wrap">{selectedHistory.length ? <table className="data-table"><thead><tr><th>Round</th><th>Download</th><th>Upload</th><th>Round total</th><th>Cumulative</th><th>Test MAE</th><th>Test RMSE</th></tr></thead><tbody>{selectedHistory.map(row => <tr key={row.round}><td>{row.round}</td><td>{formatBytes(row.download_bytes)}</td><td>{formatBytes(row.upload_bytes)}</td><td>{formatBytes(row.total_round_bytes)}</td><td>{formatBytes(row.cumulative_bytes)}</td><td>{row.test_mae.toFixed(6)}</td><td>{row.test_rmse.toFixed(6)}</td></tr>)}</tbody></table> : <EmptyState message="No communication CSV rows are available." />}</div></section>
    </div>
  );
}
