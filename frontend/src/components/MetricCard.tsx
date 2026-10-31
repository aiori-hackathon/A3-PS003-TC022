interface Props {
  label: string;
  value: string | number;
  unit?: string;
  highlight?: boolean;
}

export function MetricCard({ label, value, unit, highlight }: Props) {
  return (
    <div className={`metric-card${highlight ? ' highlight' : ''}`}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {unit && <div className="metric-unit">{unit}</div>}
    </div>
  );
}
