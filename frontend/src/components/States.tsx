interface LoadingProps { message?: string }
interface ErrorProps { message: string; onRetry?: () => void }
interface EmptyProps { message?: string }

export function LoadingState({ message = 'Loading data…' }: LoadingProps) {
  return <div className="state-container" role="status" aria-live="polite"><div className="spinner" /><div className="state-title">{message}</div></div>;
}

export function ErrorState({ message, onRetry }: ErrorProps) {
  return <div className="state-container" role="alert"><div className="state-icon">!</div><div className="state-title">Unable to load data</div><div className="state-body">{message}</div>{onRetry && <button className="btn btn-ghost" onClick={onRetry}>Retry</button>}</div>;
}

export function EmptyState({ message = 'No data available.' }: EmptyProps) {
  return <div className="state-container"><div className="state-icon">∅</div><div className="state-title">No saved data</div><div className="state-body">{message}</div></div>;
}
