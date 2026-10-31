import { lazy, Suspense, useEffect, useMemo, useState } from 'react';
import type { CSSProperties } from 'react';
import { Route, Routes, useLocation } from 'react-router-dom';
import { Sidebar } from './components/Sidebar';
import { api } from './api/client';
import { LoadingState } from './components/States';

const OverviewPage = lazy(() => import('./pages/OverviewPage').then(module => ({ default: module.OverviewPage })));
const TrafficPredictionPage = lazy(() => import('./pages/TrafficPredictionPage').then(module => ({ default: module.TrafficPredictionPage })));
const FederatedLearningPage = lazy(() => import('./pages/FederatedLearningPage').then(module => ({ default: module.FederatedLearningPage })));
const CommunicationCostPage = lazy(() => import('./pages/CommunicationCostPage').then(module => ({ default: module.CommunicationCostPage })));
const ExperimentsPage = lazy(() => import('./pages/ExperimentsPage').then(module => ({ default: module.ExperimentsPage })));
const AboutPage = lazy(() => import('./pages/AboutPage').then(module => ({ default: module.AboutPage })));

type Theme = 'dark' | 'light';
const accents = [
  { name: 'Electric Cyan', color: '#38bdf8' },
  { name: 'Neon Purple', color: '#a78bfa' },
  { name: 'Cyber Green', color: '#34d399' },
  { name: 'Electric Blue', color: '#4f8cff' },
];
const titles: Record<string, string> = {
  '/': 'Overview', '/prediction': 'Traffic Intelligence', '/federated': 'Federated Learning',
  '/communication': 'Communication Cost', '/experiments': 'Experiments', '/about': 'About & Methodology',
};

function getColorChannels(hex: string) {
  const normalized = hex.replace('#', '');
  const parsed = Number.parseInt(normalized, 16);
  return `${(parsed >> 16) & 255}, ${(parsed >> 8) & 255}, ${parsed & 255}`;
}

export default function App() {
  const location = useLocation();
  const [theme, setTheme] = useState<Theme>(() => (localStorage.getItem('ncs-theme') as Theme) || 'dark');
  const [accent, setAccent] = useState(() => localStorage.getItem('ncs-accent') || accents[0].color);
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem('ncs-sidebar') === 'collapsed');
  const [mobileOpen, setMobileOpen] = useState(false);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const channels = useMemo(() => getColorChannels(accent), [accent]);

  useEffect(() => {
    localStorage.setItem('ncs-theme', theme);
    localStorage.setItem('ncs-accent', accent);
    localStorage.setItem('ncs-sidebar', collapsed ? 'collapsed' : 'expanded');
  }, [theme, accent, collapsed]);

  useEffect(() => {
    let active = true;
    const check = async () => {
      try { await api.health(); if (active) setBackendOnline(true); }
      catch { if (active) setBackendOnline(false); }
    };
    void check();
    const timer = window.setInterval(() => { void check(); }, 15000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => { if (event.key === 'Escape') setMobileOpen(false); };
    window.addEventListener('keydown', handleEscape);
    document.body.style.overflow = mobileOpen ? 'hidden' : '';
    return () => { window.removeEventListener('keydown', handleEscape); document.body.style.overflow = ''; };
  }, [mobileOpen]);

  return (
    <div
      className={`app-shell ${collapsed ? 'nav-collapsed' : ''} ${mobileOpen ? 'mobile-nav-open' : ''}`}
      data-theme={theme}
      style={{ '--color-accent': accent, '--accent-rgb': channels } as CSSProperties}
    >
      <Sidebar collapsed={collapsed} mobileOpen={mobileOpen} onToggle={() => setCollapsed(value => !value)} onClose={() => setMobileOpen(false)} />
      {mobileOpen && <button className="mobile-scrim" aria-label="Close navigation" onClick={() => setMobileOpen(false)} />}
      <div className="main-content">
        <header className="topbar">
          <div className="topbar-leading">
            <button className="icon-button mobile-menu-button" aria-label="Open navigation" onClick={() => setMobileOpen(true)}>☰</button>
            <div className="breadcrumb"><span className="breadcrumb-root">NoCivicSense</span><span className="breadcrumb-sep">/</span><strong>{titles[location.pathname] ?? 'Dashboard'}</strong></div>
          </div>
          <div className="topbar-actions">
            <div className="dataset-pill"><span className="status-dot online" />PeMS04 <span className="pill-divider" />Public dataset</div>
            <div className={`backend-status ${backendOnline === null ? 'checking' : backendOnline ? 'online' : 'offline'}`} title={backendOnline ? 'API connected' : backendOnline === null ? 'Checking API' : 'API unavailable'}>
              <span className="status-dot" />{backendOnline === null ? 'Checking' : backendOnline ? 'API connected' : 'API offline'}
            </div>
            <button className="icon-button theme-toggle" aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`} title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`} onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
              {theme === 'dark' ? '☼' : '☾'}
            </button>
            <details className="accent-menu">
              <summary className="accent-trigger" aria-label="Accent color settings"><span className="accent-swatch" style={{ background: accent }} />Accent</summary>
              <div className="accent-popover">
                <div className="popover-title">Interface accent</div>
                {accents.map(option => <button key={option.name} className={`accent-option${accent === option.color ? ' selected' : ''}`} onClick={() => setAccent(option.color)}><span className="accent-swatch" style={{ background: option.color }} />{option.name}{accent === option.color && <span className="check-mark">✓</span>}</button>)}
                <label className="custom-accent">Custom colour<input type="color" value={accent} aria-label="Choose custom accent color" onChange={event => setAccent(event.target.value)} /></label>
              </div>
            </details>
          </div>
        </header>
        <Suspense fallback={<div className="page-body"><LoadingState message="Opening research view…" /></div>}>
          <Routes>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/prediction" element={<TrafficPredictionPage />} />
            <Route path="/federated" element={<FederatedLearningPage />} />
            <Route path="/communication" element={<CommunicationCostPage />} />
            <Route path="/experiments" element={<ExperimentsPage />} />
            <Route path="/about" element={<AboutPage />} />
            <Route path="*" element={<OverviewPage />} />
          </Routes>
        </Suspense>
        <footer className="app-footer"><span>PS-003 · Federated Traffic-Flow Prediction</span><span>Research prototype · PeMS04</span></footer>
      </div>
    </div>
  );
}
