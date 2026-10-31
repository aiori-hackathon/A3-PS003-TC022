import { NavLink } from 'react-router-dom';

const NAV = [
  { to: '/', icon: '◈', label: 'Overview' },
  { to: '/prediction', icon: '⌁', label: 'Traffic Intelligence' },
  { to: '/federated', icon: '⇄', label: 'Federated Learning' },
  { to: '/communication', icon: '⇅', label: 'Communication Cost' },
  { to: '/experiments', icon: '◷', label: 'Experiments' },
  { to: '/about', icon: '⌘', label: 'About & Methodology' },
];

interface Props {
  collapsed: boolean;
  mobileOpen: boolean;
  onToggle: () => void;
  onClose: () => void;
}

export function Sidebar({ collapsed, mobileOpen, onToggle, onClose }: Props) {
  return (
    <aside className={`sidebar${mobileOpen ? ' mobile-open' : ''}`}>
      <div className="brand-block">
        <div className="brand-mark">N<span>.</span></div>
        <div className="brand-copy"><div className="brand-name">NoCivicSense</div><div className="brand-subtitle">Traffic intelligence lab</div></div>
        <button className="icon-button collapse-button" onClick={onToggle} aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'} title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}>{collapsed ? '»' : '«'}</button>
        <button className="icon-button sidebar-close" onClick={onClose} aria-label="Close navigation">×</button>
      </div>
      <div className="nav-label">Workspace</div>
      <nav className="sidebar-nav" aria-label="Main navigation">
        {NAV.map(({ to, icon, label }) => (
          <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => `sidebar-link${isActive ? ' active' : ''}`} title={collapsed ? label : undefined} onClick={onClose}>
            <span className="nav-icon" aria-hidden="true">{icon}</span><span className="nav-text">{label}</span><span className="nav-indicator" />
          </NavLink>
        ))}
      </nav>
      <div className="sidebar-bottom">
        <div className="sidebar-dataset-icon">D4</div>
        <div className="sidebar-dataset-copy"><strong>PeMS04</strong><span>California · 307 sensors</span></div>
        <div className="dataset-live-dot" title="Static research dataset" />
      </div>
      <div className="sidebar-version">PS-003 <span>•</span> Research prototype</div>
    </aside>
  );
}
