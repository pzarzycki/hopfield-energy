import { Navigate, NavLink, Route, Routes } from "react-router-dom";

import "./App.css";
import DenseAssociativeMemoryPage from "./pages/DenseAssociativeMemory";
import DenseHopfieldNetworkPage from "./pages/DenseHopfieldNetwork";
import HopfieldNetworkPage from "./pages/HopfieldNetwork";
import { PlaceholderNetworkPage } from "./pages/PlaceholderNetwork";
import RestrictedBoltzmannMachinePage from "./pages/RestrictedBoltzmannMachine";
import { MODEL_CATALOG, formatPrimaryTasks } from "./core/modelCatalog";

const SHARE_URL = "https://pzarzycki.github.io/hopfield-energy/";
const SHARE_TITLE = "Energy-Based Memory Models — interactive Hopfield, Dense Associative Memory & RBM demos";

function ShareButtons() {
  const linkedin = `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(SHARE_URL)}`;
  const twitter  = `https://twitter.com/intent/tweet?url=${encodeURIComponent(SHARE_URL)}&text=${encodeURIComponent(SHARE_TITLE)}`;
  return (
    <div className="network-nav__share">
      <a href={linkedin} target="_blank" rel="noopener noreferrer"
         className="share-btn share-btn--linkedin"
         aria-label="Share on LinkedIn" title="Share on LinkedIn">
        <svg viewBox="0 0 24 24" aria-hidden="true" fill="currentColor">
          <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
        </svg>
      </a>
      <a href={twitter} target="_blank" rel="noopener noreferrer"
         className="share-btn share-btn--twitter"
         aria-label="Share on X / Twitter" title="Share on X / Twitter">
        <svg viewBox="0 0 24 24" aria-hidden="true" fill="currentColor">
          <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-4.714-6.231-5.401 6.231H2.744l7.737-8.835L2.25 2.25h6.928l4.279 5.655zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
        </svg>
      </a>
    </div>
  );
}

function App() {
  return (
    <div className="app-shell">
      <header className="network-nav">
        <div className="network-nav__eyebrow">Network Type</div>
        <div className="network-nav__row">
          <nav className="network-nav__list" aria-label="Network type selection">
            {MODEL_CATALOG.map((route) => (
              <NavLink
                key={route.path}
                to={route.path}
                className={({ isActive }) => `network-nav__link${isActive ? " is-active" : ""}`}
                title={`Primary task: ${formatPrimaryTasks(route.primaryTasks)}`}
              >
                {route.label}
              </NavLink>
            ))}
          </nav>
          <ShareButtons />
        </div>
      </header>

      <main className="app-main">
        <Routes>
          <Route path="/" element={<Navigate to="/networks/hopfield-network" replace />} />
          <Route path="/networks/hopfield-network" element={<HopfieldNetworkPage />} />
          <Route path="/networks/dense-hopfield-network" element={<DenseHopfieldNetworkPage />} />
          <Route path="/networks/dense-associative-memory" element={<DenseAssociativeMemoryPage />} />
          <Route
            path="/networks/boltzmann-machine"
            element={
              <PlaceholderNetworkPage
                title="Boltzmann Machine"
                urlPath="/networks/boltzmann-machine"
                description="Placeholder page. Boltzmann Machine is not implemented yet."
              />
            }
          />
          <Route
            path="/networks/restricted-boltzmann-machine"
            element={<RestrictedBoltzmannMachinePage />}
          />
        </Routes>
      </main>

      <footer className="app-footer">
        <span className="app-footer__label">GitHub:</span>
        <a
          className="app-footer__link"
          href="https://github.com/pzarzycki/hopfield-energy"
          target="_blank"
          rel="noopener noreferrer"
        >
          github.com/pzarzycki/hopfield-energy
        </a>
      </footer>
    </div>
  );
}

export default App;
