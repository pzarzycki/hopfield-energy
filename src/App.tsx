import { Navigate, NavLink, Route, Routes } from "react-router-dom";

import "./App.css";
import DenseAssociativeMemoryPage from "./pages/DenseAssociativeMemory";
import DenseHopfieldNetworkPage from "./pages/DenseHopfieldNetwork";
import HopfieldNetworkPage from "./pages/HopfieldNetwork";
import { PlaceholderNetworkPage } from "./pages/PlaceholderNetwork";
import RestrictedBoltzmannMachinePage from "./pages/RestrictedBoltzmannMachine";
import { MODEL_CATALOG, formatPrimaryTasks } from "./core/modelCatalog";

function App() {
  return (
    <div className="app-shell">
      <header className="network-nav">
        <div className="network-nav__eyebrow">Network Type</div>
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
      </header>

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
    </div>
  );
}

export default App;
