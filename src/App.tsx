import { Navigate, NavLink, Route, Routes } from "react-router-dom";

import "./App.css";
import HopfieldNetworkPage from "./pages/HopfieldNetwork";
import { PlaceholderNetworkPage } from "./pages/PlaceholderNetwork";

const NETWORK_ROUTES = [
  { path: "/networks/hopfield-network", label: "Hopfield Network" },
  { path: "/networks/dense-hopfield-network", label: "Dense Hopfield Network" },
  { path: "/networks/boltzmann-machine", label: "Boltzmann Machine" },
  { path: "/networks/restricted-boltzmann-machine", label: "Restricted Boltzmann Machine" },
] as const;

function App() {
  return (
    <div className="app-shell">
      <header className="network-nav">
        <div className="network-nav__eyebrow">Network Type</div>
        <nav className="network-nav__list" aria-label="Network type selection">
          {NETWORK_ROUTES.map((route) => (
            <NavLink
              key={route.path}
              to={route.path}
              className={({ isActive }) => `network-nav__link${isActive ? " is-active" : ""}`}
            >
              {route.label}
            </NavLink>
          ))}
        </nav>
      </header>

      <Routes>
        <Route path="/" element={<Navigate to="/networks/hopfield-network" replace />} />
        <Route path="/networks/hopfield-network" element={<HopfieldNetworkPage />} />
        <Route
          path="/networks/dense-hopfield-network"
          element={
            <PlaceholderNetworkPage
              title="Dense Hopfield Network"
              urlPath="/networks/dense-hopfield-network"
              description="Placeholder page. Dense Hopfield Network is not implemented yet."
            />
          }
        />
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
          element={
            <PlaceholderNetworkPage
              title="Restricted Boltzmann Machine"
              urlPath="/networks/restricted-boltzmann-machine"
              description="Placeholder page. Restricted Boltzmann Machine is not implemented yet."
            />
          }
        />
      </Routes>
    </div>
  );
}

export default App;
