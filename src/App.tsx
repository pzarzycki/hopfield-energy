import { HashRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import ScrollToTop from './components/ScrollToTop';
import Introduction from './pages/Introduction';
import HopfieldNetworks from './pages/HopfieldNetworks';
import BoltzmannMachines from './pages/BoltzmannMachines';
import RestrictedBoltzmannMachines from './pages/RestrictedBoltzmannMachines';
import DenseHopfieldNetworks from './pages/DenseHopfieldNetworks';
import MultiLayerRBMs from './pages/MultiLayerRBMs';
import EnergyBasedModels from './pages/EnergyBasedModels';
import './App.css';

function App() {
  return (
    <Router>
      <ScrollToTop />
      <Layout>
        <Routes>
          <Route path="/" element={<Introduction />} />
          <Route path="/hopfield-networks" element={<HopfieldNetworks />} />
          <Route path="/boltzmann-machines" element={<BoltzmannMachines />} />
          <Route path="/restricted-boltzmann-machines" element={<RestrictedBoltzmannMachines />} />
          <Route path="/dense-hopfield-networks" element={<DenseHopfieldNetworks />} />
          <Route path="/multi-layer-rbms" element={<MultiLayerRBMs />} />
          <Route path="/energy-based-models" element={<EnergyBasedModels />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
