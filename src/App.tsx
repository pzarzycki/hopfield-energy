import { HashRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import ScrollToTop from './components/ScrollToTop';
import HopfieldNetwork from './pages/HopfieldNetwork';
import './App.css';

function App() {
    return (
        <Router>
            <ScrollToTop />
            <Layout>
                <Routes>
                    <Route path="/" element={<HopfieldNetwork />} />
                    <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
            </Layout>
        </Router>
    );
}

export default App;
