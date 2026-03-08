import LaTeX from '../components/LaTeX';

export default function Introduction() {
    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">Energy-Based Memory Models</h1>
                <p className="page-intro">
                    Welcome to this comprehensive showcase of energy-based memory models.
                    These powerful architectures form the foundation of modern machine learning,
                    combining principles from statistical physics and neural computation.
                </p>
            </header>

            <div className="content-card">
                <h2>What are Energy-Based Models?</h2>
                <p>
                    Energy-based models (EBMs) are a class of machine learning models that learn to
                    associate low energy values with correct or desired configurations, and high energy
                    values with incorrect ones. This framework provides a unified view of many learning
                    algorithms and neural architectures.
                </p>
                <p>
                    The key concept is the <strong>energy function</strong> <LaTeX math="E(x)" />, which maps any
                    configuration of variables to a scalar value. During training, the model learns
                    to shape this energy landscape such that desired patterns occupy low-energy states.
                </p>
            </div>

            <div className="content-card">
                <h2>Historical Context</h2>
                <p>
                    Energy-based models have their roots in statistical physics, particularly in the
                    study of spin glasses and the Ising model. The connection between physics and
                    neural networks was pioneered by physicists like John Hopfield in the 1980s.
                </p>
                <p>
                    These models have evolved from simple associative memories to sophisticated
                    architectures capable of learning complex probability distributions and
                    generating new data.
                </p>
            </div>

            <div className="content-card">
                <h2>Key Concepts</h2>
                <ul>
                    <li><strong>Energy Function:</strong> A mathematical function that assigns scalar energy values to system states</li>
                    <li><strong>Attractor States:</strong> Low-energy configurations that the system naturally settles into</li>
                    <li><strong>Boltzmann Distribution:</strong> The probability distribution over states at thermal equilibrium</li>
                    <li><strong>Associative Memory:</strong> The ability to retrieve complete patterns from partial or noisy inputs</li>
                    <li><strong>Learning Rules:</strong> Algorithms for adjusting model parameters to shape the energy landscape</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Applications</h2>
                <p>
                    Energy-based memory models have found applications across diverse domains:
                </p>
                <ul>
                    <li>Pattern recognition and completion</li>
                    <li>Optimization problems (traveling salesman, graph coloring)</li>
                    <li>Generative modeling and data synthesis</li>
                    <li>Feature learning and dimensionality reduction</li>
                    <li>Recommendation systems and collaborative filtering</li>
                    <li>Image denoising and restoration</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Navigation Guide</h2>
                <p>
                    This showcase is organized to guide you through the evolution and variants of
                    energy-based models:
                </p>
                <ol>
                    <li><strong>Hopfield Networks:</strong> The foundational associative memory model</li>
                    <li><strong>Boltzmann Machines:</strong> Stochastic neural networks with hidden units</li>
                    <li><strong>Restricted Boltzmann Machines:</strong> Simplified architecture enabling efficient training</li>
                    <li><strong>Dense Hopfield Networks:</strong> Modern variants with increased capacity</li>
                    <li><strong>Multi-layer RBMs:</strong> Deep architectures for hierarchical learning</li>
                    <li><strong>Energy-Based Models:</strong> General theoretical framework and modern approaches</li>
                </ol>
                <p>
                    Use the navigation menu on the left to explore each topic in detail.
                </p>
            </div>
        </div>
    );
}
