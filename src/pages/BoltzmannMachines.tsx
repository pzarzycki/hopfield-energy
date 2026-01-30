export default function BoltzmannMachines() {
    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">Boltzmann Machines</h1>
                <p className="page-intro">
                    Boltzmann Machines are stochastic recurrent neural networks that can learn
                    probability distributions over binary vectors. They extend Hopfield networks
                    by introducing hidden units and stochastic dynamics.
                </p>
            </header>

            <div className="content-card">
                <h2>Architecture</h2>
                <p>
                    Unlike Hopfield networks, Boltzmann Machines consist of two types of units:
                </p>
                <ul>
                    <li><strong>Visible units:</strong> Represent observable data</li>
                    <li><strong>Hidden units:</strong> Capture latent structure and correlations in data</li>
                </ul>
                <p>
                    Units can be connected in arbitrary ways, forming a general undirected graph.
                    This flexibility allows the model to capture complex dependencies.
                </p>
            </div>

            <div className="content-card">
                <h2>Energy Function</h2>
                <p>
                    The energy of a configuration is defined as:
                </p>
                <pre><code>E(v, h) = -Σᵢⱼ wᵢⱼ sᵢ sⱼ - Σᵢ bᵢ sᵢ</code></pre>
                <p>
                    where v represents visible units, h represents hidden units, w are weights,
                    and b are biases. The probability of a configuration follows the Boltzmann distribution.
                </p>
            </div>

            <div className="content-card">
                <h2>Boltzmann Distribution</h2>
                <p>
                    At thermal equilibrium, the probability of a state is given by:
                </p>
                <pre><code>P(v, h) = (1/Z) exp(-E(v, h) / T)</code></pre>
                <p>
                    where T is temperature and Z is the partition function (normalization constant):
                </p>
                <pre><code>Z = Σᵥ Σₕ exp(-E(v, h) / T)</code></pre>
                <p>
                    Lower energy states are exponentially more probable than high energy states.
                </p>
            </div>

            <div className="content-card">
                <h2>Stochastic Dynamics</h2>
                <p>
                    Unlike deterministic Hopfield networks, Boltzmann Machines update units stochastically:
                </p>
                <pre><code>P(sᵢ = 1) = σ(Σⱼ wᵢⱼ sⱼ + bᵢ)</code></pre>
                <p>
                    where σ is the logistic sigmoid function. This stochasticity allows the network to:
                </p>
                <ul>
                    <li>Escape local minima during learning</li>
                    <li>Sample from the learned probability distribution</li>
                    <li>Explore the state space more thoroughly</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Learning Algorithm</h2>
                <p>
                    Boltzmann Machines learn by maximizing the log-likelihood of the training data.
                    The weight update rule is:
                </p>
                <pre><code>Δwᵢⱼ ∝ ⟨sᵢ sⱼ⟩ₐₐₜₐ - ⟨sᵢ sⱼ⟩ₘₒₐₑₗ</code></pre>
                <p>
                    This requires computing two expectations:
                </p>
                <ul>
                    <li><strong>Positive phase:</strong> Statistics when visible units are clamped to data</li>
                    <li><strong>Negative phase:</strong> Statistics when the network runs freely</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Challenges</h2>
                <h3>Computational Complexity</h3>
                <p>
                    Training Boltzmann Machines is notoriously difficult:
                </p>
                <ul>
                    <li>Computing the partition function Z is intractable for large networks</li>
                    <li>Requires extensive sampling (MCMC) to estimate expectations</li>
                    <li>Convergence to equilibrium can be extremely slow</li>
                    <li>Training time scales poorly with network size</li>
                </ul>

                <h3>Solutions</h3>
                <ul>
                    <li>Restricted Boltzmann Machines (RBMs) simplify the architecture</li>
                    <li>Contrastive Divergence provides faster approximate learning</li>
                    <li>Mean-field approximations estimate expectations analytically</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Advantages</h2>
                <ul>
                    <li>Can learn complex probability distributions</li>
                    <li>Hidden units capture latent structure in data</li>
                    <li>Generative model: can synthesize new samples</li>
                    <li>Theoretical foundation in statistical physics</li>
                    <li>Bidirectional connections enable rich dynamics</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Historical Significance</h2>
                <p>
                    Boltzmann Machines, introduced by Hinton and Sejnowski in 1985, were among the
                    first neural networks with hidden units and a principled learning algorithm. While
                    general Boltzmann Machines proved too slow for practical use, they inspired:
                </p>
                <ul>
                    <li>Restricted Boltzmann Machines (more practical variant)</li>
                    <li>Deep Belief Networks (stacked RBMs)</li>
                    <li>Modern energy-based models and score-based generative models</li>
                </ul>
            </div>
        </div>
    );
}
