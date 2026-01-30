export default function HopfieldNetworks() {
    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">Hopfield Networks</h1>
                <p className="page-intro">
                    Hopfield networks are recurrent neural networks that serve as content-addressable
                    memory systems. Introduced by John Hopfield in 1982, they demonstrate how neural
                    networks can store and retrieve patterns.
                </p>
            </header>

            <div className="content-card">
                <h2>Architecture</h2>
                <p>
                    A Hopfield network consists of a set of N binary neurons, where each neuron is
                    connected to every other neuron (fully connected). The network has no hidden layers—all
                    neurons are both input and output units.
                </p>
                <h3>Key Components</h3>
                <ul>
                    <li><strong>Neurons:</strong> Binary units that can be in state +1 or -1</li>
                    <li><strong>Weights:</strong> Symmetric connections (w<sub>ij</sub> = w<sub>ji</sub>) between neurons</li>
                    <li><strong>No self-connections:</strong> w<sub>ii</sub> = 0 for all neurons</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Energy Function</h2>
                <p>
                    The Hopfield network defines an energy function that decreases with each update:
                </p>
                <pre><code>E = -½ Σᵢⱼ wᵢⱼ sᵢ sⱼ - Σᵢ θᵢ sᵢ</code></pre>
                <p>
                    where <code>sᵢ</code> is the state of neuron i, <code>wᵢⱼ</code> are the connection weights,
                    and <code>θᵢ</code> are neuron thresholds. The network evolves toward local energy minima,
                    which correspond to stored patterns.
                </p>
            </div>

            <div className="content-card">
                <h2>Update Rule</h2>
                <p>
                    Neurons update their states asynchronously based on the current network state:
                </p>
                <pre><code>sᵢ(t+1) = sign(Σⱼ wᵢⱼ sⱼ(t) - θᵢ)</code></pre>
                <p>
                    This update rule guarantees that the energy function never increases, ensuring
                    convergence to a stable state (attractor).
                </p>
            </div>

            <div className="content-card">
                <h2>Learning: Hebbian Rule</h2>
                <p>
                    To store patterns, Hopfield networks use Hebbian learning: "neurons that fire together,
                    wire together." The weight matrix is computed as:
                </p>
                <pre><code>wᵢⱼ = (1/N) Σₚ xᵢᵖ xⱼᵖ</code></pre>
                <p>
                    where <code>xᵖ</code> represents the p-th pattern to be stored, and N is the number of neurons.
                </p>
            </div>

            <div className="content-card">
                <h2>Storage Capacity</h2>
                <p>
                    A critical limitation of classical Hopfield networks is their storage capacity.
                    For a network with N neurons:
                </p>
                <ul>
                    <li>Maximum capacity: approximately 0.15N patterns</li>
                    <li>Beyond this limit, spurious states (unintended attractors) emerge</li>
                    <li>Pattern retrieval becomes unreliable with too many stored patterns</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Properties</h2>
                <h3>Advantages</h3>
                <ul>
                    <li>Robust to noise: can retrieve complete patterns from partial inputs</li>
                    <li>Content-addressable memory: access by content rather than address</li>
                    <li>Guaranteed convergence to stable states</li>
                    <li>Simple and biologically plausible learning rule</li>
                </ul>

                <h3>Limitations</h3>
                <ul>
                    <li>Limited storage capacity (0.15N patterns)</li>
                    <li>Spurious states can trap the network in incorrect patterns</li>
                    <li>Synchronous updates may lead to oscillations</li>
                    <li>All-to-all connectivity scales poorly (O(N²) connections)</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Applications</h2>
                <ul>
                    <li>Pattern completion and restoration</li>
                    <li>Error correction in noisy data</li>
                    <li>Combinatorial optimization problems</li>
                    <li>Associative memory tasks</li>
                    <li>Modeling aspects of biological memory</li>
                </ul>
            </div>
        </div>
    );
}
