import LaTeX from '../components/LaTeX';

export default function DenseHopfieldNetworks() {
    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">Dense Hopfield Networks</h1>
                <p className="page-intro">
                    Dense (or Modern) Hopfield Networks represent a significant advancement over classical
                    Hopfield networks, featuring exponentially larger storage capacity and continuous
                    energy landscapes. Introduced by Krotov and Hopfield (2016) and refined by Ramsauer et al. (2020).
                </p>
            </header>

            <div className="content-card">
                <h2>Motivation</h2>
                <p>
                    Classical Hopfield networks suffer from severe limitations:
                </p>
                <ul>
                    <li>Storage capacity of only ~0.15N patterns for N neurons</li>
                    <li>Spurious states interfere with retrieval</li>
                    <li>Binary states limit expressiveness</li>
                </ul>
                <p>
                    Dense Hopfield Networks overcome these limitations through a modernized energy function
                    and update rules.
                </p>
            </div>

            <div className="content-card">
                <h2>Modern Energy Function</h2>
                <p>
                    The energy function uses higher-order polynomial interactions:
                </p>
                <LaTeX block math="E(\xi) = -\text{log} \sum_n \exp(\beta \xi^T X_n)" />
                <p>
                    where:
                </p>
                <ul>
                    <li><LaTeX math="\xi" /> is the current state (continuous-valued)</li>
                    <li><LaTeX math="X_n" /> are stored patterns</li>
                    <li><LaTeX math="\beta" /> controls the separation between energy levels</li>
                </ul>
                <p>
                    This formulation is equivalent to the Log-Sum-Exp function, connecting to modern
                    attention mechanisms.
                </p>
            </div>

            <div className="content-card">
                <h2>Update Rule and Retrieval</h2>
                <p>
                    The update rule is derived from minimizing the energy function:
                </p>
                <LaTeX block math="\xi(t+1) = \text{softmax}(\beta X \xi(t))^T X" />
                <p>
                    This can be recognized as a form of <strong>attention mechanism</strong>:
                </p>
                <ul>
                    <li>Compute similarities between current state and stored patterns</li>
                    <li>Apply softmax to get attention weights</li>
                    <li>Weighted combination of stored patterns</li>
                </ul>
                <p>
                    This connection links Hopfield networks to Transformer architectures!
                </p>
            </div>

            <div className="content-card">
                <h2>Exponential Storage Capacity</h2>
                <p>
                    Dense Hopfield Networks achieve remarkable storage capacity:
                </p>
                <ul>
                    <li><strong>Classical Hopfield:</strong> ~0.15N patterns</li>
                    <li><strong>Dense Hopfield:</strong> exponential in N (e.g., e^N with proper scaling)</li>
                </ul>
                <p>
                    The separation between energy levels grows exponentially with the number of neurons,
                    enabling reliable storage and retrieval of vastly more patterns.
                </p>

                <h3>Theoretical Guarantees</h3>
                <p>
                    With appropriate parameter settings:
                </p>
                <ul>
                    <li>Stored patterns are global energy minima</li>
                    <li>Exponentially small retrieval error</li>
                    <li>Robust to noise and perturbations</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Connection to Attention Mechanisms</h2>
                <p>
                    The update rule of Dense Hopfield Networks is mathematically equivalent to
                    attention in Transformers:
                </p>
                <LaTeX block math="\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V" />
                <p>
                    Mapping to Hopfield retrieval:
                </p>
                <ul>
                    <li>Query Q <LaTeX math="\leftrightarrow" /> Current state <LaTeX math="\xi" /></li>
                    <li>Keys K <LaTeX math="\leftrightarrow" /> Stored patterns X</li>
                    <li>Values V <LaTeX math="\leftrightarrow" /> Stored patterns X</li>
                    <li>Temperature <LaTeX math="\sqrt{d} \leftrightarrow 1/\beta" /></li>
                </ul>
                <p>
                    This reveals that <strong>attention is a form of associative memory retrieval</strong>!
                </p>
            </div>

            <div className="content-card">
                <h2>Continuous States</h2>
                <p>
                    Unlike classical binary Hopfield networks, dense variants operate on continuous states:
                </p>
                <ul>
                    <li>States and patterns are real-valued vectors</li>
                    <li>Enables integration with modern neural architectures</li>
                    <li>Allows gradient-based optimization</li>
                    <li>Natural handling of continuous data (images, embeddings)</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Hopfield Layers in Deep Learning</h2>
                <p>
                    Dense Hopfield Networks can be integrated as layers in deep neural networks:
                </p>
                <h3>Hopfield Layer</h3>
                <pre><code>{`class HopfieldLayer(nn.Module):
    def forward(self, xi, X):
        # Compute attention weights
        scores = beta * torch.matmul(xi, X.T)
        weights = torch.softmax(scores, dim=-1)

        # Retrieve weighted combination
        output = torch.matmul(weights, X)
        return output`}</code></pre>

                <h3>Applications</h3>
                <ul>
                    <li>Associative memory in neural architectures</li>
                    <li>Improving Transformer efficiency</li>
                    <li>Few-shot learning (store examples in memory)</li>
                    <li>Prototype-based classification</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Metastable States</h2>
                <p>
                    An interesting property of Dense Hopfield Networks:
                </p>
                <ul>
                    <li><strong>Metastable states:</strong> The energy landscape can have shallow local minima</li>
                    <li><strong>Compositional retrieval:</strong> Can retrieve mixtures of patterns</li>
                    <li><strong>Smooth interpolation:</strong> Gradual transitions between stored patterns</li>
                </ul>
                <p>
                    This enables more flexible and nuanced memory retrieval compared to classical models.
                </p>
            </div>

            <div className="content-card">
                <h2>Recent Developments</h2>
                <ul>
                    <li><strong>Hopfield Networks is All You Need (2020):</strong> Unified view with attention</li>
                    <li><strong>Continuous attractors:</strong> Extensions to manifolds of stable states</li>
                    <li><strong>Hopfield Layers:</strong> Integration with deep learning frameworks</li>
                    <li><strong>Biological plausibility:</strong> Connections to cortical computations</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Advantages Over Classical Hopfield Networks</h2>
                <ul>
                    <li>✓ Exponential storage capacity vs. linear</li>
                    <li>✓ Continuous states and patterns</li>
                    <li>✓ Stronger theoretical guarantees</li>
                    <li>✓ Direct connection to modern deep learning</li>
                    <li>✓ Differentiable and trainable end-to-end</li>
                    <li>✓ No spurious states with proper parametrization</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Impact and Future Directions</h2>
                <p>
                    Dense Hopfield Networks bridge classical energy-based models with modern deep learning:
                </p>
                <ul>
                    <li>Provide theoretical foundations for attention mechanisms</li>
                    <li>Enable new architectures combining memory and learning</li>
                    <li>Inspire biologically-motivated neural models</li>
                    <li>Open research directions in associative memory and beyond</li>
                </ul>
            </div>
        </div>
    );
}
