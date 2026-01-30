export default function RestrictedBoltzmannMachines() {
    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">Restricted Boltzmann Machines</h1>
                <p className="page-intro">
                    Restricted Boltzmann Machines (RBMs) are a simplified variant of Boltzmann Machines
                    with a bipartite structure. This restriction makes training tractable while
                    maintaining powerful representational capabilities.
                </p>
            </header>

            <div className="content-card">
                <h2>Architecture</h2>
                <p>
                    The key restriction in RBMs is their bipartite structure:
                </p>
                <ul>
                    <li><strong>Two layers:</strong> visible layer (v) and hidden layer (h)</li>
                    <li><strong>No intra-layer connections:</strong> visible units don't connect to each other,
                        nor do hidden units</li>
                    <li><strong>Fully connected between layers:</strong> each visible unit connects to all hidden units</li>
                </ul>
                <p>
                    This restriction dramatically simplifies inference and learning.
                </p>
            </div>

            <div className="content-card">
                <h2>Energy Function</h2>
                <p>
                    The energy of an RBM configuration is:
                </p>
                <pre><code>E(v, h) = -Σᵢⱼ vᵢ wᵢⱼ hⱼ - Σᵢ aᵢ vᵢ - Σⱼ bⱼ hⱼ</code></pre>
                <p>
                    where:
                </p>
                <ul>
                    <li>v = visible layer state</li>
                    <li>h = hidden layer state</li>
                    <li>w = weight matrix</li>
                    <li>a, b = visible and hidden biases</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Conditional Independence</h2>
                <p>
                    The bipartite structure creates a crucial property: <strong>conditional independence</strong>.
                    Given one layer, all units in the other layer are independent:
                </p>
                <pre><code>P(h | v) = ∏ⱼ P(hⱼ | v)
                    P(v | h) = ∏ᵢ P(vᵢ | h)</code></pre>
                <p>
                    This allows parallel computation of all units in a layer:
                </p>
                <pre><code>P(hⱼ = 1 | v) = σ(Σᵢ wᵢⱼ vᵢ + bⱼ)
                    P(vᵢ = 1 | h) = σ(Σⱼ wᵢⱼ hⱼ + aᵢ)</code></pre>
            </div>

            <div className="content-card">
                <h2>Contrastive Divergence</h2>
                <p>
                    Geoffrey Hinton introduced Contrastive Divergence (CD), a breakthrough training algorithm
                    for RBMs that makes learning practical:
                </p>
                <h3>CD-k Algorithm</h3>
                <ol>
                    <li>Initialize visible units with training data</li>
                    <li>Compute hidden probabilities: P(h | v⁰)</li>
                    <li>Sample hidden states: h⁰ ~ P(h | v⁰)</li>
                    <li>Reconstruct visible: v¹ ~ P(v | h⁰)</li>
                    <li>Recompute hidden: h¹ ~ P(h | v¹)</li>
                    <li>Repeat steps 4-5 for k iterations (typically k=1)</li>
                    <li>Update weights: Δw ∝ ⟨v⁰h⁰⟩ₐₐₜₐ - ⟨vᵏhᵏ⟩ₘₒₐₑₗ</li>
                </ol>
                <p>
                    CD-1 (k=1) is remarkably effective despite being an approximation.
                </p>
            </div>

            <div className="content-card">
                <h2>Training Details</h2>
                <h3>Weight Update Rule</h3>
                <pre><code>Δwᵢⱼ = ε (⟨vᵢ hⱼ⟩ₐₐₜₐ - ⟨vᵢ hⱼ⟩ᵣₑ꜀ₒₙ)
                    Δaᵢ = ε (⟨vᵢ⟩ₐₐₜₐ - ⟨vᵢ⟩ᵣₑ꜀ₒₙ)
                    Δbⱼ = ε (⟨hⱼ⟩ₐₐₜₐ - ⟨hⱼ⟩ᵣₑ꜀ₒₙ)</code></pre>

                <h3>Common Enhancements</h3>
                <ul>
                    <li><strong>Momentum:</strong> Smooth weight updates over time</li>
                    <li><strong>Weight decay:</strong> L2 regularization to prevent overfitting</li>
                    <li><strong>Sparsity:</strong> Encourage sparse hidden representations</li>
                    <li><strong>Mini-batches:</strong> Process multiple examples simultaneously</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Applications</h2>
                <h3>Unsupervised Feature Learning</h3>
                <p>
                    RBMs excel at discovering useful features from unlabeled data:
                </p>
                <ul>
                    <li>Pre-training for deep neural networks</li>
                    <li>Dimensionality reduction</li>
                    <li>Feature extraction for classification</li>
                </ul>

                <h3>Collaborative Filtering</h3>
                <p>
                    RBMs power recommendation systems:
                </p>
                <ul>
                    <li>Netflix Prize winning solution used RBMs</li>
                    <li>Model user preferences and item features</li>
                    <li>Handle missing data naturally</li>
                </ul>

                <h3>Other Applications</h3>
                <ul>
                    <li>Image denoising and completion</li>
                    <li>Topic modeling for text</li>
                    <li>Motion capture data analysis</li>
                    <li>Anomaly detection</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Variants</h2>
                <ul>
                    <li><strong>Gaussian RBM:</strong> Real-valued visible units for continuous data</li>
                    <li><strong>Sparse RBM:</strong> Enforces sparse hidden representations</li>
                    <li><strong>Convolutional RBM:</strong> Weight sharing for image data</li>
                    <li><strong>Conditional RBM:</strong> Incorporates additional conditional inputs</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Advantages</h2>
                <ul>
                    <li>Efficient training via Contrastive Divergence</li>
                    <li>Tractable inference due to conditional independence</li>
                    <li>Powerful unsupervised feature learning</li>
                    <li>Can be stacked to form Deep Belief Networks</li>
                    <li>Handles both discrete and continuous data (with variants)</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Legacy and Impact</h2>
                <p>
                    RBMs played a pivotal role in the deep learning revolution:
                </p>
                <ul>
                    <li>Enabled training of deep neural networks (pre-training era ~2006-2012)</li>
                    <li>Demonstrated the power of unsupervised learning</li>
                    <li>Inspired modern generative models (VAEs, GANs, diffusion models)</li>
                    <li>Showed that energy-based models could scale to practical problems</li>
                </ul>
                <p>
                    While modern deep learning has moved beyond RBMs, their conceptual contributions
                    remain influential in contemporary machine learning.
                </p>
            </div>
        </div>
    );
}
