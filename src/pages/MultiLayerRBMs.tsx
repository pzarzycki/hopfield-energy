export default function MultiLayerRBMs() {
    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">Multi-layer RBMs</h1>
                <p className="page-intro">
                    Multi-layer RBMs, particularly Deep Belief Networks (DBNs), stack multiple
                    Restricted Boltzmann Machines to create hierarchical representations. They
                    pioneered the deep learning revolution by showing how to train deep networks effectively.
                </p>
            </header>

            <div className="content-card">
                <h2>Deep Belief Networks (DBNs)</h2>
                <p>
                    A Deep Belief Network is created by stacking RBMs on top of each other:
                </p>
                <ul>
                    <li>Each RBM layer learns features from the layer below</li>
                    <li>Lower layers capture low-level features (edges, textures)</li>
                    <li>Higher layers learn abstract, compositional features</li>
                    <li>Hybrid architecture: directed at top, undirected below</li>
                </ul>
                <p>
                    The result is a deep probabilistic model with powerful representational capabilities.
                </p>
            </div>

            <div className="content-card">
                <h2>Greedy Layer-wise Pre-training</h2>
                <p>
                    The breakthrough training algorithm for DBNs, introduced by Hinton et al. (2006):
                </p>
                <h3>Algorithm Steps</h3>
                <ol>
                    <li><strong>Train first RBM:</strong> Learn features from input data using CD</li>
                    <li><strong>Freeze first layer:</strong> Fix the learned weights</li>
                    <li><strong>Generate features:</strong> Pass data through trained RBM to get hidden activations</li>
                    <li><strong>Train second RBM:</strong> Use hidden activations as input data</li>
                    <li><strong>Repeat:</strong> Continue stacking RBMs layer by layer</li>
                    <li><strong>Fine-tune:</strong> (Optional) Use supervised learning to adjust all layers jointly</li>
                </ol>

                <h3>Key Insight</h3>
                <p>
                    Each layer improves the lower bound on the log-likelihood of the data, guaranteeing
                    that adding layers doesn't hurt performance (in theory).
                </p>
            </div>

            <div className="content-card">
                <h2>Architecture</h2>
                <p>
                    A typical DBN structure:
                </p>
                <pre><code>Input Layer (visible)
                    ↕ (RBM 1)
                    Hidden Layer 1
                    ↕ (RBM 2)
                    Hidden Layer 2
                    ↕ (RBM 3)
                    Hidden Layer 3
                    ↕
                    Top Layer (associative memory)</code></pre>

                <h3>Directional Interpretation</h3>
                <ul>
                    <li><strong>Bottom-up (recognition):</strong> Input → Features → Abstract concepts</li>
                    <li><strong>Top-down (generation):</strong> Abstract concepts → Features → Reconstructed input</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Hybrid Generative Model</h2>
                <p>
                    The complete DBN has a unique structure:
                </p>
                <ul>
                    <li><strong>Top two layers:</strong> Form an RBM (undirected, associative memory)</li>
                    <li><strong>Lower layers:</strong> Directed belief network (top-down generative model)</li>
                </ul>
                <p>
                    Joint probability distribution:
                </p>
                <pre><code>P(v, h¹, h², h³) = P(v|h¹) P(h¹|h²) P(h², h³)</code></pre>
                <p>
                    where P(h², h³) is defined by the top-level RBM.
                </p>
            </div>

            <div className="content-card">
                <h2>Why Pre-training Worked</h2>
                <p>
                    In the mid-2000s, greedy layer-wise pre-training solved critical challenges:
                </p>

                <h3>The Gradient Problem</h3>
                <ul>
                    <li>Random initialization led to poor local minima</li>
                    <li>Vanishing/exploding gradients in deep networks</li>
                    <li>Supervised training alone failed for deep architectures</li>
                </ul>

                <h3>Pre-training Benefits</h3>
                <ul>
                    <li><strong>Better initialization:</strong> Weights start in a good region of parameter space</li>
                    <li><strong>Feature learning:</strong> Unsupervised learning discovers useful representations</li>
                    <li><strong>Regularization:</strong> Acts as a strong prior on what features are meaningful</li>
                    <li><strong>Data efficiency:</strong> Leverages unlabeled data effectively</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Historical Impact (2006-2012)</h2>
                <p>
                    DBNs catalyzed the deep learning revolution:
                </p>
                <ul>
                    <li><strong>2006:</strong> Hinton's Science paper showed deep networks could be trained</li>
                    <li><strong>MNIST:</strong> Achieved state-of-the-art results on digit recognition</li>
                    <li><strong>Speech recognition:</strong> Microsoft's breakthrough using DBNs</li>
                    <li><strong>ImageNet:</strong> Initial deep learning attempts used pre-training</li>
                </ul>
                <p>
                    These successes demonstrated that "depth matters" and reignited interest in neural networks.
                </p>
            </div>

            <div className="content-card">
                <h2>Applications</h2>
                <h3>Unsupervised Pre-training</h3>
                <ul>
                    <li>Initialize deep networks for supervised tasks</li>
                    <li>Transfer learning before large labeled datasets</li>
                    <li>Domain adaptation and semi-supervised learning</li>
                </ul>

                <h3>Generative Modeling</h3>
                <ul>
                    <li>Generate synthetic images, audio, text</li>
                    <li>Data augmentation for training</li>
                    <li>Anomaly detection via reconstruction error</li>
                </ul>

                <h3>Dimensionality Reduction</h3>
                <ul>
                    <li>Learn compact representations of high-dimensional data</li>
                    <li>Visualization of complex datasets</li>
                    <li>Feature extraction for downstream tasks</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Decline and Modern Alternatives</h2>
                <p>
                    After ~2012, DBNs were largely superseded by newer techniques:
                </p>

                <h3>Why RBM Pre-training Fell Out of Favor</h3>
                <ul>
                    <li><strong>Better initialization:</strong> Xavier/He initialization works well</li>
                    <li><strong>ReLU activations:</strong> Mitigate vanishing gradients</li>
                    <li><strong>Batch normalization:</strong> Stabilizes deep network training</li>
                    <li><strong>More data:</strong> Large labeled datasets (ImageNet) enable pure supervised learning</li>
                    <li><strong>Residual connections:</strong> Enable training of very deep networks</li>
                </ul>

                <h3>Modern Alternatives</h3>
                <ul>
                    <li><strong>Autoencoders & VAEs:</strong> Simpler unsupervised learning</li>
                    <li><strong>GANs:</strong> Superior generative models</li>
                    <li><strong>Self-supervised learning:</strong> Contrastive methods, masked modeling</li>
                    <li><strong>Diffusion models:</strong> State-of-the-art generation</li>
                    <li><strong>Transformers:</strong> Dominant architecture across domains</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Conceptual Legacy</h2>
                <p>
                    While DBNs are rarely used today, their conceptual contributions endure:
                </p>
                <ul>
                    <li><strong>Hierarchical features:</strong> Deep networks learn layer-by-layer abstractions</li>
                    <li><strong>Unsupervised pre-training:</strong> Returns with self-supervised learning (BERT, GPT)</li>
                    <li><strong>Modular training:</strong> Inspired curriculum learning and progressive training</li>
                    <li><strong>Probabilistic modeling:</strong> Energy-based view of deep learning</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Technical Challenges</h2>
                <ul>
                    <li>Training requires multiple passes through data (pre-train each layer)</li>
                    <li>Contrastive Divergence is approximate, may not converge to true distribution</li>
                    <li>Difficult to train very deep DBNs ({'>'}4-5 layers)</li>
                    <li>Slow inference compared to feed-forward networks</li>
                    <li>Sampling from the model is expensive (MCMC required)</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>When to Consider Multi-layer RBMs</h2>
                <p>
                    Despite being superseded, DBNs may still be relevant in specific scenarios:
                </p>
                <ul>
                    <li>Very small labeled datasets with abundant unlabeled data</li>
                    <li>Interpretable hierarchical feature learning</li>
                    <li>Research on probabilistic models and energy-based learning</li>
                    <li>Educational purposes to understand deep learning history</li>
                    <li>Hybrid models combining modern and classical techniques</li>
                </ul>
            </div>
        </div>
    );
}
