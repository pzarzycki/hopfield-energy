import LaTeX from '../components/LaTeX';

export default function EnergyBasedModels() {
    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">Energy-Based Models</h1>
                <p className="page-intro">
                    Energy-Based Models (EBMs) provide a unified framework for understanding learning
                    as energy minimization. This general view encompasses many classical and modern
                    machine learning approaches.
                </p>
            </header>

            <div className="content-card">
                <h2>The Energy-Based Framework</h2>
                <p>
                    At the core of EBMs is the concept of an <strong>energy function</strong> <LaTeX math="E(x, y; \theta)" />:
                </p>
                <ul>
                    <li><strong>x:</strong> Input variables (observed data)</li>
                    <li><strong>y:</strong> Output variables (predictions, labels, or latent variables)</li>
                    <li><strong>θ:</strong> Model parameters</li>
                    <li><strong>E:</strong> Scalar energy value</li>
                </ul>
                <p>
                    The energy function captures the <em>compatibility</em> or <em>goodness</em> of a
                    configuration—low energy means high compatibility.
                </p>
            </div>

            <div className="content-card">
                <h2>Core Principle</h2>
                <p>
                    Learning consists of shaping the energy landscape:
                </p>
                <ul>
                    <li><strong>Desired configurations:</strong> Should have low energy</li>
                    <li><strong>Undesired configurations:</strong> Should have high energy</li>
                </ul>
                <p>
                    Inference is finding low-energy configurations:
                </p>
                <LaTeX block math="y^* = \text{argmin}_y E(x, y; \theta)" />
                <p>
                    This framework is remarkably general and applies to diverse learning problems.
                </p>
            </div>

            <div className="content-card">
                <h2>Probabilistic Interpretation</h2>
                <p>
                    EBMs connect to probability via the Gibbs distribution:
                </p>
                <LaTeX block math="P(y | x) = \frac{\exp(-E(x, y))}{Z(x)}" />
                <LaTeX block math="Z(x) = \sum_y \exp(-E(x, y))" />
                <p>
                    where Z(x) is the partition function. This means:
                </p>
                <ul>
                    <li>Lower energy → Higher probability</li>
                    <li>Energy differences determine probability ratios</li>
                    <li>Temperature parameter can control sharpness of distribution</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Loss Functions and Training</h2>
                <h3>Maximum Likelihood Training</h3>
                <p>
                    Maximize probability of correct outputs:
                </p>
                <LaTeX block math="\mathcal{L}_{ML} = -\log P(y | x) = E(x, y) + \log Z(x)" />
                <p>
                    Challenge: computing <LaTeX math="Z(x)" /> is often intractable!
                </p>

                <h3>Contrastive Methods</h3>
                <p>
                    Push down energy of correct answers, pull up energy of incorrect ones:
                </p>
                <LaTeX block math="\mathcal{L} = E(x, y^+) - E(x, y^-)" />
                <ul>
                    <li><LaTeX math="y^+" />: positive example (correct answer)</li>
                    <li><LaTeX math="y^-" />: negative example (contrastive sample)</li>
                </ul>

                <h3>Score Matching</h3>
                <p>
                    Match the gradient of the energy function to data distribution:
                </p>
                <LaTeX block math="\mathcal{L}_{SM} = \frac{1}{2} ||\nabla_x E(x) - \nabla_x \log p_{data}(x)||^2" />
                <p>
                    Avoids computing the partition function entirely.
                </p>
            </div>

            <div className="content_card">
                <h2>Examples of Energy-Based Models</h2>

                <h3>Classical EBMs</h3>
                <ul>
                    <li><strong>Hopfield Networks:</strong> <LaTeX math="E = -\sum_{i,j} w_{ij} s_i s_j" /></li>
                    <li><strong>Boltzmann Machines:</strong> Stochastic EBMs with Boltzmann distribution</li>
                    <li><strong>CRFs (Conditional Random Fields):</strong> Structured output prediction</li>
                    <li><strong>Maximum Entropy Models:</strong> Feature-based EBMs</li>
                </ul>

                <h3>Modern EBMs</h3>
                <ul>
                    <li><strong>Deep Energy Models:</strong> Neural networks as energy functions</li>
                    <li><strong>Score-Based Models:</strong> Learn energy gradients (diffusion models)</li>
                    <li><strong>Transformer Attention:</strong> Softmax over energy scores</li>
                    <li><strong>Contrastive Learning:</strong> SimCLR, MoCo (metric learning)</li>
                    <li><strong>Neural ODEs:</strong> Continuous-time energy minimization</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Unified View of Learning</h2>
                <p>
                    The EBM framework reveals connections between seemingly disparate methods:
                </p>

                <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '1rem' }}>
                    <thead>
                        <tr style={{ backgroundColor: 'var(--bg-tertiary)', borderBottom: '2px solid var(--border-color)' }}>
                            <th style={{ padding: '0.75rem', textAlign: 'left' }}>Model</th>
                            <th style={{ padding: '0.75rem', textAlign: 'left' }}>Energy Function</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                            <td style={{ padding: '0.75rem' }}>Linear Regression</td>
                            <td style={{ padding: '0.75rem' }}><LaTeX math="E(x,y) = \frac{1}{2}||y - w^Tx||^2" /></td>
                        </tr>
                        <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                            <td style={{ padding: '0.75rem' }}>Logistic Regression</td>
                            <td style={{ padding: '0.75rem' }}><LaTeX math="E(x,y) = -y \cdot w^Tx" /></td>
                        </tr>
                        <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                            <td style={{ padding: '0.75rem' }}>SVM</td>
                            <td style={{ padding: '0.75rem' }}><LaTeX math="E(x,y) = \max(0, 1 - y \cdot w^Tx)" /></td>
                        </tr>
                        <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                            <td style={{ padding: '0.75rem' }}>Neural Networks</td>
                            <td style={{ padding: '0.75rem' }}><LaTeX math="E(x,y) = -f_\theta(x, y)" /></td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div className="content-card">
                <h2>Modern Resurgence: Score-Based Models</h2>
                <p>
                    Diffusion models and score-based generative models represent a major EBM revival:
                </p>

                <h3>Key Ideas</h3>
                <ul>
                    <li><strong>Score function:</strong> <LaTeX math="s(x) = -\nabla_x E(x) = \nabla_x \log p(x)" /></li>
                    <li><strong>Learn score instead of energy:</strong> Avoids partition function</li>
                    <li><strong>Langevin dynamics:</strong> Sample by following score gradients</li>
                    <li><strong>Noise conditioning:</strong> Multi-scale scores for better training</li>
                </ul>

                <h3>Denoising Diffusion Models</h3>
                <p>
                    Add noise incrementally, learn to reverse the process:
                </p>
                <ol>
                    <li>Forward process: gradually add Gaussian noise to data</li>
                    <li>Reverse process: learn to denoise at each step (predict score)</li>
                    <li>Generation: start from noise, iteratively denoise</li>
                </ol>
                <p>
                    Result: state-of-the-art image generation (DALL-E 2, Stable Diffusion, Imagen).
                </p>
            </div>

            <div className="content-card">
                <h2>Advantages of the EBM Framework</h2>
                <ul>
                    <li><strong>Flexibility:</strong> Can model complex dependencies and constraints</li>
                    <li><strong>Generality:</strong> Unified view of diverse learning algorithms</li>
                    <li><strong>Structured prediction:</strong> Natural for outputs with rich structure</li>
                    <li><strong>Uncertainty quantification:</strong> Energy landscape shows confidence</li>
                    <li><strong>Interpretability:</strong> Energy provides semantic meaning</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Challenges</h2>
                <ul>
                    <li><strong>Intractable partition function:</strong> Computing Z is often exponentially hard</li>
                    <li><strong>Inference complexity:</strong> Finding argmin may require iterative optimization</li>
                    <li><strong>Sampling difficulty:</strong> MCMC can be slow to mix</li>
                    <li><strong>Negative sampling:</strong> Choosing good contrastive examples is crucial</li>
                    <li><strong>Training instability:</strong> Energy can be unbounded</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Current Research Directions</h2>
                <ul>
                    <li><strong>Tractable EBMs:</strong> Architectures with efficient partition functions</li>
                    <li><strong>Hybrid models:</strong> Combining EBMs with other approaches</li>
                    <li><strong>Continuous normalizing flows:</strong> Invertible transformations</li>
                    <li><strong>Implicit EBMs:</strong> Learning without explicit energy function</li>
                    <li><strong>Neural implicit representations:</strong> Coordinate-based networks</li>
                    <li><strong>Energy-based control:</strong> Reinforcement learning and robotics</li>
                </ul>
            </div>

            <div className="content-card">
                <h2>Philosophical Perspective</h2>
                <p>
                    The energy-based view offers deep insights into learning:
                </p>
                <ul>
                    <li><strong>Learning = Landscape shaping:</strong> Making good configurations attractive</li>
                    <li><strong>Inference = Optimization:</strong> Finding valleys in the landscape</li>
                    <li><strong>Generalization:</strong> Smooth energy landscapes extrapolate well</li>
                    <li><strong>Connection to physics:</strong> Thermodynamics, statistical mechanics</li>
                    <li><strong>Bayesian interpretation:</strong> Energy relates to negative log probability</li>
                </ul>
                <p>
                    This perspective unifies optimization, probability, and learning into a coherent framework.
                </p>
            </div>

            <div className="content-card">
                <h2>Conclusion</h2>
                <p>
                    Energy-Based Models provide not just a class of algorithms, but a <em>way of thinking</em>
                    about machine learning. From Hopfield's associative memories to modern diffusion models,
                    the principle of learning through energy minimization continues to drive innovation.
                </p>
                <p>
                    As we develop more sophisticated architectures and training methods, the energy-based
                    framework remains a powerful conceptual tool for understanding and designing learning systems.
                </p>
            </div>
        </div>
    );
}
