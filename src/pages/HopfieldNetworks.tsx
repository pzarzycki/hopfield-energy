import { useState, useEffect, useRef } from 'react';
import PageNav from '../components/PageNav';
import DrawingCanvas from '../components/DrawingCanvas';
import NetworkVisualizer from '../components/NetworkVisualizer';
import AnimationControls from '../components/AnimationControls';
import LaTeX from '../components/LaTeX';
import { HopfieldNetwork } from '../utils/hopfield-network';
import { getMNISTDigits } from '../utils/mnist-data';

export default function HopfieldNetworks() {
    const [network] = useState(() => new HopfieldNetwork(784));
    const [memories, setMemories] = useState<number[][]>([]);
    const [queryPattern, setQueryPattern] = useState<number[]>([]);
    const [currentState, setCurrentState] = useState<number[]>([]);
    const [energyHistory, setEnergyHistory] = useState<number[]>([]);
    const [currentStep, setCurrentStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [speed, setSpeed] = useState(5);
    const [isConverged, setIsConverged] = useState(false);
    const [matchedDigit, setMatchedDigit] = useState(-1);
    const [hasStarted, setHasStarted] = useState(false);
    const animationRef = useRef<number | null>(null);
    const lastUpdateRef = useRef<number>(0);

    const sections = [
        { id: 'overview', label: 'Overview' },
        { id: 'memories', label: 'Stored Memories' },
        { id: 'retrieval', label: 'Pattern Retrieval' },
        { id: 'theory', label: 'Theory' }
    ];

    // Initialize network with MNIST digits on mount
    useEffect(() => {
        const digits = getMNISTDigits();
        const patterns = digits.map(d => d.pattern);
        network.train(patterns);
        setMemories(patterns);
    }, [network]);

    // Handle pattern change from drawing canvas
    const handlePatternChange = (pattern: number[]) => {
        setQueryPattern(pattern);
    };

    // Start retrieval process
    const handleSubmit = () => {
        if (queryPattern.length === 0) return;

        // Reset network state
        network.setState(queryPattern);
        setCurrentState([...queryPattern]);
        setEnergyHistory([network.getCurrentEnergy()]);
        setCurrentStep(0);
        setIsConverged(false);
        setMatchedDigit(network.getMatchedPattern());
        setHasStarted(true);
        setIsPlaying(false);

        // Scroll to visualization
        setTimeout(() => {
            document.getElementById('visualization')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    };

    // Animation loop
    useEffect(() => {
        if (!isPlaying) {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
                animationRef.current = null;
            }
            return;
        }

        const animate = (timestamp: number) => {
            if (!lastUpdateRef.current) {
                lastUpdateRef.current = timestamp;
            }

            const elapsed = timestamp - lastUpdateRef.current;
            const interval = 1000 / speed;

            if (elapsed >= interval) {
                if (!network.hasConverged()) {
                    const state = network.step();
                    setCurrentState([...state.state]);
                    setEnergyHistory(prev => [...prev, state.energy]);
                    setCurrentStep(state.step);
                    setMatchedDigit(network.getMatchedPattern());
                    lastUpdateRef.current = timestamp;
                } else {
                    setIsConverged(true);
                    setIsPlaying(false);
                    return;
                }
            }

            animationRef.current = requestAnimationFrame(animate);
        };

        animationRef.current = requestAnimationFrame(animate);

        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
        };
    }, [isPlaying, speed, network]);

    // Control handlers
    const handlePlay = () => {
        if (!hasStarted || isConverged) return;
        setIsPlaying(true);
        lastUpdateRef.current = 0;
    };

    const handlePause = () => {
        setIsPlaying(false);
    };

    const handleStepForward = () => {
        if (network.hasConverged()) {
            setIsConverged(true);
            return;
        }

        const state = network.step();
        setCurrentState([...state.state]);
        setEnergyHistory(prev => [...prev, state.energy]);
        setCurrentStep(state.step);
        setMatchedDigit(network.getMatchedPattern());

        if (network.hasConverged()) {
            setIsConverged(true);
        }
    };

    const handleStepBackward = () => {
        if (currentStep <= 0) return;

        const prevStep = currentStep - 1;
        network.resetToStep(prevStep);
        const prevState = network.history[prevStep];
        setCurrentState([...prevState.state]);
        setEnergyHistory(prev => prev.slice(0, prevStep + 1));
        setCurrentStep(prevStep);
        setIsConverged(false);
        setMatchedDigit(network.getMatchedPattern());
    };

    const handleReset = () => {
        if (!hasStarted) return;

        network.resetToStep(0);
        const initialState = network.history[0];
        setCurrentState([...initialState.state]);
        setEnergyHistory([initialState.energy]);
        setCurrentStep(0);
        setIsConverged(false);
        setMatchedDigit(network.getMatchedPattern());
        setIsPlaying(false);
    };

    const handleRunToEnd = () => {
        setIsPlaying(false);

        let iterations = 0;
        const maxIterations = 100;

        while (!network.hasConverged() && iterations < maxIterations) {
            const state = network.step();
            setCurrentState([...state.state]);
            setEnergyHistory(prev => [...prev, state.energy]);
            setCurrentStep(state.step);
            iterations++;
        }

        setIsConverged(true);
        setMatchedDigit(network.getMatchedPattern());
    };

    const handleSpeedChange = (newSpeed: number) => {
        setSpeed(newSpeed);
    };

    // Calculate evolution history for filmstrip (take up to 6 evenly spaced snapshots)
    const getEvolutionHistory = () => {
        const history = network.history;
        if (history.length === 0) return [];

        // Always include start and current end
        if (history.length <= 6) return history;

        const snapshots = [];
        // Add start
        snapshots.push(history[0]);

        // Add minimal intermediate points
        const count = 4; // plus start and end = 6
        const step = (history.length - 1) / (count + 1);

        for (let i = 1; i <= count; i++) {
            const index = Math.floor(i * step);
            if (index > 0 && index < history.length - 1) {
                snapshots.push(history[index]);
            }
        }

        // Add end
        snapshots.push(history[history.length - 1]);

        // Deduplicate just in case
        return Array.from(new Set(snapshots));
    };

    return (
        <div>
            <PageNav sections={sections} />

            <header className="page-header" id="overview">
                <h1 className="page-title">Hopfield Networks</h1>
                <p className="page-intro">
                    Hopfield networks are recurrent neural networks that serve as content-addressable
                    memory systems. Introduced by John Hopfield in 1982, they demonstrate how neural
                    networks can store and retrieve patterns through energy minimization.
                </p>
            </header>

            {/* Stored Memories Section */}
            <section id="memories" className="demo-section">
                <h2>Stored Memories</h2>
                <div className="content-card">
                    <p>
                        This network has been trained to store 10 MNIST digit patterns (0-9).
                        Each pattern is a 28×28 binary image (784 neurons). The network uses Hebbian
                        learning to compute connection weights:
                    </p>

                    <LaTeX
                        block
                        math="w_{ij} = \frac{1}{P} \sum_{p=1}^{P} x_i^p x_j^p"
                    />

                    <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                        where <LaTeX math="P" /> is the number of patterns, <LaTeX math="x_i^p" /> is
                        the <LaTeX math="i" />-th neuron value in pattern <LaTeX math="p" />,
                        and <LaTeX math="w_{ij}" /> are the connection weights with <LaTeX math="w_{ii} = 0" /> (no self-connections).
                    </p>

                    {memories.length > 0 && (
                        <div style={{ marginTop: '2rem' }}>
                            <h3 style={{ fontSize: '1rem', marginBottom: '1rem' }}>Memory Patterns</h3>
                            <div className="pattern-grid">
                                {memories.map((memory, idx) => (
                                    <MemoryPattern
                                        key={idx}
                                        pattern={memory}
                                        label={idx}
                                        isActive={false}
                                    />
                                ))}
                            </div>
                        </div>
                    )}
                </div>

                <div className="content-card">
                    <h3>Network Architecture</h3>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '2rem' }}>
                        <div>
                            <h4 style={{ fontSize: '0.95rem', marginBottom: '1rem' }}>Network Specifications</h4>
                            <table style={{ width: '100%', fontSize: '0.9rem' }}>
                                <tbody>
                                    <tr>
                                        <td style={{ padding: '0.5rem', borderBottom: '1px solid var(--border-color)', fontWeight: '600' }}>Neurons</td>
                                        <td style={{ padding: '0.5rem', borderBottom: '1px solid var(--border-color)' }}>784 (28 × 28)</td>
                                    </tr>
                                    <tr>
                                        <td style={{ padding: '0.5rem', borderBottom: '1px solid var(--border-color)', fontWeight: '600' }}>Connections</td>
                                        <td style={{ padding: '0.5rem', borderBottom: '1px solid var(--border-color)' }}>
                                            <LaTeX math="784 \times 783 / 2 = 307{,}056" />
                                        </td>
                                    </tr>
                                    <tr>
                                        <td style={{ padding: '0.5rem', borderBottom: '1px solid var(--border-color)', fontWeight: '600' }}>Weight Matrix</td>
                                        <td style={{ padding: '0.5rem', borderBottom: '1px solid var(--border-color)' }}>
                                            784 × 784 (symmetric)
                                        </td>
                                    </tr>
                                    <tr>
                                        <td style={{ padding: '0.5rem', borderBottom: '1px solid var(--border-color)', fontWeight: '600' }}>Patterns Stored</td>
                                        <td style={{ padding: '0.5rem', borderBottom: '1px solid var(--border-color)' }}>10 (digits 0-9)</td>
                                    </tr>
                                    <tr>
                                        <td style={{ padding: '0.5rem', fontWeight: '600' }}>Topology</td>
                                        <td style={{ padding: '0.5rem' }}>Fully connected, recurrent</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                        <div>
                            <h4 style={{ fontSize: '0.95rem', marginBottom: '1rem' }}>Structure</h4>
                            <div style={{
                                background: 'var(--bg-secondary)',
                                padding: '1.5rem',
                                borderRadius: 'var(--radius-md)',
                                textAlign: 'center'
                            }}>
                                <div style={{ marginBottom: '1rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                                    Single-layer recurrent architecture
                                </div>
                                <div style={{
                                    border: '2px dashed var(--primary)',
                                    borderRadius: 'var(--radius-md)',
                                    padding: '1.5rem',
                                    background: 'var(--bg-primary)'
                                }}>
                                    <div style={{ fontSize: '1.1rem', fontWeight: '600', marginBottom: '0.5rem' }}>
                                        784 Binary Neurons
                                    </div>
                                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                                        <LaTeX math="s_i \in \{-1, +1\}" />
                                    </div>
                                    <div style={{ margin: '1rem 0', color: 'var(--text-tertiary)' }}>⟷</div>
                                    <div style={{ fontSize: '0.85rem' }}>
                                        All-to-all connections (no self-connections)
                                    </div>
                                    <div style={{ fontSize: '0.85rem', marginTop: '0.5rem' }}>
                                        <LaTeX math="w_{ij} = w_{ji}, \quad w_{ii} = 0" />
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                        This network has <strong>no hidden layers</strong> — all neurons are both input and output units.
                        The recurrent connections allow the network to settle into stable states (attractors) that
                        correspond to stored patterns.
                    </p>
                </div>
            </section>

            {/* Pattern Retrieval Section */}
            <section id="retrieval" className="demo-section">
                <h2>Pattern Retrieval</h2>

                <div className="content-card">
                    <h3>Draw a Query Pattern</h3>
                    <p style={{ marginBottom: '1.5rem', fontSize: '0.95rem' }}>
                        Draw a digit (0-9) on the canvas below. The network will retrieve the closest
                        stored memory by iteratively minimizing its energy function. Try drawing a noisy
                        or incomplete digit to observe the network's error-correction capabilities.
                    </p>

                    <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '2rem', alignItems: 'start' }}>
                        <DrawingCanvas onPatternChange={handlePatternChange} />
                        <div>
                            <button
                                onClick={handleSubmit}
                                className="control-btn"
                                disabled={queryPattern.length === 0}
                                style={{
                                    padding: '0.75rem 2rem',
                                    fontSize: '1rem',
                                    fontWeight: '600',
                                    backgroundColor: queryPattern.length > 0 ? 'var(--primary)' : 'var(--bg-tertiary)',
                                    color: queryPattern.length > 0 ? 'white' : 'var(--text-tertiary)',
                                    border: 'none',
                                    marginBottom: '1rem'
                                }}
                            >
                                Start Retrieval
                            </button>
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                                The network will update neuron states asynchronously according to:
                            </p>
                            <LaTeX
                                block
                                math="s_i(t+1) = \text{sign}\left(\sum_j w_{ij} s_j(t)\right)"
                            />
                        </div>
                    </div>
                </div>

                {hasStarted && (
                    <>
                        <div id="visualization" style={{ scrollMarginTop: '100px' }}>
                            <NetworkVisualizer
                                memories={memories}
                                currentState={currentState}
                                queryPattern={queryPattern}
                                energy={energyHistory[energyHistory.length - 1] || 0}
                                step={currentStep}
                                isConverged={isConverged}
                                matchedDigit={matchedDigit}
                                energyHistory={energyHistory}
                                evolutionHistory={getEvolutionHistory()}
                            />
                        </div>

                        <AnimationControls
                            isPlaying={isPlaying}
                            currentStep={currentStep}
                            totalSteps={network.history.length - 1}
                            speed={speed}
                            canStepBackward={currentStep > 0}
                            canStepForward={!isConverged}
                            onPlay={handlePlay}
                            onPause={handlePause}
                            onStepForward={handleStepForward}
                            onStepBackward={handleStepBackward}
                            onReset={handleReset}
                            onSpeedChange={handleSpeedChange}
                            onRunToEnd={handleRunToEnd}
                        />
                    </>
                )}
            </section>

            {/* Theory Section */}
            <section id="theory" className="demo-section">
                <h2>Theory and Background</h2>

                <div className="content-card">
                    <h3>Architecture</h3>
                    <p>
                        A Hopfield network consists of <LaTeX math="N" /> binary neurons in a fully connected
                        recurrent architecture. Key properties:
                    </p>
                    <ul>
                        <li><strong>Binary Neurons:</strong> Each unit has state <LaTeX math="s_i \in \{-1, +1\}" /></li>
                        <li><strong>Symmetric Weights:</strong> <LaTeX math="w_{ij} = w_{ji}" /> for all <LaTeX math="i, j" /></li>
                        <li><strong>No Self-Connections:</strong> <LaTeX math="w_{ii} = 0" /> for all <LaTeX math="i" /></li>
                        <li><strong>Asynchronous Updates:</strong> Neurons update one at a time in random order</li>
                    </ul>
                </div>

                <div className="content-card">
                    <h3>Energy Function</h3>
                    <p>
                        The network defines a global energy function (Lyapunov function):
                    </p>
                    <LaTeX
                        block
                        math="E = -\frac{1}{2} \sum_{i,j} w_{ij} s_i s_j"
                    />
                    <p>
                        This energy <strong>never increases</strong> during asynchronous updates, guaranteeing
                        convergence to a local minimum (fixed point). Stored patterns correspond to local
                        energy minima.
                    </p>
                </div>

                <div className="content-card">
                    <h3>Update Rule</h3>
                    <p>
                        Each neuron updates its state based on the weighted sum of inputs from all other neurons:
                    </p>
                    <LaTeX
                        block
                        math="s_i(t+1) = \text{sign}\left(\sum_{j=1}^{N} w_{ij} s_j(t)\right)"
                    />
                    <p>
                        where <LaTeX math="\text{sign}(x) = +1" /> if <LaTeX math="x \geq 0" />,
                        and <LaTeX math="\text{sign}(x) = -1" /> otherwise. This guarantees that energy
                        decreases or remains constant with each update.
                    </p>
                </div>

                <div className="content-card">
                    <h3>Hebbian Learning</h3>
                    <p>
                        To store a set of patterns <LaTeX math="\{x^1, x^2, \ldots, x^P\}" />, the weight
                        matrix is computed using Hebb's rule: "neurons that fire together, wire together."
                    </p>
                    <LaTeX
                        block
                        math="w_{ij} = \frac{1}{P} \sum_{p=1}^{P} x_i^p x_j^p \quad \text{for} \quad i \neq j"
                    />
                    <p>
                        This learning rule is:
                    </p>
                    <ul>
                        <li><strong>Local:</strong> Only requires pre- and post-synaptic neuron states</li>
                        <li><strong>Unsupervised:</strong> No error signal or labels needed</li>
                        <li><strong>One-shot:</strong> Weights computed in single pass</li>
                        <li><strong>Biologically plausible:</strong> Resembles synaptic plasticity</li>
                    </ul>
                </div>

                <div className="content-card">
                    <h3>Storage Capacity</h3>
                    <p>
                        A critical limitation is the maximum number of patterns that can be reliably stored.
                        For a network with <LaTeX math="N" /> neurons:
                    </p>
                    <ul>
                        <li>
                            <strong>Theoretical capacity:</strong> Approximately <LaTeX math="C \approx 0.138N" /> patterns
                            can be stored with high retrieval accuracy
                        </li>
                        <li>
                            <strong>Beyond capacity:</strong> Spurious states (unintended attractors) emerge when
                            too many patterns are stored
                        </li>
                        <li>
                            <strong>Pattern correlation:</strong> Highly correlated patterns reduce effective capacity
                        </li>
                    </ul>
                    <p>
                        For this demo with <LaTeX math="N = 784" /> neurons, the capacity is roughly 108 patterns.
                        We store only 10 digits, well within the safe range.
                    </p>
                </div>

                <div className="content-card">
                    <h3>Properties</h3>
                    <h4>Advantages</h4>
                    <ul>
                        <li><strong>Robust to noise:</strong> Can retrieve complete patterns from corrupted or partial inputs</li>
                        <li><strong>Content-addressable:</strong> Memory accessed by similarity, not address</li>
                        <li><strong>Guaranteed convergence:</strong> Energy minimization ensures fixed point</li>
                        <li><strong>Simple learning rule:</strong> Hebbian learning is biologically plausible</li>
                        <li><strong>Distributed representation:</strong> Information stored across all connections</li>
                    </ul>

                    <h4>Limitations</h4>
                    <ul>
                        <li><strong>Limited capacity:</strong> Only <LaTeX math="0.138N" /> patterns can be reliably stored</li>
                        <li><strong>Spurious states:</strong> Network may converge to unintended patterns</li>
                        <li><strong>Quadratic complexity:</strong> <LaTeX math="O(N^2)" /> connections required</li>
                        <li><strong>Symmetric weights:</strong> Limits biological realism (real synapses are directional)</li>
                        <li><strong>Binary states:</strong> Limited expressiveness compared to continuous activations</li>
                    </ul>
                </div>

                <div className="content-card">
                    <h3>Applications</h3>
                    <ul>
                        <li><strong>Pattern completion:</strong> Restore damaged images, audio, or text</li>
                        <li><strong>Error correction:</strong> Retrieve clean data from noisy observations</li>
                        <li><strong>Associative memory:</strong> Content-addressable storage systems</li>
                        <li><strong>Optimization:</strong> Solve combinatorial problems (TSP, graph coloring)</li>
                        <li><strong>Modeling memory:</strong> Understanding biological memory and recall</li>
                    </ul>
                </div>
            </section>
        </div>
    );
}

// Memory pattern display component
const MemoryPattern: React.FC<{ pattern: number[]; label: number; isActive: boolean }> = ({
    pattern,
    label,
    isActive
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const size = Math.sqrt(pattern.length);
        canvas.width = size * 2;
        canvas.height = size * 2;

        // Create ImageData
        const imageData = ctx.createImageData(size, size);
        for (let i = 0; i < pattern.length; i++) {
            const pixelIndex = i * 4;
            const value = pattern[i] === 1 ? 0 : 255;
            imageData.data[pixelIndex] = value;
            imageData.data[pixelIndex + 1] = value;
            imageData.data[pixelIndex + 2] = value;
            imageData.data[pixelIndex + 3] = 255;
        }

        // Draw scaled
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = size;
        tempCanvas.height = size;
        const tempCtx = tempCanvas.getContext('2d');
        if (!tempCtx) return;

        tempCtx.putImageData(imageData, 0, 0);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(tempCanvas, 0, 0, size, size, 0, 0, size * 2, size * 2);
    }, [pattern]);

    return (
        <div className={`pattern-item ${isActive ? 'active' : ''}`}>
            <canvas
                ref={canvasRef}
                style={{
                    border: '1px solid var(--border-color)',
                    borderRadius: '4px',
                    backgroundColor: 'white'
                }}
            />
            <span style={{ fontSize: '0.75rem', fontWeight: '600', marginTop: '0.25rem' }}>{label}</span>
        </div>
    );
};
