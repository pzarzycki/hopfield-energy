import { useState, useEffect, useRef } from 'react';

import DrawingCanvas from '../components/DrawingCanvas';
import AnimationControls from '../components/AnimationControls';

import LaTeX from '../components/LaTeX';
import { HopfieldNetwork } from '../utils/hopfield-network';
import { getDatasetSamples, grayscaleToBipolar, patternToImageData } from '../utils/mnist-data';

export default function HopfieldNetwork_Page() {

    const [network] = useState(() => new HopfieldNetwork(784));
    const [memories, setMemories] = useState<number[][]>([]);
    const [queryPattern, setQueryPattern] = useState<number[]>([]);
    const [currentStep, setCurrentStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [speed, setSpeed] = useState(5);
    const [isConverged, setIsConverged] = useState(false);
    const [matchedDigit, setMatchedDigit] = useState(-1);
    const [hasStarted, setHasStarted] = useState(false);
    const animationRef = useRef<number | null>(null);
    const lastUpdateRef = useRef<number>(0);



    useEffect(() => {
        const samples = getDatasetSamples('mnist');
        const grayscalePatterns = samples.map(d => d.pattern);
        // Binarize grayscale to bipolar for classical Hopfield network training
        const bipolarPatterns = grayscalePatterns.map(p => grayscaleToBipolar(p));
        network.train(bipolarPatterns);
        // Store original grayscale for display
        setMemories(grayscalePatterns);
        setHasStarted(false);
        setQueryPattern([]);
        setCurrentStep(0);
        setIsConverged(false);
        setMatchedDigit(-1);
    }, [network]);

    const handlePatternChange = (pattern: number[]) => setQueryPattern(pattern);

    const handleSubmit = () => {
        if (queryPattern.length === 0) return;
        network.setState(queryPattern);
        setCurrentStep(0);
        setIsConverged(false);
        setMatchedDigit(network.getMatchedPattern());
        setHasStarted(true);
        setIsPlaying(false);
    };

    useEffect(() => {
        if (!isPlaying) {
            if (animationRef.current) { cancelAnimationFrame(animationRef.current); animationRef.current = null; }
            return;
        }
        const animate = (timestamp: number) => {
            if (!lastUpdateRef.current) lastUpdateRef.current = timestamp;
            const elapsed = timestamp - lastUpdateRef.current;
            if (elapsed >= 1000 / speed) {
                if (!network.hasConverged()) {
                    const state = network.step();
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
        return () => { if (animationRef.current) cancelAnimationFrame(animationRef.current); };
    }, [isPlaying, speed, network]);

    const handlePlay = () => { if (!hasStarted || isConverged) return; setIsPlaying(true); lastUpdateRef.current = 0; };
    const handlePause = () => setIsPlaying(false);
    const handleStepForward = () => {
        if (network.hasConverged()) { setIsConverged(true); return; }
        const state = network.step();
        setCurrentStep(state.step);
        setMatchedDigit(network.getMatchedPattern());
        if (network.hasConverged()) setIsConverged(true);
    };
    const handleStepBackward = () => {
        if (currentStep <= 0) return;
        network.resetToStep(currentStep - 1);
        setCurrentStep(currentStep - 1);
        setIsConverged(false);
        setMatchedDigit(network.getMatchedPattern());
    };
    const handleReset = () => {
        if (!hasStarted) return;
        network.resetToStep(0);
        setCurrentStep(0);
        setIsConverged(false);
        setMatchedDigit(network.getMatchedPattern());
        setIsPlaying(false);
    };
    const handleRunToEnd = () => {
        setIsPlaying(false);
        let iterations = 0;
        while (!network.hasConverged() && iterations < 100) {
            const state = network.step();
            setCurrentStep(state.step);
            iterations++;
        }
        setIsConverged(true);
        setMatchedDigit(network.getMatchedPattern());
    };



    const getLabel = (idx: number) => `${idx}`;

    return (
        <div>


            <header className="page-header" id="overview" style={{ paddingBottom: '0' }}>
                <h1 className="page-title" style={{ marginBottom: '1rem' }}>Hopfield Network</h1>
            </header>

            <section id="demo" className="demo-section" style={{ borderTop: 'none', marginTop: 0, paddingTop: 0 }}>

                <div className="content-card">
                    <h3 style={{ marginTop: 0 }}>Stored Memories</h3>
                    <p>
                        The network has been trained on 10 binarized MNIST digit patterns
                        using Hebbian learning.
                    </p>
                    {memories.length > 0 && (
                        <div className="pattern-grid">
                            {memories.map((memory, idx) => (
                                <MemoryPattern key={idx} pattern={memory} label={getLabel(idx)} isActive={idx === matchedDigit && isConverged} />
                            ))}
                        </div>
                    )}
                </div>

                <div className="content-card">
                    <h3>Draw a Query Pattern</h3>
                    <p>Draw a pattern on the canvas. The network will retrieve the closest stored memory.</p>
                    <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '2rem', alignItems: 'start' }}>
                        <DrawingCanvas onPatternChange={handlePatternChange} />
                        <div>
                            <button onClick={handleSubmit} className="control-btn primary" disabled={queryPattern.length === 0}
                                style={{ padding: '0.75rem 2rem', fontSize: '1rem', marginBottom: '1rem' }}>
                                Start Retrieval
                            </button>
                            <p style={{ fontSize: '0.85rem' }}>Update rule:</p>
                            <LaTeX block math="s_i(t+1) = \text{sign}\left(\sum_j w_{ij} s_j(t)\right)" />
                        </div>
                    </div>
                </div>

                {hasStarted && (
                    <>

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
                            onSpeedChange={setSpeed}
                            onRunToEnd={handleRunToEnd}
                        />
                    </>
                )}
            </section>
        </div>
    );
}

// Memory pattern display
const MemoryPattern: React.FC<{ pattern: number[]; label: string; isActive: boolean }> = ({
    pattern, label, isActive
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
        const imageData = patternToImageData(pattern);
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
            <canvas ref={canvasRef} style={{ width: '56px', height: '56px' }} />
            <span style={{ fontSize: '0.7rem', fontWeight: '600', marginTop: '0.25rem', color: 'var(--text-secondary)' }}>{label}</span>
        </div>
    );
};
