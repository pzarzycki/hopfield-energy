import React, { useRef, useEffect } from 'react';
import { patternToImageData } from '../utils/mnist-data';

interface NetworkVisualizerProps {
    memories: number[][];
    currentState: number[];
    queryPattern: number[];
    energy: number;
    step: number;
    isConverged: boolean;
    matchedDigit: number;
    energyHistory: number[];
    evolutionHistory: { step: number; state: number[] }[];
}

const NetworkVisualizer: React.FC<NetworkVisualizerProps> = ({
    memories,
    currentState,
    queryPattern,
    energy,
    step,
    isConverged,
    matchedDigit,
    energyHistory,
    evolutionHistory
}) => {
    const queryCanvasRef = useRef<HTMLCanvasElement>(null);
    const currentCanvasRef = useRef<HTMLCanvasElement>(null);
    const targetCanvasRef = useRef<HTMLCanvasElement>(null);
    const energyCanvasRef = useRef<HTMLCanvasElement>(null);

    // Render pattern to canvas
    const renderPattern = (canvas: HTMLCanvasElement | null, pattern: number[], scale: number = 2) => {
        if (!canvas || pattern.length === 0) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const size = Math.sqrt(pattern.length);
        canvas.width = size * scale;
        canvas.height = size * scale;

        const imageData = patternToImageData(pattern);
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = size;
        tempCanvas.height = size;
        const tempCtx = tempCanvas.getContext('2d');
        if (!tempCtx) return;

        tempCtx.putImageData(imageData, 0, 0);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(tempCanvas, 0, 0, size, size, 0, 0, size * scale, size * scale);
    };

    // Render energy graph
    const renderEnergyGraph = () => {
        const canvas = energyCanvasRef.current;
        if (!canvas || energyHistory.length === 0) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.fillStyle = '#ffffff'; // Ensure white background for contrast
        ctx.fillRect(0, 0, width, height);

        // Find min and max energy for scaling
        const minEnergy = Math.min(...energyHistory);
        const maxEnergy = Math.max(...energyHistory);
        const energyRange = maxEnergy - minEnergy || 1;

        // Add padding to range
        const paddedMin = minEnergy - (energyRange * 0.1);
        const paddedMax = maxEnergy + (energyRange * 0.1);
        const paddedRange = paddedMax - paddedMin;

        // Draw grid
        ctx.strokeStyle = '#e5e7eb';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        for (let i = 0; i <= 4; i++) {
            const y = height - ((i / 4) * (height - 20) + 10);
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
        ctx.setLineDash([]);

        // Draw energy curve
        ctx.strokeStyle = '#2563eb'; // Primary blue
        ctx.lineWidth = 2;
        ctx.beginPath();

        energyHistory.forEach((e, i) => {
            const x = (width * i) / (energyHistory.length - 1 || 1);
            // Invert y-axis: higher energy -> smaller y (top), lower energy -> larger y (bottom)
            // But standard graph: min at bottom, max at top.
            // Canvas y=0 is top. So max energy should be at y=small, min energy at y=large
            const normalizedEnergy = (e - paddedMin) / paddedRange;
            const y = height - (normalizedEnergy * (height - 20) + 10);

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });

        ctx.stroke();

        // Draw current point
        if (energyHistory.length > 0) {
            const currentE = energyHistory[energyHistory.length - 1];
            const normalizedE = (currentE - paddedMin) / paddedRange;
            const lastX = width; // Right edge
            const lastY = height - (normalizedE * (height - 20) + 10);

            ctx.fillStyle = '#2563eb';
            ctx.beginPath();
            ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
            ctx.fill();
        }
    };

    // Render evolution filmstrip
    const renderFilmstrip = () => {
        return (
            <div style={{ display: 'flex', gap: '1rem', overflowX: 'auto', padding: '1rem 0' }}>
                {evolutionHistory.length === 0 && <p style={{ color: 'var(--text-secondary)' }}>Waiting for evolution...</p>}
                {evolutionHistory.length > 0 && (
                    <>
                        <div style={{ flex: '0 0 auto', textAlign: 'center' }}>
                            <MemoryPattern
                                pattern={queryPattern}
                                label={0}
                                isActive={false}
                            />
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                                Start
                            </div>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', color: '#9ca3af' }}>→</div>
                    </>
                )}
                {evolutionHistory.map((snapshot, idx) => (
                    <div key={idx} style={{ flex: '0 0 auto', textAlign: 'center', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <div>
                            <MemoryPattern
                                pattern={snapshot.state}
                                label={snapshot.step}
                                isActive={false}
                            />
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                                Step {snapshot.step}
                            </div>
                        </div>
                        {idx < evolutionHistory.length - 1 && (
                            <div style={{ color: '#9ca3af' }}>→</div>
                        )}
                    </div>
                ))}
            </div>
        );
    };

    // Update visualizations
    useEffect(() => {
        renderPattern(queryCanvasRef.current, queryPattern, 3);
    }, [queryPattern]);

    useEffect(() => {
        renderPattern(currentCanvasRef.current, currentState, 3);
    }, [currentState]);

    useEffect(() => {
        if (matchedDigit >= 0 && matchedDigit < memories.length) {
            renderPattern(targetCanvasRef.current, memories[matchedDigit], 3);
        }
    }, [matchedDigit, memories]);

    useEffect(() => {
        renderEnergyGraph();
    }, [energyHistory]);

    return (
        <div className="demo-card">
            <h3>Network Visualization</h3>

            {/* Stored Memories */}
            <div style={{ marginBottom: '2rem' }}>
                <h4 style={{ fontSize: '0.95rem', marginBottom: '1rem', color: 'var(--text-secondary)' }}>
                    Stored Memories
                </h4>
                <div className="pattern-grid">
                    {memories.map((memory, idx) => (
                        <MemoryPattern
                            key={idx}
                            pattern={memory}
                            label={idx}
                            isActive={idx === matchedDigit && isConverged}
                        />
                    ))}
                </div>
            </div>

            {/* Retrieval Process */}
            {queryPattern.length > 0 && (
                <div style={{ marginBottom: '2rem' }}>
                    <h4 style={{ fontSize: '0.95rem', marginBottom: '1rem', color: 'var(--text-secondary)' }}>
                        Retrieval Process
                    </h4>
                    <div className="visualization-row">
                        <div className="visualization-stage">
                            <canvas ref={queryCanvasRef} style={{ border: '1px solid var(--border-color)', borderRadius: '4px' }} />
                            <span style={{ fontSize: '0.85rem', color: 'var(--text-tertiary)' }}>Query</span>
                        </div>
                        <div style={{ fontSize: '1.5rem', color: 'var(--text-tertiary)' }}>→</div>
                        <div className="visualization-stage">
                            <canvas ref={currentCanvasRef} style={{ border: '1px solid var(--border-color)', borderRadius: '4px' }} />
                            <span style={{ fontSize: '0.85rem', color: 'var(--text-tertiary)' }}>Current State</span>
                        </div>
                        <div style={{ fontSize: '1.5rem', color: 'var(--text-tertiary)' }}>→</div>
                        <div className="visualization-stage">
                            <canvas ref={targetCanvasRef} style={{ border: '1px solid var(--border-color)', borderRadius: '4px' }} />
                            <span style={{ fontSize: '0.85rem', color: 'var(--text-tertiary)' }}>Target</span>
                        </div>
                    </div>
                </div>
            )}

            {/* Pattern Evolution Filmstrip */}
            {queryPattern.length > 0 && (
                <div style={{ marginBottom: '2rem' }}>
                    <h4 style={{ fontSize: '0.95rem', marginBottom: '1rem', color: 'var(--text-secondary)' }}>
                        Pattern Evolution
                    </h4>
                    {renderFilmstrip()}
                </div>
            )}

            {/* Energy Graph */}
            {energyHistory.length > 0 && (
                <div style={{ marginBottom: '2rem' }}>
                    <h4 style={{ fontSize: '0.95rem', marginBottom: '1rem', color: 'var(--text-secondary)' }}>
                        Energy Over Time
                    </h4>
                    <canvas
                        ref={energyCanvasRef}
                        width={600}
                        height={200}
                        style={{ width: '100%', height: 'auto', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-md)' }}
                    />
                </div>
            )}

            {/* Statistics */}
            {step >= 0 && (
                <div className="stats-grid">
                    <div className="stat-item">
                        <span className="stat-label">Step</span>
                        <span className="stat-value">{step}</span>
                    </div>
                    <div className="stat-item">
                        <span className="stat-label">Energy</span>
                        <span className="stat-value">{energy.toFixed(2)}</span>
                    </div>
                    <div className="stat-item">
                        <span className="stat-label">Status</span>
                        <span className="stat-value" style={{ color: isConverged ? 'var(--primary)' : 'var(--text-secondary)' }}>
                            {isConverged ? 'Converged ✓' : 'Running...'}
                        </span>
                    </div>
                    {isConverged && matchedDigit >= 0 && (
                        <div className="stat-item">
                            <span className="stat-label">Matched Digit</span>
                            <span className="stat-value" style={{ color: 'var(--primary)' }}>{matchedDigit}</span>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

// Memory pattern component
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
        canvas.width = size * 2; // High DPI
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
            <canvas
                ref={canvasRef}
                style={{
                    border: '1px solid var(--border-color)',
                    borderRadius: '4px',
                    backgroundColor: 'white',
                    width: '56px',
                    height: '56px'
                }}
            />
            {label >= 0 && (
                <span style={{ fontSize: '0.75rem', fontWeight: '600', marginTop: '0.25rem' }}>{label}</span>
            )}
        </div>
    );
};

export default NetworkVisualizer;
