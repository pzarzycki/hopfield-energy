import React from 'react';

interface AnimationControlsProps {
    isPlaying: boolean;
    currentStep: number;
    totalSteps: number;
    speed: number;
    canStepBackward: boolean;
    canStepForward: boolean;
    onPlay: () => void;
    onPause: () => void;
    onStepForward: () => void;
    onStepBackward: () => void;
    onReset: () => void;
    onSpeedChange: (speed: number) => void;
    onRunToEnd: () => void;
}

const AnimationControls: React.FC<AnimationControlsProps> = ({
    isPlaying,
    currentStep,
    totalSteps,
    speed,
    canStepBackward,
    canStepForward,
    onPlay,
    onPause,
    onStepForward,
    onStepBackward,
    onReset,
    onSpeedChange,
    onRunToEnd
}) => {
    return (
        <div className="demo-card">
            <h3>Animation Controls</h3>

            <div className="controls-bar">
                <div className="control-buttons">
                    <button
                        className="control-btn"
                        onClick={onReset}
                        disabled={currentStep === 0}
                        title="Reset to initial state"
                    >
                        ⏮ Reset
                    </button>

                    <button
                        className="control-btn"
                        onClick={onStepBackward}
                        disabled={!canStepBackward || isPlaying}
                        title="Step backward"
                    >
                        ◀ Back
                    </button>

                    {isPlaying ? (
                        <button
                            className="control-btn"
                            onClick={onPause}
                            title="Pause"
                            style={{ backgroundColor: 'var(--primary)', color: 'white' }}
                        >
                            ⏸ Pause
                        </button>
                    ) : (
                        <button
                            className="control-btn"
                            onClick={onPlay}
                            disabled={!canStepForward}
                            title="Play"
                        >
                            ▶ Play
                        </button>
                    )}

                    <button
                        className="control-btn"
                        onClick={onStepForward}
                        disabled={!canStepForward || isPlaying}
                        title="Step forward"
                    >
                        Forward ▶
                    </button>

                    <button
                        className="control-btn"
                        onClick={onRunToEnd}
                        disabled={!canStepForward || isPlaying}
                        title="Run to convergence"
                    >
                        ⏭ Run to End
                    </button>
                </div>

                <div className="speed-control">
                    <label htmlFor="speed-slider" style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                        Speed:
                    </label>
                    <input
                        id="speed-slider"
                        type="range"
                        min="1"
                        max="10"
                        value={speed}
                        onChange={(e) => onSpeedChange(parseInt(e.target.value))}
                        className="speed-slider"
                        disabled={isPlaying}
                    />
                    <span style={{ fontSize: '0.9rem', fontWeight: '600', minWidth: '3rem' }}>
                        {speed} steps/s
                    </span>
                </div>
            </div>

            {totalSteps > 0 && (
                <div style={{ marginTop: '1rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                    Step {currentStep} / {totalSteps}
                </div>
            )}
        </div>
    );
};

export default AnimationControls;
