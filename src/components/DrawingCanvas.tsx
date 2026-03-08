import React, { useRef, useEffect, useState } from 'react';

interface DrawingCanvasProps {
    onPatternChange: (pattern: number[]) => void;
    size?: number;
}

const DrawingCanvas: React.FC<DrawingCanvasProps> = ({ onPatternChange, size = 280 }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const patternSize = 28; // MNIST size

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Initialize with white background
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, size, size);
    }, [size]);

    const getMousePos = (e: React.MouseEvent<HTMLCanvasElement>): { x: number; y: number } => {
        const canvas = canvasRef.current;
        if (!canvas) return { x: 0, y: 0 };

        const rect = canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    };

    const getTouchPos = (e: React.TouchEvent<HTMLCanvasElement>): { x: number; y: number } => {
        const canvas = canvasRef.current;
        if (!canvas) return { x: 0, y: 0 };

        const rect = canvas.getBoundingClientRect();
        const touch = e.touches[0];
        return {
            x: touch.clientX - rect.left,
            y: touch.clientY - rect.top
        };
    };

    const draw = (x: number, y: number) => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        ctx.fillStyle = 'black';
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.fill();
    };

    const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
        setIsDrawing(true);
        const pos = getMousePos(e);
        draw(pos.x, pos.y);
    };

    const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
        if (!isDrawing) return;
        const pos = getMousePos(e);
        draw(pos.x, pos.y);
    };

    const handleMouseUp = () => {
        setIsDrawing(false);
        extractPattern();
    };

    const handleTouchStart = (e: React.TouchEvent<HTMLCanvasElement>) => {
        e.preventDefault();
        setIsDrawing(true);
        const pos = getTouchPos(e);
        draw(pos.x, pos.y);
    };

    const handleTouchMove = (e: React.TouchEvent<HTMLCanvasElement>) => {
        e.preventDefault();
        if (!isDrawing) return;
        const pos = getTouchPos(e);
        draw(pos.x, pos.y);
    };

    const handleTouchEnd = (e: React.TouchEvent<HTMLCanvasElement>) => {
        e.preventDefault();
        setIsDrawing(false);
        extractPattern();
    };

    const extractPattern = () => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Get the full canvas image data
        const imageData = ctx.getImageData(0, 0, size, size);

        // Downsample to 28x28
        const pattern: number[] = [];
        const scale = size / patternSize;

        for (let y = 0; y < patternSize; y++) {
            for (let x = 0; x < patternSize; x++) {
                // Sample from the center of each cell
                let sum = 0;
                let count = 0;

                const startX = Math.floor(x * scale);
                const startY = Math.floor(y * scale);
                const endX = Math.floor((x + 1) * scale);
                const endY = Math.floor((y + 1) * scale);

                for (let py = startY; py < endY; py++) {
                    for (let px = startX; px < endX; px++) {
                        const idx = (py * size + px) * 4;
                        const brightness = (imageData.data[idx] + imageData.data[idx + 1] + imageData.data[idx + 2]) / 3;
                        sum += brightness;
                        count++;
                    }
                }

                const avgBrightness = sum / count;
                // Convert to binary: darker pixels (drawn) = 1, lighter (background) = -1
                pattern.push(avgBrightness < 128 ? 1 : -1);
            }
        }

        onPatternChange(pattern);
    };

    const handleClear = () => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, size, size);

        // Notify with empty pattern
        const emptyPattern = Array(patternSize * patternSize).fill(-1);
        onPatternChange(emptyPattern);
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <canvas
                ref={canvasRef}
                width={size}
                height={size}
                className="drawing-canvas"
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                onTouchStart={handleTouchStart}
                onTouchMove={handleTouchMove}
                onTouchEnd={handleTouchEnd}
                style={{
                    border: '2px solid var(--border-color)',
                    borderRadius: 'var(--radius-md)',
                    cursor: 'crosshair',
                    touchAction: 'none'
                }}
            />
            <button
                onClick={handleClear}
                className="control-btn"
                style={{ width: 'fit-content' }}
            >
                Clear Canvas
            </button>
        </div>
    );
};

export default DrawingCanvas;
