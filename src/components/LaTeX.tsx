import React from 'react';
import 'katex/dist/katex.min.css';
import katex from 'katex';

interface LaTeXProps {
    math: string;
    block?: boolean;
}

const LaTeX: React.FC<LaTeXProps> = ({ math, block = false }) => {
    const html = React.useMemo(() => {
        try {
            return katex.renderToString(math, {
                displayMode: block,
                throwOnError: false,
                strict: false,
            });
        } catch (error) {
            console.error('LaTeX rendering error:', error);
            return math;
        }
    }, [math, block]);

    if (block) {
        return (
            <div
                className="latex-block"
                dangerouslySetInnerHTML={{ __html: html }}
                style={{
                    margin: '1rem 0',
                    padding: '1rem',
                    background: 'var(--bg-tertiary)',
                    borderRadius: 'var(--radius-md)',
                    overflow: 'auto',
                    textAlign: 'center',
                }}
            />
        );
    }

    return (
        <span
            className="latex-inline"
            dangerouslySetInnerHTML={{ __html: html }}
        />
    );
};

export default LaTeX;
