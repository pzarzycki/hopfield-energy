import type { ReactNode } from "react";
import { X } from "lucide-react";

interface DatasetDialogProps {
  title: string;
  summary: string;
  facts?: string[];
  onClose: () => void;
  children: ReactNode;
}

export function DatasetDialog({ title, summary, facts = [], onClose, children }: DatasetDialogProps) {
  return (
    <div className="modal-backdrop" onClick={onClose} role="presentation">
      <section
        className="modal-dialog modal-dialog--dataset"
        role="dialog"
        aria-modal="true"
        aria-labelledby="dataset-dialog-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <div>
            <h2 id="dataset-dialog-title">{title}</h2>
            <p>{summary}</p>
            {facts.length > 0 ? (
              <div className="dataset-facts" aria-label="Dataset statistics">
                {facts.map((fact) => (
                  <span key={fact}>{fact}</span>
                ))}
              </div>
            ) : null}
          </div>
          <button type="button" className="modal-close-btn" onClick={onClose} aria-label="Close dataset dialog">
            <X size={16} />
          </button>
        </div>
        <div className="modal-body">{children}</div>
      </section>
    </div>
  );
}
