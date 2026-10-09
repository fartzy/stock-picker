import { useEffect, useRef } from "react";
import Overnight from "./Overnight";

export default function OvernightQuickLook({
  ticker,
  onClose,
  onOpenFull,
}: {
  ticker: string;
  onClose: () => void;
  onOpenFull: () => void;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    dialog.showModal();
    dialog.querySelector<HTMLInputElement>('input[name="assumed-close"]')?.focus();
    return () => dialog.close();
  }, []);

  return (
    <dialog
      ref={dialogRef}
      className="overnight-quick-dialog"
      aria-label={`${ticker} next-open estimate`}
      onCancel={(event) => { event.preventDefault(); onClose(); }}
      onClick={(event) => { if (event.target === event.currentTarget) onClose(); }}
    >
      <div className="overnight-quick-inner">
        <div className="overnight-quick-head">
          <div><span>Next-open estimate</span><h2>{ticker}</h2></div>
          <button type="button" className="overnight-quick-close" aria-label="Close next-open estimate" onClick={onClose}>×</button>
        </div>
        <Overnight initialTicker={ticker} compact />
        <button className="overnight-quick-full" type="button" onClick={onOpenFull}>Open full Overnight tab <span aria-hidden="true">↗</span></button>
      </div>
    </dialog>
  );
}
