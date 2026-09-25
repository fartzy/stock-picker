import type { ReactNode } from "react";

export default function TogglePill({
  on,
  onToggle,
  children,
  extra,
}: {
  on: boolean;
  onToggle: () => void;
  children: ReactNode;
  extra?: ReactNode;
}) {
  return (
    <div className={`toggle-pill${on ? " on" : ""}`}>
      <button type="button" className={`toggle-pill-btn${on ? " active" : ""}`} aria-pressed={on} onClick={onToggle}>
        {children}
      </button>
      {extra}
    </div>
  );
}
