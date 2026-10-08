import type { ReactNode, Ref } from "react";

/** Shared presentation for feature groups; callers supply only their own metadata. */
export function FeatureCategory({
  title,
  summaryExtra,
  children,
  categoryRef,
  className = "",
}: {
  title: string;
  summaryExtra?: ReactNode;
  children: ReactNode;
  categoryRef?: Ref<HTMLDetailsElement>;
  className?: string;
}) {
  return (
    <details className={`view-card registry-category ${className}`.trim()} ref={categoryRef}>
      <summary>
        <strong style={{ color: "var(--accent)" }}>{title}</strong>
        {summaryExtra}
      </summary>
      <div className="view-features">{children}</div>
    </details>
  );
}

/** A disclosure row shared by the trainable Registry and read-only model inputs. */
export function FeatureRow({
  name,
  expanded,
  onToggle,
  children,
  leading,
  actions,
  stats,
  highlighted = false,
  pruned = false,
  rowRef,
}: {
  name: string;
  expanded: boolean;
  onToggle: () => void;
  children: ReactNode;
  leading?: ReactNode;
  actions?: ReactNode;
  stats?: ReactNode;
  highlighted?: boolean;
  pruned?: boolean;
  rowRef?: Ref<HTMLDivElement>;
}) {
  return (
    <div className={`feature-row row-hover${highlighted ? " feature-row-highlight" : ""}`} ref={rowRef}>
      <div className="feature-row-header">
        <span className="feature-row-main">
          {leading}
          <button
            type="button"
            className="feature-row-toggle"
            onClick={onToggle}
            aria-label={`${expanded ? "Hide" : "Show"} details for ${name}`}
            aria-expanded={expanded}
          >
            {expanded ? "▾" : "▸"}
          </button>
          <button
            type="button"
            className={`feature-name feature-name-button${pruned ? " pruned-feature" : ""}`}
            onClick={onToggle}
            aria-expanded={expanded}
          >
            {name}
          </button>
          {actions}
        </span>
        {stats && <span className="feature-stats">{stats}</span>}
      </div>
      {expanded && children}
    </div>
  );
}
