import { useEffect, useState } from "react";
import { fetchCatalog, fetchCoverage, type CatalogResponse, type CoverageResponse } from "../api";

function coverageColor(pct: number | undefined): string {
  if (pct === undefined) return "#555b6b";
  const t = Math.max(0, Math.min(1, (pct - 0.4) / 0.6));
  const bad = [193, 97, 90];
  const good = [95, 174, 140];
  const rgb = bad.map((c, i) => Math.round(c + (good[i] - c) * t));
  return `rgb(${rgb.join(",")})`;
}

export default function FeatureCatalog() {
  const [catalog, setCatalog] = useState<CatalogResponse | null>(null);
  const [coverage, setCoverage] = useState<CoverageResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([fetchCatalog(), fetchCoverage()])
      .then(([catalogData, coverageData]) => {
        setCatalog(catalogData);
        setCoverage(coverageData);
      })
      .catch((err) => setError(String(err)));
  }, []);

  if (error) return <p className="error">{error}</p>;
  if (!catalog || !coverage) return <p className="muted">Loading catalog...</p>;

  return (
    <div className="category-grid">
      {Object.entries(catalog.catalog).map(([category, features]) => (
        <details className="category" key={category}>
          <summary>
            {category} ({features.length})
          </summary>
          {features.map((feature) => {
            const pct = coverage.coverage[feature];
            return (
              <div className="feature-row" key={feature}>
                <span style={{ color: coverageColor(pct) }}>&#9679;</span>{" "}
                <span className="feature-name">{feature}</span>{" "}
                <span className="muted">{pct !== undefined ? `${Math.round(pct * 100)}%` : "design-only"}</span>
                <div className="feature-desc">{catalog.descriptions[feature]}</div>
              </div>
            );
          })}
        </details>
      ))}
    </div>
  );
}
