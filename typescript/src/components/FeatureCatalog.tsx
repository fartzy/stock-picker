import { fetchCatalog, fetchCoverage, type CatalogResponse, type CoverageResponse } from "../api";
import { themeColor, themeRgb } from "../theme";
import { useFetchData } from "../useFetchData";

const COVERAGE_GRADIENT_MIN_MAX: [number, number] = [0.4, 1.0];

function coverageColor(pct: number | undefined): string {
  if (pct === undefined) return themeColor("neutral");
  const [gradientMin, gradientMax] = COVERAGE_GRADIENT_MIN_MAX;
  const t = Math.max(0, Math.min(1, (pct - gradientMin) / (gradientMax - gradientMin)));
  const bad = themeRgb("bad");
  const good = themeRgb("good");
  const rgb = bad.map((c, i) => Math.round(c + (good[i] - c) * t));
  return `rgb(${rgb.join(",")})`;
}

export default function FeatureCatalog() {
  const { data: catalog, error: catalogError } = useFetchData<CatalogResponse>(fetchCatalog);
  const { data: coverage, error: coverageError } = useFetchData<CoverageResponse>(fetchCoverage);
  const error =
    catalogError && coverageError ? `${catalogError}; ${coverageError}` : (catalogError ?? coverageError);

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
