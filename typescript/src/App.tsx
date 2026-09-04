import CoverageChart from "./components/CoverageChart";
import CorrelationHeatmap from "./components/CorrelationHeatmap";
import FeatureCatalog from "./components/FeatureCatalog";
import Registry from "./components/Registry";

export default function App() {
  return (
    <div className="page">
      <header>
        <h1>stock-picker</h1>
        <p className="muted">Feature catalog, coverage, correlation, and registry -- served live from the FastAPI backend.</p>
      </header>

      <section>
        <h2>Registry</h2>
        <div className="panel">
          <Registry />
        </div>
      </section>

      <section>
        <h2>Feature Views</h2>
        <FeatureCatalog />
      </section>

      <section>
        <h2>Coverage, worst first</h2>
        <div className="panel">
          <CoverageChart />
        </div>
      </section>

      <section>
        <h2>Correlation &amp; redundancy</h2>
        <div className="panel">
          <CorrelationHeatmap />
        </div>
      </section>
    </div>
  );
}
