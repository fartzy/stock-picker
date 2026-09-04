import { useEffect, useRef, useState } from "react";
import {
  fetchCorrelation,
  fetchPrunedFeatures,
  pruneFeature,
  unpruneFeature,
  type CorrelationResponse,
  type PrunedFeaturesResponse,
} from "../api";
import { themeColor, themeRgb } from "../theme";
import { useFetchData } from "../useFetchData";

const CELL_SIZE_PX = 7;
const VISIBLE_PAIR_COUNT = 15;
// The heatmap's natural size (columns * CELL_SIZE_PX) can run much taller
// than the fixed-length pairs list next to it -- cap it and let it scroll
// internally instead of dictating a mostly-empty panel height.
const MAX_HEATMAP_HEIGHT_PX = 480;

function FeatureLabel({
  name,
  pruned,
  onToggle,
  muted,
}: {
  name: string;
  pruned: boolean;
  onToggle: () => void;
  muted?: boolean;
}) {
  return (
    <span className={muted ? "muted" : undefined}>
      <span className={pruned ? "pruned-feature" : undefined}>{name}</span>{" "}
      <button className="prune-toggle" onClick={onToggle}>
        {pruned ? "restore" : "prune"}
      </button>
    </span>
  );
}

function corrColor(v: number | null): string {
  if (v === null) return themeColor("surfaceRaised");
  const t = (v + 1) / 2;
  const neg = themeRgb("corrNegative");
  const mid = themeRgb("corrNeutral");
  const pos = themeRgb("accent");
  let c1: number[], c2: number[], tt: number;
  if (t < 0.5) {
    c1 = neg;
    c2 = mid;
    tt = t / 0.5;
  } else {
    c1 = mid;
    c2 = pos;
    tt = (t - 0.5) / 0.5;
  }
  const rgb = c1.map((c, i) => Math.round(c + (c2[i] - c) * tt));
  return `rgb(${rgb.join(",")})`;
}

export default function CorrelationHeatmap() {
  const { data, error } = useFetchData<CorrelationResponse>(fetchCorrelation);
  const { data: prunedData } = useFetchData<PrunedFeaturesResponse>(fetchPrunedFeatures);
  const [prunedOverride, setPrunedOverride] = useState<Set<string> | null>(null);
  const pruned = prunedOverride ?? new Set(prunedData?.pruned_features ?? []);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  async function togglePrune(feature: string) {
    const next = new Set(pruned);
    if (next.has(feature)) {
      next.delete(feature);
      await unpruneFeature(feature);
    } else {
      next.add(feature);
      await pruneFeature(feature);
    }
    setPrunedOverride(next);
  }

  useEffect(() => {
    if (!data || !canvasRef.current) return;
    const cell = CELL_SIZE_PX;
    const n = data.columns.length;
    const canvas = canvasRef.current;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = n * cell * dpr;
    canvas.height = n * cell * dpr;
    canvas.style.width = `${n * cell}px`;
    canvas.style.height = `${n * cell}px`;
    const ctx = canvas.getContext("2d")!;
    ctx.scale(dpr, dpr);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        ctx.fillStyle = corrColor(data.matrix[i][j]);
        ctx.fillRect(j * cell, i * cell, cell - 0.5, cell - 0.5);
      }
    }
  }, [data]);

  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="muted">Loading correlation...</p>;

  return (
    <div>
      <div style={{ overflow: "auto", maxHeight: MAX_HEATMAP_HEIGHT_PX, marginBottom: "var(--space-4)" }}>
        <canvas ref={canvasRef} />
      </div>
      <div className="muted" style={{ marginBottom: 8 }}>
        Top correlated pairs (pruning candidates)
      </div>
      <div className="pair-grid">
        {data.top_pairs.slice(0, VISIBLE_PAIR_COUNT).map((pair) => (
          <div className="pair-row" key={`${pair.a}-${pair.b}`}>
            <span>
              <FeatureLabel name={pair.a} pruned={pruned.has(pair.a)} onToggle={() => togglePrune(pair.a)} />
              <br />
              <FeatureLabel name={pair.b} pruned={pruned.has(pair.b)} onToggle={() => togglePrune(pair.b)} muted />
            </span>
            <span style={{ color: pair.correlation >= 0 ? "var(--accent)" : "var(--corr-negative)" }}>
              {pair.correlation >= 0 ? "+" : ""}
              {pair.correlation.toFixed(3)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
