import { useEffect, useRef } from "react";
import { fetchCorrelation, type CorrelationResponse } from "../api";
import { themeColor, themeRgb } from "../theme";
import { useFetchData } from "../useFetchData";

const CELL_SIZE_PX = 7;
const VISIBLE_PAIR_COUNT = 15;

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
  const canvasRef = useRef<HTMLCanvasElement>(null);

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
    <div style={{ display: "grid", gridTemplateColumns: "1.3fr 1fr", gap: 20 }}>
      <div style={{ overflowX: "auto" }}>
        <canvas ref={canvasRef} />
      </div>
      <div>
        <div className="muted" style={{ marginBottom: 8 }}>
          Top correlated pairs (pruning candidates)
        </div>
        {data.top_pairs.slice(0, VISIBLE_PAIR_COUNT).map((pair) => (
          <div className="pair-row" key={`${pair.a}-${pair.b}`}>
            <span>
              {pair.a}
              <br />
              <span className="muted">{pair.b}</span>
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
