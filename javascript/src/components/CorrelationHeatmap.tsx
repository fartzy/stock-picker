import { useEffect, useRef, useState } from "react";
import { fetchCorrelation, type CorrelationResponse } from "../api";

function corrColor(v: number | null): string {
  if (v === null) return "#1b2230";
  const t = (v + 1) / 2;
  const neg = [91, 127, 191];
  const mid = [42, 50, 66];
  const pos = [217, 164, 65];
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
  const [data, setData] = useState<CorrelationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    fetchCorrelation().then(setData).catch((err) => setError(String(err)));
  }, []);

  useEffect(() => {
    if (!data || !canvasRef.current) return;
    const cell = 7;
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
        {data.top_pairs.slice(0, 15).map((pair) => (
          <div className="pair-row" key={`${pair.a}-${pair.b}`}>
            <span>
              {pair.a}
              <br />
              <span className="muted">{pair.b}</span>
            </span>
            <span style={{ color: pair.correlation >= 0 ? "var(--accent)" : "#5b7fbf" }}>
              {pair.correlation >= 0 ? "+" : ""}
              {pair.correlation.toFixed(3)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
