import { useState } from "react";
import { createTrade, type TradeCreate } from "../api";

interface FormState {
  ticker: string;
  side: "buy" | "sell";
  shares: string;
  price: string;
  executedAt: string;
}

function pad(value: number): string {
  return String(value).padStart(2, "0");
}

function localDateTimeValue(date: Date): string {
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function toIsoWithOffset(localDateTime: string): string {
  const date = new Date(localDateTime);
  const offsetMinutes = -date.getTimezoneOffset();
  const sign = offsetMinutes >= 0 ? "+" : "-";
  const hours = pad(Math.floor(Math.abs(offsetMinutes) / 60));
  const minutes = pad(Math.abs(offsetMinutes) % 60);
  return `${localDateTime}:00${sign}${hours}:${minutes}`;
}

function emptyForm(): FormState {
  return { ticker: "", side: "buy", shares: "", price: "", executedAt: localDateTimeValue(new Date()) };
}

export default function AddTradeForm({ onAdded }: { onAdded: () => void }) {
  const [form, setForm] = useState<FormState>(emptyForm);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canSubmit =
    form.ticker.trim() !== "" &&
    Number(form.shares) > 0 &&
    Number(form.price) > 0 &&
    form.executedAt !== "";

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit || submitting) return;

    setSubmitting(true);
    setError(null);
    const trade: TradeCreate = {
      ticker: form.ticker.trim().toUpperCase(),
      side: form.side,
      shares: Number(form.shares),
      price: Number(form.price),
      executed_at: toIsoWithOffset(form.executedAt),
    };
    try {
      await createTrade(trade);
      setForm({ ...emptyForm(), executedAt: form.executedAt, side: form.side });
      onAdded();
    } catch (err) {
      setError(String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="add-trade-form">
      <span className="meta-label">Log a trade</span>
      <div className="form-row">
        <input
          className="form-input"
          placeholder="Ticker"
          value={form.ticker}
          onChange={(e) => setForm({ ...form, ticker: e.target.value })}
          style={{ width: 90 }}
        />
        <select
          className="form-select"
          value={form.side}
          onChange={(e) => setForm({ ...form, side: e.target.value as "buy" | "sell" })}
        >
          <option value="buy">Buy</option>
          <option value="sell">Sell</option>
        </select>
        <input
          className="form-input"
          type="number"
          placeholder="Shares"
          value={form.shares}
          onChange={(e) => setForm({ ...form, shares: e.target.value })}
          style={{ width: 90 }}
        />
        <input
          className="form-input"
          type="number"
          step="0.01"
          placeholder="Price"
          value={form.price}
          onChange={(e) => setForm({ ...form, price: e.target.value })}
          style={{ width: 100 }}
        />
        <input
          className="form-input"
          type="datetime-local"
          value={form.executedAt}
          onChange={(e) => setForm({ ...form, executedAt: e.target.value })}
          title="Fill date and time in your local timezone"
        />
        <button className="btn-primary" type="submit" disabled={!canSubmit || submitting}>
          {submitting ? "Adding..." : "Add trade"}
        </button>
      </div>
      <p className="muted" style={{ marginTop: 8 }}>
        Date and time are your local clock. Leave as-is for now, or set the actual fill.
      </p>
      {error && (
        <p className="error" style={{ marginTop: 8 }}>
          {error}
        </p>
      )}
    </form>
  );
}
