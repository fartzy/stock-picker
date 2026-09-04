import { useState } from "react";
import { createTrade, type TradeCreate } from "../api";

interface FormState {
  ticker: string;
  side: "buy" | "sell";
  shares: string;
  price: string;
}

const EMPTY_FORM: FormState = { ticker: "", side: "buy", shares: "", price: "" };

export default function AddTradeForm({ onAdded }: { onAdded: () => void }) {
  const [form, setForm] = useState<FormState>(EMPTY_FORM);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canSubmit = form.ticker.trim() !== "" && Number(form.shares) > 0 && Number(form.price) > 0;

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
    };
    try {
      await createTrade(trade);
      setForm(EMPTY_FORM);
      onAdded();
    } catch (err) {
      setError(String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit}>
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
          placeholder="Price"
          value={form.price}
          onChange={(e) => setForm({ ...form, price: e.target.value })}
          style={{ width: 100 }}
        />
        <button className="btn-primary" type="submit" disabled={!canSubmit || submitting}>
          {submitting ? "Adding..." : "Add trade"}
        </button>
      </div>
      {error && (
        <p className="error" style={{ marginTop: 8 }}>
          {error}
        </p>
      )}
    </form>
  );
}
