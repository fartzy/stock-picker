import { useId, useState, type KeyboardEvent } from "react";
import type { TickerSuggestion } from "../overnightTickers";

const MAX_SUGGESTIONS = 8;

export default function OvernightTickerInput({ value, suggestions, onChange, onSelect }: {
  value: string;
  suggestions: readonly TickerSuggestion[];
  onChange: (value: string) => void;
  onSelect: (ticker: string) => void;
}) {
  const id = useId();
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  const search = value.trim().toUpperCase();
  const matches = suggestions.filter(({ ticker }) => !search || ticker.includes(search)).slice(0, MAX_SUGGESTIONS);
  const visible = open && matches.length > 0;

  function choose(ticker: string) {
    setOpen(false);
    onSelect(ticker);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Escape") {
      setOpen(false);
    } else if (event.key === "ArrowDown" && matches.length) {
      event.preventDefault();
      setOpen(true);
      setActive((index) => (index + 1) % matches.length);
    } else if (event.key === "ArrowUp" && matches.length) {
      event.preventDefault();
      setOpen(true);
      setActive((index) => index < 0 ? matches.length - 1 : (index - 1 + matches.length) % matches.length);
    } else if (event.key === "Enter" && visible) {
      event.preventDefault();
      choose(matches[active < 0 || active >= matches.length ? 0 : active].ticker);
    }
  }

  return (
    <div className="overnight-ticker-field" onBlur={(event) => {
      if (!event.currentTarget.contains(event.relatedTarget)) setOpen(false);
    }}>
      <label htmlFor={`${id}-input`}>Ticker</label>
      <input
        id={`${id}-input`}
        className="form-input"
        value={value}
        onChange={(event) => { onChange(event.target.value.toUpperCase()); setOpen(true); setActive(-1); }}
        onFocus={() => { setOpen(true); setActive(-1); }}
        onKeyDown={handleKeyDown}
        placeholder="Search or enter ticker"
        role="combobox"
        aria-autocomplete="list"
        aria-expanded={visible}
        aria-controls={`${id}-options`}
        aria-activedescendant={visible && active >= 0 && active < matches.length ? `${id}-option-${active}` : undefined}
        autoComplete="off"
        required
      />
      {visible && <div id={`${id}-options`} className="overnight-ticker-options" role="listbox" aria-label="Traded tickers">
        {matches.map(({ ticker, detail }, index) => (
          <button
            id={`${id}-option-${index}`}
            key={ticker}
            type="button"
            role="option"
            aria-selected={index === active}
            onMouseEnter={() => setActive(index)}
            onClick={() => choose(ticker)}
          >
            <strong>{ticker}</strong><small>{detail}</small>
          </button>
        ))}
      </div>}
    </div>
  );
}
