import { fetchNewsOutcomes, type NewsOutcomeGroup, type NewsOutcomesResponse } from "../api";
import { useFetchData } from "../useFetchData";
import { StatStrip } from "./StatStrip";

function pct(value: number | null, signed = false): string {
  if (value === null) return "None yet";
  const text = `${(value * 100).toFixed(signed ? 2 : 0)}%`;
  return signed && value > 0 ? `+${text}` : text;
}

function group(label: string, g: NewsOutcomeGroup, dropLabel: string): string {
  return `${label} ${g.n} · ${pct(g.avg, true)} avg · ${pct(g.win_rate)} up · ${pct(g.drop_rate)} ${dropLabel}`;
}

/** Did the news flag point at the worse days? Scored on every judged live pick. */
export default function NewsFlagOutcome() {
  const { data } = useFetchData<NewsOutcomesResponse>(fetchNewsOutcomes, { deps: [] });
  if (!data || data.flagged.n + data.clear.n === 0) return null;
  const right = data.flag_right_rate;
  const dropLabel = `fell ${Math.round(Math.abs(data.drop_threshold) * 100)}%+`;
  return (
    <StatStrip className="news-flag-outcome" items={[
      { key: "title", title: true, align: "start", value: <strong>News flag</strong> },
      { key: "flagged", align: "start", value: group("Flagged", data.flagged, dropLabel) },
      { key: "clear", align: "start", value: group("Clear", data.clear, dropLabel) },
      {
        key: "right",
        value: right === null ? "Not enough picks" : (
          <span className={right > 0.5 ? "quote-diff-up" : "quote-diff-down"}>
            Right {pct(right)} of the time
          </span>
        ),
      },
    ]} />
  );
}
