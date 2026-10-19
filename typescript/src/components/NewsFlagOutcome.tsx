import { fetchNewsOutcomes, type NewsOutcomesResponse } from "../api";
import { useFetchData } from "../useFetchData";

function Avg({ value }: { value: number | null }) {
  if (value === null) return <span className="muted">None yet</span>;
  const up = value >= 0;
  return <span className={up ? "quote-diff-up" : "quote-diff-down"}>{up ? "▲" : "▼"} {(Math.abs(value) * 100).toFixed(2)}%</span>;
}

/** Average day for news-flagged picks vs clear ones. Flagged should be lower. */
export default function NewsFlagOutcome() {
  const { data } = useFetchData<NewsOutcomesResponse>(fetchNewsOutcomes, { deps: [] });
  if (!data || data.flagged.n === 0) return null;
  return (
    <p className="news-flag-outcome">
      <strong>News flag</strong> Flagged <Avg value={data.flagged.avg} /> · Clear <Avg value={data.clear.avg} />
    </p>
  );
}
