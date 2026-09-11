import { fetchPipelineFreshness, type PipelineFreshnessResponse } from "../api";
import { useFetchData } from "../useFetchData";

const POLL_INTERVAL_MS = 30_000;

function formatIsoDate(iso: string | null): string {
  if (!iso) return "none";
  const [year, month, day] = iso.split("-");
  if (!year || !month || !day) return iso;
  return new Date(Number(year), Number(month) - 1, Number(day)).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

export default function FreshnessBadge() {
  const { data, error } = useFetchData<PipelineFreshnessResponse>(fetchPipelineFreshness, {
    intervalMs: POLL_INTERVAL_MS,
  });

  if (error) return <p className="error">Could not load pipeline freshness.</p>;
  if (!data) return <p className="muted">Checking whether last night's pipeline finished...</p>;

  const ready = data.ready_for_inference;
  return (
    <p className={ready ? "freshness-ok" : "freshness-stale"} title={data.detail}>
      {ready ? "Ready to score" : "Not ready to score"}
      {" · "}
      features {formatIsoDate(data.feature_snapshot_date)}
      {" · "}
      model through {formatIsoDate(data.model_trained_through)}
      {!ready && <span className="muted"> — {data.detail}</span>}
    </p>
  );
}
