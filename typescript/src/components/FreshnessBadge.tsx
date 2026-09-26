import { fetchPipelineFreshness, type PipelineFreshnessResponse } from "../api";
import { StatStrip } from "./StatStrip";
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
      <StatStrip
        items={[
          { key: "ready", title: true, align: "start", value: ready ? "Ready to score" : "Not ready to score" },
          { key: "features", label: "Features", value: formatIsoDate(data.feature_snapshot_date) },
          { key: "model", label: "Model through", value: formatIsoDate(data.model_trained_through) },
          ...(!ready ? [{ key: "detail", align: "start" as const, value: data.detail }] : []),
        ]}
      />
    </p>
  );
}
