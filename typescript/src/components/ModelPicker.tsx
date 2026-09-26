import {
  clearModelSelection,
  fetchModelSelection,
  fetchModelTypes,
  setModelSelection,
  type ModelSelectionResponse,
  type ModelTypeInfo,
  type ModelTypesResponse,
} from "../api";
import { useEffect, useState } from "react";
import { useFetchData } from "../useFetchData";

function SourceLink({ info }: { info: ModelTypeInfo }) {
  const location = `${info.source_file}:${info.source_line}`;
  // No origin remote resolved (e.g. no git available) -- plain text instead
  // of a dead link.
  if (!info.github_url) return <span>{location}</span>;
  return (
    <a href={info.github_url} target="_blank" rel="noreferrer">
      {location}
    </a>
  );
}

export default function ModelPicker() {
  const { data: modelTypes, error: modelTypesError } = useFetchData<ModelTypesResponse>(fetchModelTypes);
  const { data: modelSelection, error: selectionError } =
    useFetchData<ModelSelectionResponse>(fetchModelSelection);
  // undefined = not yet initialized from the fetch; null = no explicit
  // choice (every available model type included); Set = an explicit choice.
  // Nothing else in the app mutates this concurrently, so (like Registry's
  // feature selection) it's seeded once and then owned locally.
  const [chosenModelTypes, setChosenModelTypes] = useState<Set<string> | null | undefined>(undefined);

  useEffect(() => {
    if (chosenModelTypes === undefined && modelSelection) {
      setChosenModelTypes(
        modelSelection.model_choices ? new Set(modelSelection.model_choices.map((c) => c.model_type)) : null,
      );
    }
  }, [modelSelection, chosenModelTypes]);

  async function toggleModelType(modelType: string) {
    if (!modelSelection) return;
    const next = new Set(chosenModelTypes ?? modelSelection.available_model_types);
    if (next.has(modelType)) {
      // Always leave at least one model type selected -- an empty ensemble
      // has nothing to blend and nothing to train.
      if (next.size === 1) return;
      next.delete(modelType);
    } else {
      next.add(modelType);
    }
    if (next.size === modelSelection.available_model_types.length) {
      setChosenModelTypes(null);
      await clearModelSelection();
    } else {
      setChosenModelTypes(next);
      await setModelSelection([...next].map((type) => ({ model_type: type, weight: 1.0 })));
    }
  }

  const error = [modelTypesError, selectionError].filter(Boolean).join("; ") || null;
  if (error) return <p className="error">{error}</p>;
  if (!modelTypes || !modelSelection || chosenModelTypes === undefined) {
    return <p className="muted">Loading model types...</p>;
  }

  const chosen = chosenModelTypes ?? new Set(modelSelection.available_model_types);
  // model-types describes every trainer (including logistic_regression, a
  // diagnostic-only fit). The picker offers TRAINABLE_MODEL_TYPES -- return
  // models plus the ListFold rank model. Rank is trained as its own pickle, not blended.
  const pickable = modelTypes.model_types.filter((info) =>
    modelSelection.available_model_types.includes(info.model_type),
  );

  return (
    <div>
      <h3>Model families for training</h3>
      {chosenModelTypes === null && (
        <p className="muted" style={{ marginTop: 0 }}>
          All checked = no override (solo LightGBM + Rank pickle). Rank is
          never averaged into the 0.5% return blend.
        </p>
      )}
      {pickable.map((info) => (
        <div className="model-type-row row-hover" key={info.model_type}>
          <label>
            <input
              type="checkbox"
              checked={chosen.has(info.model_type)}
              onChange={() => toggleModelType(info.model_type)}
            />{" "}
            {info.display_name}
          </label>
          <div className="model-type-caption">
            {info.package}=={info.package_version} &middot; <SourceLink info={info} />
          </div>
        </div>
      ))}
    </div>
  );
}
