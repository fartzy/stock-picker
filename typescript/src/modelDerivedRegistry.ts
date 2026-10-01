// The API may be older than the frontend during a rolling restart. Only
// surface a model-derived feature when both registry and catalog advertise it.
export function modelDerivedRegistrySections<TView extends { features: string[] }, TFeature>(
  registry: { model_derived_views?: TView[] },
  catalog: { model_derived_features?: Record<string, TFeature> },
): Array<{ view: TView; features: Record<string, TFeature> }> {
  const features = catalog.model_derived_features ?? {};
  return (registry.model_derived_views ?? [])
    .map((view) => ({
      view: { ...view, features: view.features.filter((name) => name in features) } as TView,
      features,
    }))
    .filter(({ view }) => view.features.length > 0);
}

export function selectableFeatureNames(
  registry: { feature_views: Array<{ features: string[] }>; model_derived_views?: Array<{ features: string[] }> },
  catalog: { model_derived_features?: Record<string, unknown> },
): Set<string> {
  return new Set([
    ...registry.feature_views.flatMap((view) => view.features),
    ...modelDerivedRegistrySections(registry, catalog).flatMap(({ view }) => view.features),
  ]);
}

export function sameFeatureSet(left: Set<string>, right: Set<string>): boolean {
  return left.size === right.size && [...left].every((name) => right.has(name));
}

export function hasUnprunedMarketFeature(
  selected: Set<string>,
  marketFeatures: Set<string>,
  pruned: Set<string>,
): boolean {
  return [...selected].some((name) => marketFeatures.has(name) && !pruned.has(name));
}
