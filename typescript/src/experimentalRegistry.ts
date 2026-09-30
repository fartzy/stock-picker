// A refreshed frontend can briefly be paired with an older API process.
// Hide unavailable research metadata without changing production categories.
export function experimentalRegistrySections<TView extends { features: string[] }, TFeature>(
  registry: { experimental_views?: TView[] },
  catalog: { experimental_features?: Record<string, TFeature> },
): Array<{ view: TView; features: Record<string, TFeature> }> {
  const features = catalog.experimental_features ?? {};
  return (registry.experimental_views ?? [])
    .map((view) => ({
      view: { ...view, features: view.features.filter((name) => name in features) } as TView,
      features,
    }))
    .filter(({ view }) => view.features.length > 0);
}
