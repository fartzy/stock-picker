import assert from "node:assert/strict";
import test from "node:test";
import { experimentalRegistrySections } from "../src/experimentalRegistry.js";

test("older API responses leave experimental sections empty", () => {
  assert.deepEqual(experimentalRegistrySections({ feature_views: [] }, { catalog: {} }), []);
  assert.deepEqual(
    experimentalRegistrySections(
      { experimental_views: [{ name: "svm_derived", features: ["svr_oof_pred"] }] },
      { experimental_features: {} },
    ),
    [],
  );
});

test("available research metadata is separate from production views", () => {
  const feature = { name: "svr_oof_pred", description: "Forecast", computation: "Earlier data", example: "0.01" };
  const registry = {
    feature_views: [{ name: "momentum", features: ["return_1d"] }],
    experimental_views: [{ name: "svm_derived", features: ["svr_oof_pred", "missing"] }],
  };
  const sections = experimentalRegistrySections(registry, {
    experimental_features: { svr_oof_pred: feature },
  });

  assert.equal(registry.feature_views.length, 1);
  assert.deepEqual(sections.map(({ view }) => view.features), [["svr_oof_pred"]]);
  assert.equal(sections[0].features.svr_oof_pred, feature);
});
