import assert from "node:assert/strict";
import test from "node:test";
import {
  hasUnprunedMarketFeature,
  modelDerivedRegistrySections,
  sameFeatureSet,
  selectableFeatureNames,
} from "../src/modelDerivedRegistry.js";

const feature = {
  name: "svc_direction_margin",
  description: "Earlier-fold direction score",
  computation: "Out-of-fold LinearSVC decision margin",
  example: "0.2",
  status: "production_eligible",
};

test("older APIs do not show model-derived views or change selection", () => {
  const registry = { feature_views: [{ features: ["return_1d"] }] };
  assert.deepEqual(modelDerivedRegistrySections(registry, {}), []);
  assert.deepEqual([...selectableFeatureNames(registry, {})], ["return_1d"]);
});

test("only advertised model-derived features are selectable, not research outputs", () => {
  const registry = {
    feature_views: [{ features: ["return_1d"] }],
    model_derived_views: [{ name: "direction_svc", features: ["svc_direction_margin", "missing"] }],
    experimental_views: [{ name: "svm_research", features: ["svr_oof_pred"] }],
  };
  const catalog = {
    model_derived_features: { svc_direction_margin: feature },
    experimental_features: { svr_oof_pred: { name: "svr_oof_pred" } },
  };
  assert.deepEqual(modelDerivedRegistrySections(registry, catalog).map(({ view }) => view.features), [["svc_direction_margin"]]);
  assert.deepEqual([...selectableFeatureNames(registry, catalog)], ["return_1d", "svc_direction_margin"]);
});

test("catalog metadata alone cannot make a feature selectable", () => {
  const registry = { feature_views: [{ features: ["return_1d"] }] };
  assert.deepEqual([...selectableFeatureNames(registry, { model_derived_features: { svc_direction_margin: feature } })], ["return_1d"]);
});

test("selection needs an unpruned market feature beside the model-derived margin", () => {
  const market = new Set(["return_1d"]);
  const selected = new Set(["return_1d", "svc_direction_margin"]);
  assert.equal(hasUnprunedMarketFeature(selected, market, new Set()), true);
  assert.equal(hasUnprunedMarketFeature(selected, market, new Set(["return_1d"])), false);
  assert.equal(hasUnprunedMarketFeature(new Set(["svc_direction_margin"]), market, new Set()), false);
});

test("reset-to-all compares members, not only set sizes", () => {
  const available = new Set(["return_1d", "svc_direction_margin"]);
  assert.equal(sameFeatureSet(new Set(["return_1d", "svc_direction_margin"]), available), true);
  assert.equal(sameFeatureSet(new Set(["return_1d", "stale_feature"]), available), false);
});
