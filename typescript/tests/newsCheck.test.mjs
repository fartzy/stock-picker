import assert from "node:assert/strict";
import test from "node:test";
import { newsCheckPresentation } from "../src/newsCheck.js";

const check = (status, extra = {}) => ({
  status, article_count: 5, reviewed_count: 5, sources: ["finnhub", "polygon"], issues: [], ...extra,
});

test("legacy, pending, empty and failed checks are distinct", () => {
  assert.equal(newsCheckPresentation().label, "details unavailable");
  assert.equal(newsCheckPresentation(null, true).label, "checked (older scan)");
  assert.equal(newsCheckPresentation(null, false).label, "pending");
  assert.equal(newsCheckPresentation(null, false).warning, true);
  assert.equal(newsCheckPresentation(null, true, "material headline").label, "");
  assert.equal(newsCheckPresentation(check("no_news", { article_count: 0, reviewed_count: 0 })).label, "no recent articles");
  assert.equal(newsCheckPresentation(check("not_checked", { article_count: null })).label, "not checked");
  assert.equal(newsCheckPresentation(check("not_checked", { article_count: null })).warning, true);
  assert.equal(newsCheckPresentation(check("error", { article_count: 0 })).label, "news check failed");
});

test("degraded review remains visible alongside a material-news flag", () => {
  const result = newsCheckPresentation(check("degraded", {
    flag: "DEA rescheduling pause", issues: ["llm_unavailable", "polygon_unavailable"], sources: ["finnhub"],
  }), true, "DEA rescheduling pause");
  assert.equal(result.label, "");
  assert.equal(result.warning, true);
  assert.match(result.detail, /local classifier used/);
  assert.match(result.detail, /Backup feed unavailable/);
});

test("classifier-only checks explain the article count without implying AI review", () => {
  const result = newsCheckPresentation(check("degraded", {
    article_count: 3, reviewed_count: 3, judge: "classifier", issues: ["llm_unavailable"],
  }));
  assert.equal(result.label, "AI unavailable · 3 articles screened");
  assert.match(result.detail, /3 of 3 articles screened by local classifier/);
});

test("successful review gives counts without promising the stock is clear", () => {
  const result = newsCheckPresentation(check("complete", { article_count: 1, reviewed_count: 1 }));
  assert.equal(result.label, "reviewed · 1 article");
  assert.equal(result.warning, false);
  assert.match(result.detail, /1 of 1 articles reviewed/);
});

test("a skip decision stays on one line while its review evidence remains in the tooltip", () => {
  const result = newsCheckPresentation(check("complete", { article_count: 14, reviewed_count: 14 }), true,
    "district court patent infringement ruling");
  assert.equal(result.label, "");
  assert.match(result.detail, /14 of 14 articles reviewed/);
});
