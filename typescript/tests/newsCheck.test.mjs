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
  assert.equal(newsCheckPresentation(null, true, "material headline").label, "");
  assert.equal(newsCheckPresentation(check("no_news", { article_count: 0, reviewed_count: 0 })).label, "no articles found · 0 articles");
  assert.equal(newsCheckPresentation(check("not_checked", { article_count: null })).label, "not checked");
  assert.equal(newsCheckPresentation(check("error", { article_count: 0 })).label, "check failed · 0 articles");
});

test("degraded review remains visible alongside a material-news flag", () => {
  const result = newsCheckPresentation(check("degraded", {
    flag: "DEA rescheduling pause", issues: ["llm_unavailable", "polygon_unavailable"], sources: ["finnhub"],
  }));
  assert.equal(result.label, "limited check · 5 articles");
  assert.equal(result.warning, true);
  assert.match(result.detail, /local classifier used/);
  assert.match(result.detail, /Backup feed unavailable/);
});

test("successful review gives counts without promising the stock is clear", () => {
  const result = newsCheckPresentation(check("complete", { article_count: 1, reviewed_count: 1 }));
  assert.equal(result.label, "reviewed · 1 article");
  assert.equal(result.warning, false);
  assert.match(result.detail, /1 of 1 articles reviewed/);
});
