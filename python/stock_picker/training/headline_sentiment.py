"""Material-negative headline classifier for recommended names only.

Tfidf + logistic regression on labeled examples so "CEO pauses to thank
staff" does not look like "pauses Phase 3 trial". Fit once at first use.
"""

from __future__ import annotations

from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

_LABELED: list[tuple[str, int]] = [
    ("Xenon pauses Phase 3 clinical trial after safety review", 1),
    ("Company pauses Phase 3 clinical trial", 1),
    ("FDA places clinical hold on lead drug candidate", 1),
    ("Company halted the trial following patient deaths", 1),
    ("Complete response letter from FDA rejects application", 1),
    ("Biotech announces secondary offering and dilution", 1),
    ("Going concern warning in annual report", 1),
    ("Firm files for bankruptcy protection", 1),
    ("FDA issues warning letter over manufacturing", 1),
    ("Trial discontinued after futility analysis", 1),
    ("Company terminates the trial citing safety", 1),
    ("Shares halted pending news of clinical hold", 1),
    ("Recall of product after contamination found", 1),
    ("Phase 3 study paused for safety", 1),
    ("Clinical trial paused after adverse events", 1),
    ("Company reports quarterly earnings beat estimates", 0),
    ("Analyst upgrades shares to buy on growth", 0),
    ("CEO pauses to thank employees at conference", 0),
    ("Firm pauses hiring to focus on product launch", 0),
    ("New partnership announced with larger drug maker", 0),
    ("Dividend increased for the third consecutive year", 0),
    ("Company to present data at medical conference", 0),
    ("Stock added to a major index", 0),
    ("Management to host investor day next month", 0),
    ("Revenue guidance raised for the full year", 0),
    ("Board authorizes share repurchase program", 0),
    ("Routine quarterly 10-Q filing submitted", 0),
    ("Company pauses to celebrate anniversary", 0),
]


@lru_cache(maxsize=1)
def _pipeline() -> Pipeline:
    texts = [row[0] for row in _LABELED]
    labels = [row[1] for row in _LABELED]
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=400, class_weight="balanced", C=4.0)),
        ]
    )
    pipe.fit(texts, labels)
    return pipe


def material_negative_score(text: str) -> float:
    blob = (text or "").strip()
    if not blob:
        return 0.0
    proba = _pipeline().predict_proba([blob])[0]
    classes = list(_pipeline().named_steps["clf"].classes_)
    return float(proba[classes.index(1)]) if 1 in classes else 0.0


def is_material_negative(text: str) -> bool:
    blob = (text or "").strip()
    if not blob:
        return False
    return int(_pipeline().predict([blob])[0]) == 1


def news_flag_from_articles(articles: list[dict]) -> str | None:
    best_headline = None
    best_score = 0.0
    for article in articles:
        headline = str(article.get("headline") or "").strip()
        summary = str(article.get("summary") or "").strip()
        blob = f"{headline}. {summary}".strip()
        if not is_material_negative(blob):
            continue
        score = material_negative_score(blob)
        if score >= best_score:
            best_score = score
            best_headline = headline[:160] or None
    return best_headline
