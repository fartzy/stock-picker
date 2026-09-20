"""Material-news detector for recommended names.

Not "any headline" and not keyword "pause". Tfidf (1-3grams) plus explicit
event-phrase features, logistic regression. Label 1 = BIG move-the-name
news, good or bad (trial blow-up, FDA reject, dilution, bankruptcy, OR
takeover, FDA approval, halt-pending-news, melt-up). Label 0 = routine
(analyst initiate, modest beat, conference, roundup, CEO thanks staff).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

# Phrase hits that almost always mean "do not fade this as a pattern trade".
_MATERIAL_PHRASES = (
    "clinical hold",
    "clinical trial pause",
    "trial pause",
    "phase 3 pause",
    "phase iii pause",
    "halted the trial",
    "halts trial",
    "discontinued the trial",
    "terminated the trial",
    "complete response letter",
    "crl ",
    "fda rejects",
    "fda rejection",
    "fda approval",
    "fda approves",
    "accelerated approval",
    "warning letter",
    "going concern",
    "files for bankruptcy",
    "bankruptcy protection",
    "chapter 11",
    "secondary offering",
    "follow-on offering",
    "dilutive",
    "dilution",
    "stock offering",
    "to acquire",
    "will acquire",
    "acquisition of",
    "buyout",
    "taken private",
    "tender offer",
    "merger agreement",
    "definitive agreement",
    "trading halt",
    "halted pending",
    "halted for news",
    "going private",
    "restatement",
    "accounting probe",
    "doj investigation",
    "sec investigation",
    "fraud",
    "cook the books",
    "guidance slashed",
    "cuts guidance",
    "withdraws guidance",
    "ceo resigns",
    "ceo steps down",
    "short squeeze",
    "soars after",
    "surges after",
    "plunges after",
    "crashes after",
    "halts phase",
)

_LABELED: list[tuple[str, int]] = [
    # Big bad
    ("Xenon pauses Phase 3 clinical trial after safety review", 1),
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
    ("CEO resigns effective immediately amid probe", 1),
    ("Company slashes full-year guidance", 1),
    ("SEC opens investigation into accounting", 1),
    ("Stock plunges after failed late-stage study", 1),
    # Big good / mania
    ("FDA approves first-in-class treatment", 1),
    ("Company to be acquired at a 40 percent premium", 1),
    ("Board agrees to cash buyout", 1),
    ("Shares halted pending material news", 1),
    ("Stock soars after surprise earnings blowout", 1),
    ("Firm announces definitive merger agreement", 1),
    ("Tender offer launched at substantial premium", 1),
    ("Short squeeze sends shares surging after", 1),
    # Routine -- do NOT avoid
    ("Company reports quarterly earnings beat estimates", 0),
    ("Analyst upgrades shares to buy on growth", 0),
    ("Wells Fargo initiates coverage with overweight rating", 0),
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
    ("Stocks moving lower in Friday pre-market session", 0),
    ("Xenon Pharmaceuticals and other big stocks moving lower", 0),
    ("Applied Digital and Terawulf climb as oversold AI names bounce", 0),
    ("Ackman took a stake in Netflix years after a losing bet", 0),
    ("Should investors be worried about a competitor trial", 0),
]


def _phrase_features(texts):
    rows = []
    for raw in texts:
        blob = f" {(raw or '').lower()} "
        rows.append([1.0 if phrase in blob else 0.0 for phrase in _MATERIAL_PHRASES])
    return np.asarray(rows, dtype=float)


@lru_cache(maxsize=1)
def _pipeline() -> Pipeline:
    texts = [row[0] for row in _LABELED]
    labels = [row[1] for row in _LABELED]
    pipe = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "tfidf",
                            TfidfVectorizer(
                                ngram_range=(1, 3),
                                min_df=1,
                                sublinear_tf=True,
                            ),
                        ),
                        (
                            "phrases",
                            FunctionTransformer(_phrase_features, validate=False),
                        ),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    C=2.0,
                    solver="liblinear",
                ),
            ),
        ]
    )
    pipe.fit(texts, labels)
    return pipe


def material_news_score(text: str) -> float:
    blob = (text or "").strip()
    if not blob:
        return 0.0
    proba = _pipeline().predict_proba([blob])[0]
    classes = list(_pipeline().named_steps["clf"].classes_)
    return float(proba[classes.index(1)]) if 1 in classes else 0.0


def is_material_news(text: str) -> bool:
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
        if not is_material_news(blob):
            continue
        score = material_news_score(blob)
        if score >= best_score:
            best_score = score
            best_headline = headline[:160] or None
    return best_headline


# Back-compat names used by older tests.
def material_negative_score(text: str) -> float:
    return material_news_score(text)


def is_material_negative(text: str) -> bool:
    return is_material_news(text)
