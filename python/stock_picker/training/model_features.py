"""Which inputs each model actually reads, for the Feature Store's by-model view.

Names come from the saved artifacts (each ensemble member's persisted
`feature_names`, the overnight model's pinned `feature_columns`), not from
current selection or pruning state, which may have changed since training.
A feature can belong to several models; Fit and Rank share most of theirs.

News deciders read article text, not feature columns, so they are described
by the article fields they consume.
"""

from __future__ import annotations

from stock_picker.features.news import NEWS_FEATURE_COLUMNS
from stock_picker.features.stacked_svm import STACKED_SVM_COLUMNS
from stock_picker.training.ensemble import ensemble_feature_names
from stock_picker.training.news_day_judge import MAX_HEADLINES
from stock_picker.training.overnight_model import STACKED_FEATURE_COLUMNS

# Article fields the news deciders read (news_policy, news_day_judge, headline_sentiment).
NEWS_ARTICLE_FIELDS = ("headline", "summary")
# Stored per-article values the news_* features are built from (features/news.py).
NEWS_FEATURE_SOURCES = ("published_at", "material_score", "is_material", "phrase_hits")


def _ensemble_inputs(model) -> list[str]:
    return sorted(ensemble_feature_names(model)) if model is not None else []


def model_inputs(fit=None, rank=None, overnight=None) -> list[dict]:
    """One entry per model: key, label, saved input names, and how many are model outputs."""
    fit_inputs = _ensemble_inputs(fit)
    overnight_inputs = list(overnight.feature_columns) if overnight is not None else list(STACKED_FEATURE_COLUMNS)
    return [
        {
            "key": "fit",
            "label": "Fit",
            "saved": fit is not None,
            "features": fit_inputs,
            "model_outputs": [name for name in fit_inputs if name in STACKED_SVM_COLUMNS],
        },
        {
            "key": "rank",
            "label": "Rank",
            "saved": rank is not None,
            "features": _ensemble_inputs(rank),
            "model_outputs": [],
        },
        {
            "key": "overnight",
            "label": "Overnight",
            "saved": overnight is not None,
            "features": overnight_inputs,
            "model_outputs": [name for name in overnight_inputs if name.startswith("day_")],
        },
        {
            "key": "news",
            "label": "News",
            "saved": True,
            # News deciders read article text; the news_* columns are what the
            # headline classifier's scores become as Fit/Rank features.
            "features": list(NEWS_FEATURE_COLUMNS),
            "model_outputs": [],
            "articles": {
                "fields": list(NEWS_ARTICLE_FIELDS),
                "max_articles": MAX_HEADLINES,
                "feature_sources": list(NEWS_FEATURE_SOURCES),
            },
        },
    ]
