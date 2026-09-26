# 0008: Model families earn ensemble weight empirically; solos always searched

Date: 2026-09-06 (backfilled 2026-09-26)
Status: Accepted

## Context

It's tempting to assume more model families blended together = better. Each
family adds training time, persistence surface, and failure modes; the
assumption needs a test, not a vibe.

## Decision

All predictive trainers return a common `TrainedModel` shape so
`ensemble.py` can blend any mix by weighted average. Membership and weights
are decided by search (`tune_experiment.py`, later `hour_search.py` /
`grid_search.py`): every solo is always a candidate, so the search can never
recommend a blend worse than the best solo. Logistic regression is excluded
from blending on type grounds (predicts binarized direction, not continuous
return) and kept as a standalone importance lens.

## Consequences

- RandomForest (#38), NeuralNet, and Ridge (#48/#49) all went through the
  search and all lost to solo LightGBM -- production Fit is
  `DEFAULT_MODEL_SPECS = [ModelSpec("lightgbm")]`, decided by numbers.
- The pattern is permanent: a new family (e.g. SVR) gets a trainer + a
  search run before any picker/production exposure.
