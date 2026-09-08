# Onboarding

You just cloned `stock-picker`. This gets you from a fresh clone to the app
running with real data.

## 1. Install bazelisk

```
brew install bazelisk
```

Bazel's version is pinned in `.bazelversion` -- always invoke `bazelisk`,
never plain `bazel`.

## 2. Bootstrap the data pipeline (first time only)

Nothing ships with pretrained data -- `data/` is entirely gitignored except
`data/training_runs/` (historical run metadata, kept so you can see prior
results even before you've trained anything yourself) and `data/trades/`
(the actual trade log, if the person who last committed had any logged).
Run these three, in order, once:

```
bazelisk run //python/stock_picker/ingestion:main   # pulls the 500-ticker universe + 6mo price history (network-bound, a few minutes)
bazelisk run //python/stock_picker/features:main    # computes ~107 engineered features per ticker from that history
bazelisk run //python/stock_picker/training:main    # trains the ensemble, persists it, records a run
```

Each step reads the previous step's output from disk
(`data/{universe,prices,features,models}/`), so they have to run in this
order the first time. After that, re-running any step just refreshes its
own data.

## 3. Run the app

```
bazelisk run //python/stock_picker/api:main   # backend, :8000
bazelisk run //typescript:dev                  # frontend, :5173, proxies /api -> :8000
```

Open `localhost:5173`. The API server has no auto-reload -- after changing
any Python file, kill and re-run `//python/stock_picker/api:main` for the
change to take effect. The frontend hot-reloads on its own.

## Next

- `README.md` -- what the app does, current model performance, roadmap.
- `CLAUDE.md` -- architecture, conventions, and the sharper gotchas for
  ongoing development (read this before making changes, not just to run it).
