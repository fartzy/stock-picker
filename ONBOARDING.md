# Onboarding

You just cloned `stock-picker`. This gets you from a fresh clone to the app
running with real data.

## 1. Install bazelisk

```
brew install bazelisk
```

Bazel's version is pinned in `.bazelversion` -- always invoke `bazelisk`,
never plain `bazel`.

## 2. Data is already there -- skip straight to step 3

`data/` (universe, prices, features, the trained model, trade log, run
history) is committed to git, not gitignored -- a fresh clone already has
a working, already-trained app. You do not need to run anything before
step 3.

The tradeoff: this data is only as fresh as whenever it was last
committed. If you want current prices/predictions rather than whatever
snapshot shipped with the clone, refresh the pipeline yourself, in order:

```
bazelisk run //python/stock_picker/ingestion:main   # pulls the 500-ticker universe + latest 6mo price history (network-bound, a few minutes)
bazelisk run //python/stock_picker/features:main    # recomputes ~107 engineered features per ticker from that history
bazelisk run //python/stock_picker/training:main    # retrains the ensemble, persists it, records a new run
```

Each step reads the previous step's output from disk
(`data/{universe,prices,features,models}/`), so they have to run in this
order. Re-running any of them just refreshes its own data -- and since
that data is tracked, refreshing it shows up as a real diff to commit.

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
