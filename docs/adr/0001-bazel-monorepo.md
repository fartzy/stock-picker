# 0001: Bazel monorepo, pinned via bazelisk, gazelle-generated BUILD files

Date: 2026-09-02 (backfilled 2026-09-26)
Status: Accepted

## Context

One repo holds a Python backend, a React/TS frontend, data, and jobs. Builds
and tests need to run identically across sessions and machines without a
per-directory toolchain dance.

## Decision

Bazel for the whole stack, version pinned in `.bazelversion` and enforced by
invoking `bazelisk` (never plain `bazel`). Python BUILD files are generated
by gazelle (`bazelisk run //:gazelle`) rather than written by hand.

## Consequences

- One command (`bazelisk test //...`) covers Python and TS.
- Gazelle has known sharp edges: a py_binary whose module is also a
  py_library source can't be import-resolved (documented in CLAUDE.md; the
  workaround is to not import across that boundary). As of 2026-09-26 gazelle
  also fails repo-wide on a phantom `stock_picker.log` ambiguity, so BUILD
  edits are currently manual, mirroring existing patterns.
