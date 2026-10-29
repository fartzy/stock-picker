# Local work reconciliation — October 28, 2026

This archive preserves historical work discovered while checking in the local
workspace. The September handoff describes the state on September 29; it is
not the current task list.

- `session-handoff-2026-09-29.md` is the exact note from commit `4d0d238f0`.
  The SVM work later merged in PR #150. Its outstanding benchmark fix is
  recovered separately as part of this reconciliation.
- `generated-worktree-build.patch` is the exact uncommitted diff of ten
  BUILD files in `.claude/worktrees/feature+models-tab-run-history`, based on
  `d58d4633d` (`fix/training-panel-spacing`, already merged in PR #34).
  Gazelle had replaced repository labels with paths into that nested
  worktree and duplicated some training dependencies. These paths are
  invalid in a normal checkout. Preserve the diff here rather than apply
  it to current BUILD files. The nested worktree can return to its committed
  files once this archive is committed. `.bazelignore` already excludes it.
- The previously unpublished `picks/2026/09/29.txt` from
  `experiment/svm-stacked-features` is restored to its original dated path
  with the paper-book snapshot.

Other old data branches contain earlier snapshots of paths already present
on main and subsequently refreshed. The October 7 quote/fee work is included
in PR #198; the October 8 model and reconciled trades are included in
PRs #209, #210 and #213. Their historical branches are retained.

Validation of the October 28 snapshot: all 3,995 changed Parquet files decode;
all three SQLite stores pass integrity checks; the trade CSV matches all
1,340 database fills. Market and model artifacts remain tracked under ADR 0010.
