# 0026: Score the news flag on outcomes; collect data for a learned drop model

Date: 2026-10-18
Status: Implemented (measurement and data only)

## Problem

A news skip exists to avoid buying a stock that is about to plummet. Across
the 305 saved news decisions from Sep 21 to Oct 16, the flag pointed the
wrong way: flagged picks had better days than clear ones. Changing the
question did not fix it:

| Question | Flagged | P(flagged did worse) |
|---|---|---|
| Current: "avoid big news, good or bad" | 58 | 0.43 |
| "Avoid only bad news" (Grok) | 15 | 0.26 |
| "Will it fall today?" with gap and recent moves (Jev) | 211 | 0.42 |

The worst days came after news the price had already absorbed before the
open (FRMI +20% on a deal, then -11%; BSP +41% on Berkshire, then -14%).
Headline-reading judges do not see that. We will not keep rewording prompts
without evidence.

## Decision

- `paper/news_outcomes.py` joins every judged, closed, live paper-book pick
  to its outcome. Each row holds the headlines, gap, judge, flag, Jev news
  score, open-to-close return, and `dropped` (fell 5% or more).
- `GET /api/paper-book/news-outcomes` scores the flag as the share of
  (flagged, clear) pairs where the flagged pick did worse; 0.5 is no skill.
  A one-line **News flag** strip in What If shows it.
- `GET /api/paper-book/news-outcomes/rows` exports the labelled rows. This is
  the training set for a learned "will it drop" news model. No new storage:
  the paper book already keeps the news check, and the scans keep Jev.

News skipping and every buy stay unchanged. The trader makes each decision;
these numbers inform it.

## Next

Train the drop model once the rows cover enough flagged drops to validate
walk-forward by date. At about 25 judged picks a day, that takes months.
Backfilling the universe's past news from `news.db` against price history
would shorten it, and should be a separate ADR.
