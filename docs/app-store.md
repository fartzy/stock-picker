# App Store (other machine)

Do this on a **separate machine** with a **personal cloud account**. This
Mac stays the trading box (8:31 click, parquet, SQLite, launchd). Do not
put cloud keys, Apple certs, or API secrets in this repo.

Home Screen on Wi‑Fi is already shipped (`http://<LAN>:8000` → Share →
Add to Home Screen). The store is a different product.

## What Apple wants

A signed iOS binary talking to **HTTPS on the public internet**. Not
`localhost:5173`, not `192.168.x.x:8000`, not a lid-sleeping Mac.

## Plan

1. **Apple Developer Program** (~$99/year) on the personal Apple ID.
2. **Host FastAPI** (Fly, Railway, a always-on VPS — pick one on the
   other machine). HTTPS required. This Mac is not the production host.
3. **Data the phone needs** is small: Rank/Fit JSON, paper book, trade
   log. Do **not** ship 2,000 parquet files to the cloud on day one.
   Options: rsync/SQLite snapshot from this Mac after the morning job, or
   run a thin API that only reads `data/buy_signals/` + `data/trades/`.
4. **Wrap the existing React UI** with Capacitor (`typescript/`). Same
   tabs. Point `api.ts` at the hosted origin, not Vite’s `/api` proxy.
5. **Auth** before anything public. Even a single-user password is enough
   for TestFlight; App Review will ask.
6. **Privacy nutrition labels + finance copy.** This looks like investing
   advice. No “guaranteed picks.” Trading tab is a journal, not a broker.
7. **Xcode → TestFlight → App Store Connect → review.** Expect a reject
   on localhost, missing privacy, or “buy these stocks.”

## Out of scope here

- Morning Rank+Fit still runs on **this** Mac (quotes, models, news).
  Cloud only **serves** the lists the phone reads.
- Native Swift rewrite — only if Capacitor review fails.
- Polygon paid keys, Grok keys, Finnhub — stay in env on the host, never
  git.

## First slice on the other machine

```
# 1. clone this repo
# 2. pnpm --dir typescript build
# 3. wrap typescript/ with Capacitor (ios)
# 4. host //python/stock_picker/api:main behind HTTPS
# 5. TestFlight to your phone
```

Stop after TestFlight works on cellular. Store submit is a later PR.
