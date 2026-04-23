# Live data & session-style forecasting — brainstorm notes

*Captured for later discussion. Not implementation spec.*

---

## 1. Getting live match data

**Feasible**, but depends on **source, cost, latency, and legal/T&Cs**.

| Approach | Notes |
|----------|--------|
| **Official / commercial feeds** | Leagues / data providers sell real-time ball-by-ball APIs with licensing. Best for production reliability and rights. |
| **Scraping public sites** | Often against ToS, fragile, incomplete or delayed. Poor foundation for production. |
| **Community / open data** | Often delayed; OK for experiments, not trading-grade “live.” |
| **Own / partner scoring** | If you control or integrate the scorer feed, you control quality and latency. |

The real constraint is usually **rights + SLA + latency**, not whether bytes exist.

---

## 2. How odds work on books like Stake (general industry picture)

No access to any one operator’s internal stack; sportsbooks typically combine:

1. **Implied probabilities** from pricing models (stats, market, conditions, injuries, etc.).
2. **Decimal odds (simplified):** fair odds ≈ 1 / probability, before margin.
3. **Margin (vig / overround):** prices shortened so implied probs sum to **> 100%** across a market — that gap is the book’s edge.
4. **Flow / liability:** odds move as money arrives — balance risk, incorporate sharp signals; large operators may hedge elsewhere.
5. **Live:** continuous updates from **live models + trader oversight**; in-play is harder because state changes every ball.

So: **model + margin + risk management**, not one public formula.

---

## 3. Session betting in cricket (concept)

**Session betting** = short-segment markets (runs in next over / next X balls / overs 7–10, wickets in a block, etc.).

**Pricing idea (conceptual):**

- Build a **distribution of runs** (or wickets) for that segment **conditional on** current score, wickets, overs left, phase (PP / middle / death), venue, matchup, aggression.
- Convert to probabilities for lines (e.g. over/under 8.5 runs in the over).
- Apply **margin** like any other market.
- **In-play:** update **very frequently** (often every ball); variance in 6 balls is huge (wicket or six swings everything).

Informal “fancy” markets can be similar in theory but often more manual and higher model risk.

---

## 4. API “hits” per innings (if *you* provide an API)

Depends on **delivery model**:

| Model | Rough picture |
|--------|----------------|
| **WebSocket / SSE / push** | One long-lived connection; events per ball/update (often **~100–250+** events per T20 innings including extras/revisions — depends on provider). Not really “hits” per ball in the HTTP sense. |
| **REST polling ball-by-ball** | Up to **~120** requests per **full** T20 innings (fewer if innings ends early). |
| **Time-based polling (e.g. every 3–5 s)** | Can land around **hundreds** of requests per innings depending on length and rate limits. |
| **Batch “since cursor”** | Fewer HTTP calls, but latency must still meet product needs. |

**Budget mentally:** at least **one update per legal delivery** for ball-by-ball modeling, or a **streaming** feed with low latency.

---

## 5. Product idea: continuous session-style *forecasts* (not micromarkets)

**Windows discussed (examples):** 0–6 overs, 1–10, 1–12, 1–15, full innings — definitions must be nailed down (inclusive/exclusive of partial overs, wides/no-balls in “delivery countdown”).

**Requirement:** session betting sometimes **expires 10–15 deliveries before** the end of the session → forecasts must target **runs in the *remainder*** of the window given **only** the balls left until cutoff (and wickets can end the innings early).

**Technically:** after each ball, maintain state and output a **distribution** (not only a mean) for remaining runs in the active/upcoming windows — calibration matters if compared to binary thresholds.

**“Beat Stake” / beat the book:** not something to promise. Books add **margin**, manage **liability**, and often have **better or faster data**. Strong models still face **vig** and **adverse selection**. Useful framing: separate **analytics / fantasy / broadcast** use cases from **systematic profit vs retail lines** (high legal and financial risk; assume structural edge to the house unless proven otherwise on held-out data with full costs).

---

## 6. Open questions for next conversation

- **Format:** T20 only vs ODIs?
- **API shape:** ball events + timestamps vs over summaries only? (Summaries may block “10–15 balls before expiry” logic.)
- **Exact window definitions** and alignment with how *your* market defines a “session.”
- **Goal:** research / product display vs comparison to published lines (compliance and ToS for any third-party data).

---

*Last consolidated: 2026-03-28.*
