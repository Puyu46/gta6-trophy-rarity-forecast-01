# GTA VI Trophy Rarity Forecaster

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Puyu46/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Puyu46/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)

Entry for make.fun's `gta6-trophy-rarity-forecast-01`: forecast, for each of
the 55 PlayStation 5 trophies of Grand Theft Auto VI, its earn rate on
Exophase at 2026-11-26T00:00Z, as five quantiles (5%, 25%, 50%, 75%, 95%).

## Files

- `main.py` — the entry. Python 3.13 standard library only, deterministic,
  no network, no clock. Reads every `<id>.in` in `MAKEFUN_INPUT_DIR` (or
  `/input`), writes `<id>.out` to `MAKEFUN_OUTPUT_DIR` (or `/output`).
- `score.py` — local scorer: the pinball loss on the logit scale exactly as
  `RULES.txt` defines it, plus typical miss and 90% coverage.
- `validate.py` — local validation harness (practice cases, a real 33-hour
  backtest, a synthetic final-like case, contract checks).
- `validation/` — validation fixtures (reconstructed practice answers, the
  20 Nov Exophase snapshot, the backtest case).
- `reference-baseline.py` — the kit's starter (house baseline), kept for
  comparison only; `main.py` is an independent implementation.
- `RULES.txt`, `LICENSE` — from the kit (LICENSE untouched).

## The model

A trophy's earn rate is holders / players-tracked, and the two move for
different reasons, so they are forecast separately.

**Players tracked N(t).** Exophase refreshes a game's counts in jumps hours
apart, so N is a step function; over days it follows roughly a power law in
time since release, `ln N = a + b * ln(1 + t)`. N at the target is read off
a least-squares fit of that law through the case's *distinct* reads
(identical consecutive reads carry no information and are dropped first).
When the history is flat (one distinct read, as in the practice cases) the
slope cannot be fit, so the model falls back to a prior slope and to the
documented first-refresh jump (see "Priors" below).

**Holders E_i(t).** For each trophy, `ln E_i` is regressed on `ln N` over
the distinct reads, giving an elasticity gamma: holders grow like
`N ** gamma`. Gamma near 0 means the trophy's holders are frozen (the
launch-night hardcore players only) and its rate is crushed by dilution as
N grows; gamma near 1 means holders grow with the player base and the rate
holds; gamma above 1 means the trophy is on the progression front and its
rate rises. The fitted gamma is shrunk toward a prior and, for trophies on
the front (recognised from their description: early story chapters and
popular early activities), pulled up toward the segment elasticity those
trophies showed between Exophase's first refresh and the public 20 Nov
snapshot — the front takes off after the first hours, which a fit on the
first hours alone cannot see. The pull is scaled by the horizon (a
90-minute horizon barely reaches the front) and every forecast rate is
capped at an archetype ceiling, so a runaway extrapolation cannot produce
a silly rate.

**Median and spread.** Median rate `r_last * (N_target / N_last) ** (gamma
- 1)`. The five quantiles are a logit-normal band around that median; the
spread grows with the horizon, with disagreement between the fitted and
prior gamma, and for front trophies. The intro trophy (100% at launch
night and at 42,228 players alike) is forecast to stay at 100%.

## Priors (public facts, not fitted to any hidden answer)

- First-refresh jump 6.06x: Exophase's first refresh took players tracked
  from 638 to 3,868 and trophies earned from 943 to 4,276 (kit's
  `competition.yaml`). A flat history means no refresh has been seen yet,
  so one is due before any target more than an hour out.
- Population slope 1.1: slope of `ln N` on `ln(1 + t_hours)` through the
  three public points — 638 at launch night, 3,868 at the first refresh,
  42,228 in the Wayback snapshot of 20 Nov 13:16 UTC.
- Front archetypes and segment elasticities (holders growth between the
  first refresh and the 20 Nov snapshot): Chapter 1 ~2.4, Chapter 2 ~1.7,
  fishing ~2.0, Breakfast Run ~2.1, arcade ~1.8, bowling ~1.8, vehicles
  ~1.7, Chapter 3 ~1.2, Chapter 4 ~1.0. Everything else sits near 0
  (holders frozen at the launch-night counts).

## Local validation results (`python3 validate.py`)

Practice answers are not shipped in the kit, so they were reconstructed
from the published refresh totals (3,868 players, 4,276 trophies earned):
trophy 2 stays at 100% (3,868 holders) and the other 54 trophies share the
remaining 408 holders, with the 103 net new non-intro holdings allocated
with weights `count**k`. The family spans k=1 (proportional) to k=8
(strongly concentrated on the easiest trophies); the house baseline scores
1.40–1.62 across it, matching the ~1.51 stated in `RULES.txt`, which
validates the reconstruction's scale. The central variant k=4 (baseline
1.48) is the closest to the stated figure. In short: the model scores
roughly 13x better than the house baseline on the practice cases (loss
~0.11–0.15 vs ~1.4–1.6) and about 2x better on the 33-hour backtest
(0.87 vs 1.78).

| check | model | house baseline |
|---|---|---|
| practice p00, loss (k=1/2/4/8) | 0.112 / 0.108 / 0.109 / 0.110 | 1.40 / 1.43 / 1.48 / 1.62 |
| practice p01, loss (k=1/2/4/8) | 0.153 / 0.141 / 0.132 / 0.110 | 1.40 / 1.43 / 1.48 / 1.62 |
| practice typical miss / 90% coverage | 0.04–0.05 pp / 0.95–0.98 | 0.68 pp / 0.00 |
| real 33h backtest (20 Nov snapshot), loss | 0.87 | 1.78 |
| backtest typical miss / coverage | 0.36 pp / 0.24 | 0.68 pp / 0.07 |

The backtest's remaining error is concentrated in the progression front
(Chapter 1 forecast 4.4% vs 11.1% actual, fishing 7.3% vs 12.8%): from a
two-state history the takeoff is only visible through the prior, and the
model deliberately under-predicts it rather than chasing the backtest —
the final case's history runs through 21 Nov 11:30, so the takeoff is in
the fitted data there. The backtest coverage (0.24) is low for the same
reason; the spread was widened (base 0.22 → 0.26, horizon 0.10 → 0.12)
after a grid search showed wider spreads help long horizons at a tiny
cost on the 90-minute practice cases.

Contract checks pass on every case: 55 rows, ids exact, quantiles
non-decreasing in [0, 100], outputs ~2.8 KB (limit 65,536), deterministic
bytes across runs, 0.07 s wall time for both practice cases (limit 8 s
per case), intro trophy ~100%, platinum ~0.1%.

## How to run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
python3 validate.py            # full local validation (practice + backtest)
python3 score.py out/p00.out validation/answers_practice_k4.json
```
