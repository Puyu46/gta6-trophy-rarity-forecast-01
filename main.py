"""GTA VI trophy rarity forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Forecasts, for each of the game's PlayStation 5 trophies, five quantiles of
its Exophase earn rate at the case's target, from the case's history of reads
of the trophy page. Python 3.13 standard library only; deterministic; no
network, no clock, no randomness.

Model (holders / players decomposition)
----------------------------------------
A trophy's earn rate is holders / players-tracked, and the two move for
different reasons, so they are forecast separately:

* Players tracked N(t). Exophase refreshes a game's counts in jumps hours
  apart, so N is a step function. Over days it follows roughly a power law
  in time since release, ln N = a + b * ln(1 + t). N at the target is read
  off a least-squares fit of that law through the case's distinct reads
  (identical consecutive reads carry no information and are dropped first).
  When the history is flat (a single distinct read, as in the practice
  cases), the slope cannot be fit: the fit falls back to a prior slope and
  to the documented first-refresh jump (see JUMP_PRIOR below).

* Holders E_i(t) per trophy. For each trophy, ln E_i is regressed on ln N
  over the distinct reads, giving an elasticity gamma_i: holders grow like
  N ** gamma_i. gamma near 0 means the trophy's holders are frozen (the
  launch-night hardcore players only) and its rate is crushed by dilution
  as N grows; gamma near 1 means holders grow with the player base and the
  rate holds; gamma above 1 means the trophy is on the progression front
  and its rate rises. The fitted gamma is shrunk toward a prior and, for
  trophies on the front (recognised from their description: early story
  chapters and popular early activities), pulled up toward the segment
  elasticity those trophies showed between Exophase's first refresh and
  the public 20 Nov snapshot -- the front takes off after the first hours,
  which a fit on the first hours alone cannot see. The pull is scaled by
  the horizon (a 90-minute horizon barely reaches the front) and every
  forecast rate is capped at an archetype ceiling, so a runaway
  extrapolation cannot produce a silly rate.

* The median rate is r_last * (N_target / N_last) ** (gamma - 1). The
  five quantiles are a logit-normal band around that median; the spread
  grows with the horizon, with disagreement between the fitted and prior
  gamma, and for front trophies, whose takeoff is the least certain part.

Calibration priors (public facts, not fitted to any hidden answer)
------------------------------------------------------------------
* FIRST_REFRESH_JUMP = 6.06: Exophase's first refresh took players tracked
  from 638 to 3,868 (kit's competition.yaml) and trophies earned from 943
  to 4,276. A flat history means no refresh has been seen yet, so one is
  due before any target more than an hour out.
* POP_GROWTH_B = 1.1: slope of ln N on ln(1 + t_hours) through the three
  public points (638 at launch night, 3,868 at the first refresh, 42,228
  in the Wayback snapshot of 20 Nov 13:16 UTC).
* Front archetypes and their segment elasticities (holders growth between
  the first refresh and the 20 Nov snapshot, same public data): Chapter 1
  ~2.4, Chapter 2 ~1.7, fishing ~2.0, Breakfast Run ~2.1, arcade ~1.8,
  bowling ~1.8, vehicles ~1.7, Chapter 3 ~1.2, Chapter 4 ~1.0. Everything
  else sits near 0 (holders frozen at the launch-night counts).

Grader contract: reads every <id>.in in MAKEFUN_INPUT_DIR (or /input),
writes <id>.out to MAKEFUN_OUTPUT_DIR (or /output) as
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}, one row per
trophy, five non-decreasing quantiles in [0, 100], file well under 65,536
bytes, run in far under 8 seconds per case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports. (This file is self-contained.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [NormalDist().inv_cdf(tau) for tau in LEVELS]

# --- calibration priors (see module docstring) ---------------------------
FIRST_REFRESH_JUMP = 6.06   # 638 -> 3,868 players at the first refresh
POP_GROWTH_B = 1.1          # slope of ln N on ln(1 + t_hours)
FLAT_GAMMA = 0.16           # elasticity implied by the practice refresh totals
SHRINK_TARGET = 0.30        # centre the fitted gamma is shrunk toward
SPREAD_BASE = 0.26          # logit spread at zero horizon
SPREAD_HORIZON = 0.12       # extra logit spread per sqrt(day of horizon)
SPREAD_DISAGREE = 0.25      # extra spread per unit of fit/prior disagreement
SPREAD_FRONT = 0.15         # extra spread for front trophies
SPREAD_MIN = 0.15
SPREAD_MAX = 1.10
SPREAD_INTRO = 0.02         # the 100% intro trophy barely moves
MAX_PLAYERS = 3_000_000     # sanity cap on extrapolated players tracked

# Front archetypes, recognised from a trophy's name/description (lowercase
# substring match, first match wins): (segment elasticity, rate ceiling %).
# Segment elasticities are holders growth between Exophase's first refresh
# (19 Nov 04:30 UTC) and the public 20 Nov 13:16 UTC snapshot.
FRONT_TABLE = (
    ("chapter 1", 2.40, 55.0),
    ("chapter 2", 1.70, 35.0),
    ("breakfast run", 2.10, 35.0),
    ("fish", 2.00, 45.0),
    ("arcade", 1.80, 30.0),
    ("bowl", 1.80, 30.0),
    ("vehicle", 1.70, 30.0),
    ("chapter 3", 1.20, 20.0),
    ("chapter 4", 1.00, 15.0),
)


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    return math.log(clamp(pct) / (100.0 - clamp(pct)))


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text):
    """ISO 8601 UTC timestamp -> epoch seconds (naive datetimes as UTC)."""
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def ols_slope(xs, ys):
    """Least-squares slope of ys on xs; None when xs carry no variance."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def distinct_states(case):
    """Collapse the history to its distinct reads.

    Exophase holds a game's counts unchanged for hours, then jumps them at
    a refresh; the flat reads between jumps carry no information. A read is
    distinct when its players-tracked count or any rate differs from the
    previous distinct read. Returns [(t_hours, players, rates)] with
    t measured from the game's release date at 00:00 UTC.
    """
    release = case.get("game", {}).get("release_date", "2026-11-19")
    t0 = parse_time(release + "T00:00:00Z")
    states = []
    prev_key = None
    for read in case.get("history", []):
        players = read.get("players_tracked")
        rates = read.get("rates") or []
        if not isinstance(players, (int, float)) or players <= 0:
            continue
        key = (players, tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        t = (parse_time(read["at"]) - t0) / 3600.0
        states.append((t, float(players), list(rates)))
    return states


def front_archetype(name, description):
    """(segment elasticity, rate ceiling) for a front trophy, else None."""
    hay = (name + " " + description).lower()
    for needle, gamma, ceiling in FRONT_TABLE:
        if needle in hay:
            return gamma, ceiling
    return None


def last_rate(states, index):
    """The trophy's rate in the latest read that reports it, else 0.01."""
    for _, _, rates in reversed(states):
        if index < len(rates) and rates[index] is not None:
            return float(rates[index])
    return 0.01


def forecast_players(states, t_target, horizon_hours):
    """Players tracked at the target: power-law fit, with a flat-history prior."""
    n_last = states[-1][1]
    if len(states) >= 2:
        xs = [math.log(max(1.0 + t, 0.5)) for t, _, _ in states]
        ys = [math.log(n) for _, n, _ in states]
        slope = ols_slope(xs, ys)
        if slope is not None:
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            intercept = my - slope * mx
            pred = math.exp(intercept + slope * math.log(max(1.0 + t_target, 0.5)))
            return min(max(pred, n_last), MAX_PLAYERS)
    # Flat history: no slope to fit. Anchor a prior power law at the last
    # read, but never below the documented first-refresh jump -- a history
    # that has sat unchanged for hours is overdue a refresh.
    t_last = states[-1][0]
    prior = n_last * math.exp(
        POP_GROWTH_B
        * (math.log(max(1.0 + t_target, 0.5)) - math.log(max(1.0 + t_last, 0.5)))
    )
    if horizon_hours >= 1.0:
        prior = max(prior, n_last * FIRST_REFRESH_JUMP)
    return min(max(prior, n_last), MAX_PLAYERS)


def forecast_case(case):
    if case.get("levels", LEVELS) != LEVELS:
        raise ValueError(f"{case.get('id')} asks for levels {case.get('levels')}")
    trophies = case["trophies"]
    states = distinct_states(case)
    if not states:
        raise ValueError(f"{case.get('id')} has no usable history")
    horizon_minutes = float(case["horizon_minutes"])
    horizon_hours = horizon_minutes / 60.0
    horizon_days = horizon_minutes / 1440.0
    release = case.get("game", {}).get("release_date", "2026-11-19")
    t0 = parse_time(release + "T00:00:00Z")
    t_target = (parse_time(case["target"]) - t0) / 3600.0
    n_target = forecast_players(states, t_target, horizon_hours)
    n_last = states[-1][1]
    growth = n_target / n_last
    # The front takes off over about a day; a short horizon barely reaches it.
    front_scale = min(1.0, horizon_hours / 24.0)

    # Per-trophy log-holders and log-players over the distinct reads, for
    # the elasticity fit. Holders are whole players; the page prints rates
    # to 2 decimals, so counts below ~1 are floored at 0.5 for the log.
    log_n = [math.log(n) for _, n, _ in states]
    rows = []
    for i, trophy in enumerate(trophies):
        r_last = last_rate(states, i)
        arch = front_archetype(trophy.get("name", ""), trophy.get("description", ""))
        if r_last >= 99.9:
            # The intro trophy: every tracked player holds it, at launch
            # night and at 42k players alike. It stays at 100%.
            median, spread = 100.0, SPREAD_INTRO
        else:
            counts = []
            for _, n, rates in states:
                if i < len(rates) and rates[i] is not None:
                    counts.append(max(round(float(rates[i]) * n / 100.0), 0.5))
                else:
                    counts.append(None)
            pairs = [(ln, math.log(c)) for ln, c in zip(log_n, counts) if c is not None]
            gamma_fit = None
            if len(pairs) >= 2:
                gamma_fit = ols_slope([p[0] for p in pairs], [p[1] for p in pairs])
            if gamma_fit is None:
                # No movement to learn from: the refresh totals imply an
                # effective elasticity near FLAT_GAMMA for a typical trophy.
                gamma = FLAT_GAMMA
            else:
                # Trust the fit more the more distinct reads support it,
                # but always keep some prior weight: a two-read fit is one
                # refresh, and the front's takeoff is not in it yet.
                w = min(0.8, 0.25 + 0.08 * (len(states) - 1))
                gamma = w * gamma_fit + (1.0 - w) * SHRINK_TARGET
            if arch is not None and arch[0] >= 0.5:
                # Front trophy: the fitted elasticity is a lagging average
                # over the dip and the takeoff, so pull it up to the
                # segment elasticity the front actually showed.
                gamma = max(gamma, arch[0] * front_scale)
            median = r_last * growth ** (gamma - 1.0)
            ceiling = arch[1] if arch is not None else 99.99
            median = min(max(median, 0.01), ceiling)
            spread = SPREAD_BASE + SPREAD_HORIZON * math.sqrt(horizon_days)
            if gamma_fit is not None:
                spread += SPREAD_DISAGREE * abs(gamma - gamma_fit)
            if arch is not None:
                spread += SPREAD_FRONT
            spread = min(max(spread, SPREAD_MIN), SPREAD_MAX)
        lg = logit(median)
        rows.append({
            "id": trophy["id"],
            "q": [round4(100.0 / (1.0 + math.exp(-(lg + z * spread)))) for z in Z],
        })
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
