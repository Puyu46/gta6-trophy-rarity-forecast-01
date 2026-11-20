"""Local scorer for gta6-trophy-rarity-forecast-01.

Implements the pinball loss on the logit scale exactly as RULES.txt's
"Scoring" section defines it, plus the board's diagnostics (typical miss,
90% coverage). Used only for local validation; the server grades entries.

Usage:
    python3 score.py <forecast.out.json> <answers.json> [--baseline-rates rates.json]

answers.json: {"<trophy id>": <earn rate in percent>, ...}
With --baseline-rates, also scores a persistence forecast (every quantile
equal to the given last-read rate) for comparison.
"""

import json
import math
import sys

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]


def clamp(x):
    """Clamp a percent to the scorer's [0.01, 99.99] window."""
    return min(99.99, max(0.01, x))


def logit(pct):
    return math.log(clamp(pct) / (100.0 - clamp(pct)))


def pinball(y, q, tau):
    """One item's pinball loss: 2 * max(tau*d, (tau-1)*d), d = logit(y)-logit(q)."""
    d = logit(y) - logit(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def score(forecast, answers):
    """forecast: {id: [q05, q25, q50, q75, q95]}; answers: {id: rate}.

    Returns (loss, typical_miss_pp, coverage_90, n_items).
    """
    total = 0.0
    miss = 0.0
    covered = 0
    n = 0
    for tid, y in answers.items():
        qs = forecast[tid]
        for tau, q in zip(LEVELS, qs):
            total += pinball(y, q, tau)
        miss += abs(y - qs[2])
        if qs[0] <= y <= qs[4]:
            covered += 1
        n += 1
    return total / (n * len(LEVELS)), miss / n, covered / n, n


def load_forecast(path):
    doc = json.load(open(path, encoding="utf-8"))
    return {row["id"]: row["q"] for row in doc["trophies"]}


def load_answers(path):
    raw = json.load(open(path, encoding="utf-8"))
    return {int(k): float(v) for k, v in raw.items()}


def main():
    if len(sys.argv) < 3:
        print("usage: score.py <forecast.out.json> <answers.json> "
              "[--baseline-rates rates.json]")
        return 2
    forecast = load_forecast(sys.argv[1])
    answers = load_answers(sys.argv[2])
    loss, miss, cov, n = score(forecast, answers)
    print(f"loss={loss:.4f} typical_miss={miss:.2f}pp coverage_90={cov:.3f} "
          f"trophies={n}")
    if "--baseline-rates" in sys.argv:
        i = sys.argv.index("--baseline-rates")
        last = load_answers(sys.argv[i + 1])
        base = {tid: [r] * 5 for tid, r in last.items()}
        bl, bm, bc, _ = score(base, answers)
        print(f"baseline(persistence, zero spread): loss={bl:.4f} "
              f"typical_miss={bm:.2f}pp coverage_90={bc:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
