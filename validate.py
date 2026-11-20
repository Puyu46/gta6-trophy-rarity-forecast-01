"""Local validation harness for main.py.

Checks, all against data that is public or reconstructed from public totals
(see README.md for provenance):

V1  practice cases p00/p01 scored against the reconstructed 04:30 refresh
    answers (family k=1,2,4,8: the 103 net new non-intro holdings allocated
    with weights count**k; every variant honours the published totals
    3,868 players / 4,276 trophies earned). The house baseline scores
    1.40-1.62 on this family (RULES.txt states ~1.51).
V2  a real 33-hour backtest: a case whose history is the flat launch night
    plus the reconstructed 04:30 read, forecasting the exact rates of the
    Wayback snapshot of 20 Nov 13:16 UTC (42,228 players tracked).
V3  edge cases and grader-contract checks: quantiles non-decreasing and in
    [0, 100], 55 rows, ids exact, output JSON valid and <= 65,536 bytes,
    the 100% intro trophy stays ~100%, the platinum stays tiny,
    deterministic bytes across runs, runtime under 8 s per case.

Usage: python3 validate.py [--spread SCALE] [--no-v2]
With --spread SCALE the spread constants are multiplied (grid-search aid);
the graded program is main.py with its default constants.
"""

import importlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import main as forecaster  # noqa: E402
import score as scorer  # noqa: E402


def baseline_forecast(last_rates, spread):
    """Persistence forecast: every quantile = last rate, logit band `spread`."""
    from score import logit
    rows = {}
    for tid, r in last_rates.items():
        lg = logit(r)
        rows[int(tid)] = [
            100.0 / (1.0 + math.exp(-(lg + z * spread))) for z in scorer.Z
        ]
    return rows


def run_main_on(case, spread_scale=1.0):
    """Forecast one case dict with main.py's forecast_case (scaled spread)."""
    if spread_scale != 1.0:
        for name in ("SPREAD_BASE", "SPREAD_HORIZON", "SPREAD_DISAGREE", "SPREAD_FRONT"):
            setattr(forecaster, name, getattr(forecaster, name) * spread_scale)
    try:
        out = forecaster.forecast_case(case)
    finally:
        if spread_scale != 1.0:
            importlib.reload(forecaster)
    return {row["id"]: row["q"] for row in out["trophies"]}


def check_contract(forecast, case):
    problems = []
    ids = [t["id"] for t in case["trophies"]]
    if sorted(forecast) != sorted(ids):
        problems.append("trophy ids mismatch")
    for tid, qs in forecast.items():
        if len(qs) != 5:
            problems.append(f"trophy {tid}: {len(qs)} quantiles, not 5")
        if any((not isinstance(q, (int, float))) or q != q or abs(q) == float("inf")
               for q in qs):
            problems.append(f"trophy {tid}: non-finite quantile")
        if any(q < 0 or q > 100 for q in qs):
            problems.append(f"trophy {tid}: quantile outside [0, 100]")
        if any(b < a for a, b in zip(qs, qs[1:])):
            problems.append(f"trophy {tid}: quantiles not non-decreasing")
    return problems


def main():
    spread_scale = 1.0
    do_v2 = True
    args = sys.argv[1:]
    if "--spread" in args:
        spread_scale = float(args[args.index("--spread") + 1])
    if "--no-v2" in args:
        do_v2 = False

    p00 = json.load(open(ROOT / "input" / "p00.in"))
    p01 = json.load(open(ROOT / "input" / "p01.in"))
    last_practice = scorer.load_answers(ROOT / "validation" / "last_rates_practice.json")

    print(f"== V1 practice (spread x{spread_scale}) ==")
    for case in (p00, p01):
        fc = run_main_on(case, spread_scale)
        problems = check_contract(fc, case)
        print(f"-- {case['id']}: contract problems: {problems or 'none'}")
        for k in (1, 2, 4, 8):
            ans = scorer.load_answers(ROOT / "validation" / f"answers_practice_k{k}.json")
            loss, miss, cov, _ = scorer.score(fc, ans)
            print(f"   vs k={k} answers: loss={loss:.4f} miss={miss:.2f}pp cov={cov:.2f}")
        base = baseline_forecast(last_practice, 0.5 * math.sqrt(90 / 1440))
        ans4 = scorer.load_answers(ROOT / "validation" / "answers_practice_k4.json")
        bl, bm, bc, _ = scorer.score(base, ans4)
        print(f"   baseline (k=4 answers): loss={bl:.4f} miss={bm:.2f}pp cov={bc:.2f}")

    if do_v2:
        print(f"== V2 real 33h backtest (spread x{spread_scale}) ==")
        v2 = json.load(open(ROOT / "validation" / "v2_case.json"))
        ans = scorer.load_answers(ROOT / "validation" / "v2_answers.json")
        fc = run_main_on(v2, spread_scale)
        problems = check_contract(fc, v2)
        print(f"-- v2: contract problems: {problems or 'none'}")
        loss, miss, cov, _ = scorer.score(fc, ans)
        print(f"   model: loss={loss:.4f} miss={miss:.2f}pp cov={cov:.2f}")
        last_v2 = scorer.load_answers(ROOT / "validation" / "last_rates_v2.json")
        base = baseline_forecast(last_v2, 0.5 * math.sqrt(1966 / 1440))
        bl, bm, bc, _ = scorer.score(base, ans)
        print(f"   baseline: loss={bl:.4f} miss={bm:.2f}pp cov={bc:.2f}")
        for tid in (3, 20, 4, 8):
            print(f"   trophy {tid}: forecast median={fc[tid][2]:.2f}% "
                  f"answer={ans[tid]:.2f}%")

    # V4: synthetic final-like case -- three distinct states (launch night,
    # the 04:30 refresh, the exact 20 Nov snapshot), target 21 Nov 11:30.
    # No true answers exist for the target, so this only exercises the
    # multi-state fit path and checks the forecasts are sane.
    print("== V4 synthetic final-like case (3 states, no answers) ==")
    v2 = json.load(open(ROOT / "validation" / "v2_case.json"))
    ans20 = scorer.load_answers(ROOT / "validation" / "v2_answers.json")
    v4 = json.loads(json.dumps(v2))
    v4["id"] = "v4"
    v4["at"] = "2026-11-20T13:16:00Z"
    v4["horizon_minutes"] = 1334
    v4["target"] = "2026-11-21T11:30:00Z"
    v4["history"] = v2["history"] + [{
        "slot": "2026-11-20T13:16:00Z", "at": "2026-11-20T13:16:24.000Z",
        "players_tracked": 42228, "trophies_earned": 57008,
        "rates": [ans20[t["id"]] for t in v2["trophies"]],
    }]
    fc4 = run_main_on(v4, spread_scale)
    problems = check_contract(fc4, v4)
    print(f"-- v4: contract problems: {problems or 'none'}")
    for tid in (1, 2, 3, 4, 5, 20, 46):
        print(f"   trophy {tid}: last={ans20[tid]:.2f}% -> "
              f"median={fc4[tid][2]:.2f}% q05={fc4[tid][0]:.2f} q95={fc4[tid][4]:.2f}")
    meds = [fc4[t["id"]][2] for t in v4["trophies"]]
    print(f"   medians: min={min(meds):.2f}% max={max(meds):.2f}% "
          f"(intro trophy excluded from max: "
          f"{max(m for i, m in enumerate(meds) if i != 1):.2f}%)")

    # CLI run: determinism + runtime + output bytes (the exact grader path).
    print("== CLI run (grader path) ==")
    outdir = ROOT / "out"
    t0 = time.time()
    subprocess.run(
        [sys.executable, "main.py"], cwd=ROOT, check=True,
        env={"PATH": "/usr/bin:/bin", "MAKEFUN_INPUT_DIR": str(ROOT / "input"),
             "MAKEFUN_OUTPUT_DIR": str(outdir)},
    )
    dt = time.time() - t0
    sizes = {p.name: p.stat().st_size for p in sorted(outdir.glob("*.out"))}
    print(f"   wall time for {len(sizes)} cases: {dt:.2f}s; sizes: {sizes}")
    # determinism: run again into another dir and compare bytes.
    outdir2 = ROOT / "out2"
    subprocess.run(
        [sys.executable, "main.py"], cwd=ROOT, check=True,
        env={"PATH": "/usr/bin:/bin", "MAKEFUN_INPUT_DIR": str(ROOT / "input"),
             "MAKEFUN_OUTPUT_DIR": str(outdir2)},
    )
    same = all(
        (outdir / n).read_bytes() == (outdir2 / n).read_bytes() for n in sizes
    )
    print(f"   deterministic across runs: {same}")
    for n in sizes:
        doc = json.load(open(outdir / n))
        print(f"   {n}: {len(doc['trophies'])} rows, "
              f"trophy 2 q={doc['trophies'][1]['q']}, "
              f"platinum q={doc['trophies'][0]['q']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
