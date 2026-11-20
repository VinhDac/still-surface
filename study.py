"""The study. Run: python study.py   (writes results.json; a few minutes on six cores)

A. The paper's numerical example (section 3.2) on a still surface, 2000 paths as the paper
   uses, judged against the paper's own Figure 1.
B. The test the example does not run: the same trades while the implied vol surface moves
   through the half day (a parallel shift sized from the VIX, a change of skew that is a
   labelled scenario, and both; and the shift weighted by maturity, "term"), and what the
   book held at the close gains or loses if the surface then moves once more. Each
   strategy's mean, standard deviation and 1% tail.
C. The paper's own lever for risk during the session, the running penalty beta, which the
   example sets to 0: what each penalty buys (the in-session risk the move brings, the risk
   carried past the close) and what it costs (mean value on a still surface), the terminal
   penalty scaled against running penalties on the paper's buckets, on the book's total vega
   and on the vega the term-weighted move loads on.
D. How much less a longer maturity moves: the VIX3M against the VIX (data.term_loading).
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # one thread a worker: the workers are the parallelism

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import data
import model
import simulate
import surface

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS = 2000  # the paper's number of paths
SEED_A = 7
SEEDS_B = (7, 8, 9, 10)  # B pools four runs of 2000 paths, so a 1% tail rests on 80 paths
SCENARIOS = ("still", "parallel", "skew", "both", "parallel 15-20", "term")
FRONTIER_SCENARIOS = ("still", "parallel", "term")  # the penalties of C are run on these
# ASSUMPTION, a labelled scenario, not sized from data: the half-day change of the 3-month
# 90% minus 110% implied vol spread has a standard deviation of 1 vol point (surface.skew_shape).
SKEW_STD = 0.01
# "parallel" draws a day's change of the VIX from every day since 1990; "parallel 15-20" only
# from days after a close of 15 to 20, the band around this surface's at-the-money vol (17.6).
BAND = (15.0, 20.0)
SNAPSHOTS = tuple(range(model.STEPS // 10, model.STEPS + 1, model.STEPS // 10))  # every tenth of the session

# Figure 1(b), read by eye off the paper: where each density of terminal value peaks, and the
# peak's height. A normal density of height h has standard deviation 1 / (h sqrt(2 pi)).
FIGURE_1B = {
    "optimal baseline": (167, 0.083), "optimal high (x10)": (167, 0.080), "optimal low (x0.1)": (167, 0.083),
    "ZI 0.1% vega": (110, 0.158), "ZI 0.5% vega": (137, 0.065), "ZI 1% vega": (52, 0.079),
}
# Figure 1(d), read by eye: where each bucket's terminal percentage-vega peaks under ZI 0.1%.
FIGURE_1D = (5, 10, 4, 16, 25, 15)
OPTIMAL = ("optimal baseline", "optimal high (x10)", "optimal low (x0.1)")
ZI = ("ZI 0.1% vega", "ZI 0.5% vega", "ZI 1% vega")
BASELINE = "optimal baseline"
# C's penalties. A running penalty is priced as r / T: r is what the terminal penalty alpha would
# charge for the same vega at the close, so beta = alpha / T charges a session's worth of holding
# it what alpha charges for holding it past the close. Every one keeps the paper's alpha unless
# named "terminal".
TERMINAL = (3.0, 30.0, 100.0)  # with the paper's 1, 10 and 0.1, the terminal penalty's line
TOTAL = (0.01, 0.05, 0.2, 0.8)  # r of a running penalty on the book's total percentage-vega
BUCKETS = (0.1, 1.0, 10.0)  # m of a running penalty m alpha_k / T on each of the paper's buckets
TERM = (0.05, 0.2, 0.8)  # r of a running penalty on the vega the term-weighted move loads on
BOTH = (10.0, 0.2)  # the terminal penalty x10 and a running penalty 0.2 / T on total vega together


def terminal_group(book):
    return [model.Optimal(book, a, f"terminal x{a:g}") for a in TERMINAL]


def total_group(book):
    out = [model.Optimal(book, 1.0, f"running total {r:g}/T", beta_total=r / model.T) for r in TOTAL]
    a, r = BOTH
    return out + [model.Optimal(book, a, f"terminal x{a:g} + running total {r:g}/T", beta_total=r / model.T)]


def bucket_and_term_group(book):
    alpha = np.array(model.ALPHA)
    loading = data.term_loading()[0]
    out = [model.Optimal(book, 1.0, f"running buckets {m:g} alpha/T", beta=m * alpha / model.T) for m in BUCKETS]
    return out + [model.Optimal(book, 1.0, f"running term {r:g}/T", beta_total=r / model.T, loading=loading(book.tau))
                  for r in TERM]


GROUPS = {"paper": model.strategies, "terminal": terminal_group, "total": total_group,
          "buckets and term": bucket_and_term_group}


def job(task):
    """One run of a group of strategies: a scenario of the surface and a seed. The fills and the
    spot are drawn from the seed alone, whatever the group: every strategy of every group meets
    the same order flow and the same spot path, and the surface's move is drawn apart."""
    scenario, seed, group = task
    book = model.Book()
    strategies = GROUPS[group](book)
    _, changes = data.daily_changes()
    term = data.term_loading()[0]
    rng = np.random.default_rng(1000 + seed)  # the surface's own draws: fills and spot keep theirs
    moves = None
    if scenario != "still":
        pool = data.changes_from(*BAND) if scenario == "parallel 15-20" else changes
        shift = data.half_day(rng.choice(pool, PATHS)) if scenario != "skew" else np.zeros(PATHS)
        skew = rng.normal(0.0, SKEW_STD, PATHS) if scenario in ("skew", "both") else np.zeros(PATHS)
        moves = simulate.Moves(shift, skew, rng, loading=term if scenario == "term" else None)
    out = simulate.run(book, strategies, PATHS, seed, moves, snapshots=SNAPSHOTS)
    later = np.random.default_rng(2000 + seed)  # the next half day's move, on the closing book
    shift_next, skew_next = data.half_day(later.choice(changes, PATHS)), later.normal(0.0, SKEW_STD, PATHS)
    carry = simulate.carried(book, out, shift_next, skew_next)
    carry_term = simulate.carried(book, out, shift_next, skew_next, loading=term)
    keep = ("value", "spread", "hedged", "trades", "bucket_vega", "vega_path")
    return scenario, seed, group, out["names"], {k: out[k] for k in keep} | {"carried": carry, "carried_term": carry_term}


def judge(rep):
    """The reproduction, judged by what the paper's text and Figure 1 say."""
    mean = {n: rep[n]["value"]["mean"] for n in rep}
    out = []

    def say(rule, holds, detail):
        out.append({"rule": rule, "holds": bool(holds), "detail": detail})

    say("every optimal strategy beats every zero-intelligence one on mean terminal value (Fig 1b)",
        min(mean[n] for n in OPTIMAL) > max(mean[n] for n in ZI),
        f"lowest optimal {min(mean[n] for n in OPTIMAL):.1f}, highest ZI {max(mean[n] for n in ZI):.1f}")
    say("ZI with a 1% vega spread does poorly: the lowest mean of all six",
        min(mean, key=mean.get) == "ZI 1% vega", f"ZI 1% {mean['ZI 1% vega']:.1f}")
    for n in ("ZI 0.1% vega", "ZI 0.5% vega"):
        bv = rep[n]["bucket_vega"]
        say(f"{n} builds a positive vega in every bucket, wider than the optimal's (Fig 1d, 1e)",
            all(m > 0 for m in bv["mean"]) and min(bv["std"]) > max(rep["optimal baseline"]["bucket_vega"]["std"][k] for k in (0, 2, 3, 5)),
            "bucket means " + ", ".join(f"{m:.1f}" for m in bv["mean"]))
    ob = rep["optimal baseline"]["bucket_vega"]
    near = [abs(ob["mean"][k]) < 1.0 for k in (0, 2, 3, 5)]
    narrow = max(ob["std"][k] for k in (0, 2, 3, 5)) < min(ob["std"][k] for k in (1, 4))
    say("optimal baseline: buckets 1, 3, 4, 6 end near zero vega and narrower than buckets 2 and 5 (Fig 1c)",
        all(near) and narrow,
        "means " + ", ".join(f"{m:.2f}" for m in ob["mean"]) + "; std " + ", ".join(f"{s:.2f}" for s in ob["std"]))
    for n, (peak, height) in FIGURE_1B.items():
        sd_fig = 1 / (height * np.sqrt(2 * np.pi))
        m, s = rep[n]["value"]["mean"], rep[n]["value"]["std"]
        say(f"{n}: mean within 5% and std within 30% of Figure 1(b) read by eye ({peak}, about {sd_fig:.1f})",
            abs(m / peak - 1) < 0.05 and abs(s / sd_fig - 1) < 0.30, f"mean {m:.1f}, std {s:.2f}")
    zi = rep["ZI 0.1% vega"]["bucket_vega"]["mean"]
    say("ZI 0.1%: each bucket's mean vega within 4 points of Figure 1(d) read by eye",
        all(abs(a - b) < 4 for a, b in zip(zi, FIGURE_1D)),
        "this study " + ", ".join(f"{a:.1f}" for a in zi) + "; figure " + ", ".join(str(b) for b in FIGURE_1D))
    return out


def stats(runs, names, seeds):
    """Pool the given seeds of one scenario, strategy by strategy."""
    out = {}
    for i, n in enumerate(names):
        value = np.concatenate([runs[s]["value"][i] for s in seeds])
        bv = np.concatenate([runs[s]["bucket_vega"][i] for s in seeds])
        vp = np.concatenate([runs[s]["vega_path"][:, i] for s in seeds], axis=1)  # (snapshots, paths)
        out[n] = {
            "vega_path": {"fraction": [k / model.STEPS for k in SNAPSHOTS][:vp.shape[0]],
                          "mean": vp.mean(1).tolist(), "rms": np.sqrt((vp ** 2).mean(1)).tolist()},
            "value": simulate.summary(value),
            "spread": float(np.concatenate([runs[s]["spread"][i] for s in seeds]).mean()),
            "hedged": float(np.concatenate([runs[s]["hedged"][i] for s in seeds]).mean()),
            "trades": float(np.concatenate([runs[s]["trades"][i] for s in seeds]).mean()),
            "bucket_vega": {"mean": bv.mean(0).tolist(), "std": bv.std(0, ddof=1).tolist()},
            "carried": simulate.summary(np.concatenate([runs[s]["carried"][i] for s in seeds])),
            "carried_term": simulate.summary(np.concatenate([runs[s]["carried_term"][i] for s in seeds])),
            "with_carried": simulate.summary(np.concatenate([runs[s]["value"][i] + runs[s]["carried"][i] for s in seeds])),
        }
    return out


def lean(book):
    """How far one more lot of an option moves its own bid (and its ask the other way), as a share
    of 1/kappa, at four times of the session, for each penalty level: -2 Theta_2[n, n] kappa_n."""
    picks = {"1y at the money (bucket 5)": (1.0, 1.0), "3m 90% (bucket 1)": (0.9, 0.25)}
    when = {"open": 0, "mid-session": model.STEPS // 2, "10 minutes before the close": model.STEPS - 40,
            "last step": model.STEPS - 1}
    out = {}
    for scale, name in zip((1.0, 10.0, 0.1), OPTIMAL):
        o = model.Optimal(book, scale)
        out[name] = {}
        for label, (m, tau) in picks.items():
            n = int(np.flatnonzero(np.isclose(book.moneyness, m) & np.isclose(book.tau, tau))[0])
            out[name][label] = {w: float(-2 * o.theta2[j][n, n] * book.kappa[n]) for w, j in when.items()}
    return out


def moved_by(runs, still, names, seeds):
    """Per path, what the move did: the same trades, value with the move minus value without."""
    return {n: simulate.summary(np.concatenate([runs[s]["value"][i] - still[s]["value"][i] for s in seeds]))
            for i, n in enumerate(names)}


def frontier(runs, names):
    """C. Every optimal strategy of every group, on the same paths as the baseline (the fills and
    the spot are the seed's): what it costs, the baseline's mean on a still surface minus its own,
    path by path; what it buys, the in-session risk each move brings to its own trades (value
    with the move minus value without) and the risk its closing book carries into the next half
    day; and the book's vega through the session."""
    seeds = SEEDS_B
    where = {n: (g, i) for g, nm in names.items() for i, n in enumerate(nm) if not n.startswith("ZI")}

    def pooled(n, sc, key):
        g, i = where[n]
        return np.concatenate([runs[g][sc][s][key][i] for s in seeds])

    base = pooled(BASELINE, "still", "value")
    out = {}
    for n, (g, i) in where.items():
        value = pooled(n, "still", "value")
        paid = base - value
        e = {"group": g, "mean": float(value.mean()), "std": float(value.std(ddof=1)),
             "cost": float(paid.mean()), "cost_se": float(paid.std(ddof=1) / np.sqrt(paid.size)),
             "cost_share": float(paid.mean() / base.mean()), "trades": float(pooled(n, "still", "trades").mean())}
        for sc in ("parallel", "term"):
            moved = pooled(n, sc, "value") - value
            e[f"in_session_{sc}"] = simulate.summary(moved)
            e[f"in_session_{sc}_cut"] = float(1 - moved.std(ddof=1) / (pooled(BASELINE, sc, "value") - base).std(ddof=1))
        for key, label in (("carried", "parallel"), ("carried_term", "term")):
            c = pooled(n, "still", key)
            e[f"carried_{label}"] = simulate.summary(c)
            e[f"carried_{label}_cut"] = float(1 - c.std(ddof=1) / pooled(BASELINE, "still", key).std(ddof=1))
        vp = np.concatenate([runs[g]["still"][s]["vega_path"][:, i] for s in seeds], axis=1)
        e["vega_rms"] = np.sqrt((vp ** 2).mean(1)).tolist()
        out[n] = e
    return out


def term_structure():
    """D. The VIX3M's daily changes against the VIX's, the same days: the loading at 93 days, how
    steady it is, and what the power law through it gives each maturity of the book."""
    both, x, y = data.paired_changes()
    load, slope, p = data.term_loading()
    day = np.array(both[1:])
    periods = {}
    for lo, hi, label in (("2007", "2013", "2007-2012"), ("2013", "2020", "2013-2019"), ("2020", "2027", "2020-2026")):
        m = (day >= lo) & (day < hi)
        periods[label] = float(np.cov(x[m], y[m])[0, 1] / np.var(x[m], ddof=1))
    calm = np.abs(x) < 3
    return {"source": data.URL_3M, "first": both[0], "last": both[-1], "changes": int(x.size),
            "slope": slope, "p": p, "corr": float(np.corrcoef(x, y)[0, 1]), "slope_by_period": periods,
            "calm_days": int(calm.sum()), "slope_calm_days": float(np.cov(x[calm], y[calm])[0, 1] / np.var(x[calm], ddof=1)),
            "loading": {str(t): float(load(t)) for t in model.TAUS}}


def main():
    t0 = time.time()
    dates, changes = data.daily_changes()  # fetched once here, before the workers read the cache
    data.term_loading()  # and the VIX3M
    tasks = [(sc, s, "paper") for sc in SCENARIOS for s in SEEDS_B]
    tasks += [(sc, s, g) for g in GROUPS if g != "paper" for sc in FRONTIER_SCENARIOS for s in SEEDS_B]
    with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as pool:
        done = list(pool.map(job, tasks))
    runs = {g: {sc: {} for sc in SCENARIOS} for g in GROUPS}
    names = {}
    for sc, s, g, nm, out in done:
        runs[g][sc][s] = out
        names[g] = nm
    paper, pnames = runs["paper"], names["paper"]
    reproduction = stats(paper["still"], pnames, (SEED_A,))
    verdict = judge(reproduction)
    b = {sc: stats(paper[sc], pnames, SEEDS_B) for sc in SCENARIOS}
    moved = {sc: moved_by(paper[sc], paper["still"], pnames, SEEDS_B) for sc in SCENARIOS if sc != "still"}
    same_trades = all(np.array_equal(runs[g][sc][s]["bucket_vega"], runs[g]["still"][s]["bucket_vega"])
                      and np.array_equal(runs[g][sc][s]["trades"], runs[g]["still"][s]["trades"])
                      for g in GROUPS for sc in (SCENARIOS if g == "paper" else FRONTIER_SCENARIOS) for s in SEEDS_B)
    params = surface.PARAMS
    book = model.Book()
    fitted = {str(t): [float(surface.implied_vol(100 * m, t)) for m in surface.MONEYNESS_READ] for t in surface.FIGURE_1A}
    resid = max(abs(fit - read) for t, row in surface.FIGURE_1A.items() for fit, read in zip(fitted[str(t)], row))
    half = data.half_day(changes)
    results = {
        "paper": "Lucic and Tse, Option market-making and vol arbitrage, Risk.net 2025, author manuscript 30 Jan 2025",
        "settings": {"paths_A": PATHS, "seed_A": SEED_A, "paths_B": PATHS * len(SEEDS_B), "seeds_B": list(SEEDS_B),
                     "steps": model.STEPS, "T_years": model.T, "dt_years": model.DT, "sigma": model.SIGMA,
                     "alpha": list(model.ALPHA), "skew_std_assumed": SKEW_STD},
        "surface": {"params": [float(p) for p in params], "max_abs_residual_vs_figure_read": float(resid),
                    "vol_range_of_44": [float(book.vol0.min()), float(book.vol0.max())],
                    "options_cheaper_than_view": int((book.vol0 < model.SIGMA).sum())},
        "vix": {"source": data.URL, "first": dates[0], "last": dates[-1], "changes": int(changes.size),
                "std_points": float(changes.std(ddof=1)), "q01_points": float(np.quantile(changes, 0.01)),
                "q99_points": float(np.quantile(changes, 0.99)), "min_points": float(changes.min()),
                "max_points": float(changes.max()), "half_day_std_vol": float(half.std(ddof=1))},
        "A": {"strategies": reproduction, "verdict": verdict, "matches": all(v["holds"] for v in verdict)},
        "vix_band": {"band": list(BAND), "changes": int(data.changes_from(*BAND).size),
                     "std_points": float(data.changes_from(*BAND).std(ddof=1))},
        "B": {"scenarios": b, "moved_by": moved, "same_trades_in_every_scenario": bool(same_trades),
              "lean": lean(book)},
        "C": {"strategies": frontier(runs, names), "terminal": [0.1, 1.0, 10.0, *TERMINAL], "total": list(TOTAL),
              "buckets": list(BUCKETS), "term": list(TERM), "both": list(BOTH)},
        "D": term_structure(),
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    report(results)


def report(r):
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"A. Reproduction on a still surface, {r['settings']['paths_A']} paths")
    for n, s in r["A"]["strategies"].items():
        v = s["value"]
        print(f"  {n:20s} mean {v['mean']:7.2f}  std {v['std']:5.2f}  trades {s['trades']:6.0f}  "
              f"bucket vega " + " ".join(f"{m:5.1f}" for m in s["bucket_vega"]["mean"]))
    for v in r["A"]["verdict"]:
        print(f"  {'holds' if v['holds'] else 'FAILS'}  {v['rule']}: {v['detail']}")
    print("  the reproduction " + ("matches Figure 1" if r["A"]["matches"] else "DOES NOT MATCH Figure 1"))
    x = r["vix"]
    print(f"\nVIX {x['first']} to {x['last']}: {x['changes']} daily changes, std {x['std_points']:.2f} points, "
          f"1% {x['q01_points']:.2f}, 99% {x['q99_points']:.2f}; half-day std {100 * x['half_day_std_vol']:.2f} vol points")
    print(f"\nB. The surface moves, {r['settings']['paths_B']} paths; same trades in every scenario: "
          f"{r['B']['same_trades_in_every_scenario']}")
    for sc, block in r["B"]["scenarios"].items():
        print(f"  {sc}")
        for n, s in block.items():
            v, c = s["value"], s["carried"]
            print(f"    {n:20s} mean {v['mean']:7.2f}  std {v['std']:5.2f}  1% {v['q01']:7.2f}  worst 1% {v['es01']:7.2f}"
                  f"  | next half day on the closing book: std {c['std']:5.2f}  1% {c['q01']:6.2f}")
    for sc, block in r["B"]["moved_by"].items():
        print(f"  what the {sc} move did to the same trades")
        for n, v in block.items():
            print(f"    {n:20s} mean {v['mean']:6.2f}  std {v['std']:5.2f}  1% {v['q01']:7.2f}  worst 1% {v['es01']:7.2f}")
    print(f"  ({r['vix_band']['changes']} changes after a VIX close of 15 to 20: std {r['vix_band']['std_points']:.2f} points)")
    print("  one more lot leans its own quote by this share of 1/kappa (open, mid-session, 10 min before the close, last step)")
    for n, block in r["B"]["lean"].items():
        for label, v in block.items():
            print(f"    {n:20s} {label:27s} " + "  ".join(f"{100 * x:6.2f}%" for x in v.values()))
    print("  the book's total percentage-vega through the session, still surface: mean / root mean square")
    for n, s in r["B"]["scenarios"]["still"].items():
        vp = s["vega_path"]
        print(f"    {n:20s} " + "  ".join(f"{m:5.1f}/{q:5.1f}" for m, q in zip(vp["mean"], vp["rms"])))
    print(f"\nC. What each penalty costs and buys, {r['settings']['paths_B']} paths, against the baseline")
    print(f"    {'':38s} {'cost':>6s} {'(se)':>6s} | in-session std, cut: parallel     term | carried std, cut: parallel     term")
    for n, e in r["C"]["strategies"].items():
        ip, it, cp, ct = e["in_session_parallel"], e["in_session_term"], e["carried_parallel"], e["carried_term"]
        print(f"    {n:38s} {e['cost']:6.2f} {e['cost_se']:6.3f} | {ip['std']:5.2f} {100 * e['in_session_parallel_cut']:4.0f}%"
              f"  {it['std']:5.2f} {100 * e['in_session_term_cut']:4.0f}% | {cp['std']:5.2f} {100 * e['carried_parallel_cut']:4.0f}%"
              f"  {ct['std']:5.2f} {100 * e['carried_term_cut']:4.0f}%  vega rms " + " ".join(f"{x:4.1f}" for x in e["vega_rms"]))
    d = r["D"]
    print(f"\nD. VIX3M on VIX, {d['changes']} days {d['first']} to {d['last']}: slope {d['slope']:.3f} (corr {d['corr']:.3f}), "
          f"p {d['p']:.3f}; by period " + ", ".join(f"{k} {v:.3f}" for k, v in d["slope_by_period"].items())
          + f"; calm days {d['slope_calm_days']:.3f}; loading " + ", ".join(f"{k}y {v:.3f}" for k, v in d["loading"].items()))
    print(f"\n{r['seconds']} s")


if __name__ == "__main__":
    main()
