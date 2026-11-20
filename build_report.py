"""Build the validation note, report/note.pdf, from results.json, hjb_results.json and the counts
checks.py and mutations.py leave (checks.json, mutations.json).

Every figure in the note's text is a macro this script writes from those files (\\val{name}), an
equation number of the paper (\\eq{12}), a section or figure of the paper (\\paper{...}), or one
of the paper's own parameters (\\given{2000}, checked against GIVEN). The text is searched for
any other digit: a number typed by hand stops the build. The sentences that compare numbers are
checked against the results first (claims), and the note must come out at three pages or fewer.

Run: python build_report.py   (needs pdflatex; writes report/numbers.tex, report/fig-*.pdf,
report/note.pdf and a copy at ../Dac-Vinh-Nguyen-Still-Surface.pdf)
"""
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import study  # noqa: E402  (the figure read by eye, FIGURE_1B, is the study's own constant)

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "report")
PDF_COPY = os.path.join(os.path.dirname(HERE), "Dac-Vinh-Nguyen-Still-Surface.pdf")
PAGES = 3
REPO = "github.com/VinhDac/still-surface"
PAGE = "vinhdac.github.io/notes/still-surface"
MONTHS = "January February March April May June July August September October November December".split()
# The only numbers the text may carry outside \val: the paper's parameters and the two horizons
# of the indices (each written as \given{...}).
GIVEN = {"2025", "2024", "17", "0.1", "0.2", "1", "30", "93"}
NAMED = ("VIX3M", "VXVCLS", "\\Theta_2")  # names and symbols with a digit in them, not numbers
NAMES = {"optimal baseline": "optimal, the paper's $\\alpha$", "optimal high (x10)": "optimal, $\\alpha\\times 10$",
         "optimal low (x0.1)": "optimal, $\\alpha\\times 0.1$", "ZI 0.1% vega": "zero-intelligence, 0.1\\% vega",
         "ZI 0.5% vega": "zero-intelligence, 0.5\\% vega", "ZI 1% vega": "zero-intelligence, 1\\% vega"}
RT = "running term 0.2/T"  # the running penalty the summary names: on the loaded vega, r = 0.2
RTOT = "running total 0.2/T"
X10, X100 = "optimal high (x10)", "terminal x100"


def load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


def num(x, d=2):
    s = f"{abs(x):,.{d}f}".replace(",", "{,}")
    return ("$-$" if x < 0 and round(abs(x), d) > 0 else "") + s


def pct(x):
    return f"{round(100 * x):d}\\%"


def day(iso):
    d = dt.date.fromisoformat(iso)
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


class Numbers:
    """The note's macros: a name and the value as printed."""

    def __init__(self):
        self.values = {}

    def __setitem__(self, name, text):
        assert re.fullmatch(r"[a-z0-9-]+", name), f"bad macro name {name}"
        assert name not in self.values, f"{name} twice"
        self.values[name] = str(text)

    def tex(self):
        lines = ["% written by build_report.py from the results: do not edit"]
        lines += [f"\\expandafter\\def\\csname val@{k}\\endcsname{{{v}}}" for k, v in self.values.items()]
        return "\n".join(lines) + "\n"


SEVEN_TENTHS = 7  # the vega snapshots are at every tenth of the session: the first seven


def x10_gap(c):
    """How far ten times the terminal penalty's vega is from the baseline's, the most over the first
    seven tenths of the session, as a share of the baseline's."""
    vb, vx = c["optimal baseline"]["vega_rms"], c[X10]["vega_rms"]
    return max(abs(x / y - 1) for x, y in zip(vx[:SEVEN_TENTHS], vb[:SEVEN_TENTHS]))


def claims(r, h, checks, faults):
    """The note's sentences that compare numbers, checked before anything is written."""
    a, b, c = r["A"], r["B"], r["C"]["strategies"]
    assert a["matches"], "the reproduction no longer matches Figure 1"
    assert b["same_trades_in_every_scenario"], "a moving surface changed the trades"
    sc = b["scenarios"]
    base_still, base_term, base_par = (sc[k]["optimal baseline"]["value"]["std"] for k in ("still", "term", "parallel"))
    assert base_still < base_term < base_par, "the moves no longer add risk in this order"
    assert c[X10]["in_session_term_cut"] < 0.1 and c[X10]["in_session_parallel_cut"] < 0.1, "x10 now cuts in-session risk"
    assert c[X100]["in_session_term_cut"] < 0.1, "x100 now cuts in-session risk"
    assert c[X10]["carried_term_cut"] > 2 * c[X10]["in_session_term_cut"], "x10 no longer works mainly at the close"
    assert c[RT]["in_session_term_cut"] > 3 * c[X10]["in_session_term_cut"], "the running penalty no longer does the in-session work"
    assert c[RT]["cost"] < c[X10]["cost"], "the running penalty is no longer the cheaper"
    assert c[RT]["cost"] < c[RTOT]["cost"], "the loaded factor is no longer cheaper than the whole book's vega"
    assert c["running buckets 1 alpha/T"]["in_session_term_cut"] < c[RT]["in_session_term_cut"] or \
        c["running buckets 1 alpha/T"]["cost"] > c[RT]["cost"], "the paper's buckets now buy as much for less"
    vb, vx = c["optimal baseline"]["vega_rms"], c[X10]["vega_rms"]
    assert x10_gap(c) < 0.05, "x10's vega parts from the baseline's within the first seven tenths"
    assert vx[-1] < 0.5 * vb[-1], "x10 no longer pulls the vega in at the close"
    vt = c[RTOT]["vega_rms"]
    assert max(vt) < 1.15 * min(vt) and max(vt) < 0.5 * max(vb), "the running penalty no longer holds the whole vega level and low"
    skew = sc["skew"]["optimal baseline"]["value"]["std"]
    assert skew < base_still * 1.1, "the skew scenario no longer adds little"
    assert checks["held"] == checks["of"] and faults["caught"] == faults["of"], "a check fails or a fault is missed"
    hjb_claims(h)


HORIZONS = ("0.5 day", "1 day", "5 days")


def hjb_cases(h):
    """The approximation's cases with beta = 0, by option and horizon."""
    return {opt: {hz: cases[hz] for hz in HORIZONS} for opt, cases in h["cases"].items()}


def worst_gap(case):
    """The largest quote gap, in units of 1/kappa, where the book is at the last step (its 1st to
    99th percentile of inventory under the exact policy), bid or ask."""
    g = case["largest_gap_last_step_p01_to_p99"]
    return max(abs(g["bid"]["gap"]), abs(g["ask"]["gap"]))


def hjb_claims(h):
    """The approximation section's sentences: tiny at the half day, larger by five days, and the
    exact solution's own checks (the grid is wide enough, the exact policy valued is h)."""
    assert h is not None, "hjb_results.json is missing"
    for opt, cases in hjb_cases(h).items():
        for hz, c in cases.items():
            assert c["value_lost"] >= -1e-9, f"{opt} {hz}: the closed form beats the exact solution"
            assert abs(c["exact_quotes_evaluated_minus_h00"]) < 1e-6, f"{opt} {hz}: the exact policy is not worth h"
            assert c["h_moves_when_Q_doubles"] < 1e-6, f"{opt} {hz}: the inventory grid is too narrow"
        assert cases["0.5 day"]["value_lost_share"] < 1e-4, f"{opt}: the half day loses more than a ten-thousandth"


def numbers(r, h, checks, faults):
    n = Numbers()
    n["date"] = day(dt.date.today().isoformat())
    n["repo"], n["repo-url"] = REPO, "https://" + REPO
    n["page"], n["page-url"] = PAGE, "https://" + PAGE
    a, b, c, d = r["A"], r["B"], r["C"]["strategies"], r["D"]
    s = a["strategies"]
    n["a-base-mean"] = num(s["optimal baseline"]["value"]["mean"], 1)
    n["fig-base"] = str(study.FIGURE_1B["optimal baseline"][0])
    n["a-rules"] = str(len(a["verdict"]))
    n["a-paths"] = f"{r['settings']['paths_A']:,}".replace(",", "{,}")
    rows = []
    for key, label in NAMES.items():
        v = s[key]["value"]
        rows.append(f"{label} & {study.FIGURE_1B[key][0]} & {num(v['mean'], 1)} & {num(v['std'])} & {num(s[key]['trades'], 0)} \\\\")
    n["t1-rows"] = "\n".join(rows)
    n["zi01-vega"] = num(sum(s["ZI 0.1% vega"]["bucket_vega"]["mean"]), 1)
    n["base-vega-close"] = num(c["optimal baseline"]["vega_rms"][-1], 1)
    sur = r["surface"]
    n["surface-points"] = str(sum(len(v) for v in study.surface.FIGURE_1A.values()))
    n["surface-resid"] = num(100 * sur["max_abs_residual_vs_figure_read"], 2)
    x = r["vix"]
    n["vix-days"] = num(x["changes"], 0)
    n["vix-first"], n["vix-last"] = day(x["first"]), day(x["last"])
    n["vix-half"] = num(100 * x["half_day_std_vol"], 2)
    n["d-days"] = num(d["changes"], 0)
    n["d-slope"] = num(d["slope"], 2)
    n["d-corr"] = num(d["corr"], 2)
    per = list(d["slope_by_period"].values())
    n["d-low"], n["d-high"] = num(min(per), 2), num(max(per), 2)
    for key, tau in (("load-3m", "0.25"), ("load-6m", "0.5"), ("load-1y", "1.0"), ("load-3y", "3.0")):
        n[key] = num(d["loading"][tau], 2)
    n["same-trades"] = "checked on every path" if b["same_trades_in_every_scenario"] else "NOT SO"
    sc = b["scenarios"]
    n["skew-std"] = "standard deviation " + num(sc["skew"]["optimal baseline"]["value"]["std"]) + " against " + \
        num(sc["still"]["optimal baseline"]["value"]["std"]) + " still"
    n["b-paths"] = num(r["settings"]["paths_B"], 0)
    for key, scen in (("b-still-std", "still"), ("b-term-std", "term"), ("b-par-std", "parallel")):
        n[key] = num(sc[scen]["optimal baseline"]["value"]["std"])
    rows = []
    for key, label in (("optimal baseline", "optimal, the paper's $\\alpha$"), ("ZI 0.1% vega", "zero-intelligence, 0.1\\% vega")):
        for scen, where in (("still", "still surface"), ("term", "moving, by maturity"), ("parallel", "moving, parallel")):
            v = sc[scen][key]["value"]
            rows.append(f"{label}, {where} & {num(v['mean'], 1)} & {num(v['std'])} & {num(v['q01'], 1)} & {num(v['es01'], 1)} \\\\")
        rows.append("\\hline")
    n["t2-rows"] = "\n".join(rows[:-1])
    for key, name in (("x10", X10), ("x100", X100), ("rt", RT), ("rtot", RTOT)):
        e = c[name]
        n[f"{key}-cost"] = num(e["cost"])
        n[f"{key}-in-term-cut"] = pct(e["in_session_term_cut"])
        n[f"{key}-carried-term-cut"] = pct(e["carried_term_cut"])
        n[f"{key}-carried-par-cut"] = pct(e["carried_parallel_cut"])
    mid = len(c["optimal baseline"]["vega_rms"]) // 2 - 1  # the snapshot at half the session
    n["base-mid"] = num(c["optimal baseline"]["vega_rms"][mid], 1)
    n["x10-mid"] = num(c[X10]["vega_rms"][mid], 1)
    n["x10-close"] = num(c[X10]["vega_rms"][-1], 1)
    n["x10-gap"] = pct(x10_gap(c))
    n["rtot-mid"] = num(c[RTOT]["vega_rms"][mid], 1)
    rows = []
    table = ((X10, "terminal, $\\alpha\\times 10$"), (X100, "terminal, $\\alpha\\times 100$"),
             ("running buckets 1 alpha/T", "running, the paper's buckets, $\\alpha_k/T$"),
             (RTOT, "running, the book's whole vega, $0.2/T$"), (RT, "running, the vega the move loads on, $0.2/T$"),
             ("terminal x10 + running total 0.2/T", "terminal $\\times 10$ and running on the whole vega"))
    for name, label in table:
        e = c[name]
        rows.append(f"{label} & {num(e['cost'])} & {pct(e['in_session_term_cut'])} & {pct(e['in_session_parallel_cut'])}"
                    f" & {pct(e['carried_term_cut'])} & {pct(e['carried_parallel_cut'])} \\\\")
    n["t3-rows"] = "\n".join(rows)
    hjb_numbers(n, h)
    n["checks-line"] = (f"{checks['held']} checks of the code hold, and all {faults['caught']} faults planted in it on purpose, "
                        "one at a time, are caught.")
    return n


def tiny(x):
    """A small share as a percentage, two significant figures, never in scientific notation."""
    p = 100 * x
    if p < 1e-4:
        return "less than 0.0001\\%"
    digits = max(0, 1 - int(np.floor(np.log10(p))))
    s = f"{p:.{digits}f}"
    return (s.rstrip("0").rstrip(".") if "." in s else s) + "\\%"


def ppm(x):
    """A small share in parts per million, two significant figures (below 0.01 said so)."""
    v = 1e6 * x
    if v < 0.01:
        return "$<$0.01"
    digits = max(0, 1 - int(np.floor(np.log10(v))))
    s = f"{v:.{digits}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


HJB_NAMES = {"3m ATM (bucket 2)": "3 months, at the money", "3m 90% (bucket 1)": "3 months, 90\\% strike",
             "1y ATM (bucket 5)": "1 year, at the money", "3y 110% (bucket 6)": "3 years, 110\\% strike"}


def hjb_numbers(n, h):
    """The approximation's sentences and table, from hjb_results.json."""
    cases = hjb_cases(h)
    half = max(c["0.5 day"]["value_lost_share"] for c in cases.values())
    five = max(c["5 days"]["value_lost_share"] for c in cases.values())
    gap = max(worst_gap(c["0.5 day"]) for c in cases.values())
    n["hjb-summary"] = (f"Against the exact HJB \\eq{{6}}, solved for one option at a time, the closed form's quotes give up at "
                        f"most {tiny(half)} of the value over the half day and {tiny(five)} over five days.")
    rows = []
    for opt, label in HJB_NAMES.items():
        c = cases[opt]
        inv = c["0.5 day"]["inventory_exact"]
        rows.append(f"{label} & {' & '.join(ppm(c[hz]['value_lost_share']) for hz in HORIZONS)} & "
                    f"{tiny(worst_gap(c['0.5 day']))} & {num(inv['p01'], 0)} to {num(inv['p99'], 0)} \\\\")
    table = ("\\begin{center}\\small\n\\begin{tabular}{lrrrrr}\n\\heavyrule\n"
             "& \\multicolumn{3}{c}{value given up, parts per million} & quote gap at the close, & inventory, \\\\\n"
             "option & half day & one day & five days & share of $1/\\kappa$ & 1st to 99th percentile \\\\\n\\hline\n"
             + "\n".join(rows) + "\n\\heavyrule\n\\end{tabular}\n\\end{center}")
    n["hjb-text"] = (
        "The quotes \\eq{12} rest on a second-order expansion of the exponentials in \\eq{6}, approximation \\eq{7}; the paper's "
        "footnote calls them very satisfactory over a short horizon. For one option at a time, with gamma frozen as \\eq{20} "
        "freezes it, I solved \\eq{6} exactly on an inventory grid wide enough that doubling it changes nothing, valued the "
        "closed form's quotes by the same equation's linear twin, and set the two side by side at the paper's parameters. "
        "Where the book actually is at the close, the first to ninety-ninth percentile of its inventory, the two sets of "
        f"quotes differ by at most {tiny(gap)} of $1/\\kappa$, the spread's own scale.\n" + table)


def figures(r):
    """Two panels: the book's vega through the session, and each penalty's cost against the
    in-session risk it leaves (moves by maturity)."""
    plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                         "axes.unicode_minus": False, "axes.formatter.use_mathtext": True, "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False})
    still, terminal, running, light = "#8A94A3", "#B5541C", "#1F6FB2", "#8DB3D9"
    c = r["C"]["strategies"]
    frac = [0.0] + r["B"]["scenarios"]["still"]["optimal baseline"]["vega_path"]["fraction"]
    fig, ax = plt.subplots(figsize=(3.3, 2.25))
    for name, label, color, style in (("optimal baseline", "the paper's penalty", still, "-"),
                                      (X10, "terminal $\\times 10$", terminal, "-"),
                                      (RTOT, "running, whole vega, $0.2/T$", light, "--"),
                                      (RT, "running, loaded vega, $0.2/T$", running, "-")):
        ax.plot(frac, [0.0] + c[name]["vega_rms"], color=color, lw=1.4, ls=style, label=label)
    ax.set_xlabel("share of the half day")
    ax.set_ylabel("percentage-vega, root mean square")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.45 * max(max(c[k]["vega_rms"]) for k in ("optimal baseline", X10)))
    ax.legend(loc="upper center", ncol=2, fontsize=6.3, handlelength=1.8, columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig-vega.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.3, 2.25))

    def line(names, color, label, marker="o"):
        xs = [c[k]["cost"] for k in names]
        ys = [c[k]["in_session_term"]["std"] for k in names]
        ax.plot(xs, ys, color=color, lw=1.2, marker=marker, ms=3, label=label)

    line(["optimal low (x0.1)", "optimal baseline", "terminal x3", X10, "terminal x30", X100], terminal, "terminal, $\\alpha\\times 0.1$ to $\\times 100$")
    line(["optimal baseline"] + [f"running total {x:g}/T" for x in r["C"]["total"]], light, "running, whole vega")
    line(["optimal baseline"] + [f"running term {x:g}/T" for x in r["C"]["term"]], running, "running, loaded vega")
    line(["optimal baseline"] + [f"running buckets {m:g} alpha/T" for m in r["C"]["buckets"]], still, "running, the paper's buckets", "s")
    base = c["optimal baseline"]
    ax.plot([base["cost"]], [base["in_session_term"]["std"]], "o", color="black", ms=3.5, zorder=5)
    ax.annotate("the paper's example", (base["cost"], base["in_session_term"]["std"]), xytext=(7, -11),
                textcoords="offset points", fontsize=6.5)
    ax.set_xlabel("cost: mean value given up on a still surface")
    ax.set_ylabel("in-session risk left (std)")
    top = max(c[k]["in_session_term"]["std"] for k in c)
    ax.set_ylim(0, 1.5 * top)
    ax.legend(loc="upper right", ncol=2, fontsize=6.3, handlelength=1.8, columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig-frontier.pdf"))
    plt.close(fig)


def lint(tex, n):
    """Every digit in the body must sit in \\val, \\eq, \\paper or \\given (and a \\given must be in
    GIVEN); every \\val must be a macro this build wrote."""
    body = tex.split("\\begin{document}", 1)[1]
    body = re.sub(r"(?<!\\)%.*", "", body)
    problems = [f"\\val{{{k}}} has no value" for k in re.findall(r"\\val\{([^}]*)\}", body) if k not in n.values]
    problems += [f"\\given{{{g}}} is not a parameter of the paper" for g in re.findall(r"\\given\{([^}]*)\}", body) if g not in GIVEN]
    s = re.sub(r"\\href\{[^}]*\}", "", body)
    for name in NAMED:  # names that hold a digit: an index, a series
        s = s.replace(name, "")
    s = re.sub(r"\\(val|eq|given|paper|url|label|ref)\{[^}]*\}", "", s)
    s = re.sub(r"\\includegraphics\[[^\]]*\]\{[^}]*\}", "", s)
    s = re.sub(r"\\(vspace|hspace|setlength\{[^}]*\})\{[^}]*\}", "", s)
    s = re.sub(r"\\multicolumn\{\d+\}", "", s)
    for m in re.finditer(r"\d", s):
        problems.append("a number typed by hand: ..." + s[max(0, m.start() - 50):m.end() + 20].replace("\n", " ") + "...")
        break
    return problems


def latex(name):
    """pdflatex twice, never installing a package (a missing one fails the build); the page count."""
    for _ in range(2):
        done = subprocess.run(["pdflatex", "--disable-installer", "-interaction=nonstopmode", "-halt-on-error", name],
                              cwd=OUT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if done.returncode != 0:
            raise SystemExit("pdflatex failed:\n" + "\n".join(done.stdout.splitlines()[-30:]))
    log = open(os.path.join(OUT, name[:-4] + ".log"), encoding="utf-8", errors="replace").read()
    m = re.search(r"Output written on .*?\((\d+) pages?", log)
    undefined = re.findall(r"\?\?[a-z0-9-]+", open(os.path.join(OUT, name), encoding="utf-8").read())
    return int(m.group(1)) if m else None, undefined


def counts():
    """The checks that hold and the planted faults caught, over both suites: checks.py with
    mutations.py, and hjb_checks.py with hjb_mutations.py (run apart, the HJB's being slow)."""
    checks, faults = load("checks.json"), load("mutations.json")
    hc, hf = load("hjb_checks.json"), load("hjb_mutations.json")
    return ({"held": checks["held"] + hc["held"], "of": checks["of"] + hc["of"]},
            {"caught": faults["caught"] + hf["caught"], "of": faults["of"] + hf["of"]})


def main():
    r = load("results.json")
    h = load("hjb_results.json") if os.path.exists(os.path.join(HERE, "hjb_results.json")) else None
    checks, faults = counts()
    claims(r, h, checks, faults)
    n = numbers(r, h, checks, faults)
    tex = open(os.path.join(OUT, "note.tex"), encoding="utf-8").read()
    problems = lint(tex, n)
    if problems:
        raise SystemExit("the note's text fails its lint:\n  " + "\n  ".join(problems))
    with open(os.path.join(OUT, "numbers.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write(n.tex())
    figures(r)
    pages, _ = latex("note.tex")
    if pages is None or pages > PAGES:
        raise SystemExit(f"the note runs to {pages} pages, more than {PAGES}")
    shutil.copyfile(os.path.join(OUT, "note.pdf"), PDF_COPY)
    print(f"report/note.pdf: {pages} pages, {len(n.values)} numbers from the results; copied to {PDF_COPY}")


if __name__ == "__main__":
    main()
