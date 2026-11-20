"""Build the page, site/index.html, from the same results as the note: every number on it is read
from results.json, hjb_results.json and the checks' counts, and the claims build_report.py checks
are checked again before anything is written.

Run: python build_page.py   (writes site/index.html)
"""
import html
import json
import os
import re
import sys

import build_report as rep

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, "site")
PDF_URL = "https://vinhdac.github.io/notes/still-surface/Dac-Vinh-Nguyen-Still-Surface.pdf"
MINUS = "−"


def plain(tex):
    """A value written for the note, made fit for HTML: LaTeX's thousands brace, minus and percent."""
    s = tex.replace("{,}", ",").replace("$-$", MINUS).replace("\\%", "%").replace("\\_", "_").replace("$<$", "&lt;")
    s = re.sub(r"\$\\alpha\\times ?([0-9.]+)\$", r"&alpha;&times;\1", s)
    s = s.replace("$\\alpha$", "&alpha;").replace("$\\alpha_k/T$", "&alpha;<sub>k</sub>/T").replace("$\\times 10$", "&times;10")
    s = re.sub(r"\$([0-9.]+)/T\$", r"\1/T", s)
    s = re.sub(r"\\eq\{(\d+)\}", r"(\1)", s)
    assert "$" not in s and "\\" not in s, f"LaTeX left in {s!r}"
    return s


def rows(tex_rows):
    """The note's table rows, as HTML rows."""
    out = []
    for line in tex_rows.split("\n"):
        line = line.strip()
        if line == "\\hline":
            continue
        cells = [plain(c.strip()) for c in line.rstrip("\\").rstrip().rstrip("\\").split("&")]
        out.append("<tr>" + "".join(f"<td>{cells[0]}</td>" if i == 0 else f'<td class="num">{c}</td>'
                                    for i, c in enumerate(cells)) + "</tr>")
    return "\n".join(out)


def hjb_html(h):
    """Section 5 of the page: the closed form against the exact HJB, from hjb_results.json."""
    cases = rep.hjb_cases(h)
    gap = max(rep.worst_gap(c["0.5 day"]) for c in cases.values())
    body = []
    for opt, label in rep.HJB_NAMES.items():
        c = cases[opt]
        inv = c["0.5 day"]["inventory_exact"]
        cells = [plain(label)] + [plain(rep.ppm(c[hz]["value_lost_share"])) for hz in rep.HORIZONS] + \
            [plain(rep.tiny(rep.worst_gap(c["0.5 day"]))), plain(f"{rep.num(inv['p01'], 0)} to {rep.num(inv['p99'], 0)}")]
        body.append("<tr>" + "".join(f"<td>{x}</td>" if i == 0 else f'<td class="num">{x}</td>' for i, x in enumerate(cells)) + "</tr>")
    return (
        "<p>The quotes (12) rest on a second-order expansion of the exponentials in the HJB equation (6); the paper's footnote "
        "calls them very satisfactory over a short horizon. For one option at a time, with gamma frozen as (20) freezes it, I "
        "solved (6) exactly on an inventory grid wide enough that doubling it changes nothing, valued the closed form's quotes "
        "by the same equation's linear twin, and set the two side by side at the paper's parameters. Where the book actually "
        f"is at the close, the first to ninety-ninth percentile of its inventory, the quotes differ by at most "
        f'<span class="num">{plain(rep.tiny(gap))}</span> of 1/&kappa;, the spread\'s own scale.</p>\n'
        '<div class="scroll"><table><thead><tr><th>option</th><th>value given up, ppm: half day</th><th>one day</th>'
        "<th>five days</th><th>quote gap at the close, share of 1/&kappa;</th><th>inventory, 1st to 99th percentile</th></tr></thead>"
        "<tbody>" + "\n".join(body) + "</tbody></table></div>")


def readme(v):
    """README.md for the code's repository, from the same values as the page."""
    with open(os.path.join(HERE, "page", "README.tpl.md"), encoding="utf-8") as f:
        text = f.read()
    keys = dict(v, page_url="https://" + rep.PAGE)
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", text)) - set(keys))
    assert not missing, f"the README asks for {missing}"
    text = re.sub(r"\{\{(\w+)\}\}", lambda m: keys[m.group(1)], text)
    assert "&" not in text.replace("&amp;", ""), "an HTML entity in the README"
    return text


PICKS = (("optimal baseline", "the paper's penalty", "still"), (rep.X10, "terminal ×10", "terminal"),
         (rep.X100, "terminal ×100", "terminal"), ("running term 0.05/T", "running 0.05/T, loaded vega", "running"),
         (rep.RT, "running 0.2/T, loaded vega", "running"), (rep.RTOT, "running 0.2/T, whole vega", "running"))


def data(r):
    """What the drawings need, and nothing else."""
    c = r["C"]["strategies"]
    families = [{"key": "terminal", "label": "terminal α, ×0.1 to ×100",
                 "names": ["optimal low (x0.1)", "optimal baseline", "terminal x3", rep.X10, "terminal x30", rep.X100]},
                {"key": "total", "label": "running, whole vega",
                 "names": ["optimal baseline"] + [f"running total {x:g}/T" for x in r["C"]["total"]]},
                {"key": "term", "label": "running, loaded vega",
                 "names": ["optimal baseline"] + [f"running term {x:g}/T" for x in r["C"]["term"]]},
                {"key": "buckets", "label": "running, the paper's buckets",
                 "names": ["optimal baseline"] + [f"running buckets {m:g} alpha/T" for m in r["C"]["buckets"]]}]
    labels = {"optimal baseline": "the paper's example", "optimal low (x0.1)": "terminal ×0.1",
              rep.X10: "terminal ×10", "terminal x3": "terminal ×3", "terminal x30": "terminal ×30", rep.X100: "terminal ×100"}
    points = {}
    for fam in families:
        for i, name in enumerate(fam["names"]):
            e = c[name]
            label = labels.get(name) or re.sub(r"^running (total|term|buckets) ", lambda m: {
                "total": "running, whole vega, ", "term": "running, loaded vega, ", "buckets": "running, the paper's buckets, "}[m.group(1)],
                name).replace(" alpha/T", " α/T")
            p = points.setdefault(name, {"key": name, "label": label,
                                         "cost": round(e["cost"], 4), "risk": round(e["in_session_term"]["std"], 4),
                                         "cut": round(e["in_session_term_cut"], 4), "carried": round(e["carried_term_cut"], 4),
                                         "families": [], "order": {}})
            p["families"].append(fam["key"])
            p["order"][fam["key"]] = i
    return {"surface": [round(x, 6) for x in r["surface"]["params"]],
            "loading": [round(r["D"]["loading"][t], 4) for t in ("0.25", "0.5", "1.0", "3.0")],
            "half_std": round(r["vix"]["half_day_std_vol"], 5),
            "fraction": r["B"]["scenarios"]["still"]["optimal baseline"]["vega_path"]["fraction"],
            "vega": {k: [round(x, 3) for x in c[k]["vega_rms"]] for k, _, _ in PICKS},
            "picks": [{"key": k, "label": label, "kind": kind} for k, label, kind in PICKS],
            "families": [{"key": f["key"], "label": f["label"]} for f in families],
            "frontier": list(points.values())}


def main():
    r = rep.load("results.json")
    h = rep.load("hjb_results.json") if os.path.exists(os.path.join(HERE, "hjb_results.json")) else None
    checks, faults = rep.counts()
    rep.claims(r, h, checks, faults)
    n = rep.numbers(r, h, checks, faults)
    latex_only = {"t1-rows", "t2-rows", "t3-rows", "hjb-text"}
    v = {k.replace("-", "_"): plain(x) for k, x in n.values.items() if k not in latex_only}
    v["t1_rows"], v["t2_rows"], v["t3_rows"] = rows(n.values["t1-rows"]), rows(n.values["t2-rows"]), rows(n.values["t3-rows"])
    v["repo_url"], v["pdf_url"] = n.values["repo-url"], PDF_URL
    v["checks_held"], v["faults_of"] = str(checks["held"]), str(faults["of"])
    v["picks"] = "".join(f'<button type="button" data-key="{html.escape(k)}" data-kind="{kind}" aria-pressed="{str(i == 1).lower()}">{html.escape(label)}</button>'
                         for i, (k, label, kind) in enumerate(PICKS))
    v["hjb_html"] = hjb_html(h)
    v["data_json"] = json.dumps(data(r), separators=(",", ":"))
    with open(os.path.join(HERE, "page", "note.js"), encoding="utf-8") as f:
        v["script"] = f.read()
    with open(os.path.join(HERE, "page", "note.tpl.html"), encoding="utf-8") as f:
        page = f.read()
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", page)) - set(v))
    assert not missing, f"the page asks for {missing}"
    page = re.sub(r"\{\{(\w+)\}\}", lambda m: v[m.group(1)], page)
    os.makedirs(SITE, exist_ok=True)
    with open(os.path.join(SITE, "index.html"), "w", encoding="utf-8", newline="\n") as f:
        f.write(page)
    with open(os.path.join(HERE, "README.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(readme(v))
    print(f"site/index.html: {len(page):,} bytes; README.md written")


if __name__ == "__main__":
    main()
