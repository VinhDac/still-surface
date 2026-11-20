"""Daily closes of the VIX from FRED (series VIXCLS, Cboe), cached in data/, which is not
redistributed. They size the parallel shift of the surface: a day's change of the VIX, in
vol points, scaled to half a day. The VIX3M (series VXVCLS, Cboe's 3-month index) beside it
says how much less a longer maturity moves (term_loading).

One honest request with a plain client name. If FRED refuses it, this module says so and
stops; nothing tries another way in.
"""
import csv
import io
import math
import os
import urllib.error
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data")
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
PATH = os.path.join(CACHE, "VIXCLS.csv")
HEADERS = {"User-Agent": "still-surface-validation/1.0 (research script)"}


class Unavailable(RuntimeError):
    pass


URL_3M = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VXVCLS"
PATH_3M = os.path.join(CACHE, "VXVCLS.csv")
DAYS_VIX, DAYS_3M = 30, 93  # the calendar days each index looks ahead (Cboe's definitions)


def fetch(series="VIXCLS"):
    url, path = (URL, PATH) if series == "VIXCLS" else (URL_3M, PATH_3M)
    if os.path.exists(path):
        return path
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
            body = r.read().decode("utf-8")
    except (urllib.error.URLError, OSError) as e:
        raise Unavailable(f"FRED did not give {series} ({e}); the sizing of the surface's move stops here") from e
    if series not in body.splitlines()[0]:
        raise Unavailable(f"FRED answered, but not with the {series} series; the sizing stops here")
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(body)
    return path


def closes_of(series):
    """Dates (ISO strings) and closes, the days FRED has a value for."""
    with open(fetch(series), encoding="utf-8") as f:
        rows = list(csv.reader(io.StringIO(f.read())))
    dates, closes = [], []
    for row in rows[1:]:
        if len(row) == 2 and row[1] not in ("", "."):
            dates.append(row[0])
            closes.append(float(row[1]))
    return dates, np.array(closes)


def vix():
    return closes_of("VIXCLS")


def paired_changes():
    """The VIX's and the VIX3M's changes over the same pairs of consecutive days, the days both
    indices closed (a day either misses is skipped by both)."""
    d1, c1 = vix()
    d3, c3 = closes_of("VXVCLS")
    both = sorted(set(d1) & set(d3))
    one, three = dict(zip(d1, c1)), dict(zip(d3, c3))
    a = np.array([one[d] for d in both])
    b = np.array([three[d] for d in both])
    return both, np.diff(a), np.diff(b)


def term_loading(slope=None):
    """How far an implied vol of maturity tau moves when the VIX moves one point. The slope of the
    VIX3M's daily changes on the VIX's, the same days, is the loading at 93 days; a power law
    through it, (tau / 30 days) ** -p, carries it to every maturity. Past 93 days that is an
    ASSUMPTION: no index of longer maturity is read. Returns the function, the slope and p."""
    if slope is None:
        _, x, y = paired_changes()
        slope = float(np.cov(x, y)[0, 1] / np.var(x, ddof=1))
    p = -math.log(slope) / math.log(DAYS_3M / DAYS_VIX)

    def loading(tau):
        return (np.asarray(tau, dtype=float) * 365 / DAYS_VIX) ** -p

    return loading, slope, p


def daily_changes():
    """Close-to-close changes between consecutive trading days, in vol points."""
    dates, closes = vix()
    return dates, np.diff(closes)


def changes_from(lo, hi, closes=None):
    """Changes on the days after a close within [lo, hi): how far the VIX moves from about
    the level of a given surface."""
    closes = vix()[1] if closes is None else np.asarray(closes, dtype=float)
    prev, changes = closes[:-1], np.diff(closes)
    return changes[(prev >= lo) & (prev < hi)]


def half_day(changes):
    """A day's changes scaled to half a day, as a Brownian move scales: by sqrt(1/2).
    In decimals of vol (one VIX point is 0.01)."""
    return np.asarray(changes) * math.sqrt(0.5) / 100
