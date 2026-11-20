"""Approximation (7) against the exact HJB (6), for one option at a time.

Lucic and Tse, "Option market-making and vol arbitrage", Risk.net 2025, author manuscript; the
equation numbers are its own. To reach the closed-form quotes (12) the paper replaces the
exponentials of (6) by their second-order expansion (7). Footnote 7 says (12) is therefore not
optimal, and that against the PDE it did very well at least over a short horizon; the conclusion
leaves the quality of the approximation to be examined more rigorously. This file measures it.

The problem solved is the one behind the multi-option closed form (20): gamma frozen at t = 0, so
the carry rate C(t, s) is the constant C0 = book.C0[n] and the value function depends on (t, q)
alone. The s-derivative term of (6) drops, and approximation (7) is all that separates (6) from
(12). Option n alone in its bucket: the penalty alpha_k (sum omega Q)^2 of section 3.2 becomes
alpha q^2 with alpha = alpha_k omega_n^2, and a running penalty beta_k becomes beta_k omega_n^2.

    exact(n, T)      (6) as 2Q+1 ODEs on q = -Q..Q from h(T, q) = -alpha q^2, and its quotes (5)
    closed(n, T)     psi2 by its closed form after (9), psi1 by the second equation of (9), quotes (12)
    evaluate(...)    the expected objective g(t, q) of given quotes, by the linear equation of the process
    inventory(...)   the law of q at T starting flat, by the forward (master) equation of the process
    compare(n, T)    the value (12) gives up against (6), its quote gaps, and where the book is

On the grid the book cannot buy at q = Q nor sell at q = -Q. All three equations are stiff at large
|q| (the exponentials grow like exp(2 kappa alpha |q|)), so they are solved by Radau with their
tridiagonal Jacobians; the backward ones in the time to go s = T - t, because near T the fastest
states move on time scales below the spacing of floating-point numbers near T.

Run: python hjb.py   (writes hjb_results.json and prints the tables; a couple of minutes)
"""
import json
import math
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # small problems: the cases run side by side instead

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.integrate import OdeSolution, Radau, solve_ivp

import model

HERE = os.path.dirname(os.path.abspath(__file__))
E = math.e
C = model.C  # the 2 e^-1 of (9)
RTOL, ATOL = 1e-10, 1e-11  # the backward equations, whose values are of a few units
LAW_RTOL, LAW_ATOL = 1e-9, 1e-14  # the forward equation, of probabilities
MAX_STEPS = 50_000  # Radau steps allowed in one run
Q_START, Q_MOST = 40, 640  # grids tried, doubling, until the answer near q = 0 stays put
NEAR = 20  # the answer that must not move: h(0, q) for |q| <= NEAR ...
Q_TOL = 1e-9  # ... by more than this share of its largest size there, when Q doubles
SHOWN = (-20, -10, -5, 0, 5, 10, 20)  # the inventories at which quote gaps are reported
DAYS = (0.5, 1.0, 5.0)  # the horizons, in trading days of 1/252 year
PICKS = {  # label: (K/S0, maturity in years); buckets numbered as the paper's Table 1, 1 to 6
    "3m ATM (bucket 2)": (1.00, 0.25),
    "3m 90% (bucket 1)": (0.90, 0.25),
    "1y ATM (bucket 5)": (1.00, 1.00),
    "3y 110% (bucket 6)": (1.10, 3.00),
}


@dataclass(frozen=True)
class Params:
    """One option's problem; alpha and beta are the effective penalties on q^2 (alpha_k omega^2, beta_k omega^2)."""
    lam_a: float
    lam_b: float
    kappa: float
    C0: float
    alpha: float
    beta: float = 0.0

    @property
    def D(self):
        """lam_a kappa + lam_b kappa, the D of psi2's closed form (kappa is the same on both sides)."""
        return self.lam_a * self.kappa + self.lam_b * self.kappa


_BOOK = []


def book():
    """The section 3.2 book, built once."""
    if not _BOOK:
        _BOOK.append(model.Book())
    return _BOOK[0]


def index(moneyness, tau):
    b = book()
    return int(np.flatnonzero(np.isclose(b.moneyness, moneyness) & np.isclose(b.tau, tau))[0])


def option(n, beta_k=0.0):
    """Option n of the book alone: its flow and kappa, its carry rate C0 (gamma frozen at t = 0), its
    bucket's alpha_k times omega_n^2, and beta_k times omega_n^2."""
    b = book()
    w2 = float(b.w[n]) ** 2
    return Params(float(b.lam_a[n]), float(b.lam_b[n]), float(b.kappa[n]), float(b.C0[n]),
                  model.ALPHA[int(b.bucket[n])] * w2, beta_k * w2)


def params(n, beta_k=0.0):
    """n is an option of the book (beta_k then sets its running penalty) or a Params of its own."""
    if isinstance(n, Params):
        if beta_k:
            raise ValueError("a Params carries its own beta")
        return n
    return option(int(n), beta_k)


def grid(Q):
    return np.arange(-Q, Q + 1, dtype=float)


def running(p, q):
    """The running gain of holding q: the carry C0 q less the running penalty beta q^2."""
    return p.C0 * q - p.beta * q * q


def _first_step(jac, s, y, span):
    """A first step the fastest state can follow: a tenth of 1 / (largest row sum of |J|)."""
    r = float(abs(jac(s, y)).sum(axis=1).max())
    return span if r == 0 else min(span, max(0.1 / r, 100 * np.spacing(s)))


def _radau(rhs, jac, lo, hi, y, rtol, atol):
    """One run of Radau from y at lo to hi, step by step as solve_ivp takes it, with its dense output.
    A run that needs more than MAX_STEPS steps has gone wrong (the hardest case here takes about
    6,500) and stops with an error rather than crawl on."""
    solver = Radau(rhs, lo, np.array(y, dtype=float), hi, rtol=rtol, atol=atol, jac=jac,
                   first_step=_first_step(jac, lo, y, hi - lo))
    ts, parts = [lo], []
    while solver.status == "running":
        if len(parts) == MAX_STEPS:
            raise RuntimeError(f"Radau took more than {MAX_STEPS} steps on [{lo:.4g}, {hi:.4g}]")
        message = solver.step()
        if solver.status == "failed":
            raise RuntimeError(f"Radau stopped on [{lo:.4g}, {hi:.4g}]: {message}")
        ts.append(solver.t)
        parts.append(solver.dense_output())
    return np.array(solver.y), OdeSolution(ts, parts), len(parts)


class Path:
    """y(s) for s in [0, S], solved by Radau from y(0) in pieces that end at the given stops, so that
    the values at the stops are the solver's own steps rather than its interpolation."""

    def __init__(self, rhs, jac, y0, S, stops=(), rtol=RTOL, atol=ATOL):
        self.S = float(S)
        cuts = sorted({0.0, self.S} | {float(s) for s in stops if 0.0 < s < S})
        y = np.asarray(y0, dtype=float)
        self.ends = {0.0: y}
        self.pieces = []
        self.steps = 0
        for lo, hi in zip(cuts, cuts[1:]):
            y, dense, steps = _radau(rhs, jac, lo, hi, y, rtol, atol)
            self.ends[hi] = y
            self.pieces.append((lo, hi, dense))
            self.steps += steps

    def __call__(self, s):
        for cut, y in self.ends.items():
            if abs(s - cut) <= 1e-14 * self.S:
                return y.copy()
        for lo, hi, f in self.pieces:
            if lo <= s <= hi:
                return f(s)
        raise ValueError(f"s = {s} is outside [0, {self.S}]")


class Backward:
    """The solution on q = -Q..Q of a backward equation from -alpha q^2 at T, read at any t of [0, T]."""

    def __init__(self, p, T, Q, rhs, jac, stops=()):
        self.p, self.T, self.Q, self.q = p, T, Q, grid(Q)
        self.path = Path(rhs, jac, -p.alpha * self.q ** 2, T, [T - t for t in stops])

    def at(self, t):
        return self.path(self.T - t)


# --- the exact problem (6) ---------------------------------------------------------------------

def hjb_equations(p, Q):
    """(6), gamma frozen, on q = -Q..Q in the time to go s = T - t (dh/dt is minus this), and its Jacobian:

        dh/ds(q) = C0 q - beta q^2 + lam_a/(e kappa) exp(-kappa (h(q) - h(q-1)))
                                   + lam_b/(e kappa) exp(-kappa (h(q) - h(q+1)))

    At q = Q the book cannot buy: no bid term. At q = -Q it cannot sell: no ask term."""
    q = grid(Q)
    run = running(p, q)
    k = p.kappa
    ca, cb = p.lam_a / (E * k), p.lam_b / (E * k)
    can_sell, can_buy = q > -Q, q < Q

    def terms(h):
        down, up = np.r_[h[0], h[:-1]], np.r_[h[1:], h[-1]]  # h(q - 1), h(q + 1); the pads are masked off
        with np.errstate(over="ignore"):
            ask = np.where(can_sell, ca * np.exp(-k * (h - down)), 0.0)
            bid = np.where(can_buy, cb * np.exp(-k * (h - up)), 0.0)
        return ask, bid

    def rhs(s, h):
        ask, bid = terms(h)
        return run + ask + bid

    def jac(s, h):
        ask, bid = terms(h)
        return sparse.diags_array([k * ask[1:], -k * (ask + bid), k * bid[:-1]], offsets=(-1, 0, 1), format="csc")

    return rhs, jac


class Exact(Backward):
    """h(t, q) of (6) on q = -Q..Q, and its optimal quotes (5)."""

    def __init__(self, p, T, Q, stops=()):
        super().__init__(p, T, Q, *hjb_equations(p, Q), stops)

    def quotes(self, t):
        """(5): bid 1/kappa + h(q) - h(q+1), ask 1/kappa + h(q) - h(q-1); none (inf) at q = Q, q = -Q."""
        h = self.at(t)
        bid, ask = np.full_like(h, np.inf), np.full_like(h, np.inf)
        bid[:-1] = 1 / self.p.kappa + h[:-1] - h[1:]
        ask[1:] = 1 / self.p.kappa + h[1:] - h[:-1]
        return bid, ask


def exact(n, T, beta_k=0.0, Q=Q_START, stops=()):
    """(6) solved on q = -Q..Q; stops are times t at which h is wanted at the solver's own accuracy."""
    return Exact(params(n, beta_k), T, Q, stops)


# --- the closed form (9) to (12) ---------------------------------------------------------------

class Closed:
    """The closed form of section 2.1, gamma frozen: psi2 after (9), psi1 by (9), quotes (12)."""

    def __init__(self, p, T):
        self.p, self.T = p, T
        self.cD = C * p.D
        sol = solve_ivp(self._psi1_slope, (0.0, T), [0.0], method="DOP853", rtol=1e-13, atol=1e-16,
                        dense_output=True)
        if sol.status != 0:
            raise RuntimeError(f"psi1: {sol.message}")
        self._psi1 = sol.sol

    def psi2(self, t):
        """psi2(t) by the closed form after (9). With s = T - t, Upsilon = 1/(c D), eta = sqrt(beta/Upsilon)
        and zeta+- = sqrt(Upsilon beta) +- alpha the paper writes it sqrt(Upsilon beta) (zeta- e^(-eta s)
        - zeta+ e^(eta s)) / (zeta- e^(-eta s) + zeta+ e^(eta s)); divided through by cosh(eta s) it is

            psi2 = -(alpha + beta s g) / (1 + alpha c D s g),    g = tanh(eta s) / (eta s), 1 at eta s = 0,

        with no division by zero at beta = 0, where it is -alpha / (1 + c alpha D (T - t))."""
        p = self.p
        s = np.asarray(self.T - t, dtype=float)
        x = math.sqrt(p.beta * self.cD) * s
        small = x < 1e-6
        g = np.where(small, 1.0 - x * x / 3.0, np.tanh(x) / np.where(small, 1.0, x))
        return -(p.alpha + p.beta * s * g) / (1.0 + p.alpha * self.cD * s * g)

    def _psi1_slope(self, s, y):
        """The second equation of (9) with the gamma term frozen, in s = T - t (dpsi1/dt is minus this):
        C0 + c D psi2 psi1 + c (lam_b - lam_a) psi2 + c (lam_b kappa - lam_a kappa) psi2^2."""
        p = self.p
        p2 = self.psi2(self.T - s)
        return [p.C0 + self.cD * p2 * y[0] + C * (p.lam_b - p.lam_a) * p2
                + C * (p.lam_b * p.kappa - p.lam_a * p.kappa) * p2 * p2]

    def psi1(self, t):
        """psi1(t), psi1(T) = 0, by DOP853 at a relative tolerance of 1e-13."""
        return self._psi1(np.asarray(self.T - t, dtype=float))[0]

    def quotes_at(self, t, q):
        """(12): bid 1/kappa - psi1 - (2q + 1) psi2, ask 1/kappa + psi1 + (2q - 1) psi2."""
        q = np.asarray(q, dtype=float)
        p1, p2 = self.psi1(t), self.psi2(t)
        bid = 1 / self.p.kappa - p1 - (2 * q + 1) * p2
        ask = 1 / self.p.kappa + p1 + (2 * q - 1) * p2
        return bid, ask

    def on(self, Q):
        """The quotes (12) on q = -Q..Q as a function of t, for evaluate and inventory."""
        q = grid(Q)
        return lambda t: self.quotes_at(t, q)


def closed(n, T, beta_k=0.0):
    return Closed(params(n, beta_k), T)


# --- any quotes: their value and where they take the book --------------------------------------

def fill_rates(p, Q, bid, ask):
    """Fill intensities lam e^(-kappa delta) of quotes on q = -Q..Q; none on the sides the grid closes."""
    q = grid(Q)
    with np.errstate(over="ignore"):
        ra = np.where(q > -Q, p.lam_a * np.exp(-p.kappa * ask), 0.0)
        rb = np.where(q < Q, p.lam_b * np.exp(-p.kappa * bid), 0.0)
    return ra, rb


def generator(ra, rb):
    """The generator of q under fill rates ra (a sale takes q to q - 1) and rb (a purchase takes q to
    q + 1), as its three diagonals: below it ra[1:], on it -(ra + rb), above it rb[:-1]. The value
    equation uses it and the forward equation its transpose, each with it as its Jacobian."""
    return ra[1:], -(ra + rb), rb[:-1]


def tridiagonal(lower, diag, upper, y):
    """The product with y of the tridiagonal matrix of these three diagonals."""
    out = diag * y
    out[1:] += lower * y[:-1]
    out[:-1] += upper * y[1:]
    return out


def value_equations(p, Q, quotes, T):
    """The expected objective of given quotes, in s = T - t (dg/dt is minus this), and its Jacobian:

        dg/ds(q) = C0 q - beta q^2 + lam_a e^(-kappa da) (da + g(q-1) - g(q)) + lam_b e^(-kappa db) (db + g(q+1) - g(q))

    with (db, da) = quotes(t), both arrays on the grid; no ask at q = -Q, no bid at q = Q."""
    run = running(p, grid(Q))

    def parts(s):
        bid, ask = quotes(T - s)
        ra, rb = fill_rates(p, Q, bid, ask)
        with np.errstate(invalid="ignore"):
            earn = np.where(ra > 0, ra * ask, 0.0) + np.where(rb > 0, rb * bid, 0.0)
        return generator(ra, rb), run + earn

    def rhs(s, g):
        bands, gain = parts(s)
        return tridiagonal(*bands, g) + gain

    def jac(s, g):
        return sparse.diags_array(list(parts(s)[0]), offsets=(-1, 0, 1), format="csc")

    return rhs, jac


def evaluate(n, T, quotes, beta_k=0.0, Q=Q_START, stops=()):
    """g(t, q): the expected objective of following quotes(t) = (bid, ask) arrays on q = -Q..Q from (t, q)
    to T, under the same carry, penalties and grid as (6). Quotes may be negative."""
    p = params(n, beta_k)
    return Backward(p, T, Q, *value_equations(p, Q, quotes, T), stops)


def law_equations(p, Q, quotes):
    """The forward (master) equation of q under given quotes, in t, and its Jacobian: the transpose
    of the generator,

        dP(q)/dt = rb(q-1) P(q-1) + ra(q+1) P(q+1) - (ra(q) + rb(q)) P(q)."""

    def bands(t):
        lower, diag, upper = generator(*fill_rates(p, Q, *quotes(t)))
        return upper, diag, lower  # transposed: into q come purchases from q - 1 and sales from q + 1

    def rhs(t, P):
        return tridiagonal(*bands(t), P)

    def jac(t, P):
        return sparse.diags_array(list(bands(t)), offsets=(-1, 0, 1), format="csc")

    return rhs, jac


def inventory_at(n, T, quotes, Q, times):
    """The law of q on q = -Q..Q at each of the given times of [0, T], for a book that starts flat at
    t = 0 and follows quotes(t)."""
    p = params(n)
    P0 = np.zeros(2 * Q + 1)
    P0[Q] = 1.0
    path = Path(*law_equations(p, Q, quotes), P0, T, times, rtol=LAW_RTOL, atol=LAW_ATOL)
    return [path(t) for t in times]


def inventory(n, T, quotes, Q):
    """The law of q at T on q = -Q..Q for a book that starts flat at t = 0 and follows quotes(t)."""
    return inventory_at(n, T, quotes, Q, (T,))[0]


def percentiles(q, P, *levels):
    """The smallest q at which the distribution P on q reaches each level."""
    cdf = np.cumsum(P)
    return [int(q[min(int(np.searchsorted(cdf, x)), q.size - 1)]) for x in levels]


def moments(q, P):
    mean = float(P @ q)
    return mean, float(math.sqrt(max(P @ (q - mean) ** 2, 0.0)))


# --- the comparison ----------------------------------------------------------------------------

def choose_Q(p, T, stops=()):
    """The first Q of 40, 80, 160, ... on which h(0, q) for |q| <= NEAR moves by at most Q_TOL of its
    largest size there when Q doubles. Returns Q, the solutions on Q and 2Q, and that move."""
    Q = Q_START
    ex = Exact(p, T, Q, stops)
    while True:
        ex2 = Exact(p, T, 2 * Q, stops)
        a = ex.at(0.0)[Q - NEAR:Q + NEAR + 1]
        b = ex2.at(0.0)[2 * Q - NEAR:2 * Q + NEAR + 1]
        moved = float(np.max(np.abs(a - b)))
        if moved <= Q_TOL * float(np.max(np.abs(b))):
            return Q, ex, ex2, moved
        if 2 * Q >= Q_MOST:
            raise RuntimeError(f"h(0, q) still moves by {moved:.3g} when Q doubles to {2 * Q}")
        Q, ex = 2 * Q, ex2


def compare(n, T, beta_k=0.0):
    """How (12) does against (6) for option n over [0, T]: the values at (0, 0), the quote gaps
    kappa (delta_closed - delta_exact) at q in SHOWN at t = 0, T/2 and the last 15-second step, and
    where the book is under each policy (at T, and at T/2 for context)."""
    t0 = time.time()
    p = params(n, beta_k)
    times = {"t = 0": 0.0, "t = T/2": T / 2, "last step": T - model.DT}
    Q, ex, ex2, moved = choose_Q(p, T, tuple(times.values()))
    cf = Closed(p, T)
    q = grid(Q)
    h00, h00_2Q = float(ex.at(0.0)[Q]), float(ex2.at(0.0)[2 * Q])
    g00 = float(evaluate(p, T, cf.on(Q), Q=Q).at(0.0)[Q])
    g00_2Q = float(evaluate(p, T, cf.on(2 * Q), Q=2 * Q).at(0.0)[2 * Q])
    own00 = float(evaluate(p, T, ex.quotes, Q=Q).at(0.0)[Q])  # the exact quotes, evaluated: h again
    gap, gaps = {}, {}
    for name, t in times.items():
        eb, ea = ex.quotes(t)
        cb, ca = cf.on(Q)(t)
        gap[name] = (p.kappa * (cb - eb), p.kappa * (ca - ea))
        gaps[name] = {side: {f"q={j}": float(g[Q + j]) for j in SHOWN} for side, g in zip(("bid", "ask"), gap[name])}
    mid, law = inventory_at(p, T, ex.quotes, Q, (T / 2, T))
    mid_cf, law_cf = inventory_at(p, T, cf.on(Q), Q, (T / 2, T))

    def largest(pair, P):
        """The gap of largest size on each side over q from the 1st to the 99th percentile of P."""
        lo, hi = percentiles(q, P, 0.01, 0.99)
        inside = (q >= lo) & (q <= hi)
        out = {}
        for side, g in zip(("bid", "ask"), pair):
            j = int(np.argmax(np.where(inside & np.isfinite(g), np.abs(g), -1.0)))
            out[side] = {"gap": float(g[j]), "q": int(q[j])}
        return out

    def where(P, P_mid):
        a, b = percentiles(q, P, 0.01, 0.99)
        mean, std = moments(q, P)
        c, d = percentiles(q, P_mid, 0.01, 0.99)
        return {"p01": a, "p99": b, "mean": mean, "std": std, "mass": float(P.sum()), "p01_at_T/2": c, "p99_at_T/2": d}

    return {
        "T_years": T, "T_days": T * 252, "beta_k": beta_k,
        "params": {"lam_a": p.lam_a, "lam_b": p.lam_b, "kappa": p.kappa, "C0": p.C0, "alpha": p.alpha,
                   "beta": p.beta, "kappa_alpha": p.kappa * p.alpha, "c_alpha_D_T": C * p.alpha * p.D * T,
                   "kappa_C0_T": p.kappa * p.C0 * T},
        "Q": Q, "h_moves_when_Q_doubles": moved,
        "h00": h00, "g00": g00, "value_lost": h00 - g00, "value_lost_share": (h00 - g00) / h00,
        "value_lost_on_2Q": h00_2Q - g00_2Q, "exact_quotes_evaluated_minus_h00": own00 - h00,
        "quote_gap": gaps,
        "inventory_exact": where(law, mid),
        "inventory_closed": where(law_cf, mid_cf),
        "largest_gap_last_step_p01_to_p99": largest(gap["last step"], law),
        "largest_gap_at_T/2_over_p01_to_p99_at_T/2": largest(gap["t = T/2"], mid),
        "steps": {"exact_Q": ex.path.steps, "exact_2Q": ex2.path.steps},
        "seconds": round(time.time() - t0, 2),
    }


def horizon(days, beta_k):
    return f"{days:g} day{'s' if days > 1 else ''}" + (", beta = alpha/T" if beta_k else "")


def cases():
    """(label, n, T, beta_k) of every run: each option at each horizon with beta = 0, and at half a day
    with beta_k = alpha_k / T."""
    out = []
    for label, (m, tau) in PICKS.items():
        n = index(m, tau)
        for d in DAYS:
            out.append((label, n, d / 252, 0.0))
        out.append((label, n, model.T, model.ALPHA[int(book().bucket[n])] / model.T))
    return out


def job(case):
    label, n, T, beta_k = case
    return label, horizon(T * 252, beta_k), compare(n, T, beta_k)


def main():
    t0 = time.time()
    todo = sorted(cases(), key=lambda c: (option(c[1]).kappa * option(c[1]).alpha, c[2]), reverse=True)  # slowest first
    with ProcessPoolExecutor(max_workers=min(len(todo), os.cpu_count() or 1)) as pool:
        done = {(label, key): out for label, key, out in pool.map(job, todo)}
    b = book()
    options = {}
    for label, (m, tau) in PICKS.items():
        n = index(m, tau)
        options[label] = {"n": n, "moneyness": m, "tau": tau, "bucket": int(b.bucket[n]) + 1,
                          "alpha_k": model.ALPHA[int(b.bucket[n])], "omega": float(b.w[n]),
                          "vol0": float(b.vol0[n])}
    results = {
        "paper": "Lucic and Tse, Option market-making and vol arbitrage, Risk.net 2025, author manuscript",
        "what": "approximation (7): quotes (12) against the exact HJB (6) for one option, gamma frozen "
                "(C = C0, as (20) assumes), alpha = alpha_k omega^2, on q = -Q..Q",
        "units": "values in the book's currency; quote gaps kappa (delta_closed - delta_exact), in units of 1/kappa",
        "settings": {"rtol": RTOL, "atol": ATOL, "law_rtol": LAW_RTOL, "law_atol": LAW_ATOL,
                     "Q_start": Q_START, "near": NEAR, "Q_tol": Q_TOL, "dt_years": model.DT,
                     "last_step_t": "T - dt", "shown_q": list(SHOWN)},
        "options": options,
        "cases": {},
        "seconds": None,
    }
    for label, n, T, beta_k in cases():  # in the order of PICKS and DAYS
        key = horizon(T * 252, beta_k)
        results["cases"].setdefault(label, {})[key] = done[(label, key)]
    results["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(HERE, "hjb_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    report(results)


def report(r):
    sys.stdout.reconfigure(encoding="utf-8")
    print("Approximation (7): the closed-form quotes (12) against the exact HJB (6), one option, gamma frozen")
    print(f"{'option':20s} {'horizon':23s} {'Q':>4s} {'h(0,0)':>12s} {'g(0,0)':>12s} {'lost':>9s} "
          f"{'share':>9s} {'q at T p1..p99':>14s} {'largest gap there, last step: bid (q) ask (q)':>45s} {'s':>6s}")
    for label, block in r["cases"].items():
        for key, c in block.items():
            lg = c["largest_gap_last_step_p01_to_p99"]
            inv = c["inventory_exact"]
            print(f"{label:20s} {key:23s} {c['Q']:4d} {c['h00']:12.7f} {c['g00']:12.7f} {c['value_lost']:9.2e} "
                  f"{c['value_lost_share']:9.2e} {inv['p01']:7d}..{inv['p99']:<5d} "
                  f"{lg['bid']['gap']:+15.2e} ({lg['bid']['q']:+3d}) {lg['ask']['gap']:+11.2e} ({lg['ask']['q']:+3d}) "
                  f"{c['seconds']:6.1f}")
    print("\nquote gaps kappa (delta_closed - delta_exact), in units of 1/kappa, at q = "
          + " ".join(f"{j:+d}" for j in SHOWN))
    for label, block in r["cases"].items():
        for key, c in block.items():
            ie, ic = c["inventory_exact"], c["inventory_closed"]
            lm = c["largest_gap_at_T/2_over_p01_to_p99_at_T/2"]
            print(f"  {label}, {key}: Q = {c['Q']}; q at T under (5) mean {ie['mean']:+.2f} std {ie['std']:.2f}, "
                  f"under (12) mean {ic['mean']:+.2f} std {ic['std']:.2f}; q at T/2 p1..p99 under (5) "
                  f"{ie['p01_at_T/2']}..{ie['p99_at_T/2']}, largest gap there at T/2: bid {lm['bid']['gap']:+.2e} "
                  f"({lm['bid']['q']:+d}), ask {lm['ask']['gap']:+.2e} ({lm['ask']['q']:+d})")
            for when, sides in c["quote_gap"].items():
                for side, row in sides.items():
                    print(f"    {when:10s} {side:3s} " + " ".join(f"{v:+10.2e}" for v in row.values()))
    print(f"\n{r['seconds']} s")


if __name__ == "__main__":
    main()
