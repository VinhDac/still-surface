"""Checks of hjb.py: each one states a rule the comparison of the quotes (12) with the exact HJB (6)
depends on, and fails if the rule is broken.

Run: python hjb_checks.py   (exit code 1 if any check fails; hjb_mutations.py breaks rules on purpose
on copies of this folder and expects a check to fail each time)
"""
import dataclasses
import math
import sys
import types

import numpy as np
from scipy import linalg, stats

import hjb
import model

sys.stdout.reconfigure(encoding="utf-8")
CHECKS = []
T = model.T
E = math.e
ATM = hjb.index(1.00, 0.25)  # 3m at the money, bucket 2: a light penalty
WING = hjb.index(0.90, 0.25)  # 3m 90%, bucket 1: a heavy penalty and the largest edge
LONG = hjb.index(1.10, 3.00)  # 3y 110%, bucket 6: the heaviest penalty in units of kappa


def check(fn):
    CHECKS.append(fn)
    return fn


def with_beta(n):
    """Option n with the running penalty of the study's beta run, beta_k = alpha_k / T."""
    return hjb.option(n, model.ALPHA[int(hjb.book().bucket[n])] / T)


def constant(Q, db, da):
    """Quotes that never change: db on the bid and da on the ask at every q of -Q..Q."""
    return lambda t: (np.full(2 * Q + 1, db), np.full(2 * Q + 1, da))


def linear_solution(p, Q, s):
    """(6) on q = -Q..Q solved another way. With one kappa on both sides, w = exp(kappa h) makes (6)
    linear, dw/ds = kappa M w, with M tridiagonal: C0 q - beta q^2 on its diagonal, lam_a/(e kappa)
    below it (a sale takes q to q - 1), lam_b/(e kappa) above it (a purchase takes q to q + 1), and
    the grid's edges are the matrix's own. So h(T - s) = ln(expm(kappa M s) exp(-kappa alpha q^2)) / kappa."""
    q = hjb.grid(Q)
    k = p.kappa
    M = (np.diag(p.C0 * q - p.beta * q * q) + np.diag(np.full(2 * Q, p.lam_a / (E * k)), -1)
         + np.diag(np.full(2 * Q, p.lam_b / (E * k)), 1))
    return np.log(linalg.expm(k * s * M) @ np.exp(-k * p.alpha * q * q)) / k


@check
def psi2_solves_its_riccati_equation():
    """psi2's closed form ends at -alpha and solves dpsi2/dt + c D psi2^2 - beta = 0, the last equation
    of (9), for beta = 0 and beta > 0; it is -alpha / (1 + c alpha D (T - t)) at beta = 0, and the
    paper's zeta form as written after (9) at beta > 0."""
    for p in (hjb.option(WING), with_beta(WING)):
        cf = hjb.Closed(p, T)
        assert abs(cf.psi2(T) / -p.alpha - 1) < 1e-14, f"psi2(T) = {cf.psi2(T)}, not -alpha = {-p.alpha}"
        cD = model.C * p.D
        for t in (0.01 * T, 0.5 * T, 0.99 * T):
            d = 2e-5 * T
            slope = (cf.psi2(t + d) - cf.psi2(t - d)) / (2 * d)
            rhs = p.beta - cD * cf.psi2(t) ** 2
            assert abs(slope - rhs) < 1e-6 * abs(rhs), \
                f"beta {p.beta:.3g}, t/T {t / T:.2f}: slope {slope:.6g}, the equation says {rhs:.6g}"
            s = T - t
            if p.beta == 0:
                want = -p.alpha / (1 + cD * p.alpha * s)
            else:
                ups = 1 / cD
                eta, root = math.sqrt(p.beta / ups), math.sqrt(ups * p.beta)
                zm, zp = root - p.alpha, root + p.alpha
                want = root * (zm * math.exp(-eta * s) - zp * math.exp(eta * s)) \
                    / (zm * math.exp(-eta * s) + zp * math.exp(eta * s))
            assert abs(cf.psi2(t) / want - 1) < 1e-12, f"beta {p.beta:.3g}: psi2 {cf.psi2(t):.12g}, closed form {want:.12g}"


@check
def psi1_solves_its_equation():
    """psi1 is 0 at T and solves dpsi1/dt + C0 + c D psi2 psi1 + c (lam_b - lam_a) psi2
    + c (lam_b kappa - lam_a kappa) psi2^2 = 0, the second equation of (9) with gamma frozen; at
    beta = 0 it is (10)'s integral done by hand, with a = c alpha D and s = T - t:
    [C0 (s + a s^2/2) - c alpha (lam_b - lam_a) s + kappa (lam_b - lam_a) alpha ln(1 + a s) / D] / (1 + a s)."""
    for p in (hjb.option(WING), with_beta(WING)):
        cf = hjb.Closed(p, T)
        assert abs(cf.psi1(T)) < 1e-15, f"psi1(T) = {cf.psi1(T)}"
        cD = model.C * p.D
        for t in (0.01 * T, 0.5 * T, 0.99 * T):
            d = 2e-5 * T
            slope = (cf.psi1(t + d) - cf.psi1(t - d)) / (2 * d)
            p2 = cf.psi2(t)
            terms = (p.C0, cD * p2 * cf.psi1(t), model.C * (p.lam_b - p.lam_a) * p2,
                     model.C * (p.lam_b * p.kappa - p.lam_a * p.kappa) * p2 ** 2)
            assert abs(slope + sum(terms)) < 1e-6 * sum(abs(x) for x in terms), \
                f"beta {p.beta:.3g}, t/T {t / T:.2f}: slope {slope:.6g}, the equation says {-sum(terms):.6g}"
            if p.beta == 0:
                s, a = T - t, cD * p.alpha
                want = (p.C0 * (s + a * s * s / 2) - model.C * p.alpha * (p.lam_b - p.lam_a) * s
                        + p.kappa * (p.lam_b - p.lam_a) * p.alpha * math.log1p(a * s) / p.D) / (1 + a * s)
                assert abs(cf.psi1(t) / want - 1) < 1e-9, f"t/T {t / T:.2f}: psi1 {cf.psi1(t):.12g}, by hand {want:.12g}"


@check
def the_closed_quotes_are_5_on_the_ansatz():
    """The quotes (12) are (5) applied to the ansatz h = psi0 + psi1 q + psi2 q^2: the bid
    1/kappa + h(q) - h(q+1) and the ask 1/kappa + h(q) - h(q-1), at every q and t."""
    p = with_beta(WING)
    cf = hjb.Closed(p, T)
    q = np.arange(-30, 31, dtype=float)
    for t in (0.0, 0.5 * T, T - model.DT):
        def h(x):
            return cf.psi1(t) * x + cf.psi2(t) * x * x
        bid, ask = cf.quotes_at(t, q)
        assert np.max(np.abs(bid - (1 / p.kappa + h(q) - h(q + 1)))) < 1e-12 / p.kappa, f"the bid at t/T {t / T:.3f}"
        assert np.max(np.abs(ask - (1 / p.kappa + h(q) - h(q - 1)))) < 1e-12 / p.kappa, f"the ask at t/T {t / T:.3f}"


@check
def one_option_is_model_optimal_on_a_book_of_one():
    """The closed form of one option is what model.Optimal computes on the 15-second grid for a book of
    that one option alone, at the open, mid-session and the last step: by (19) and (20) when beta = 0,
    within 1e-6 of a spread (model.py integrates (20) by the trapezoid rule), and by its Runge-Kutta
    solution of (17) with the running penalty beta_k = alpha_k / T, within 1e-9 of a spread."""
    b = hjb.book()
    q = np.array([[-10.0], [-1.0], [0.0], [3.0], [10.0]])
    for n in (WING, LONG):
        one = types.SimpleNamespace(n=1, bucket=b.bucket[[n]], w=b.w[[n]], lam_a=b.lam_a[[n]],
                                    lam_b=b.lam_b[[n]], kappa=b.kappa[[n]], C0=b.C0[[n]])
        beta_k = model.ALPHA[int(b.bucket[n])] / T
        for o, cf, tol in ((model.Optimal(one), hjb.closed(n, T), 1e-6),
                           (model.Optimal(one, beta=(beta_k,) * 6), hjb.closed(n, T, beta_k), 1e-9)):
            for j in (0, model.STEPS // 2, model.STEPS - 1):
                mb, ma = o.quotes(j, q)
                cb, ca = cf.quotes_at(j * model.DT, q[:, 0])
                gap = b.kappa[n] * max(np.max(np.abs(mb[:, 0] - cb)), np.max(np.abs(ma[:, 0] - ca)))
                assert gap < tol, f"option {n}, beta {cf.p.beta:.3g}, step {j}: {gap:.2e} of a spread from model.Optimal"


@check
def without_a_penalty_6_is_solved_by_hand():
    """With alpha = beta = 0, (6) is solved by h(t, q) = C0 (T - t) q + b(t), b(t) = int_t^T
    lam_a/(e kappa) exp(-kappa C0 (T - u)) + lam_b/(e kappa) exp(kappa C0 (T - u)) du: the solution
    matches it away from the grid's edges, its quotes are 1/kappa -+ C0 (T - t), and so are those of
    (12): without a penalty, approximation (7) changes no quote."""
    p = dataclasses.replace(hjb.option(WING), alpha=0.0, beta=0.0)
    k, c0 = p.kappa, p.C0
    ex = hjb.Exact(p, T, 60, (0.5 * T,))
    cf = hjb.Closed(p, T)
    q = hjb.grid(60)
    near = np.abs(q) <= 10
    for t in (0.0, 0.5 * T):
        s = T - t
        b = (p.lam_a / (E * k)) * -math.expm1(-k * c0 * s) / (k * c0) + (p.lam_b / (E * k)) * math.expm1(k * c0 * s) / (k * c0)
        err = np.max(np.abs(ex.at(t) - (c0 * s * q + b))[near])
        assert err < 1e-9 * abs(b), f"t/T {t / T}: h is {err:.2e} from the solution by hand"
        eb, ea = ex.quotes(t)
        assert np.max(np.abs(k * eb - (1 - k * c0 * s))[near]) < 1e-7, "the exact bid is not 1/kappa - C0 (T - t)"
        assert np.max(np.abs(k * ea - (1 + k * c0 * s))[near]) < 1e-7, "the exact ask is not 1/kappa + C0 (T - t)"
        cb, ca = cf.on(60)(t)
        assert max(np.max(np.abs(k * (cb - eb))[near]), np.max(np.abs(k * (ca - ea))[near])) < 1e-7, \
            f"t/T {t / T}: (12) is not (5) without a penalty"


@check
def without_penalty_edge_or_imbalance_both_quote_one_over_kappa():
    """With alpha = beta = 0, C0 = 0 and lam_b = lam_a, the closed-form and the exact quotes are both
    1/kappa, at t = 0, T/2 and the last step, away from the grid's edges."""
    p0 = hjb.option(ATM)
    p = dataclasses.replace(p0, alpha=0.0, beta=0.0, C0=0.0, lam_b=p0.lam_a)
    ex = hjb.Exact(p, T, 60, (0.5 * T, T - model.DT))
    cf = hjb.Closed(p, T)
    near = np.abs(hjb.grid(60)) <= 20
    for t in (0.0, 0.5 * T, T - model.DT):
        for name, got in zip(("exact bid", "exact ask", "(12) bid", "(12) ask"), (*ex.quotes(t), *cf.on(60)(t))):
            err = np.max(np.abs(p.kappa * got[near] - 1))
            assert err < 1e-8, f"t/T {t / T:.3f}: the {name} is {err:.2e} of a spread from 1/kappa"


@check
def exact_is_the_linear_solution():
    """exact() agrees with the solution of (6) by its linearisation w = exp(kappa h) (linear_solution):
    on eleven states, where the edges matter, at every state; on q = -40..40 at |q| <= 20; with
    alpha, beta, C0 and the imbalance all nonzero."""
    p = with_beta(WING)
    for Q, near in ((5, 5), (40, 20)):
        ex = hjb.Exact(p, T, Q, (0.5 * T,))
        sel = np.abs(hjb.grid(Q)) <= near
        for t in (0.0, 0.5 * T):
            want = linear_solution(p, Q, T - t)[sel]
            err = np.max(np.abs(ex.at(t)[sel] - want))
            assert err < 1e-8 * np.max(np.abs(want)), f"Q {Q}, t/T {t / T}: {err:.2e} from the linear solution"


@check
def the_jacobians_are_the_derivatives():
    """The tridiagonal Jacobians handed to Radau are the derivatives of their right-hand sides, by
    central differences: of (6), of the value equation and of the forward equation."""
    p = with_beta(WING)
    Q = 6
    y = -p.alpha * hjb.grid(Q) ** 2 + 0.01 * np.random.default_rng(1).standard_normal(2 * Q + 1)
    cf = hjb.Closed(p, T)
    systems = {"(6)": hjb.hjb_equations(p, Q), "the value equation": hjb.value_equations(p, Q, cf.on(Q), T),
               "the forward equation": hjb.law_equations(p, Q, cf.on(Q))}
    eps = 1e-6
    for name, (rhs, jac) in systems.items():
        J = jac(0.3 * T, y).toarray()
        num = np.column_stack([(rhs(0.3 * T, y + eps * e) - rhs(0.3 * T, y - eps * e)) / (2 * eps) for e in np.eye(y.size)])
        err = np.max(np.abs(J - num))
        assert err < 1e-6 * np.max(np.abs(num)), f"{name}: the Jacobian is {err:.2e} from the differences"


@check
def the_exact_quotes_give_h_back():
    """Evaluating the exact quotes (5) by the linear equation of the process returns h itself, at every
    state, edges included: (5) attains the sup in (4). On eleven states and on q = -40..40."""
    p = with_beta(WING)
    for Q in (5, 40):
        ex = hjb.Exact(p, T, Q)
        err = np.max(np.abs(hjb.evaluate(p, T, ex.quotes, Q=Q).at(0.0) - ex.at(0.0)))
        assert err < 1e-8 * np.max(np.abs(ex.at(0.0))), f"Q {Q}: the exact quotes are worth {err:.2e} off h"


@check
def the_closed_form_never_beats_the_optimum():
    """The value of the closed-form quotes is nowhere above h on the grid (h is the sup), and it is
    below h at q = 0 when alpha > 0: (12) is not optimal (footnote 7)."""
    for n in (WING, LONG):
        p = hjb.option(n)
        h = hjb.Exact(p, T, 40).at(0.0)
        g = hjb.evaluate(p, T, hjb.Closed(p, T).on(40), Q=40).at(0.0)
        assert np.all(g <= h + 1e-9 * np.max(np.abs(h))), f"option {n}: (12) beats (6) by {np.max(g - h):.2e}"
        assert h[40] - g[40] > 1e-9 * h[40], f"option {n}: (12) loses only {h[40] - g[40]:.2e} at q = 0"


@check
def doubling_Q_moves_nothing_near_zero():
    """Doubling the grid leaves h(0, q), |q| <= 20, where it was: from Q = 40 to 80 over half a day
    (3m 90%); and choose_Q over a day at the money, where Q = 40 is too small, picks the first Q of
    40, 80, ... that holds still when doubled."""
    p = hjb.option(WING)
    a, b = hjb.Exact(p, T, 40).at(0.0)[20:61], hjb.Exact(p, T, 80).at(0.0)[60:101]
    assert np.max(np.abs(a - b)) <= hjb.Q_TOL * np.max(np.abs(b)), f"h moves by {np.max(np.abs(a - b)):.2e}"
    p = hjb.option(ATM)
    Q, ex, ex2, moved = hjb.choose_Q(p, 2 * T)
    a, b = ex.at(0.0)[Q - 20:Q + 21], ex2.at(0.0)[2 * Q - 20:2 * Q + 21]
    assert ex.Q == Q and ex2.Q == 2 * Q and moved == np.max(np.abs(a - b)), "choose_Q's solutions"
    assert moved <= hjb.Q_TOL * np.max(np.abs(b)), f"on choose_Q's Q = {Q} h still moves by {moved:.2e}"
    assert Q > hjb.Q_START, f"Q = {Q}: Q = 40 was expected to move h by more than Q_TOL over a day"
    half = hjb.Exact(p, 2 * T, Q // 2).at(0.0)[Q // 2 - 20:Q // 2 + 21]
    assert np.max(np.abs(half - a)) > hjb.Q_TOL * np.max(np.abs(a)), f"Q = {Q // 2} would have done"


@check
def constant_quotes_have_their_value_by_hand():
    """Constant quotes (db, da) on a grid too wide to matter: fills at ra = lam_a e^(-kappa da) and
    rb = lam_b e^(-kappa db) make Q_t a difference of two Poisson counts, so g(0, 0) =
    (ra da + rb db) T + C0 (rb - ra) T^2/2 - beta ((ra + rb) T^2/2 + (rb - ra)^2 T^3/3)
    - alpha ((ra + rb) T + (rb - ra)^2 T^2)."""
    p = with_beta(WING)
    db, da = 0.8 / p.kappa, 1.3 / p.kappa
    ra, rb = p.lam_a * math.exp(-p.kappa * da), p.lam_b * math.exp(-p.kappa * db)
    want = ((ra * da + rb * db) * T + p.C0 * (rb - ra) * T ** 2 / 2
            - p.beta * ((ra + rb) * T ** 2 / 2 + (rb - ra) ** 2 * T ** 3 / 3)
            - p.alpha * ((ra + rb) * T + (rb - ra) ** 2 * T ** 2))
    got = hjb.evaluate(p, T, constant(60, db, da), Q=60).at(0.0)[60]
    assert abs(got / want - 1) < 1e-9, f"g(0, 0) = {got:.10f}, by hand {want:.10f}"


@check
def the_law_of_constant_quotes_is_skellam():
    """Under constant quotes the inventory at T is a difference of two Poisson counts of means rb T and
    ra T: inventory() sums to one and matches the Skellam distribution at every q, and its 1st and
    99th percentiles, mean (rb - ra) T and standard deviation sqrt((ra + rb) T) are Skellam's."""
    p = hjb.option(WING)
    db, da = 0.8 / p.kappa, 1.3 / p.kappa
    ra, rb = p.lam_a * math.exp(-p.kappa * da), p.lam_b * math.exp(-p.kappa * db)
    q = hjb.grid(60)
    P = hjb.inventory(p, T, constant(60, db, da), 60)
    assert abs(P.sum() - 1) < 1e-9, f"the law sums to {P.sum():.12f}"
    err = np.max(np.abs(P - stats.skellam.pmf(q, rb * T, ra * T)))
    assert err < 1e-7, f"{err:.2e} from Skellam"
    got = hjb.percentiles(q, P, 0.01, 0.99)
    want = [int(stats.skellam.ppf(x, rb * T, ra * T)) for x in (0.01, 0.99)]
    assert got == want, f"percentiles {got}, Skellam's {want}"
    mean, std = hjb.moments(q, P)
    assert abs(mean - (rb - ra) * T) < 1e-6 and abs(std - math.sqrt((ra + rb) * T)) < 1e-6, \
        f"mean {mean:.6f} and std {std:.6f}, Skellam's {(rb - ra) * T:.6f} and {math.sqrt((ra + rb) * T):.6f}"


@check
def with_no_penalty_the_book_drifts_long():
    """Under the exact quotes the law of q at T sums to one; with no penalty and no edge
    (alpha = beta = C0 = 0) its mean is (lam_b - lam_a) e^-1 T > 0: the market sells more than it
    buys (lam_b > lam_a), so the book drifts long."""
    p = dataclasses.replace(hjb.option(ATM), alpha=0.0, beta=0.0, C0=0.0)
    P = hjb.inventory(p, T, hjb.Exact(p, T, 60).quotes, 60)
    mean, want = float(P @ hjb.grid(60)), (p.lam_b - p.lam_a) / E * T
    assert abs(P.sum() - 1) < 1e-9, f"the law sums to {P.sum():.12f}"
    assert want > 0 and abs(mean / want - 1) < 1e-6, f"mean {mean:.6f}, (lam_b - lam_a) T / e = {want:.6f}"
    pw = hjb.option(WING)
    Pw = hjb.inventory(pw, T, hjb.Exact(pw, T, 40).quotes, 40)
    assert abs(Pw.sum() - 1) < 1e-9, f"with the penalty the law sums to {Pw.sum():.12f}"


@check
def the_law_keeps_its_mass_on_a_small_grid():
    """On seven states, where the book reaches both edges within half a day, the law of q at T still
    sums to one under the closed-form quotes and under the exact ones: nothing leaves the grid."""
    p = hjb.option(ATM)
    for name, quotes in (("(12)", hjb.Closed(p, T).on(3)), ("(5)", hjb.Exact(p, T, 3).quotes)):
        P = hjb.inventory(p, T, quotes, 3)
        assert min(P[0], P[-1]) > 0.01, f"under {name} the edges hold {P[0]:.3g} and {P[-1]:.3g}"
        assert abs(P.sum() - 1) < 1e-9 and P.min() > -1e-12, f"under {name} the law sums to {P.sum():.12f}"


@check
def compare_reports_what_its_pieces_give():
    """compare() on the 3m ATM over half a day reports what exact(), closed(), evaluate() and
    inventory() give: the value lost h(0, 0) - g(0, 0) and its share of h(0, 0), the gaps
    kappa (closed - exact) at t = 0, T/2 and T - dt, the percentiles of q under (5) at T and at T/2,
    the largest gap at the last step between the percentiles at T, and at T/2 between those at T/2."""
    p = hjb.option(ATM)
    r = hjb.compare(ATM, T)
    Q = r["Q"]
    ex, cf = hjb.Exact(p, T, Q, (0.0, T / 2, T - model.DT)), hjb.Closed(p, T)
    h, g = ex.at(0.0)[Q], hjb.evaluate(p, T, cf.on(Q), Q=Q).at(0.0)[Q]
    assert abs(r["h00"] - h) < 1e-12 and abs(r["g00"] - g) < 1e-12, "h(0, 0) or g(0, 0)"
    assert abs(r["value_lost"] - (h - g)) < 1e-14 and abs(r["value_lost_share"] - (h - g) / h) < 1e-14, "the value lost"
    for name, t in (("t = 0", 0.0), ("t = T/2", T / 2), ("last step", T - model.DT)):
        (eb, ea), (cb, ca) = ex.quotes(t), cf.on(Q)(t)
        for j in hjb.SHOWN:
            for side, gap in (("bid", p.kappa * (cb - eb)), ("ask", p.kappa * (ca - ea))):
                assert abs(r["quote_gap"][name][side][f"q={j}"] - gap[Q + j]) < 1e-12, f"the {side} gap at {name}, q = {j}"
    q = hjb.grid(Q)
    mid, law = hjb.inventory_at(p, T, ex.quotes, Q, (T / 2, T))
    lo, hi = hjb.percentiles(q, law, 0.01, 0.99)
    assert (r["inventory_exact"]["p01"], r["inventory_exact"]["p99"]) == (lo, hi), "the percentiles of q at T"
    assert [r["inventory_exact"]["p01_at_T/2"], r["inventory_exact"]["p99_at_T/2"]] == hjb.percentiles(q, mid, 0.01, 0.99), \
        "the percentiles of q at T/2"
    for key, t, P in (("largest_gap_last_step_p01_to_p99", T - model.DT, law),
                      ("largest_gap_at_T/2_over_p01_to_p99_at_T/2", T / 2, mid)):
        (eb, ea), (cb, ca) = ex.quotes(t), cf.on(Q)(t)
        lo, hi = hjb.percentiles(q, P, 0.01, 0.99)
        inside = (q >= lo) & (q <= hi)
        for side, gap in (("bid", p.kappa * (cb - eb)), ("ask", p.kappa * (ca - ea))):
            got = r[key][side]
            assert abs(abs(got["gap"]) - np.max(np.abs(gap[inside]))) < 1e-12 and lo <= got["q"] <= hi, f"{key}, {side}"


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn()
            print(f"ok    {fn.__name__}")
        except Exception as e:  # a check that crashes has failed
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"{len(CHECKS) - failed} of {len(CHECKS)} checks hold")
    import json, os
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hjb_checks.json"), "w", encoding="utf-8") as f:
        json.dump({"held": len(CHECKS) - failed, "of": len(CHECKS), "checks": [fn.__name__ for fn in CHECKS]}, f, indent=1)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
