"""Checks: each one states a rule the study depends on and fails if the rule is broken.

Run: python checks.py   (exit code 1 if any check fails; mutations.py breaks rules on purpose
on copies of this folder and expects a check to fail each time)
"""
import math
import sys

import numpy as np

import data
import model
import simulate
import study
import surface

sys.stdout.reconfigure(encoding="utf-8")
CHECKS = []
BOOK = model.Book()


def check(fn):
    CHECKS.append(fn)
    return fn


def index(moneyness, tau):
    return int(np.flatnonzero(np.isclose(BOOK.moneyness, moneyness) & np.isclose(BOOK.tau, tau))[0])


def optimal(scale):
    if not hasattr(optimal, "cache"):
        optimal.cache = {}
    if scale not in optimal.cache:
        optimal.cache[scale] = model.Optimal(BOOK, scale)
    return optimal.cache[scale]


@check
def the_book_is_table_1():
    """44 options; buckets of 6, 10, 6, 6, 10, 6 cut where Table 1 cuts; order flow and kappa as section 3.2."""
    b = BOOK
    assert b.n == 44, f"{b.n} options"
    assert np.bincount(b.bucket).tolist() == [6, 10, 6, 6, 10, 6], f"bucket sizes {np.bincount(b.bucket).tolist()}"
    want = {(0.94, 0.5): 0, (0.96, 0.5): 1, (1.04, 0.25): 1, (1.06, 0.25): 2,
            (0.90, 1.0): 3, (1.00, 3.0): 4, (1.10, 3.0): 5}
    for (m, tau), k in want.items():
        assert b.bucket[index(m, tau)] == k, f"K/S0 {m}, tau {tau} in bucket {b.bucket[index(m, tau)] + 1}"
    atm, wing = np.isclose(b.moneyness, 1.0), np.isclose(b.moneyness, 0.9)
    assert np.allclose(b.lam_a[atm] * model.T, 50.0), "not 100 orders a day at the money"
    assert np.allclose(b.lam_b / b.lam_a, 1.15), "the imbalance is not 1.15"
    assert np.allclose((b.kappa * b.w)[atm], 5.0) and np.allclose((b.kappa * b.w)[wing], 2.5), "kappa is not section 3.2's"
    assert model.STEPS == 2880, f"{model.STEPS} steps of 15 seconds in half a day"


@check
def the_surface_is_figure_1a():
    """The fitted surface is within half a vol point of every point read off Figure 1(a), one
    level at the money for every maturity, and its skew falls with maturity."""
    for tau, row in surface.FIGURE_1A.items():
        fit = surface.implied_vol(100 * np.array(surface.MONEYNESS_READ), tau)
        gap = np.max(np.abs(fit - np.array(row)))
        assert gap < 0.005, f"tau {tau}: {gap:.4f} from the figure"
    assert 0.13 < BOOK.vol0.min() and BOOK.vol0.max() < 0.26, "the 44 vols leave the figure's range"
    atm = [float(surface.implied_vol(100.0, t)) for t in model.TAUS]
    assert max(atm) - min(atm) < 1e-12, f"at the money {atm}"
    spread = [float(surface.implied_vol(90.0, t) - surface.implied_vol(110.0, t)) for t in model.TAUS]
    assert all(a > b > 0 for a, b in zip(spread, spread[1:])), f"90-110 spreads {spread}"


@check
def the_greeks_agree_with_differences():
    """Delta, vega and dollar gamma agree with finite differences of the Black-Scholes price."""
    s, k, tau, vol, h = 101.3, 95.0, 0.5, 0.21, 1e-2

    def p(s_=s, v_=vol):
        return float(surface.price(s_, k, tau, v_))

    _, delta = surface.price_delta(s, k, tau, vol)
    assert abs((p(s + h) - p(s - h)) / (2 * h) - delta) < 1e-6, "delta"
    assert abs((p(v_=vol + 1e-5) - p(v_=vol - 1e-5)) / 2e-5 - surface.vega(s, k, tau, vol)) < 1e-4, "vega"
    gamma = s * s * (p(s + h) - 2 * p() + p(s - h)) / h ** 2
    assert abs(gamma / surface.dollar_gamma(s, k, tau, vol) - 1) < 1e-3, "dollar gamma"


@check
def theta2_solves_its_riccati_equation():
    """Theta_2 of (19) ends at -A, stays symmetric and negative, and solves
    dTheta2/dt + (2/e) Theta2 D Theta2 = 0, beta being 0."""
    o = optimal(1.0)
    assert np.allclose(o.theta2[-1], -o.A, atol=1e-14), "Theta2(T) is not -A"
    d = BOOK.lam_b * BOOK.kappa + BOOK.lam_a * BOOK.kappa
    for j in (5, 1440, model.STEPS - 5):
        th = o.theta2[j]
        assert np.allclose(th, th.T, atol=1e-12), "Theta2 is not symmetric"
        assert np.linalg.eigvalsh(th).max() < 1e-12, "Theta2 is not negative"
        slope = (o.theta2[j + 1] - o.theta2[j - 1]) / (2 * model.DT)
        rhs = -model.C * th @ (d[:, None] * th)
        assert np.max(np.abs(slope - rhs)) < 1e-2 * np.max(np.abs(rhs)), f"the Riccati equation fails at step {j}"


@check
def theta1_solves_its_equation():
    """theta_1 of (20) is 0 at T and solves dtheta1/dt + (2/e) Theta2 D theta1 + f = 0 (17)."""
    o, b = optimal(1.0), BOOK
    assert np.allclose(o.theta1[-1], 0.0), "theta1(T) is not 0"
    d = b.lam_b * b.kappa + b.lam_a * b.kappa
    for j in (5, 1440, model.STEPS - 5):
        th2 = o.theta2[j]
        f = b.C0 + model.C * th2 @ (b.lam_b - b.lam_a) \
            + model.C * th2 @ (np.diag(th2) * (b.lam_b * b.kappa - b.lam_a * b.kappa))
        slope = (o.theta1[j + 1] - o.theta1[j - 1]) / (2 * model.DT)
        rhs = -(model.C * th2 @ (d * o.theta1[j]) + f)
        assert np.max(np.abs(slope - rhs)) < 1e-2 * np.max(np.abs(rhs)), f"equation (17) fails at step {j}"


@check
def with_no_penalty_the_quote_is_spread_and_edge():
    """alpha = 0 and a balanced flow: Theta2 = 0 and theta1 = C0 (T - t), so the bid is
    1/kappa - C0 (T - t) and the ask 1/kappa + C0 (T - t): the edge phi of section 2.1."""
    b = model.Book()
    b.lam_b = b.lam_a.copy()
    o = model.Optimal(b, alpha_scale=0.0)
    left = model.T - np.arange(model.STEPS + 1) * model.DT
    edge = b.C0[None, :] * left[:, None]
    assert np.allclose(o.base_b, 1 / b.kappa - edge, rtol=0, atol=1e-12), "bid"
    assert np.allclose(o.base_a, 1 / b.kappa + edge, rtol=0, atol=1e-12), "ask"


@check
def the_penalty_leans_against_inventory():
    """Long one lot of a bucket 1 option near the close: its bid widens and its ask tightens,
    nothing changes outside bucket 1, and ten times the penalty leans harder."""
    n = index(0.90, 0.25)
    q, flat = np.zeros((1, 44)), np.zeros((1, 44))
    q[0, n] = 1
    j = model.STEPS - 1
    b0, a0 = optimal(1.0).quotes(j, flat)
    b1, a1 = optimal(1.0).quotes(j, q)
    assert b1[0, n] > b0[0, n] and a1[0, n] < a0[0, n], "the quotes do not lean against the inventory"
    other = BOOK.bucket != BOOK.bucket[n]
    assert np.allclose(b1[0, other], b0[0, other]) and np.allclose(a1[0, other], a0[0, other]), "other buckets moved"
    hb0, _ = optimal(10.0).quotes(j, flat)
    hb1, _ = optimal(10.0).quotes(j, q)
    assert hb1[0, n] - hb0[0, n] > b1[0, n] - b0[0, n], "ten times the penalty does not lean harder"


@check
def a_held_option_earns_the_vol_carry():
    """One 3-month 90% option held through the half day, delta-hedged with the implied delta,
    nothing traded: its mean gain is the paper's carry (sigma^2 - sigma_imp^2)/2 Gamma$ T within 5%."""
    n = index(0.90, 0.25)
    q0 = np.zeros(44)
    q0[n] = 1
    never = model.ZeroIntelligence(BOOK, 1e6)
    out = simulate.run(BOOK, [never], paths=400, seed=3, Q0=q0)
    assert out["trades"].sum() == 0, "something traded"
    want, got = BOOK.C0[n] * model.T, out["hedged"][0].mean()
    assert abs(got / want - 1) < 0.05, f"mean hedged gain {got:.5f}, carry {want:.5f}"


@check
def fills_follow_the_intensity():
    """A constant spread: the trades of half a day match 1 - exp(-lambda e^(-kappa delta) dt) a step
    and a side, within three standard errors."""
    zi = model.ZeroIntelligence(BOOK, 0.5)
    paths = 200
    out = simulate.run(BOOK, [zi], paths=paths, seed=4)
    pb = -np.expm1(-BOOK.lam_b * np.exp(-BOOK.kappa * zi.spread) * model.DT)
    pa = -np.expm1(-BOOK.lam_a * np.exp(-BOOK.kappa * zi.spread) * model.DT)
    want = model.STEPS * (pb + pa).sum()
    se = math.sqrt(model.STEPS * (pb * (1 - pb) + pa * (1 - pa)).sum() / paths)
    got = out["trades"][0].mean()
    assert abs(got - want) < 3 * se, f"{got:.1f} trades, expected {want:.1f} (se {se:.2f})"


@check
def a_move_of_nothing_changes_nothing():
    """A surface move of size zero gives the still run's values exactly; a move leaves the trades alone."""
    paths, zi = 100, model.ZeroIntelligence(BOOK, 0.5)
    still = simulate.run(BOOK, [zi], paths=paths, seed=5)
    zero = simulate.Moves(np.zeros(paths), np.zeros(paths), np.random.default_rng(0))
    same = simulate.run(BOOK, [zi], paths=paths, seed=5, moves=zero)
    assert np.allclose(still["value"], same["value"], rtol=0, atol=1e-9), "a still move changed the value"
    big = simulate.Moves(np.full(paths, 0.02), np.full(paths, 0.01), np.random.default_rng(0))
    moved = simulate.run(BOOK, [zi], paths=paths, seed=5, moves=big)
    assert np.array_equal(still["Q"], moved["Q"]) and np.array_equal(still["trades"], moved["trades"]), "the move changed the trades"


@check
def a_rising_surface_pays_a_long_vega_book():
    """The same trades while the surface rises one vol point in a straight line through the half
    day: ZI 0.1% (long vega in every bucket) gains about the vega it held on average times 0.01,
    between a third and two thirds of its closing percentage-vega, as its book grows through the day."""
    paths, zi = 200, model.ZeroIntelligence(BOOK, 0.1)
    still = simulate.run(BOOK, [zi], paths=paths, seed=6)
    up = simulate.Moves(np.full(paths, 0.01), np.zeros(paths), np.random.default_rng(0))
    moved = simulate.run(BOOK, [zi], paths=paths, seed=6, moves=up)
    gain = (moved["value"] - still["value"]).mean()
    closing = still["bucket_vega"][0].sum(-1).mean()
    assert closing > 0 and 0.35 < gain / closing < 0.65, f"gain {gain:.2f} against closing vega {closing:.2f}"


@check
def the_closing_book_carries_its_vega():
    """A small parallel move after the close changes the closing book by its vega times the move."""
    paths, zi = 100, model.ZeroIntelligence(BOOK, 0.1)
    out = simulate.run(BOOK, [zi], paths=paths, seed=8)
    x = 1e-4
    got = simulate.carried(BOOK, out, np.full(paths, x), np.zeros(paths))[0]
    vega_T = surface.vega(out["S"][:, None], BOOK.K, out["time_left"], out["vol"])
    want = (out["Q"][0] * vega_T).sum(1) * x
    assert np.allclose(got, want, rtol=1e-2, atol=1e-9), "the carried gain is not vega times the move"


@check
def the_skew_move_is_the_3_month_spread():
    """One unit of skew move changes the 3-month 90% - 110% spread by exactly one unit, the 3-year
    one by less, and leaves at-the-money vols where they are."""
    sh = surface.skew_shape
    assert abs(float(sh(90.0, 0.25) - sh(110.0, 0.25)) - 1) < 1e-12, "3-month spread"
    assert 0 < float(sh(90.0, 3.0) - sh(110.0, 3.0)) < 0.5, "3-year spread"
    assert float(sh(100.0, 1.0)) == 0.0, "at the money"


@check
def the_vix_series_is_whole():
    """FRED's VIX: daily closes from 2 Jan 1990 in order, within 5 to 90; a day's change has a
    standard deviation of 1 to 3 points; half a day is a day times sqrt(1/2)."""
    dates, closes = data.vix()
    assert dates[0] == "1990-01-02" and len(dates) > 9000, f"{len(dates)} closes from {dates[0]}"
    assert all(a < b for a, b in zip(dates, dates[1:])), "dates out of order"
    assert closes.min() > 5 and closes.max() < 90, "a close out of range"
    _, changes = data.daily_changes()
    assert 1 < changes.std() < 3, f"std {changes.std():.2f}"
    assert abs(data.half_day(np.array([1.0]))[0] - math.sqrt(0.5) / 100) < 1e-15, "half a day is not sqrt(1/2) of a day"
    toy = [10, 20, 30, 16, 18]  # changes 10, 10, -14, 2: only the last follows a close in [15, 20)
    assert data.changes_from(15, 20, toy).tolist() == [2.0], "the band does not read the day before"
    assert data.changes_from(0, 1000).size == changes.size, "the band of everything loses days"


@check
def the_vega_path_ends_at_the_close():
    """The book's vega recorded through the session ends at the closing bucket vegas' sum, and a
    book that only grows (ZI 0.1%) holds about half of it at mid-session."""
    paths, zi = 100, model.ZeroIntelligence(BOOK, 0.1)
    out = simulate.run(BOOK, [zi], paths=paths, seed=9, snapshots=(model.STEPS // 2, model.STEPS))
    assert out["vega_path"].shape == (2, 1, paths), f"shape {out['vega_path'].shape}"
    assert np.allclose(out["vega_path"][-1, 0], out["bucket_vega"][0].sum(-1)), "the last snapshot is not the close"
    half = out["vega_path"][0, 0].mean() / out["vega_path"][-1, 0].mean()
    assert 0.35 < half < 0.65, f"mid-session holds {half:.2f} of the closing vega"


@check
def the_example_reproduces():
    """300 paths of the paper's example meet every rule study.judge reads off Figure 1."""
    out = simulate.run(BOOK, model.strategies(BOOK), paths=300, seed=11)
    runs = {11: {k: out[k] for k in ("value", "spread", "hedged", "trades", "bucket_vega", "vega_path")}
            | {"carried": np.zeros_like(out["value"]), "carried_term": np.zeros_like(out["value"])}}
    failed = [v for v in study.judge(study.stats(runs, out["names"], (11,))) if not v["holds"]]
    assert not failed, "; ".join(v["rule"] + ": " + v["detail"] for v in failed)


def one_option(n):
    """Option n of the book alone, as a book of one: for the paper's single-option formulas."""
    import types
    b = BOOK
    return types.SimpleNamespace(n=1, bucket=np.array([0]), w=b.w[[n]], lam_a=b.lam_a[[n]], lam_b=b.lam_b[[n]],
                                 kappa=b.kappa[[n]], C0=b.C0[[n]])


@check
def the_penalty_matrix_is_the_papers():
    """penalty_matrix on the six buckets is the A the closed form uses; on the whole book it is
    weight omega omega'; with a loading it is weight (loading omega)(loading omega)'."""
    a = model.penalty_matrix(BOOK, per_bucket=model.ALPHA)
    assert np.allclose(a, optimal(1.0).A, rtol=0, atol=1e-15), "the bucket penalty is not Optimal's A"
    w = BOOK.w
    assert np.allclose(model.penalty_matrix(BOOK, total=3.0), 3.0 * np.outer(w, w)), "the total-vega factor"
    ell = np.linspace(1.0, 0.2, BOOK.n)
    assert np.allclose(model.penalty_matrix(BOOK, total=3.0, loading=ell), 3.0 * np.outer(ell * w, ell * w)), "the loaded factor"
    assert not model.penalty_matrix(BOOK).any(), "no penalty is not zero"


@check
def the_numerical_solve_is_the_closed_form():
    """With no running penalty, Runge-Kutta on (17) gives (19)'s Theta2 to 1e-8 and (20)'s theta1
    to 1e-4 (what is left is the trapezoid rule's error in (20))."""
    o = optimal(1.0)
    th2, th1 = model.solve_numerically(o.A, np.zeros_like(o.A), BOOK)
    gap2 = np.abs(th2 - o.theta2).max() / np.abs(o.theta2).max()
    gap1 = np.abs(th1 - o.theta1).max() / np.abs(o.theta1).max()
    assert gap2 < 1e-8 and gap1 < 1e-4, f"Theta2 {gap2:.1e}, theta1 {gap1:.1e} from the closed forms"


@check
def a_running_penalty_gives_the_papers_scalar_riccati():
    """One option with alpha and beta: Theta2 is psi2 of the closed form after (9), with
    Upsilon = 1/(c D), eta = sqrt(beta/Upsilon), zeta = sqrt(Upsilon beta) -/+ alpha."""
    n = index(0.90, 0.25)
    one = one_option(n)
    a, be = 0.2 * one.w[0] ** 2, (0.2 / model.T) * one.w[0] ** 2
    th2, _ = model.solve_numerically(np.array([[a]]), np.array([[be]]), one)
    ups = 1 / (model.C * (one.lam_a[0] + one.lam_b[0]) * one.kappa[0])
    eta, root = math.sqrt(be / ups), math.sqrt(ups * be)
    left = model.T - np.arange(model.STEPS + 1) * model.DT
    lo, hi = (root - a) * np.exp(-eta * left), (root + a) * np.exp(eta * left)
    psi2 = root * (lo - hi) / (lo + hi)
    gap = np.abs(th2[:, 0, 0] - psi2).max() / np.abs(psi2).max()
    assert gap < 1e-10, f"{gap:.1e} from the paper's psi2"
    assert eta * model.T > 1, "the test is too short for beta to matter"


@check
def theta_solve_their_equations_with_a_running_penalty():
    """A running penalty on the book's total vega: Theta2 solves dTheta2/dt = B - c Theta2 D Theta2
    and theta1 the second equation of (17), at three steps, ending at -A and 0. (Near the close
    the solution moves a tenth of itself a step, so a centred difference is good to about 1%.)"""
    o = model.Optimal(BOOK, 1.0, beta_total=0.2 / model.T)
    assert o.B.any(), "no running penalty"
    assert np.allclose(o.theta2[-1], -o.A, atol=1e-14) and np.allclose(o.theta1[-1], 0.0), "the ends"
    d = BOOK.lam_b * BOOK.kappa + BOOK.lam_a * BOOK.kappa
    for j in (5, 1440, model.STEPS - 5):
        th2, th1 = o.theta2[j], o.theta1[j]
        slope = (o.theta2[j + 1] - o.theta2[j - 1]) / (2 * model.DT)
        rhs = o.B - model.C * th2 @ (d[:, None] * th2)
        assert np.max(np.abs(slope - rhs)) < 1e-2 * np.max(np.abs(rhs)), f"the Riccati equation fails at step {j}"
        f = BOOK.C0 + model.C * th2 @ (BOOK.lam_b - BOOK.lam_a) \
            + model.C * th2 @ (np.diag(th2) * (BOOK.lam_b * BOOK.kappa - BOOK.lam_a * BOOK.kappa))
        slope1 = (o.theta1[j + 1] - o.theta1[j - 1]) / (2 * model.DT)
        rhs1 = -(model.C * th2 @ (d * th1) + f)
        assert np.max(np.abs(slope1 - rhs1)) < 1e-2 * np.max(np.abs(rhs1)), f"theta1's equation fails at step {j}"


@check
def the_running_penalty_holds_the_vega_down_all_session():
    """The mechanism of C, on 200 paths: a running penalty on the book's vega holds it below half
    the baseline's at mid-session, while ten times the terminal penalty leaves mid-session within
    a tenth of the baseline's and only pulls the book in at the close."""
    strats = [optimal(1.0), optimal(10.0), model.Optimal(BOOK, 1.0, beta_total=0.2 / model.T)]
    out = simulate.run(BOOK, strats, paths=200, seed=12, snapshots=(model.STEPS // 2, model.STEPS))
    rms = np.sqrt((out["vega_path"] ** 2).mean(-1))  # (2 snapshots, 3 strategies)
    assert rms[0, 2] < 0.5 * rms[0, 0], f"running: {rms[0, 2]:.2f} at mid-session against {rms[0, 0]:.2f}"
    assert abs(rms[0, 1] / rms[0, 0] - 1) < 0.1, f"terminal x10: {rms[0, 1]:.2f} at mid-session against {rms[0, 0]:.2f}"
    assert rms[1, 1] < 0.5 * rms[1, 0], f"terminal x10: {rms[1, 1]:.2f} at the close against {rms[1, 0]:.2f}"


@check
def the_term_loading_is_the_vix3m_slope():
    """The VIX3M's and the VIX's changes are taken on the same days; the loading is 1 at 30 days
    and the regression slope at 93, falls with maturity, and the slope is a least squares slope."""
    both, x, y = data.paired_changes()
    assert x.size == y.size == len(both) - 1 and both[0] == "2007-12-04", f"{x.size} pairs from {both[0]}"
    load, slope, p = data.term_loading()
    assert abs(slope - np.polyfit(x, y, 1)[0]) < 1e-9, "not the least squares slope"
    assert 0.4 < slope < 0.9 and p > 0, f"slope {slope:.3f}, p {p:.3f}"
    assert abs(float(load(30 / 365)) - 1) < 1e-12 and abs(float(load(93 / 365)) - slope) < 1e-12, "the two anchors"
    taus = np.array(model.TAUS)
    assert np.all(np.diff(load(taus)) < 0), "the loading does not fall with maturity"


@check
def a_term_move_moves_long_maturities_less():
    """A shift with the term loading moves each option's vol by the shift times its maturity's
    loading, and the closing book then gains vega times loading times the move."""
    load = data.term_loading()[0]
    paths = 3
    moves = simulate.Moves(np.full(paths, 0.01), np.zeros(paths), np.random.default_rng(0), loading=load)
    up = moves.vol(BOOK, model.STEPS) - BOOK.vol0[None, :]
    assert np.allclose(up, 0.01 * load(BOOK.tau)[None, :], rtol=0, atol=1e-12), "the shift is not loaded by maturity"
    out = simulate.run(BOOK, [model.ZeroIntelligence(BOOK, 0.1)], paths=50, seed=13)
    x = 1e-4
    got = simulate.carried(BOOK, out, np.full(50, x), np.zeros(50), loading=load)[0]
    vega_T = surface.vega(out["S"][:, None], BOOK.K, out["time_left"], out["vol"])
    want = (out["Q"][0] * vega_T * load(BOOK.tau)[None, :]).sum(1) * x
    assert np.allclose(got, want, rtol=1e-2, atol=1e-9), "the carried gain is not loaded vega times the move"


@check
def the_frontier_measures_against_the_baseline():
    """C's arithmetic on a made-up run: the baseline costs 0 and cuts nothing; a strategy worth 1
    less on every path costs 1; one whose move brings half the baseline's swing cuts it by half."""
    rng = np.random.default_rng(0)
    p = 400
    base, move = 100 + rng.standard_normal(p), rng.standard_normal(p)
    def out(value, moved, carried):
        z = np.zeros((2, p))
        return {"value": value, "trades": z, "carried": carried, "carried_term": carried, "vega_path": np.zeros((1, 2, p)),
                "moved": moved}
    still = out(np.vstack([base, base - 1]), None, np.vstack([move, 0.5 * move]))
    runs = {"g": {sc: {7: dict(still)} for sc in ("still", "parallel", "term")}}
    for sc in ("parallel", "term"):
        runs["g"][sc][7]["value"] = np.vstack([base + move, base - 1 + 0.5 * move])
    keep = study.SEEDS_B
    study.SEEDS_B = (7,)
    try:
        c = study.frontier(runs, {"g": [study.BASELINE, "other"]})
    finally:
        study.SEEDS_B = keep
    b, o = c[study.BASELINE], c["other"]
    assert b["cost"] == 0 and b["in_session_parallel_cut"] == 0, "the baseline is not its own zero"
    assert abs(o["cost"] - 1) < 1e-12 and abs(o["in_session_term_cut"] - 0.5) < 1e-12, f"cost {o['cost']}, cut {o['in_session_term_cut']}"
    assert abs(o["carried_parallel_cut"] - 0.5) < 1e-12, "the carried cut"


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
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "checks.json"), "w", encoding="utf-8") as f:
        json.dump({"held": len(CHECKS) - failed, "of": len(CHECKS), "checks": [fn.__name__ for fn in CHECKS]}, f, indent=1)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
