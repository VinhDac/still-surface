"""Monte Carlo of half a day of market making, run as section 3.2 runs it.

Each 15-second step: every strategy posts its quotes around the mark-to-market price of each
option; a market order arrives on each side with probability 1 - exp(-lambda e^(-kappa delta) dt)
(at most one lot a side a step); a fill earns its spread; the book is delta-hedged with the
implied-vol delta; the spot moves with the true vol, 17%; every option is marked again.

The terminal portfolio value V_T = spreads earned + hedged mark-to-market gains, as
V_t = X_t + Q_t O_t - Delta_t S_t in section 2.

All strategies of a run share the fill draws and the spot path. The quotes depend only on time
and inventory, never on the surface, so a run with a moving surface trades exactly the same
inventory path as the run with the same seed on a still surface: the difference between them
is what the surface's move does to that book.
"""
import numpy as np

import model
import surface

VOL_FLOOR = 0.01


class Moves:
    """The surface moving through the half day, path by path: a shift and a change of skew, each
    a Brownian bridge from 0 to an end value given per path. The shift is parallel, or weighted
    by maturity when a loading is given (data.term_loading: a function of tau).

    vol(K, tau, t) = vol0(K, tau) + shift(t) * loading(tau) + skew(t) * surface.skew_shape(K, tau)
    """

    def __init__(self, shift_end, skew_end, rng, steps=model.STEPS, loading=None):
        self.shift = self._bridge(np.asarray(shift_end, dtype=float), rng, steps)
        self.skew = self._bridge(np.asarray(skew_end, dtype=float), rng, steps)
        self.loading = loading

    @staticmethod
    def _bridge(end, rng, steps):
        paths = end.size
        scale = end.std() if end.std() > 0 else 0.0
        frac = np.arange(steps + 1)[:, None] / steps
        if scale == 0.0:
            return frac * end[None, :]
        w = np.vstack([np.zeros((1, paths)), np.cumsum(rng.standard_normal((steps, paths)), 0) / np.sqrt(steps)])
        return frac * end[None, :] + scale * (w - frac * w[-1][None, :])

    def vol(self, book, j):
        shape = surface.skew_shape(book.K, book.tau)
        weight = weights(book, self.loading)
        v = book.vol0[None, :] + self.shift[j][:, None] * weight[None, :] + self.skew[j][:, None] * shape[None, :]
        return np.maximum(v, VOL_FLOOR)


def weights(book, loading):
    """How far each option's vol moves with a shift of one: 1 for a parallel shift."""
    return np.ones(book.n) if loading is None else np.asarray(loading(book.tau), dtype=float)


def run(book, strategies, paths=2000, seed=7, moves=None, steps=model.STEPS, dt=model.DT,
        sigma=model.SIGMA, Q0=None, snapshots=()):
    """Simulate every strategy on the same paths. Returns terminal values and books.
    Q0, an inventory to start from (the paper starts flat), is there for the checks.
    snapshots: step counts after which the book's total percentage-vega is recorded."""
    rng = np.random.default_rng(seed)
    P, N, M = paths, book.n, len(strategies)
    K, tau = book.K, book.tau
    vol_at = (lambda j: book.vol0) if moves is None else (lambda j: moves.vol(book, j))
    S = np.full(P, surface.S0)
    O, delta = surface.price_delta(S[:, None], K, tau, vol_at(0))
    Q = np.zeros((M, P, N)) if Q0 is None else np.broadcast_to(np.asarray(Q0, dtype=float), (M, P, N)).copy()
    spread = np.zeros((M, P))
    hedged = np.zeros((M, P))
    trades = np.zeros((M, P))
    lam_b, lam_a = book.lam_b * dt, book.lam_a * dt
    still = {}  # fill probabilities of strategies whose quotes never change
    snaps, path = set(snapshots), []
    for j in range(steps):
        u = rng.random((2, P, N))
        for m, strat in enumerate(strategies):
            db, da = strat.quotes(j, Q[m])
            if np.ndim(db) == 1 and m in still:
                pb, pa = still[m]
            else:
                pb = -np.expm1(-lam_b * np.exp(-book.kappa * db))
                pa = -np.expm1(-lam_a * np.exp(-book.kappa * da))
                if np.ndim(db) == 1:
                    still[m] = (pb, pa)
            nb = u[0] < pb
            na = u[1] < pa
            spread[m] += (nb * db + na * da).sum(1)
            trades[m] += nb.sum(1) + na.sum(1)
            Q[m] += nb
            Q[m] -= na
        if j + 1 in snaps:
            path.append(book.bucket_vega(Q).sum(-1))
        z = rng.standard_normal(P)
        S_new = S * np.exp(-0.5 * sigma ** 2 * dt + sigma * np.sqrt(dt) * z)
        O_new, delta_new = surface.price_delta(S_new[:, None], K, tau - (j + 1) * dt, vol_at(j + 1))
        gain = O_new - O - delta * (S_new - S)[:, None]  # one option, delta-hedged, over the step
        hedged += np.einsum("mpn,pn->mp", Q, gain)
        S, O, delta = S_new, O_new, delta_new
    return {
        "names": [s.name for s in strategies],
        "value": spread + hedged,
        "spread": spread,
        "hedged": hedged,
        "trades": trades,
        "Q": Q,
        "S": S,
        "vol": np.broadcast_to(vol_at(steps), (P, N)).copy(),
        "bucket_vega": book.bucket_vega(Q),
        "vega_path": np.array(path).reshape(len(path), M, P),
        "time_left": tau - steps * dt,
    }


def carried(book, out, shift, skew, loading=None):
    """What the book held at T gains when the surface then moves once more by (shift, skew),
    spot held at S_T: the exposure the terminal penalty is there to control."""
    S = out["S"][:, None]
    shape = surface.skew_shape(book.K, book.tau)[None, :]
    weight = weights(book, loading)[None, :]
    before = surface.price(S, book.K, out["time_left"], out["vol"])
    after_vol = np.maximum(out["vol"] + shift[:, None] * weight + skew[:, None] * shape, VOL_FLOOR)
    after = surface.price(S, book.K, out["time_left"], after_vol)
    return np.einsum("mpn,pn->mp", out["Q"], after - before)


def summary(x):
    """Mean, standard deviation, 1% quantile and the mean of the worst 1% of a sample."""
    x = np.sort(np.asarray(x, dtype=float))
    worst = max(1, int(round(0.01 * x.size)))
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)), "q01": float(np.quantile(x, 0.01)),
            "es01": float(x[:worst].mean())}
