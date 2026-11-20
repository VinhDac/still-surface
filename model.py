"""The option book of section 3.2 and the closed-form quotes of section 3.1.

Every parameter below is the paper's, section 3.2, except the implied vol surface (surface.py,
an assumption). Equation numbers are the author manuscript's (Lucic and Tse, 30 Jan 2025).

    44 options: tau in {3, 6, 12, 36} months, K/S0 from 0.90 to 1.10 by 0.02.
    6 buckets (Table 1): short tau < 9/12 or long; K/S0 <= 0.95 low, >= 1.05 high, else moderate.
    Terminal penalty alpha_k (sum over bucket k of Q_T * 1% * Vega_0)^2, beta_k = 0.
    lambda_a = 252 * 100 * (1 - |1 - K/S0|) a year, lambda_b = 1.15 lambda_a.
    kappa = 5 (1 - 5 |1 - K/S0|) / (1% * Vega_0), the same on both sides.
    sigma = 17% (the market maker's view, and here also the true vol), S0 = 100,
    T = half a trading day = 0.5/252, a step of 15 seconds = 0.25 / (60 * 24 * 252).
"""
import math

import numpy as np

import surface

TAUS = (0.25, 0.5, 1.0, 3.0)
MONEYNESS = tuple(round(0.90 + 0.02 * i, 2) for i in range(11))
SIGMA = 0.17
T = 0.5 / 252
DT = 0.25 / (60 * 24 * 252)
STEPS = round(T / DT)  # 2880
ALPHA = (0.2, 0.01, 0.2, 0.2, 0.01, 0.2)
IMBALANCE = 1.15  # lambda_b / lambda_a: the market is a net seller of options
C = 2 / math.e  # the 2 e^-1 of equations (17) to (20)


class Book:
    """The 44 options, their buckets, order flow and initial greeks, on a given surface."""

    def __init__(self, vol_surface=surface.implied_vol):
        tau, kt = np.meshgrid(TAUS, MONEYNESS, indexing="ij")
        self.tau = tau.ravel()
        self.moneyness = kt.ravel()
        self.K = surface.S0 * self.moneyness
        self.n = self.K.size
        self.vol0 = np.asarray(vol_surface(self.K, self.tau), dtype=float)
        self.vega0 = surface.vega(surface.S0, self.K, self.tau, self.vol0)
        self.w = 0.01 * self.vega0  # omega_n = 1% * Vega_0, the percentage-vega of one option
        dist = np.abs(1 - self.moneyness)
        self.lam_a = 252 * 100 * (1 - dist)
        self.lam_b = IMBALANCE * self.lam_a
        self.kappa = 5 * (1 - 5 * dist) / self.w
        low = self.moneyness <= 0.95 + 1e-9
        high = self.moneyness >= 1.05 - 1e-9
        long_ = self.tau >= 9 / 12
        self.bucket = 3 * long_ + np.where(low, 0, np.where(high, 2, 1))  # 0..5 = Buckets 1..6
        # [C0]_n: the expected vol-arbitrage profit rate of one option, gamma frozen at t = 0.
        gamma = surface.dollar_gamma(surface.S0, self.K, self.tau, self.vol0)
        self.C0 = 0.5 * (SIGMA ** 2 - self.vol0 ** 2) * gamma

    def bucket_vega(self, Q):
        """Terminal percentage-vega of each bucket: sum of Q * 1% * Vega_0, shape (..., 6)."""
        Q = np.asarray(Q, dtype=float)
        out = np.zeros(Q.shape[:-1] + (6,))
        for k in range(6):
            sel = self.bucket == k
            out[..., k] = (Q[..., sel] * self.w[sel]).sum(-1)
        return out


def penalty_matrix(book, per_bucket=None, total=0.0, loading=None):
    """A or B of section 3.1, the sum over risk factors k of weight_k (gamma_k * omega)(gamma_k * omega)':
    per_bucket, six weights on Table 1's buckets (gamma_k the bucket's indicator); total, one weight
    on the book's whole percentage-vega, the factor section 3 names for a parallel shift (gamma = 1),
    or, given a loading (one number an option), on the vega the move loads on (gamma = loading)."""
    M = np.zeros((book.n, book.n))
    if per_bucket is not None:
        per = np.asarray(per_bucket, dtype=float)
        same = book.bucket[:, None] == book.bucket[None, :]
        M += same * per[book.bucket][:, None] * np.outer(book.w, book.w)
    if total:
        g = book.w if loading is None else book.w * np.asarray(loading, dtype=float)
        M += total * np.outer(g, g)
    return M


def solve_numerically(A, B, book, steps=STEPS, dt=DT, substeps=4):
    """Theta_2 and theta_1 of (17) with a running penalty B, which (19) and (20) leave out, by
    fourth-order Runge-Kutta backward from T on the 15-second grid, gamma frozen at t = 0 as (20)
    freezes it (so theta_1 does not depend on s):

        dTheta2/dt = B - c Theta2 D Theta2,  Theta2(T) = -A
        dtheta1/dt = -(C0 + c Theta2 D theta1 + c Theta2 (lam_b - lam_a)
                       + c Theta2 diag(Theta2) (lam_b kappa_b - lam_a kappa_a)),  theta1(T) = 0
    """
    b = book
    d = b.lam_b * b.kappa + b.lam_a * b.kappa
    flow = b.lam_b - b.lam_a
    tilt = b.lam_b * b.kappa - b.lam_a * b.kappa

    def back(th2, th1):  # the slopes in s = T - t, the time left
        return (-B + C * th2 @ (d[:, None] * th2),
                b.C0 + C * th2 @ (d * th1) + C * th2 @ flow + C * th2 @ (np.diag(th2) * tilt))

    h = dt / substeps
    th2, th1 = -np.array(A, dtype=float), np.zeros(b.n)
    theta2, theta1 = np.empty((steps + 1, b.n, b.n)), np.empty((steps + 1, b.n))
    theta2[steps], theta1[steps] = th2, th1
    for j in range(steps - 1, -1, -1):
        for _ in range(substeps):
            k1 = back(th2, th1)
            k2 = back(th2 + 0.5 * h * k1[0], th1 + 0.5 * h * k1[1])
            k3 = back(th2 + 0.5 * h * k2[0], th1 + 0.5 * h * k2[1])
            k4 = back(th2 + h * k3[0], th1 + h * k3[1])
            th2 = th2 + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            th1 = th1 + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        th2 = 0.5 * (th2 + th2.T)  # symmetric in exact arithmetic: keep it so against rounding
        theta2[j], theta1[j] = th2, th1
    return theta2, theta1


class Optimal:
    """Quotes (18), with Theta_2 from (19) and theta_1 from (20), on the 15-second grid.

    delta_b = 1/kappa_b - theta_1(t) - diag(Theta_2(t)) - 2 Theta_2(t) q
    delta_a = 1/kappa_a + theta_1(t) - diag(Theta_2(t)) + 2 Theta_2(t) q

    The paper's example has no running penalty. beta (six bucket weights) and beta_total (one
    weight on the whole book's percentage-vega, or on the vega a move loads on, given a loading)
    add one, B; with B the closed forms do not hold and (17) is solved numerically.
    """

    def __init__(self, book, alpha_scale=1.0, name=None, steps=STEPS, dt=DT,
                 beta=None, beta_total=0.0, loading=None, substeps=4):
        self.name = name or f"optimal x{alpha_scale:g}"
        b = book
        alpha = np.array(ALPHA) * alpha_scale
        same = b.bucket[:, None] == b.bucket[None, :]
        self.A = same * alpha[b.bucket][:, None] * np.outer(b.w, b.w)
        self.B = penalty_matrix(b, beta, beta_total, loading)
        if self.B.any():
            theta2, theta1 = solve_numerically(self.A, self.B, b, steps, dt, substeps)
        else:
            theta2, theta1 = self._closed_form(b, steps, dt)
        diag2 = np.einsum("jnn->jn", theta2)
        self.theta1, self.theta2 = theta1, theta2
        self.base_b = 1 / b.kappa - theta1 - diag2
        self.base_a = 1 / b.kappa + theta1 - diag2

    def _closed_form(self, b, steps, dt):
        """(19) and (20), the paper's own route when B = 0."""
        d = b.lam_b * b.kappa + b.lam_a * b.kappa  # D = diag(lambda_b kappa_b + lambda_a kappa_a)
        eye = np.eye(b.n)
        horizon = steps * dt
        times = np.arange(steps + 1) * dt
        G = eye[None] + C * (horizon - times)[:, None, None] * (self.A * d[None, :])[None]  # I + c (T-t) A D
        theta2 = -np.linalg.solve(G, np.broadcast_to(self.A, G.shape))  # (19): -A (I + c (T-t) D A)^-1
        diag2 = np.einsum("jnn->jn", theta2)
        f = b.C0[None] + C * theta2 @ (b.lam_b - b.lam_a) \
            + C * np.einsum("jnm,jm->jn", theta2, diag2 * (b.lam_b * b.kappa - b.lam_a * b.kappa))
        integrand = np.einsum("jnm,jm->jn", G, f)  # (I + c (T-u) A D) f(u)
        tail = np.zeros_like(integrand)  # the integral from t_j to T, by the trapezoid rule
        for j in range(steps - 1, -1, -1):
            tail[j] = tail[j + 1] + 0.5 * dt * (integrand[j] + integrand[j + 1])
        theta1 = np.linalg.solve(G, tail[..., None])[..., 0]  # (20)
        return theta2, theta1

    def quotes(self, j, Q):
        """Bid and ask spreads at step j for inventories Q, shape (paths, 44)."""
        shade = 2 * (Q @ self.theta2[j])  # Theta_2 is symmetric
        return self.base_b[j] - shade, self.base_a[j] + shade


class ZeroIntelligence:
    """A constant spread of `units` percentage-vega on both sides of every option."""

    def __init__(self, book, units):
        self.name = f"ZI {units:g}% vega"
        self.units = units
        self.spread = units * book.w

    def quotes(self, j, Q):
        return self.spread, self.spread


def strategies(book, scales=(1.0, 10.0, 0.1), zi=(0.1, 0.5, 1.0)):
    """The six of Figure 1(b): optimal at baseline, high (x10) and low (x0.1) penalty, and three ZI."""
    names = {1.0: "optimal baseline", 10.0: "optimal high (x10)", 0.1: "optimal low (x0.1)"}
    out = [Optimal(book, s, names.get(s)) for s in scales]
    return out + [ZeroIntelligence(book, u) for u in zi]
