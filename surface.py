"""Black-Scholes and the market's implied vol surface for the example of section 3.2.

Rates and dividends are zero, as in the paper. Calls stand for every option: delta-hedged,
a call and a put of the same strike are the same position (section 2, last remark).

ASSUMPTION. The paper shows its surface only as a picture, Figure 1(a); the numbers behind it
are in Lucic and Tse (2024), SSRN 4729290, which this study does not use. FIGURE_1A holds
points read by eye off that picture, and the surface is a four-parameter form fitted to them:

    sigma(K, tau) = a + b z + c z^2 + d z^3,    z = ln(K / S0) / sqrt(tau)

one at-the-money level for every maturity (the four curves cross near 0.176), a skew that
falls like 1/sqrt(tau), and a little smile. It is fitted on moneyness 0.85 to 1.15, which holds
all 44 strikes (0.90 to 1.10); outside that range it is not meant to be used.
"""
import math

import numpy as np
from scipy.special import ndtr

S0 = 100.0
SQRT_2PI = math.sqrt(2 * math.pi)

# Figure 1(a), read by eye: implied vol at these moneyness levels K/S0, one row per maturity.
MONEYNESS_READ = (0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15)
FIGURE_1A = {
    0.25: (0.280, 0.252, 0.215, 0.176, 0.150, 0.139, 0.143),
    0.5: (0.255, 0.235, 0.205, 0.176, 0.153, 0.142, 0.139),
    1.0: (0.232, 0.215, 0.196, 0.176, 0.158, 0.146, 0.140),
    3.0: (0.208, 0.199, 0.188, 0.176, 0.165, 0.156, 0.149),
}


def _z(k, tau):
    return np.log(np.asarray(k, dtype=float) / S0) / np.sqrt(tau)


def fit():
    """Least squares of the four-parameter form on the points read off Figure 1(a)."""
    rows, vols = [], []
    for tau, row in FIGURE_1A.items():
        for m, v in zip(MONEYNESS_READ, row):
            z = math.log(m) / math.sqrt(tau)
            rows.append((1.0, z, z * z, z ** 3))
            vols.append(v)
    params, *_ = np.linalg.lstsq(np.array(rows), np.array(vols), rcond=None)
    return params


PARAMS = fit()


def implied_vol(k, tau):
    """The fitted surface: implied vol of strike k and maturity tau (years), spot S0."""
    a, b, c, d = PARAMS
    z = _z(k, tau)
    return a + b * z + c * z * z + d * z ** 3


def skew_shape(k, tau):
    """How a change of skew moves each vol: the surface's own skew term, z, scaled so that the
    3-month 90% minus 110% spread changes by exactly one unit (longer maturities move less)."""
    ref = (math.log(0.9) - math.log(1.1)) / math.sqrt(0.25)
    return _z(k, tau) / ref


def _d1(s, k, tau, vol):
    sd = vol * np.sqrt(tau)
    return np.log(s / k) / sd + 0.5 * sd, sd


def price_delta(s, k, tau, vol):
    """Call price and delta."""
    d1, sd = _d1(s, k, tau, vol)
    n1 = ndtr(d1)
    return s * n1 - k * ndtr(d1 - sd), n1


def price(s, k, tau, vol):
    return price_delta(s, k, tau, vol)[0]


def vega(s, k, tau, vol):
    """dO/dsigma, sigma in decimals: 0.01 * vega is the paper's percentage-vega."""
    d1, _ = _d1(s, k, tau, vol)
    return s * np.sqrt(tau) * np.exp(-0.5 * d1 * d1) / SQRT_2PI


def dollar_gamma(s, k, tau, vol):
    """s^2 d2O/ds2 = K phi(d2) / (sigma sqrt(tau)), the paper's Gamma$."""
    d1, sd = _d1(s, k, tau, vol)
    d2 = d1 - sd
    return k * np.exp(-0.5 * d2 * d2) / SQRT_2PI / sd
