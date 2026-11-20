"""Break each rule on a copy of this folder, and see a check fail.

A check nobody has seen fail proves little. Run: python mutations.py
(the copies take data/ with them, so no fault fetches the VIX again)
"""
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))

MUTATIONS = [
    ("the hedge has the wrong sign", "simulate.py",
     "gain = O_new - O - delta * (S_new - S)[:, None]", "gain = O_new - O + delta * (S_new - S)[:, None]"),
    ("the spot hardly moves", "simulate.py",
     "sigma * np.sqrt(dt) * z)", "sigma * dt * z)"),
    ("a fill pays its spread away", "simulate.py",
     "spread[m] += (nb * db + na * da).sum(1)", "spread[m] -= (nb * db + na * da).sum(1)"),
    ("bids fill at the asks' rate", "simulate.py",
     "pb = -np.expm1(-lam_b * np.exp(-book.kappa * db))", "pb = -np.expm1(-lam_a * np.exp(-book.kappa * db))"),
    ("the surface never moves the marks", "simulate.py",
     "if moves is None else (lambda j: moves.vol(book, j))", "if True else None"),
    ("the closing book counts shorts as longs", "simulate.py",
     'return np.einsum("mpn,pn->mp", out["Q"], after - before)',
     'return np.einsum("mpn,pn->mp", np.abs(out["Q"]), after - before)'),
    ("the market is a net buyer", "model.py", "IMBALANCE = 1.15", "IMBALANCE = 1 / 1.15"),
    ("kappa loses the wings' discount", "model.py",
     "self.kappa = 5 * (1 - 5 * dist) / self.w", "self.kappa = 5 * (1 - dist) / self.w"),
    ("percentage-vega taken as the whole vega", "model.py",
     "self.w = 0.01 * self.vega0", "self.w = self.vega0"),
    ("the edge has the wrong sign", "model.py",
     "self.C0 = 0.5 * (SIGMA ** 2 - self.vol0 ** 2) * gamma", "self.C0 = 0.5 * (self.vol0 ** 2 - SIGMA ** 2) * gamma"),
    ("Theta2 has the wrong sign", "model.py",
     "theta2 = -np.linalg.solve(G, np.broadcast_to(self.A, G.shape))",
     "theta2 = np.linalg.solve(G, np.broadcast_to(self.A, G.shape))"),
    ("theta1 forgets the inverse in (20)", "model.py",
     "theta1 = np.linalg.solve(G, tail[..., None])[..., 0]", "theta1 = tail"),
    ("the integral of (20) runs the wrong way", "model.py",
     "tail[j] = tail[j + 1] + 0.5 * dt", "tail[j] = tail[j + 1] - 0.5 * dt"),
    ("bucket 1 ends at 0.93", "model.py",
     "low = self.moneyness <= 0.95 + 1e-9", "low = self.moneyness <= 0.93"),
    ("half a day is a whole day", "model.py", "T = 0.5 / 252", "T = 1 / 252"),
    ("the surface loses its skew", "surface.py",
     "return a + b * z + c * z * z + d * z ** 3", "return a + c * z * z + d * z ** 3"),
    ("a skew move is the same at every maturity", "surface.py",
     "return _z(k, tau) / ref", "return _z(k, 0.25) / ref"),
    ("vega forgets sqrt(tau)", "surface.py",
     "return s * np.sqrt(tau) * np.exp(-0.5 * d1 * d1) / SQRT_2PI", "return s * np.exp(-0.5 * d1 * d1) / SQRT_2PI"),
    ("half a day is half a day's move", "data.py",
     "return np.asarray(changes) * math.sqrt(0.5) / 100", "return np.asarray(changes) * 0.5 / 100"),
    ("the VIX band reads the day's own close", "data.py",
     "prev, changes = closes[:-1], np.diff(closes)", "prev, changes = closes[1:], np.diff(closes)"),
    ("the vega path misses the close", "simulate.py", "if j + 1 in snaps:", "if j in snaps:"),
    ("the running penalty pushes the wrong way", "model.py",
     "return (-B + C * th2 @ (d[:, None] * th2),", "return (B + C * th2 @ (d[:, None] * th2),"),
    ("theta1 loses its discount", "model.py",
     "b.C0 + C * th2 @ (d * th1) + C * th2 @ flow", "b.C0 + C * th2 @ flow"),
    ("Runge-Kutta loses its last stage", "model.py",
     "th2 = th2 + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])", "th2 = th2 + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k3[0])"),
    ("the running penalty is dropped", "model.py",
     "self.B = penalty_matrix(b, beta, beta_total, loading)", "self.B = penalty_matrix(b, None, 0.0, loading)"),
    ("the loaded factor forgets its loading", "model.py",
     "g = book.w if loading is None else book.w * np.asarray(loading, dtype=float)", "g = book.w"),
    ("the term move weights every maturity alike", "simulate.py",
     "return np.ones(book.n) if loading is None else np.asarray(loading(book.tau), dtype=float)", "return np.ones(book.n)"),
    ("the power law runs the wrong way", "data.py",
     "p = -math.log(slope) / math.log(DAYS_3M / DAYS_VIX)", "p = math.log(slope) / math.log(DAYS_3M / DAYS_VIX)"),
    ("the loading is the ratio of swings, not the slope", "data.py",
     "slope = float(np.cov(x, y)[0, 1] / np.var(x, ddof=1))", "slope = float(np.std(y) / np.std(x))"),
    ("days one index misses are kept", "data.py",
     "both = sorted(set(d1) & set(d3))", "both = sorted(set(d1) | set(d3))"),
    ("the cost is counted the wrong way", "study.py", "paid = base - value", "paid = value - base"),
    ("the in-session cut is measured on the whole value", "study.py",
     'float(1 - moved.std(ddof=1) / (pooled(BASELINE, sc, "value") - base).std(ddof=1))',
     'float(1 - moved.std(ddof=1) / pooled(BASELINE, sc, "value").std(ddof=1))'),
]


def run(item):
    name, fname, old, new = item
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        copy = os.path.join(tmp, "piece")
        shutil.copytree(HERE, copy, ignore=shutil.ignore_patterns("__pycache__", ".git", "results.json", "site"))
        path = os.path.join(copy, fname)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if text.count(old) != 1:
            return False, f"NOT APPLIED  {name} ({text.count(old)} matches)"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text.replace(old, new))
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        done = subprocess.run([sys.executable, "-X", "utf8", "checks.py"], cwd=copy, capture_output=True,
                              text=True, encoding="utf-8", env=env)
        failed = [line.split()[1].rstrip(":") for line in done.stdout.splitlines() if line.startswith("FAIL")]
        if done.returncode == 0:
            return False, f"MISSED  {name}"
        return True, f"caught  {name}: " + (", ".join(failed) if failed else "checks crashed")


def main():
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, MUTATIONS))
    for _, line in results:
        print(line)
    caught = sum(ok for ok, _ in results)
    print(f"{caught} of {len(MUTATIONS)} planted faults caught")
    import json
    with open(os.path.join(HERE, "mutations.json"), "w", encoding="utf-8") as f:
        json.dump({"caught": caught, "of": len(MUTATIONS), "faults": [line for _, line in results]}, f, indent=1)
    sys.exit(0 if caught == len(MUTATIONS) else 1)


if __name__ == "__main__":
    main()
