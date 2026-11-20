"""Break each rule of hjb.py on a copy of this folder, and see an hjb check fail.

A check nobody has seen fail proves little. Run: python hjb_mutations.py
(each fault is applied to a copy, never to this folder; a run that does not end counts as missed)
"""
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
LIMIT = 900  # seconds for one run of hjb_checks.py on a faulty copy

MUTATIONS = [
    ("the exponent of the ask in (6) has the wrong sign", "hjb.py",
     "ask = np.where(can_sell, ca * np.exp(-k * (h - down)), 0.0)",
     "ask = np.where(can_sell, ca * np.exp(k * (h - down)), 0.0)"),
    ("the exponent of the bid in (6) has the wrong sign", "hjb.py",
     "bid = np.where(can_buy, cb * np.exp(-k * (h - up)), 0.0)",
     "bid = np.where(can_buy, cb * np.exp(-k * (up - h)), 0.0)"),
    ("(6) drops the wrong side at the grid's edges", "hjb.py",
     "can_sell, can_buy = q > -Q, q < Q", "can_sell, can_buy = q < Q, q > -Q"),
    ("(6) closes the bid everywhere", "hjb.py",
     "can_sell, can_buy = q > -Q, q < Q", "can_sell, can_buy = q > -Q, q < -Q"),
    ("(6) sells at the bid's flow", "hjb.py",
     "ca, cb = p.lam_a / (E * k), p.lam_b / (E * k)", "ca, cb = p.lam_b / (E * k), p.lam_b / (E * k)"),
    ("(6) forgets the e of the optimal fill", "hjb.py",
     "ca, cb = p.lam_a / (E * k), p.lam_b / (E * k)", "ca, cb = p.lam_a / k, p.lam_b / k"),
    ("the carry has the wrong sign", "hjb.py",
     "return p.C0 * q - p.beta * q * q", "return -p.C0 * q - p.beta * q * q"),
    ("the running penalty rewards inventory", "hjb.py",
     "return p.C0 * q - p.beta * q * q", "return p.C0 * q + p.beta * q * q"),
    ("the terminal penalty is forgotten", "hjb.py",
     "self.path = Path(rhs, jac, -p.alpha * self.q ** 2, T,", "self.path = Path(rhs, jac, 0 * self.q, T,"),
    ("the Jacobian of (6) has the wrong sign on its diagonal", "hjb.py",
     "[k * ask[1:], -k * (ask + bid), k * bid[:-1]]", "[k * ask[1:], k * (ask + bid), k * bid[:-1]]"),
    ("the exact bid reads the wrong neighbour", "hjb.py",
     "bid[:-1] = 1 / self.p.kappa + h[:-1] - h[1:]", "bid[:-1] = 1 / self.p.kappa + h[1:] - h[:-1]"),
    ("the exact ask forgets the 1/kappa of (5)", "hjb.py",
     "ask[1:] = 1 / self.p.kappa + h[1:] - h[:-1]", "ask[1:] = h[1:] - h[:-1]"),
    ("psi2 has the wrong sign", "hjb.py",
     "return -(p.alpha + p.beta * s * g) / (1.0 + p.alpha * self.cD * s * g)",
     "return (p.alpha + p.beta * s * g) / (1.0 + p.alpha * self.cD * s * g)"),
    ("beta enters psi2 with the wrong sign", "hjb.py",
     "return -(p.alpha + p.beta * s * g) /", "return -(p.alpha - p.beta * s * g) /"),
    ("eta squared is beta / (c D) instead of beta c D", "hjb.py",
     "x = math.sqrt(p.beta * self.cD) * s", "x = math.sqrt(p.beta / self.cD) * s"),
    ("D counts the ask's flow twice", "hjb.py",
     "return self.lam_a * self.kappa + self.lam_b * self.kappa",
     "return self.lam_a * self.kappa + self.lam_a * self.kappa"),
    ("psi1 forgets the imbalance", "hjb.py",
     "self.cD * p2 * y[0] + C * (p.lam_b - p.lam_a) * p2", "self.cD * p2 * y[0]"),
    ("psi1 is integrated from the wrong end", "hjb.py",
     "sol = solve_ivp(self._psi1_slope, (0.0, T), [0.0]", "sol = solve_ivp(self._psi1_slope, (T, 0.0), [0.0]"),
    ("the closed-form bid shades by 2q - 1", "hjb.py",
     "bid = 1 / self.p.kappa - p1 - (2 * q + 1) * p2", "bid = 1 / self.p.kappa - p1 - (2 * q - 1) * p2"),
    ("the closed-form ask leans the wrong way", "hjb.py",
     "ask = 1 / self.p.kappa + p1 + (2 * q - 1) * p2", "ask = 1 / self.p.kappa + p1 - (2 * q - 1) * p2"),
    ("the ask fills at the bid's flow", "hjb.py",
     "ra = np.where(q > -Q, p.lam_a * np.exp(-p.kappa * ask), 0.0)",
     "ra = np.where(q > -Q, p.lam_b * np.exp(-p.kappa * ask), 0.0)"),
    ("the fills ignore the grid's lower edge", "hjb.py",
     "ra = np.where(q > -Q, p.lam_a", "ra = np.where(q > -Q - 1, p.lam_a"),
    ("the spread is paid on the wrong side", "hjb.py",
     "earn = np.where(ra > 0, ra * ask, 0.0) + np.where(rb > 0, rb * bid, 0.0)",
     "earn = np.where(ra > 0, ra * bid, 0.0) + np.where(rb > 0, rb * ask, 0.0)"),
    ("a fill moves the book the wrong way", "hjb.py",
     "return ra[1:], -(ra + rb), rb[:-1]", "return rb[1:], -(ra + rb), ra[:-1]"),
    ("the forward equation forgets the transpose", "hjb.py",
     "return upper, diag, lower  # transposed", "return lower, diag, upper  # transposed"),
    ("the book starts long", "hjb.py", "P0[Q] = 1.0", "P0[Q + 1] = 1.0"),
    ("h is read with time running forwards", "hjb.py", "return self.path(self.T - t)", "return self.path(t)"),
    ("psi1 is read with time running forwards", "hjb.py",
     "return self._psi1(np.asarray(self.T - t, dtype=float))[0]", "return self._psi1(np.asarray(t, dtype=float))[0]"),
    ("the spread of q forgets to centre", "hjb.py", "P @ (q - mean) ** 2", "P @ q ** 2"),
    ("the law at T/2 is the law at T", "hjb.py",
     "mid, law = inventory_at(p, T, ex.quotes, Q, (T / 2, T))", "mid, law = inventory_at(p, T, ex.quotes, Q, (T, T))"),
    ("the percentiles read the law from the top", "hjb.py", "cdf = np.cumsum(P)", "cdf = np.cumsum(P[::-1])"),
    ("a percentile is one state too low", "hjb.py",
     "q[min(int(np.searchsorted(cdf, x)), q.size - 1)]", "q[min(int(np.searchsorted(cdf, x)) - 1, q.size - 1)]"),
    ("choose_Q compares the wrong inventories", "hjb.py",
     "b = ex2.at(0.0)[2 * Q - NEAR:2 * Q + NEAR + 1]", "b = ex2.at(0.0)[Q - NEAR:Q + NEAR + 1]"),
    ("the quote gap is counted in units of kappa", "hjb.py",
     "gap[name] = (p.kappa * (cb - eb), p.kappa * (ca - ea))", "gap[name] = ((cb - eb) / p.kappa, (ca - ea) / p.kappa)"),
    ("the value lost is counted the other way", "hjb.py",
     '"value_lost": h00 - g00', '"value_lost": g00 - h00'),
    ("the last step is the first", "hjb.py", '"last step": T - model.DT', '"last step": model.DT'),
    ("the gap at T/2 is looked for where the book is at T", "hjb.py",
     'largest(gap["t = T/2"], mid)', 'largest(gap["t = T/2"], law)'),
]


def run(item):
    name, fname, old, new = item
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        copy = os.path.join(tmp, "piece")
        shutil.copytree(HERE, copy, ignore=shutil.ignore_patterns("__pycache__", ".git", "results.json",
                                                                  "hjb_results.json", "site", "data"))
        path = os.path.join(copy, fname)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if text.count(old) != 1:
            return False, f"NOT APPLIED  {name} ({text.count(old)} matches)"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text.replace(old, new))
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        try:
            done = subprocess.run([sys.executable, "-X", "utf8", "-B", "hjb_checks.py"], cwd=copy, capture_output=True,
                                  text=True, encoding="utf-8", env=env, timeout=LIMIT)
        except subprocess.TimeoutExpired:
            return False, f"TIMED OUT  {name} (more than {LIMIT} s)"
        failed = [line.split()[1].rstrip(":") for line in done.stdout.splitlines() if line.startswith("FAIL")]
        if done.returncode == 0:
            return False, f"MISSED  {name}"
        return True, f"caught  {name}: " + (", ".join(failed) if failed else "checks crashed")


def main():
    with ThreadPoolExecutor(max_workers=min(6, os.cpu_count() or 1)) as pool:
        results = list(pool.map(run, MUTATIONS))
    for _, line in results:
        print(line)
    caught = sum(ok for ok, _ in results)
    print(f"{caught} of {len(MUTATIONS)} planted faults caught")
    import json
    with open(os.path.join(HERE, "hjb_mutations.json"), "w", encoding="utf-8") as f:
        json.dump({"caught": caught, "of": len(MUTATIONS), "faults": [line for _, line in results]}, f, indent=1)
    sys.exit(0 if caught == len(MUTATIONS) else 1)


if __name__ == "__main__":
    main()
