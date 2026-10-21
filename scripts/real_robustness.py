"""Robustness of the real-cabinet regression (no agents, CPU seconds).

share_ij = alpha + beta * seat share in coalition + gamma * PM party, 195 cabinets.

1. Composition: shares, seat shares and the PM dummy each sum to one within a
   cabinet, so 1 = k*alpha + beta + gamma for a k-party cabinet; a constant
   alpha is misspecified across cabinet sizes. Variant: alpha / k instead.
2. The PM party is usually the largest coalition party: control for a
   largest-party dummy and count the cabinets that identify gamma.
3. Few clusters (31 countries): wild cluster bootstrap (restricted under
   gamma = 0, Webb six-point weights) p-value, and a bootstrap-t interval.
4. Voting weights instead of seats: minimal integer weights of the full
   parliament (integer program), as in Ansolabehere et al. (2005).

    python scripts/real_robustness.py
"""
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from seki_eval import load_cabinets  # noqa: E402

WEBB = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])


def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b, y - X @ b


def cluster_se(X, e, g):
    XtX = np.linalg.inv(X.T @ X)
    meat = np.zeros((X.shape[1],) * 2)
    for c in np.unique(g):
        s = X[g == c].T @ e[g == c]
        meat += np.outer(s, s)
    G = len(np.unique(g))
    adj = G / (G - 1) * (len(y_global) - 1) / (len(y_global) - X.shape[1])
    return np.sqrt(np.diag(adj * XtX @ meat @ XtX))


def wild_cluster(X, y, g, j, B=9999, seed=0):
    """Restricted wild cluster bootstrap-t p-value for H0: b_j = 0 (Webb weights),
    and an unrestricted bootstrap-t 95% interval."""
    rng = np.random.default_rng(seed)
    b, e = ols(X, y)
    t0 = b[j] / cluster_se(X, e, g)[j]
    Xr = np.delete(X, j, axis=1)
    br, er = ols(Xr, y)
    clusters = np.unique(g)
    idx = np.searchsorted(clusters, g)
    ts, tu = [], []
    for _ in range(B):
        v = rng.choice(WEBB, len(clusters))[idx]
        ys = Xr @ br + er * v
        bs, es = ols(X, ys)
        ts.append(bs[j] / cluster_se(X, es, g)[j])
        yu = X @ b + e * v
        bu, eu = ols(X, yu)
        tu.append((bu[j] - b[j]) / cluster_se(X, eu, g)[j])
    ts, tu = np.array(ts), np.array(tu)
    se = cluster_se(X, e, g)[j]
    p = float(np.mean(np.abs(ts) >= abs(t0)))
    lo, hi = b[j] - np.quantile(tu, 0.975) * se, b[j] - np.quantile(tu, 0.025) * se
    return float(b[j]), float(se), p, [float(lo), float(hi)]


def min_integer_weights(seats, q):
    """Smallest integer weights (and quota) that represent the same simple game."""
    n = len(seats)
    win = lambda S: seats[list(S)].sum() >= q
    A, lb, ub = [], [], []
    for r in range(n + 1):
        for S in itertools.combinations(range(n), r):
            row = np.zeros(n + 1)
            row[list(S)] = 1
            row[n] = -1
            if win(S):
                if all(not win(tuple(x for x in S if x != i)) for i in S):     # minimal winning
                    A.append(row); lb.append(0); ub.append(np.inf)
            else:
                if all(win(S + (i,)) for i in range(n) if i not in S):         # maximal losing
                    A.append(row); lb.append(-np.inf); ub.append(-1)
    c = np.r_[np.ones(n), 0]
    res = milp(c, constraints=LinearConstraint(np.array(A), lb, ub), integrality=np.ones(n + 1),
               bounds=Bounds(0, np.inf))
    return np.round(res.x[:n]).astype(int), int(round(res.x[n]))


def main():
    global y_global
    cabs = load_cabinets()
    rows = []
    for c in cabs:
        mem = np.flatnonzero(c["member"])
        k = len(mem)
        s = c["seats"][mem] / c["seats"][mem].sum()
        largest = mem[np.argmax(c["seats"][mem])]
        # parties beyond the 12 largest hold too few seats to matter for the weights
        top = min(12, len(c["seats"]))
        seats_top = np.round(c["seats"][:top]).astype(int)
        w, Q = min_integer_weights(seats_top, int(c["house"] // 2 + 1))
        wv = np.zeros(len(c["seats"]))
        wv[:top] = w
        ws = wv[mem] / max(wv[mem].sum(), 1e-12)
        for t, j in enumerate(mem):
            rows.append(dict(country=c["country"], cab=c["name"], k=k, p=c["share"][j], s=s[t],
                             pm=float(j == c["pm"]), largest=float(j == largest), w=ws[t]))
    y = np.array([r["p"] for r in rows]); y_global = y
    g = np.array([r["country"] for r in rows])
    col = lambda key: np.array([r[key] for r in rows])
    one = np.ones(len(y))
    specs = {
        "baseline": np.c_[one, col("s"), col("pm")],
        "alpha_over_k": np.c_[1 / col("k"), col("s"), col("pm")],
        "largest_party_control": np.c_[one, col("s"), col("pm"), col("largest")],
        "min_integer_weights": np.c_[one, col("w"), col("pm")],
        "weights_and_seats": np.c_[one, col("s"), col("w"), col("pm")],
    }
    pm_col = {"baseline": 2, "alpha_over_k": 2, "largest_party_control": 2, "min_integer_weights": 2,
              "weights_and_seats": 3}
    out = {}
    for name, X in specs.items():
        b, e = ols(X, y)
        gam, se, p, ci = wild_cluster(X, y, g, pm_col[name], B=1999)
        out[name] = dict(coef=[float(v) for v in b], gamma=gam, cluster_se=se, wild_p=p, wild_ci=ci,
                         r2=float(1 - e.var() / y.var()))
        print(f"{name:22s} coef={np.round(b, 3)} gamma={gam:+.3f} wild-cluster 95% CI "
              f"[{ci[0]:+.3f},{ci[1]:+.3f}] p={p:.3f} R2={out[name]['r2']:.3f}")
    cab_pm_not_largest = len({r["cab"] for r in rows if r["pm"] == 1 and r["largest"] == 0})
    dummies = sum(1 for r in rows if r["w"] == 0)
    out["cabinets"] = len(cabs)
    out["countries"] = len(set(g))
    out["cabinets_pm_not_largest"] = cab_pm_not_largest
    out["members_with_zero_weight"] = dummies
    out["members"] = len(rows)
    print(f"cabinets {len(cabs)}, PM party not the largest member in {cab_pm_not_largest}; "
          f"{dummies} of {len(rows)} members are dummies (weight 0)")
    json.dump(out, open(os.path.join(ROOT, "results", "real_robustness.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
