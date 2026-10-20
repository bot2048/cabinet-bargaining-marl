"""Baron-Ferejohn benchmarks.

* ``bf_symmetric_share``: closed form of spec Theorem 2.7 (level E0).
* ``solve_bf``: stationary payoffs of weighted closed-rule bargaining
  (spec Fact 2.8, Remark 2.9), used as the test oracle for level E1.

Fixed point:  v_i = rho_i (1 - delta c_i(v)) + delta v_i sum_k rho_k eta_ki.
We replace the minimum over winning coalitions by a soft-minimum with
temperature tau, solve the smooth system with a root finder, and follow the
solution as tau -> 0 (homotopy).  The candidate is then checked as in
Remark 2.9: with v fixed, choosing inclusion probabilities eta supported on
the cost-minimising coalitions so as to minimise the largest residual is a
linear program.  The returned ``resid`` is that minimum.
"""
from itertools import combinations

import numpy as np
from scipy.optimize import linprog, root


def bf_symmetric_share(n: int, delta: float) -> float:
    return 1.0 - delta * (n - 1) / (2 * n)


def winning_coalitions(w, q, k):
    """All partner sets S such that S + {k} is winning."""
    others = [j for j in range(len(w)) if j != k]
    out = []
    for r in range(len(others) + 1):
        for S in combinations(others, r):
            if w[k] + sum(w[j] for j in S) >= q:
                out.append(S)
    return out


def _members(w, q):
    n = len(w)
    mats = []
    for k in range(n):
        coal = winning_coalitions(w, q, k)
        A = np.zeros((len(coal), n))
        for r, S in enumerate(coal):
            A[r, list(S)] = 1.0
        mats.append(A)
    return mats


def _smooth_map(v, tau, mats, rho, delta):
    n = len(v)
    x = np.zeros(n)
    eta = np.zeros((n, n))
    for k in range(n):
        cost = mats[k] @ v
        z = -(cost - cost.min()) / tau
        p = np.exp(z)
        p /= p.sum()
        x[k] = 1.0 - delta * (p @ cost)
        eta[k] = p @ mats[k]
    incl = rho @ eta
    return rho * x + delta * v * incl, x, eta


def _lp_check(v, mats, rho, delta, tie_tol):
    """Minimise max_i |residual_i| over eta supported on near-minimal coalitions."""
    n = len(v)
    ties, x = [], np.zeros(n)
    for k in range(n):
        cost = mats[k] @ v
        idx = np.flatnonzero(cost <= cost.min() + tie_tol)
        ties.append(idx)
        x[k] = 1.0 - delta * cost.min()
    nvar = sum(len(t) for t in ties)
    off = np.cumsum([0] + [len(t) for t in ties])
    # variables: y (coalition weights per proposer), then t
    A_ub, b_ub = [], []
    for i in range(n):
        row = np.zeros(nvar + 1)
        for k in range(n):
            row[off[k]:off[k + 1]] = -delta * v[i] * rho[k] * mats[k][ties[k], i]
        base = v[i] - rho[i] * x[i]
        r1 = row.copy(); r1[-1] = -1.0
        A_ub.append(r1); b_ub.append(-base)          # base + row.y <= t
        r2 = -row; r2[-1] = -1.0
        A_ub.append(r2); b_ub.append(base)           # -(base + row.y) <= t
    A_eq, b_eq = [], []
    for k in range(n):
        row = np.zeros(nvar + 1)
        row[off[k]:off[k + 1]] = 1.0
        A_eq.append(row); b_eq.append(1.0)
    c = np.zeros(nvar + 1); c[-1] = 1.0
    res = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=np.array(A_eq), b_eq=b_eq,
                  bounds=[(0, None)] * (nvar + 1), method="highs")
    y = res.x[:-1]
    eta = np.zeros((n, n))
    for k in range(n):
        eta[k] = y[off[k]:off[k + 1]] @ mats[k][ties[k]]
    return float(res.x[-1]), x, eta


def solve_bf(w, q, rho, delta, tie_tol=1e-6):
    w = np.asarray(w)
    rho = np.asarray(rho, float)
    mats = _members(w, q)
    v = rho.copy()
    for tau in np.geomspace(0.1, 1e-6, 30):
        sol = root(lambda u: u - _smooth_map(u, tau, mats, rho, delta)[0], v, method="hybr")
        if sol.success or np.abs(sol.fun).max() < 1e-8:
            v = sol.x
    resid, x, eta = _lp_check(v, mats, rho, delta, tie_tol)
    return dict(v=v, x=x, eta=eta, offer=delta * v, resid=resid)
