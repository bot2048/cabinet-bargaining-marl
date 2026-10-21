"""Verification of the demand-protocol solver (cabinet/demand_solver.py).

1. Independent implementation: a plain recursive backward induction (one
   history at a time, its own cheapest-set routine, no vectorisation) must give
   the same ex-ante values and on-path outcome as the vectorised solver on
   every move order, both tie rules, for small grids.
2. Consistency with the environment: the cheapest-set routine must choose the
   same coalition as CabinetEnv.cheapest_close on random demand histories.
3. Convergence: the fixed point in the ex-ante values V has residual 0 (to
   machine precision) when the iteration stops.
4. Robustness: the on-path splits of the reported parliaments under every
   grid and tie rule.

    python scripts/demand_solver_check.py
"""
import itertools
import json
import math
import os
import sys
from functools import lru_cache

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from cabinet.config import EnvConfig  # noqa: E402
from cabinet.demand_solver import solve, solve_order  # noqa: E402
from cabinet.env import CabinetEnv  # noqa: E402


def cheapest(movers, demands, i, w, q):
    """Cheapest winning set of earlier movers for closer i (env rule: lowest cost,
    then fewer members, then the first subset in binary order of positions)."""
    best = None
    for mask in range(1 << len(movers)):
        S = [movers[b] for b in range(len(movers)) if mask >> b & 1]
        if sum(w[j] for j in S) + w[i] < q:
            continue
        cost = sum(demands[j] for j in S)
        if cost > 1 + 1e-9:
            continue
        key = (round(cost, 9), len(S))
        if best is None or key < best[0]:
            best = (key, S, cost)
    return None if best is None else (best[1], best[2])


def brute_order(order, w, q, grid, fail, tie):
    n = len(w)

    @lru_cache(maxsize=None)
    def value(k, hist):
        """Payoff vector when position k moves after demands hist (tuple)."""
        i = order[k]
        demands = dict(zip(order[:k], hist))
        close = cheapest(list(order[:k]), demands, i, w, q)
        cpay = None
        if close is not None:
            S, cost = close
            cpay = [0.0] * n
            for j in S:
                cpay[j] = demands[j]
            cpay[i] = 1 - cost
        options = []
        for a in grid:
            options.append(tuple(fail) if k == n - 1 else value(k + 1, hist + (a,)))
        own = [o[i] for o in options]
        best = max(own)
        idx = [t for t, x in enumerate(own) if x >= best - 1e-12]
        dval = options[idx[-1] if tie.endswith("high") else idx[0]]
        if cpay is not None:
            if (tie.startswith("close") and cpay[i] >= dval[i] - 1e-12) or cpay[i] > dval[i] + 1e-12:
                return tuple(cpay)
        return dval

    return np.array(value(0, ()))


def brute_solve(w, q, delta, recognition, step, tie, iters=200):
    n = len(w)
    w = list(w)
    grid = [round(k * step, 10) for k in range(int(round(1 / step)) + 1)]
    r = np.full(n, 1 / n) if recognition == "uniform" else np.array(w, float) / sum(w)
    orders = [(f,) + rest for f in range(n) for rest in itertools.permutations([j for j in range(n) if j != f])]
    V = np.full(n, 1 / n)
    for it in range(iters):
        newV = sum(r[o[0]] / math.factorial(n - 1) * brute_order(o, w, q, grid, delta * V, tie) for o in orders)
        res = float(np.max(np.abs(newV - V)))
        V = newV
        if res < 1e-12:
            break
    return V, res, it + 1


def env_consistency(trials=2000, seed=0):
    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(trials):
        n = int(rng.integers(3, 7))
        seats = tuple(int(x) for x in rng.integers(1, 40, size=n))
        cfg = EnvConfig(n_choices=(n,), seat_mode="fixed", seats=seats, house=sum(seats), protocol="demand")
        env = CabinetEnv(cfg, seed=int(rng.integers(1 << 30))).reset()
        k = int(rng.integers(1, n))
        env.dem_k = k
        for j in env.dem_order[:k]:
            env.demands[j] = rng.integers(0, 101) / 100
        i = env.dem_order[k]
        ours = cheapest(list(env.dem_order[:k]), {j: env.demands[j] for j in env.dem_order[:k]}, i, env.w, env.q)
        theirs = env.cheapest_close(i)
        if (ours is None) != (theirs is None) or (ours is not None and sorted(ours[0]) != sorted(theirs)):
            bad += 1
    return trials, bad


def main():
    out = {"brute_force": [], "robustness": []}
    cases = [("3 equal", [1, 1, 1], 2, "uniform"), ("45/35/20", [45, 35, 20], 51, "seat"),
             ("40/30/25/5", [40, 30, 25, 5], 51, "seat"), ("5 equal", [1] * 5, 3, "uniform")]
    for name, w, q, rec in cases:
        for step in ((0.1, 0.05) if len(w) <= 4 else (0.1,)):
            for tie in ("close_high", "close_low"):
                Vb, res, it = brute_solve(w, q, 0.9, rec, step, tie)
                Vv, outv = solve(w, q, 0.9, recognition=rec, step=step, tie=tie)
                # vectorised on-path outcome of every order vs brute force
                n = len(w)
                grid = [round(k * step, 10) for k in range(int(round(1 / step)) + 1)]
                orders = [o["order"] for o in outv] if len(outv) > 1 else \
                    [(f,) + p for f in range(n) for p in itertools.permutations([j for j in range(n) if j != f])]
                diff = 0.0
                for o in orders:
                    pb = brute_order(o, w, q, grid, 0.9 * Vb, tie)
                    _, _, pv = solve_order(list(o), np.asarray(w, float), q, len(grid), 0.9 * Vv, tie)
                    diff = max(diff, float(np.max(np.abs(pb - pv))))
                row = dict(case=name, step=step, tie=tie, max_abs_diff_V=float(np.max(np.abs(Vb - Vv))),
                           max_abs_diff_onpath=diff, fixed_point_residual=res, iterations=it)
                out["brute_force"].append(row)
                print(row, flush=True)
    for name, w, q, rec in cases:
        for step in ((0.01, 0.02) if len(w) <= 4 else (0.04, 0.025)):
            for tie in ("close_high", "close_low"):
                V, outs = solve(w, q, 0.9, recognition=rec, step=step, tie=tie)
                splits = sorted({tuple(sorted(round(float(x), 3) for x in o["pay"] if x > 1e-9)) for o in outs})
                out["robustness"].append(dict(case=name, step=step, tie=tie, V=[float(x) for x in V],
                                              splits=splits))
                print(name, step, tie, splits, flush=True)
    trials, bad = env_consistency()
    out["env_consistency"] = dict(random_histories=trials, mismatches=bad)
    print(out["env_consistency"])
    json.dump(out, open(os.path.join(ROOT, "results", "demand_solver_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
