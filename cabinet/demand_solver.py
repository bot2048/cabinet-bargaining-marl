"""Exact subgame-perfect equilibrium of our demand protocol (one-shot formation game).

The protocol (env.py, ``protocol="demand"``) differs from Morelli (1999): the
first mover is recognised, the others move in a uniformly random order (no
mover chooses the order), and a mover who closes automatically takes the
cheapest winning set of earlier demands (ties: fewer members, then the env's
subset order) and the residual. If nobody closes the round fails and every
party gets delta * V_i, its ex-ante value of the game (stationarity).

Published results (Morelli 1999; Montero & Vidal-Puga 2007, 2011) do not cover
this variant, so we solve it directly: backward induction over every history
of grid demands, for every move order, iterated to a fixed point in V.
Indifferent movers break ties by a fixed rule (``tie``); the range over rules
measures the grid multiplicity.

    python -m cabinet.demand_solver
"""
import itertools
import math

import numpy as np


def _close_table(k, order, w, q, G):
    """For the mover at position k: over every history of k earlier demands
    (grid units 0..G-1, flattened), the env's cheapest feasible close.
    Returns (feasible bool[S], payoff float[S, n]) with payoffs as shares."""
    n = len(w)
    i = order[k]
    S = G ** k
    grids = (np.indices((G,) * k, dtype=np.int16).reshape(k, -1) if k
             else np.zeros((0, 1), np.int16))
    best_cost = np.full(S, np.inf)
    best_size = np.full(S, np.inf)
    best_mask = np.full(S, -1)
    unit = G - 1
    for mask in range(1 << k):
        members = [b for b in range(k) if mask >> b & 1]
        if sum(w[order[b]] for b in members) + w[i] < q:
            continue
        cost = grids[members].sum(axis=0, dtype=np.int32) if members else np.zeros(S, np.int32)
        ok = cost <= unit
        better = ok & ((cost < best_cost) | ((cost == best_cost) & (len(members) < best_size)))
        best_cost = np.where(better, cost, best_cost)
        best_size = np.where(better, len(members), best_size)
        best_mask = np.where(better, mask, best_mask)
    feas = best_mask >= 0
    pay = np.zeros((S, n), np.float32)
    for b in range(k):
        inc = feas & ((best_mask >> b) & 1).astype(bool)
        pay[inc, order[b]] = grids[b][inc] / unit
    pay[feas, i] = 1.0 - best_cost[feas] / unit
    return feas, pay


def solve_order(order, w, q, G, fail, tie="close_high"):
    """Backward induction for one move order. ``fail``: payoff vector if the
    round fails. Returns the on-path (demands, closer position, payoffs)."""
    n = len(w)
    unit = G - 1
    # V[k]: payoff vectors for every history at the start of position k's move
    V = None
    choice = [None] * n
    for k in range(n - 1, -1, -1):
        i = order[k]
        feas, cpay = _close_table(k, order, w, q, G)
        S = G ** k
        if k == n - 1:
            dem = np.broadcast_to(fail, (S, G, n))       # demanding = failure
        else:
            dem = V.reshape(S, G, n)                     # next history = (h, a)
        own = dem[:, :, i]
        if tie.endswith("high"):
            a_best = (G - 1) - np.argmax(own[:, ::-1], axis=1)
        else:
            a_best = np.argmax(own, axis=1)
        dval = dem[np.arange(S), a_best]
        cbest = np.where(feas, cpay[:, i], -np.inf)
        if tie.startswith("close"):
            use_close = cbest >= dval[:, i] - 1e-12
        else:
            use_close = cbest > dval[:, i] + 1e-12
        V = np.where(use_close[:, None], cpay, dval)
        choice[k] = (use_close, a_best)
    # on-path play
    h, demands = 0, []
    for k in range(n):
        use_close, a_best = choice[k]
        if use_close[h]:
            feas, cpay = _close_table(k, order, w, q, G)
            return demands, k, cpay[h]
        a = int(a_best[h])
        demands.append(a / unit)
        h = h * G + a
    return demands, None, np.asarray(fail, float)


def solve(w, q, delta, recognition="uniform", step=0.01, tie="close_high", iters=50, tol=1e-9):
    """Stationary SPE of the one-shot demand game. Returns ex-ante values and
    the on-path outcome of every order (with its probability)."""
    w = np.asarray(w, float)
    n = len(w)
    G = int(round(1 / step)) + 1
    r = np.full(n, 1 / n) if recognition == "uniform" else w / w.sum()
    symmetric = np.allclose(w, w[0]) and recognition == "uniform"
    orders = [tuple(range(n))] if symmetric else [
        (f,) + rest for f in range(n) for rest in itertools.permutations([j for j in range(n) if j != f])]
    V = np.full(n, 1 / n)
    for _ in range(iters):
        out, newV = [], np.zeros(n)
        for o in orders:
            prob = 1.0 if symmetric else r[o[0]] / math.factorial(n - 1)
            dem, closer, pay = solve_order(list(o), w, q, G, delta * V, tie)
            out.append(dict(order=o, prob=prob, demands=dem, closer=closer, pay=pay))
            newV += prob * pay
        if symmetric:
            newV = np.full(n, newV.sum() / n)
        done = np.max(np.abs(newV - V)) < tol
        V = newV
        if done:
            break
    return V, out


def summary(w, q, delta, **kw):
    V, out = solve(w, q, delta, **kw)
    lines = []
    for o in out:
        C = o["pay"] > 1e-12
        pos = {p: j for j, p in enumerate(o["order"])}
        lines.append((o["prob"], tuple(sorted(np.flatnonzero(C))), tuple(np.round(o["pay"], 3)),
                      o["closer"], [o["order"][j] for j in range(len(o["demands"]))]))
    return V, lines


if __name__ == "__main__":
    cases = [("3 equal", [1, 1, 1], 2, "uniform"), ("5 equal", [1] * 5, 3, "uniform"),
             ("45/35/20", [45, 35, 20], 51, "seat"), ("40/30/25/5", [40, 30, 25, 5], 51, "seat")]
    for name, w, q, rec in cases:
        for step in ((0.01, 0.02) if len(w) < 5 else (0.04, 0.025)):
            for tie in ("close_high", "close_low"):
                V, lines = summary(w, q, 0.9, recognition=rec, step=step, tie=tie)
                print(f"{name:11s} step={step} tie={tie:10s} V={np.round(V, 3)}")
                for prob, C, pay, closer, movers in lines[:6]:
                    print(f"    p={prob:.3f} coalition={C} pay={pay} closer_pos={closer} demanders={movers}")
