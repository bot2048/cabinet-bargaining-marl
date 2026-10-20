"""Scripted, non-learning agents (spec BL1-BL4).

They read the environment state directly (they are reference points, not
learners) and are used as sanity checks, as fixed co-players and as the
positive control for the exploitability test.
"""
import numpy as np

from .bf_solver import solve_bf, winning_coalitions
from .env import PROPOSE, VOTE, GOVERN
from .metrics import hamilton_min


def random_feasible_coalition(env, f, rng):
    """Sequential include/exclude with the masks of spec Lemma 2.21."""
    C = np.zeros(env.n, bool)
    C[f] = True
    order = [j for j in rng.permutation(env.n) if j != f]
    for u, j in enumerate(order):
        rest = sum(env.w[k] for k in order[u + 1:])
        can_exclude = env.w[C].sum() + rest >= env.q
        C[j] = (rng.random() < 0.5) or not can_exclude
    return C


def _min_seat_coalition(env, f, rng):
    """Winning coalition containing f with the smallest seat total."""
    best, best_w = None, None
    others = [j for j in range(env.n) if j != f]
    for mask in range(1 << len(others)):
        C = np.zeros(env.n, bool)
        C[f] = True
        for b, j in enumerate(others):
            if mask >> b & 1:
                C[j] = True
        tot = env.w[C].sum()
        if tot >= env.q and (best is None or tot < best_w or
                             (tot == best_w and rng.random() < 0.5)):
            best, best_w = C, tot
    return best


def _allocate_proportional(env, C, rng):
    s = np.where(C, env.w / env.w[C].sum(), 0.0)
    if env.cfg.divisible:
        return s
    members = np.flatnonzero(C)
    counts, _ = hamilton_min(s[members], env.m, rng)
    counts = np.round(counts * env.m).astype(int)
    mu = np.repeat(members, counts)
    return rng.permutation(mu)


class Scripted:
    def __init__(self, name, seed=0):
        self.name = name
        self.rng = np.random.default_rng(seed)
        self._bf_cache = {}

    def act(self, env, i, kind):
        return getattr(self, f"_{self.name}")(env, i, kind)

    # BL1 -------------------------------------------------------------- BF
    def _bf_values(self, env):
        key = (tuple(env.w), env.cfg.rho, env.delta)
        if key not in self._bf_cache:
            rho = env.recognition_probs() if env.cfg.rho != "largest" else np.full(env.n, 1 / env.n)
            self._bf_cache[key] = solve_bf(env.w, env.q, rho, env.delta)
        return self._bf_cache[key]

    def _bf(self, env, i, kind):
        sol = self._bf_values(env)
        v, d = sol["v"], env.delta
        if kind == PROPOSE:
            # a cost-minimising coalition, ties broken uniformly at random
            coal = winning_coalitions(env.w, env.q, i)
            cost = np.array([v[list(S)].sum() for S in coal])
            ties = np.flatnonzero(cost <= cost.min() + 1e-7)
            C = np.zeros(env.n, bool)
            C[[i, *coal[self.rng.choice(ties)]]] = True
            shares = np.where(C, d * v, 0.0)
            shares[i] = 0.0
            shares[i] = 1.0 - shares.sum()
            return C, self._to_alloc(env, C, shares)
        if kind == VOTE:
            return int(env.proposal.p[i] >= d * v[i] - 1e-6)
        return 1

    # BL2 ---------------------------------------------------------- Gamson
    def _gamson(self, env, i, kind):
        if kind == PROPOSE:
            C = _min_seat_coalition(env, i, self.rng)
            return C, _allocate_proportional(env, C, self.rng)
        if kind == VOTE:
            p, C = env.proposal.p, env.proposal.C
            s = np.where(C, env.w / env.w[C].sum(), 0.0)
            dg = 0.5 * np.abs(p - s).sum()
            if env.cfg.divisible:
                return int(dg <= 0.02)
            _, dmin = hamilton_min(s[C], env.m, self.rng)
            return int(dg <= dmin + 1 / (2 * env.m) + 1e-9)
        return 1

    # BL3 ------------------------------------------------- myopic BF heuristic
    def _myopic(self, env, i, kind):
        vhat = env.w / env.W
        d = env.delta
        if kind == PROPOSE:
            C = _min_seat_coalition(env, i, self.rng)
            target = np.where(C, d * vhat, 0.0)
            target[i] = 0.0
            if env.cfg.divisible:
                shares = target.copy()
                shares[i] = 1.0 - target.sum()
                return C, shares
            val = env.nu if env.cfg.public_types else np.tile(env.sigma, (env.n, 1))
            mu = np.full(env.m, i)
            got = np.zeros(env.n)
            for j in np.flatnonzero(C):
                if j == i:
                    continue
                for k in np.argsort(-val[j]):
                    if got[j] >= target[j]:
                        break
                    if mu[k] == i:
                        mu[k] = j
                        got[j] += val[j, k]
            return C, mu
        if kind == VOTE:
            own = env._office(env.proposal)[i]
            return int(own >= d * vhat[i] - 1e-9)
        return 1

    # BL4 ---------------------------------------------------------- random
    def _random(self, env, i, kind):
        if kind == PROPOSE:
            C = random_feasible_coalition(env, i, self.rng)
            if env.cfg.divisible:
                return C, np.where(C, self.rng.dirichlet(np.ones(env.n)), 0.0)
            return C, self.rng.choice(np.flatnonzero(C), size=env.m)
        return int(self.rng.random() < 0.5)

    def _to_alloc(self, env, C, shares):
        if env.cfg.divisible:
            return shares
        # indivisible: assign ministries to approach the target shares
        members = np.flatnonzero(C)
        counts, _ = hamilton_min(shares[members] / shares[members].sum(), env.m, self.rng)
        return np.repeat(members, np.round(counts * env.m).astype(int))
