"""Cabinet bargaining game Gamma_theta (spec Definition 2.2).

A multi-agent environment in which parties form coalition cabinets.  Each
period without a cabinet is a formation round: nature draws a formateur, the
formateur proposes a winning coalition and an allocation, and every named
partner votes.  A sitting cabinet pays per-period flows until the term ends
or a member walks out.

The environment exposes *decisions* rather than fixed turns: at any moment a
set of agents must act (one formateur, or all voters, or all governing
members simultaneously).  ``pending()`` lists them and ``act()`` applies their
actions together.  Rewards are booked per agent between its own decisions,
which gives each agent the semi-Markov decision process of spec Lemma 3.5.
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import EnvConfig, LAM_MAX

PROPOSE, VOTE, GOVERN = 0, 1, 2
KIND_NAMES = ("propose", "vote", "govern")

FP = 14 + 3 * LAM_MAX      # party-token width
FM = 8                     # ministry-token width
FPAIR = 4                  # (ministry, party) pair-feature width
FC = 17                    # context width
FX = 8                     # decision-specific extras


def hamilton(total: int, shares: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Largest-remainder apportionment of ``total`` units by ``shares``."""
    quota = total * shares / shares.sum()
    base = np.floor(quota).astype(int)
    rem = quota - base
    left = total - base.sum()
    if left > 0:
        order = np.lexsort((rng.random(len(rem)), -rem))
        base[order[:left]] += 1
    return base


@dataclass
class Cabinet:
    C: np.ndarray            # bool mask over parties
    f: int
    mu: np.ndarray           # ministry -> party (indivisible) or None
    p: np.ndarray            # sigma-share per party (sums to 1 over C)
    start: int = 0           # global period index at installation
    log_id: int = -1


@dataclass
class Decision:
    env: "CabinetEnv"
    agent: int
    kind: int


class CabinetEnv:
    def __init__(self, cfg: EnvConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.cabinet_log = []         # every installed cabinet (R10)
        self.episode_log = []         # per-episode delay/collapse statistics

    # ------------------------------------------------------------------ setup
    def reset(self):
        c, r = self.cfg, self.rng
        self.n = int(r.choice(c.n_choices))
        self.delta = float(r.choice(c.delta_choices))
        n, m = self.n, (1 if c.divisible else c.m)
        self.m = m
        self.sigma = np.full(m, 1.0 / m)
        # types (drawn once, persist across terms)
        nu = np.tile(self.sigma, (n, 1))
        if c.xi > 0 and not c.divisible:
            dirich = r.dirichlet(np.full(m, c.kappa_nu / m), size=n)
            nu = (1 - c.xi) * nu + c.xi * dirich
            # finite support (R20); common valuations stay exact so that the
            # welfare identity (Prop. 2.11) holds without rounding error
            nu = np.stack([self._round_simplex(v, 100) for v in nu])
        self.nu = nu
        self.z = np.round(r.random((n, c.d)) * 20) / 20
        self.lead = np.full(m, -1)
        if not c.divisible:
            self.lead[: min(c.d, m)] = np.arange(min(c.d, m))
        # seats
        if c.seat_mode == "equal":
            self.W = n
            self.w = np.ones(n, dtype=int)
        else:
            self.W = c.house_size
            self.w = self._draw_seats(r.dirichlet(np.full(n, c.alpha_seat)))
        self.chi_bar = self.w / self.W
        self.q = self.W // 2 + 1
        # clock and records
        self.t, self.ell, self.period = 0, 0, 0
        self.cabinet = None
        self.history = deque(maxlen=max(c.Lam, 1))
        self.last_vote = np.zeros(n)
        self.last_offer = np.zeros((n, n, 2))   # [i, j] = (included j, share)
        self.acc_r = np.zeros(n)
        self.acc_disc = np.ones(n)
        self.done = False
        self.ep_stats = dict(rounds=0, failed=0, collapses=0, periods=0, governed=0,
                             omega=0.0)
        self.disc_global = 1.0
        self.ret = np.zeros(n)          # discounted return per seat (evaluation)
        self.bonus = np.zeros(n)
        self._start_term(first=True)
        return self

    def _round_simplex(self, v, grid):
        """Round a probability vector to multiples of 1/grid (R20)."""
        units = hamilton(grid, v, self.rng)
        return units / grid

    def _draw_seats(self, shares):
        for _ in range(1000):
            w = hamilton(self.W, shares, self.rng)
            if w.max() < self.W // 2 + 1:
                return w
            shares = self.rng.dirichlet(np.full(self.n, self.cfg.alpha_seat))
        raise RuntimeError("could not draw a parliament without a majority party")

    def _start_term(self, first=False):
        c = self.cfg
        if not first:
            chi = self.rng.dirichlet(c.kappa_w * self.chi_bar + 1e-6)
            self.w = self._draw_seats(chi)
        self.ell = 0
        self.rounds_this_term = 0
        self.rounds_since = 0
        self._begin_period()

    # ------------------------------------------------------------ recognition
    def recognition_probs(self) -> np.ndarray:
        rule = self.cfg.rho
        if rule == "uniform":
            return np.full(self.n, 1.0 / self.n)
        if rule == "seat":
            return self.w / self.W
        # largest-first: k-th round since term start / last collapse -> k-th largest
        order = self._largest_order()
        p = np.zeros(self.n)
        p[order[self.rounds_since % self.n]] = 1.0
        return p

    def _largest_order(self):
        if not hasattr(self, "_tie") or self._tie_seats is not self.w:
            self._tie = self.rng.random(self.n)
            self._tie_seats = self.w
        return np.lexsort((self._tie, -self.w))

    # ------------------------------------------------------------- dynamics
    def _begin_period(self):
        if self.cabinet is None:
            self.phase = PROPOSE
            self.f = int(self.rng.choice(self.n, p=self.recognition_probs()))
            self.proposal = None
            self.ep_stats["rounds"] += 1
            self._pending = [Decision(self, self.f, PROPOSE)]
        elif self.cfg.exits:
            self.phase = GOVERN
            members = np.flatnonzero(self.cabinet.C)
            self.bonus = np.zeros(self.n)
            if self.cfg.ebar > 0:
                e = self.rng.exponential(1.0, size=len(members)) * self.cfg.ebar
                step = self.cfg.ebar / 20
                self.bonus[members] = np.minimum(np.round(e / step) * step, 10 * self.cfg.ebar)
            self._pending = [Decision(self, int(j), GOVERN) for j in members]
        else:
            self._pay_flows(self.cabinet)
            self._end_period()

    def pending(self):
        return [] if self.done else self._pending

    def act(self, actions: dict):
        """Apply the actions of every pending decision. ``actions[agent]``:
        PROPOSE -> (C_mask bool[n], alloc) with alloc = mu int[m] or shares float[n];
        VOTE/GOVERN -> int 0/1 (1 = accept / stay)."""
        if self.phase == PROPOSE:
            C, alloc = actions[self.f]
            self._receive_proposal(np.asarray(C, bool), alloc)
        elif self.phase == VOTE:
            votes = {j: int(actions[j]) for j in self.voters}
            for j, a in votes.items():
                self.last_vote[j] = 1 if a else -1
            if all(votes.values()):
                self._install(self.proposal)
            else:
                self.ep_stats["failed"] += 1
                self._caretaker()
                self._end_period()
        elif self.phase == GOVERN:
            stay = {d.agent: int(actions[d.agent]) for d in self._pending}
            exiters = [j for j, a in stay.items() if not a]
            if exiters:
                for j in exiters:
                    self._reward(j, self.bonus[j])
                self.ep_stats["collapses"] += 1
                self._close_cabinet(collapsed_by=exiters)
                self._caretaker()
                self.rounds_since = 0
            else:
                self._pay_flows(self.cabinet)
            self._end_period()

    def _receive_proposal(self, C, alloc):
        n, f = self.n, self.f
        assert C[f] and self.w[C].sum() >= self.q, "infeasible proposal (mask bug)"
        if self.cfg.divisible:
            p = np.zeros(n)
            p[C] = np.asarray(alloc, float)[C]
            p = p / p.sum()
            mu = None
        else:
            mu = np.asarray(alloc, int)
            assert np.all(C[mu]), "ministry assigned outside coalition"
            p = np.bincount(mu, weights=self.sigma, minlength=n)
        self.proposal = Cabinet(C=C.copy(), f=f, mu=mu, p=p)
        self.last_offer[f] = 0
        self.last_offer[f, :, 0] = C
        self.last_offer[f, :, 1] = p
        self.voters = [int(j) for j in np.flatnonzero(C) if j != f]
        if not self.voters:                      # cannot happen (no majority party)
            self._install(self.proposal)
            return
        self.phase = VOTE
        self._pending = [Decision(self, j, VOTE) for j in self.voters]

    def _install(self, cab: Cabinet):
        cab.start = self.period
        self.cabinet = cab
        rec = dict(n=self.n, w=self.w.copy(), C=cab.C.copy(), f=cab.f, p=cab.p.copy(),
                   p_nu=self._office(cab), term=self.t, delay=self.rounds_this_term,
                   delta=self.delta, rho=self.cfg.rho, duration=None, collapsed=False,
                   x_bf=None)
        cab.log_id = len(self.cabinet_log)
        self.cabinet_log.append(rec)
        self.history.append(cab)
        self._pay_flows(cab)
        if self.cfg.L is None:          # cabinet governs for ever (Prop. 2.6)
            rec["duration"] = np.inf
            self._finish()
        elif self.cfg.exits:
            self._end_period()
        else:                            # no exits: governs to the end of the term
            self._end_period()

    def _office(self, cab):
        if self.cfg.divisible:
            return cab.p.copy()
        v = np.zeros(self.n)
        np.add.at(v, cab.mu, self.nu[cab.mu, np.arange(self.m)])
        return v

    def policy_vector(self, cab):
        y = np.full(self.cfg.d, self.cfg.y0)
        if cab is not None and cab.mu is not None:
            for k in np.flatnonzero(self.lead >= 0):
                h = self.lead[k]
                y[h] = self.z[cab.mu[k], h]
        return y

    def _policy_loss(self, cab):
        if self.cfg.lam == 0:
            return np.zeros(self.n)
        y = self.policy_vector(cab)
        return self.cfg.lam * np.abs(y[None, :] - self.z).mean(axis=1)

    def _pay_flows(self, cab):
        r = self._office(cab) - self._policy_loss(cab)
        self.ep_stats["governed"] += 1
        for j in range(self.n):
            self._reward(j, r[j])

    def _caretaker(self):
        r = -self._policy_loss(None)
        if self.cfg.L is None:            # per-period equivalent of one lost period
            r = (1 - self.delta) * r
        for j in range(self.n):
            self._reward(j, r[j])

    def _reward(self, j, x):
        self.acc_r[j] += self.acc_disc[j] * x
        self.ret[j] += self.disc_global * x

    def _close_cabinet(self, collapsed_by=None):
        cab = self.cabinet
        rec = self.cabinet_log[cab.log_id]
        rec["duration"] = self.period - cab.start + 1
        rec["collapsed"] = bool(collapsed_by)
        self.cabinet = None

    def _end_period(self):
        c = self.cfg
        self.period += 1
        self.ep_stats["periods"] += 1
        self.ep_stats["omega"] += self.disc_global
        self.disc_global *= self.delta
        self.acc_disc *= self.delta
        if c.L is None:
            self.rounds_this_term += 1
            self.rounds_since += 1
            if self.rounds_this_term >= c.max_rounds:
                self._finish()
            else:
                self._begin_period()
            return
        if self.cabinet is None:
            self.rounds_this_term += 1
            self.rounds_since += 1
        self.ell += 1
        if self.ell >= c.L:
            if self.cabinet is not None:
                self._close_cabinet()
            if self.rng.random() < c.beta_T:
                self.t += 1
                self._start_term()
            else:
                self._finish()
        else:
            self._begin_period()

    def _finish(self):
        if self.cfg.L is None:
            # returns are already per-period equivalents (Prop. 2.6)
            self.ep_stats["omega"] = 1.0
        self.done = True
        self._pending = []
        self.episode_log.append(dict(self.ep_stats, n=self.n, delta=self.delta, ret=self.ret.copy()))

    # --------------------------------------------------------- SMDP rewards
    def take_interval(self, j):
        """Close agent j's current interval: return (reward, discount) and reset."""
        r, d = self.acc_r[j], self.acc_disc[j]
        self.acc_r[j], self.acc_disc[j] = 0.0, 1.0
        return float(r), float(d)

    # ------------------------------------------------------------ observation
    def observe(self, i: int, kind: int) -> dict:
        c, n, m = self.cfg, self.n, self.m
        W = self.W
        prop, cab = self.proposal if self.phase == VOTE else None, self.cabinet
        pub = c.public_types
        P = np.zeros((n, FP), np.float32)
        P[:, 0] = self.w / W
        P[:, 1] = self.recognition_probs() if self.phase == PROPOSE else 0.0
        P[i, 2] = 1.0
        if self.phase in (VOTE, PROPOSE):
            P[self.f, 3] = 1.0
        if prop is not None:
            P[:, 4] = prop.C
            P[:, 6] = prop.p
            P[:, 11] = np.where(prop.C, self.w / self.w[prop.C].sum(), 0.0)
        if cab is not None:
            P[:, 5] = cab.C
            P[:, 7] = cab.p
            P[:, 12] = np.where(cab.C, self.w / self.w[cab.C].sum(), 0.0)
        if c.Lam > 0:
            # last votes and own last offers are history, so they belong to the
            # memory window: with Lambda = 0 the policy is memoryless (H7) and
            # cannot run "reject until someone has rejected" delay conventions
            P[:, 8] = self.last_vote
            P[:, 9] = self.last_offer[i, :, 0]
            P[:, 10] = self.last_offer[i, :, 1]
        if pub and c.lam > 0:
            P[:, 13] = np.abs(self.z - self.z[i]).mean(axis=1)
        if c.Lam > 0:
            for s, h in enumerate(list(self.history)[-c.Lam:][::-1]):
                P[:, 14 + 3 * s] = h.C
                P[h.f, 15 + 3 * s] = 1.0
                P[:, 16 + 3 * s] = h.p

        M = np.zeros((m, FM), np.float32)
        PR = np.zeros((m, n, FPAIR), np.float32)
        M[:, 0] = self.sigma * m
        if not c.divisible:
            M[:, 1] = self.nu[i] * m
            if cab is not None:
                M[:, 2] = cab.mu == i
                PR[np.arange(m), cab.mu, 3] = 1.0
            if prop is not None:
                M[:, 3] = prop.mu == i
                PR[np.arange(m), prop.mu, 2] = 1.0
            y = self.policy_vector(cab)
            for k in np.flatnonzero(self.lead >= 0):
                h = self.lead[k]
                M[k, 4] = 1.0
                M[k, 5] = self.z[i, h]
                M[k, 6] = y[h]
                M[k, 7] = c.y0
                if pub:
                    PR[k, :, 1] = c.lam * np.abs(self.z[:, h] - self.z[i, h]) / c.d
            if pub:
                PR[:, :, 0] = self.nu.T * m

        ctx = np.zeros(FC, np.float32)
        ctx[0] = 0.0 if c.L is None else self.ell / c.L
        ctx[1 + self.phase] = 1.0
        ctx[4] = self.q / W
        ctx[5] = self.delta
        ctx[6] = c.beta_T
        ctx[7] = c.lam
        ctx[8] = c.xi
        ctx[9] = c.ebar * 10
        ctx[10] = float(pub)
        ctx[11] = float(c.exits)
        ctx[12] = n / 10
        ctx[13] = m / 10
        ctx[14] = float(c.divisible)
        ctx[15] = float(c.L is None)
        # patience on a usable scale: delta varies only in [0.8, 0.95]
        # (a failed-round counter is deliberately absent: spec Def. 2.4)
        ctx[16] = 10 * (1 - self.delta)

        X = np.zeros(FX, np.float32)
        if kind == VOTE:
            X[0] = prop.p[i]
            X[1] = self._office(prop)[i]
            X[2] = self._believed_policy_loss(i, prop)
            X[3] = self._believed_policy_loss(i, None)
            X[4] = prop.C.sum() / n
            X[5] = self.w[i] / self.w[prop.C].sum()
            X[6] = self.w[i] / W
        elif kind == GOVERN:
            X[0] = self.bonus[i] / max(c.ebar, 1e-8) / 10
            X[1] = self._office(cab)[i]
            X[2] = self._believed_policy_loss(i, cab)
            X[3] = self._believed_policy_loss(i, None)
            X[4] = (c.L - self.ell) / c.L if c.L else 1.0
            X[5] = self.w[i] / self.w[cab.C].sum()
            X[6] = self.bonus[i]
        else:
            X[0] = self.w[i] / W
            X[1] = self.q / W
        order = None
        if kind == PROPOSE:
            others = np.array([j for j in range(n) if j != i])
            tie = self.rng.random(len(others))
            order = others[np.lexsort((tie, -self.w[others]))]
        return dict(P=P, M=M, PR=PR, ctx=ctx, X=X, kind=kind, n=n, m=m, self_idx=i,
                    w=(self.w / W).astype(np.float32), q=self.q / W, order=order,
                    divisible=c.divisible)

    def _believed_policy_loss(self, i, cab):
        """Policy loss i expects from a cabinet.  Under private types the
        holders' ideal points are unknown; i uses their prior mean distance."""
        c = self.cfg
        if c.lam == 0:
            return 0.0
        y_true = self.policy_vector(cab)
        loss = 0.0
        for h in range(c.d):
            k = np.flatnonzero(self.lead == h)
            holder = cab.mu[k[0]] if (cab is not None and cab.mu is not None and len(k)) else None
            if holder is None:
                loss += abs(c.y0 - self.z[i, h])
            elif holder == i or c.public_types:
                loss += abs(y_true[h] - self.z[i, h])
            else:
                loss += 1.0 / 3.0
        return c.lam * loss / c.d
