"""Value of proposal power (spec v0.5, Definition 1.8; Algorithm 4).

Psi_in compares the same party, in the same formation state and the same
coalition, as proposer and as partner; Psi_ex compares its payoff when it is
recognised with its payoff when someone else is. Both are counterfactual, so
they are estimated by replays from saved formation states:

* probe states: fresh episodes up to their first recognition (the first
  formation round of an episode); the state is copied with its random streams,
  so every arm of a state uses common random numbers;
* do(f = j): the copy's formateur (or first mover) is overridden;
* do(f = j, coalition = C): in addition, j's coalition decisions are forced to
  the coalition C formed from that state under the profile's own recognition;
  the allocation and all later play are sampled from the profile (Lemma 3.13);
* payoffs run to the end of the current term, in per-period units, split into
  the salience, taste, policy and exit components of Definition 2.4.

The primary Psi_in uses only pairs for which C is on path for both proposers
(estimated probability of proposing C at least ``eta_min``); the all-pairs value
is reported as a sensitivity check.

    from cabinet.proposal_power import probe
    res = probe(cfg, {"learn": net}, n_states=500)
"""
import copy

import numpy as np
import torch

from .baselines import Scripted
from .env import CabinetEnv, PROPOSE
from .policy import pad_obs, to_torch
from .ppo import _env_action

COMPONENTS = ("sigma", "nu", "pol", "e")


def _kappa(env, cfg):
    """Normaliser of Definition 2.4: discount mass from the probe state to the end
    of the term (one-shot games already pay per-period equivalents)."""
    if cfg.L is None:
        return 1.0
    d = env.delta
    return (1 - d ** cfg.L) / (1 - d)


def _run(envs, controllers, forced, cfg):
    """Play every env to the end of its current term (or of the episode).
    ``forced[e]`` = (agent, coalition mask) imposes that agent's first proposal's
    coalition decisions. Returns the component payoffs (n x 4) of each env."""
    N = cfg.n_max
    M = 1 if cfg.divisible else cfg.m
    t0 = [env.t for env in envs]
    active = [e for e, env in enumerate(envs) if not env.done]
    pending_force = dict(forced)
    mem = {}                                     # (env, seat) -> GRU state (probe states start episodes)
    while active:
        groups = {}
        for e in active:
            env = envs[e]
            for d in env.pending():
                groups.setdefault(env.seats[d.agent], []).append((e, d, env.observe(d.agent, d.kind)))
        actions = {e: {} for e in active}
        for name, items in groups.items():
            ctrl = controllers[name]
            if isinstance(ctrl, Scripted):
                for e, d, _ in items:
                    fz = pending_force.get(e)
                    if fz is not None and d.kind == PROPOSE and d.agent == fz[0]:
                        actions[e][d.agent] = ctrl.act_forced(envs[e], d.agent, fz[1])
                        pending_force.pop(e)
                    else:
                        actions[e][d.agent] = ctrl.act(envs[e], d.agent, d.kind)
                continue
            if getattr(ctrl, "recurrent", False):
                for e, d, o in items:
                    o["mem"] = mem.get((e, d.agent), np.zeros(ctrl.d, np.float32))
            batch = pad_obs([o for _, _, o in items], N, M)
            force = np.full((len(items), N - 1), -1, np.int64)
            used = []
            for b, (e, d, o) in enumerate(items):
                fz = pending_force.get(e)
                if fz is not None and d.kind == PROPOSE and d.agent == fz[0]:
                    for t, j in enumerate(o["order"]):
                        force[b, t] = int(fz[1][j])
                    used.append(e)
            if used:
                batch["force_inc"] = force
            with torch.no_grad():
                out, _, _, _ = ctrl(to_torch(batch))
            out = {k: v.numpy() for k, v in out.items()}
            for b, (e, d, o) in enumerate(items):
                actions[e][d.agent] = _env_action(o, out, b)
                if "mem" in out:
                    mem[(e, d.agent)] = out["mem"][b]
            for e in used:
                pending_force.pop(e, None)
        nxt = []
        for e in active:
            env = envs[e]
            env.act(actions[e])
            if not env.done and env.t == t0[e]:
                nxt.append(e)
        active = nxt
    return [env.comp / _kappa(env, cfg) for env in envs]


def _p_propose(net, env, j, C, samples=64):
    """Estimated probability that proposer j names coalition C in this state."""
    if isinstance(net, Scripted):
        hits = 0
        for _ in range(samples):
            Cj, _ = net.act(env, j, PROPOSE)
            hits += np.array_equal(np.asarray(Cj, bool), C)
        return hits / samples
    o = env.observe(j, PROPOSE)
    if getattr(net, "recurrent", False):
        o["mem"] = np.zeros(net.d, np.float32)
    batch = pad_obs([o] * samples, env.cfg.n_max, 1 if env.cfg.divisible else env.cfg.m)
    with torch.no_grad():
        out, _, _, _ = net(to_torch(batch))
    out = {k: v.numpy() for k, v in out.items()}
    hits = 0
    for b in range(samples):
        Cj, _ = _env_action(o, out, b)
        hits += np.array_equal(np.asarray(Cj, bool), C)
    return hits / samples


def probe(cfg, controllers, n_states=2000, n_rep=4, seed=0, eta_min=0.02, chunk=96, assign=None):
    """Algorithm 4. Returns per-state contributions and their averages."""
    rng = np.random.default_rng(seed)
    assign = assign or (lambda env: ["learn"] * env.n)
    demand = cfg.protocol == "demand"
    ex_rows, in_rows, in_all_rows, cover = [], [], [], []
    for start in range(0, n_states, chunk):
        states = []
        for b in range(start, min(n_states, start + chunk)):
            env = CabinetEnv(cfg, seed=int(rng.integers(1 << 30))).reset()
            env.seats = assign(env)
            states.append(env)
        # natural play from each state: the coalition C_b formed in the first formation
        nat = [copy.deepcopy(s) for s in states]
        _run(nat, controllers, {}, cfg)
        coal = []
        for env in nat:
            logs = [r for r in env.cabinet_log]
            coal.append(logs[0]["C"].copy() if logs else None)
        # arms do(f=j) and do(f=j, coalition=C_b)
        arms, meta = [], []
        for b, s in enumerate(states):
            rho = s.recognition_probs()
            for j in range(s.n):
                if rho[j] <= 0:
                    continue
                for _ in range(n_rep):
                    a = copy.deepcopy(s)
                    a.override_recognition(j)
                    arms.append((a, None))
                    meta.append(("ex", b, j))
            C = coal[b]
            if C is not None and not demand and C.sum() >= 2:
                for j in np.flatnonzero(C):
                    for _ in range(n_rep):
                        a = copy.deepcopy(s)
                        a.override_recognition(int(j))
                        arms.append((a, (int(j), C)))
                        meta.append(("in", b, int(j)))
        envs = [a for a, _ in arms]
        forced = {k: fz for k, (_, fz) in enumerate(arms) if fz is not None}
        pays = _run(envs, controllers, forced, cfg)
        # aggregate per state
        for b, s in enumerate(states):
            n = s.n
            rho = s.recognition_probs()
            U = {}
            for k, (kind, bb, j) in enumerate(meta):
                if bb == b:
                    U.setdefault((kind, j), []).append(pays[k])
            Ns = [i for i in range(n) if 0 < rho[i] < 1]
            if Ns:
                vals = []
                for i in Ns:
                    own = np.mean(U[("ex", i)], axis=0)[i]
                    other = sum(rho[j] / (1 - rho[i]) * np.mean(U[("ex", j)], axis=0)[i]
                                for j in range(n) if j != i and rho[j] > 0)
                    vals.append(own - other)
                ex_rows.append(np.mean(vals, axis=0))
            C = coal[b]
            if C is None or demand or C.sum() < 2:
                continue
            members = [int(j) for j in np.flatnonzero(C)]
            on = {j: _p_propose(controllers[s.seats[j]], s, j, C) >= eta_min for j in members}
            cover.append(np.mean([on[j] for j in members]))
            prim, allp = [], []
            for i in members:
                for j in members:
                    if i == j:
                        continue
                    d = np.mean(U[("in", i)], axis=0)[i] - np.mean(U[("in", j)], axis=0)[i]
                    allp.append(d)
                    if on[i] and on[j]:
                        prim.append(d)
            in_all_rows.append(np.mean(allp, axis=0))
            if prim:
                in_rows.append(np.mean(prim, axis=0))

    def summ(rows):
        if not rows:
            return None
        R = np.array(rows)
        tot = R.sum(1)
        boot = [tot[rng.integers(len(tot), size=len(tot))].mean() for _ in range(1000)]
        return dict(total=float(tot.mean()), ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                    **{c: float(R[:, k].mean()) for k, c in enumerate(COMPONENTS)}, states=len(R))

    return dict(psi_in=summ(in_rows), psi_in_all=summ(in_all_rows), psi_ex=summ(ex_rows),
                on_path_coverage=float(np.mean(cover)) if cover else None, n_states=n_states, n_rep=n_rep,
                eta_min=eta_min)
