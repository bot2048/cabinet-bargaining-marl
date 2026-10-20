"""Symmetric PSRO (spec Algorithm 2, learning regime LA4).

The population starts from scripted agents and one untrained network.  Each
iteration meta-solves the restricted game, trains a best response (PPO,
Algorithm 1) for focal seats while every other seat draws its policy from the
meta-mixture, and adds it to the population unless its estimated gain is
below the tolerance.  As the spec allows, the restricted game uses a pairwise
surrogate: A[i, j] is the payoff of a focal seat playing policy i while every
other seat plays policy j (exact for two players); the meta-mixture is the
symmetric equilibrium of A found by replicator dynamics.
"""
import copy

import numpy as np
import torch

from .baselines import Scripted
from .metrics import summarise
from .policy import PolicyNet
from .ppo import evaluate, seat_payoffs, train_selfplay


def pair_payoff(cfg, pols, a, b, episodes, seed):
    def assign(env, rng):
        seats = ["B"] * env.n
        seats[rng.integers(env.n)] = "A"
        return seats
    _, ep = evaluate(cfg, {"A": pols[a], "B": pols[b]}, assign, episodes=episodes, seed=seed)
    return seat_payoffs(ep, "A")[0]


def replicator(A, iters=20000, lr=0.5, floor=1e-4):
    """Symmetric equilibrium of the matrix game A by (exponential) replicator dynamics."""
    k = A.shape[0]
    x = np.full(k, 1.0 / k)
    scale = max(np.abs(A).max(), 1e-9)
    for _ in range(iters):
        f = A @ x
        x = x * np.exp(lr * (f - x @ f) / scale)
        x = np.maximum(x / x.sum(), floor)
        x /= x.sum()
    return x


def mixture_assign(names, w, focal=None):
    def assign(env, rng):
        seats = [names[i] for i in rng.choice(len(names), size=env.n, p=w)]
        if focal is not None:
            seats[rng.integers(env.n)] = focal
        return seats
    return assign


def run_psro(cfg, iters=6, br_iters=120, seed=0, episodes=256, eps=0.005, hp=None, log=print):
    torch.manual_seed(seed)
    pols = {"myopic": Scripted("myopic", seed), "random": Scripted("random", seed),
            "net0": PolicyNet(d=(hp or {}).get("d", 64))}
    names = list(pols)
    A = np.array([[pair_payoff(cfg, pols, a, b, episodes, seed) for b in names] for a in names])
    history = []
    last_net = pols["net0"]
    for k in range(iters):
        w = replicator(A)
        frozen = {n: pols[n] for n in names}
        br, _ = train_selfplay(cfg, br_iters, seed=seed * 100 + k, hp=hp,
                               net=copy.deepcopy(last_net), controllers=frozen,
                               assign=mixture_assign(names, w, focal="learn"), verbose=False)
        _, ep_br = evaluate(cfg, {"learn": br, **frozen}, mixture_assign(names, w, "learn"),
                            episodes=episodes * 2, seed=seed + 999)
        _, ep_mix = evaluate(cfg, frozen, mixture_assign(names, w), episodes=episodes * 2,
                             seed=seed + 999)
        u_br = seat_payoffs(ep_br, "learn")[0]
        u_mix = float(np.mean([e["ret"][j] / e["omega"] for e in ep_mix for j in range(e["n"])
                               if e["omega"] > 0]))
        gain = u_br - u_mix
        history.append(dict(iteration=k, population=list(names), meta=w.tolist(), gain=gain))
        log(f"it={k:4d}/{iters} t=0s eta=0s population={len(names)} gain={gain:+.4f} "
            f"meta={dict(zip(names, np.round(w, 3)))}")
        if gain <= eps:
            break
        name = f"br{k}"
        pols[name] = br.eval()
        last_net = br
        names.append(name)
        row = [pair_payoff(cfg, pols, name, b, episodes, seed + k) for b in names]
        col = [pair_payoff(cfg, pols, a, name, episodes, seed + k) for a in names[:-1]]
        A = np.pad(A, ((0, 1), (0, 1)))
        A[-1, :] = row
        A[:-1, -1] = col
    w = replicator(A)
    cab, ep = evaluate(cfg, {n: pols[n] for n in names}, mixture_assign(names, w),
                       episodes=2000, seed=seed + 4242)
    nets = {n: p for n, p in pols.items() if isinstance(p, PolicyNet)}
    return dict(names=names, meta=w.tolist(), A=A.tolist(), history=history,
                overall=summarise(cab, ep, cfg.m, cfg.divisible), last_gain=history[-1]["gain"]), nets
