"""Benchmark diagnostics (spec Section 3.6): probing voters' thresholds."""
import numpy as np
import torch

from .env import CabinetEnv, VOTE
from .policy import pad_obs, to_torch


def acceptance_curve(net, cfg, n, delta, offers, seed=0):
    """Pr(accept | offer x) of voter 1 in a symmetric divisible parliament.

    The formateur (party 0) names a minimal winning coalition; voter 1 is
    offered x, the other partners delta/n, and the formateur the rest."""
    cfg = cfg.with_(n_choices=(n,), delta_choices=(delta,), seat_mode="equal")
    env = CabinetEnv(cfg, seed=seed).reset()
    k = (n + 1) // 2
    C = np.zeros(n, bool)
    C[:k] = True
    probs = []
    for x in offers:
        env.f = 0
        shares = np.zeros(n)
        shares[1] = x
        shares[2:k] = delta / n
        shares[0] = max(1.0 - shares.sum(), 1e-6)
        env.phase = 0
        env._receive_proposal(C, shares)
        assert env.phase == VOTE
        o = env.observe(1, VOTE)
        b = to_torch(pad_obs([o], cfg.n_max, 1))
        with torch.no_grad():
            _, _, h = net.encode(b)
            probs.append(torch.sigmoid(net.vote(h)).item())
    return np.array(probs)


def midpoint(offers, probs):
    """Offer at which the acceptance curve crosses 1/2 (linear interpolation)."""
    above = np.flatnonzero(probs >= 0.5)
    if len(above) == 0:
        return np.nan
    i = above[0]
    if i == 0:
        return offers[0]
    x0, x1, p0, p1 = offers[i - 1], offers[i], probs[i - 1], probs[i]
    return x0 + (0.5 - p0) * (x1 - x0) / max(p1 - p0, 1e-9)
