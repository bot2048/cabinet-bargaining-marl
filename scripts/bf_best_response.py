"""Does the learned Baron-Ferejohn formateur best-respond to the learned voters?

No training: for each E0 seed and cell (n, delta), the formateur names a
minimal winning coalition and offers every partner x. The frozen voter network
gives each partner's acceptance probability on that exact proposal, so the
formateur's expected payoff is

    U(x) = P(all accept | x) (1 - k x) + (1 - P) delta V_f,   V_f = 1/n,

with k = (n-1)/2 partners. We compare U at the learned offer
x_L = (1 - learned formateur share) / k with max_x U(x). A small gap means the
formateur best-responds to the voters it faces (offers are set by voter noise,
not by a failure to explore lower offers); a large gap means it under-exploits.

    python scripts/bf_best_response.py
"""
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from cabinet.config import E0  # noqa: E402
from cabinet.env import CabinetEnv, VOTE  # noqa: E402
from cabinet.policy import pad_obs, to_torch  # noqa: E402
from cabinet.policy import load_policy  # noqa: E402

torch.set_num_threads(1)


def accept_all(net, n, delta, offers, seed=0):
    cfg = E0.with_(n_choices=(n,), delta_choices=(delta,))
    env = CabinetEnv(cfg, seed=seed).reset()
    k = (n - 1) // 2
    C = np.zeros(n, bool)
    C[:k + 1] = True
    out = []
    for x in offers:
        env.f = 0
        shares = np.zeros(n)
        shares[1:k + 1] = x
        shares[0] = max(1.0 - k * x, 1e-6)
        env.phase = 0
        env._receive_proposal(C, shares)
        assert env.phase == VOTE
        obs = [env.observe(j, VOTE) for j in range(1, k + 1)]
        b = to_torch(pad_obs(obs, cfg.n_max, len(obs)))
        with torch.no_grad():
            _, _, h = net.encode(b)
            p = torch.sigmoid(net.vote(h)).numpy().ravel()
        out.append(p)
    return np.array(out)          # [offers, k]


def main():
    offers = np.round(np.arange(0.0, 0.3001, 0.0025), 4)
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "models", "e0", "seed*.pt"))):
        seed = int(os.path.basename(path)[4:-3])
        net = load_policy(path)
        res = json.load(open(os.path.join(ROOT, "results", "e0", f"seed{seed}.json")))
        for key, cell in res["cells"].items():
            n, delta = eval(key)
            k = (n - 1) // 2
            P = accept_all(net, n, delta, offers)
            for mode, Pall in (("stochastic", P.prod(1)), ("greedy", (P > 0.5).all(1).astype(float))):
                U = Pall * (1 - k * offers) + (1 - Pall) * delta / n
                xL = (1 - cell["pf"]) / k
                UL = np.interp(xL, offers, U)
                j = int(np.argmax(U))
                rows.append(dict(seed=seed, n=n, delta=delta, mode=mode, x_learned=xL, x_best=float(offers[j]),
                                 U_learned=float(UL), U_best=float(U[j]), gain=float(U[j] - UL),
                                 p_accept_learned=float(np.interp(xL, offers, Pall)), reservation=delta / n))
    out = {}
    print(f"{'n':>2} {'delta':>5} {'mode':>10} {'x_learned':>9} {'x_best':>7} {'d/n':>6} {'gain':>7} {'gain_max':>8}")
    for (n, d) in sorted({(r["n"], r["delta"]) for r in rows}):
        for mode in ("stochastic", "greedy"):
            R = [r for r in rows if r["n"] == n and r["delta"] == d and r["mode"] == mode]
            g = np.array([r["gain"] for r in R])
            s = dict(x_learned=float(np.mean([r["x_learned"] for r in R])),
                     x_best=float(np.mean([r["x_best"] for r in R])), gain=float(g.mean()),
                     gain_max=float(g.max()), seeds=len(R))
            out[f"{n},{d},{mode}"] = s
            print(f"{n:>2} {d:>5} {mode:>10} {s['x_learned']:9.3f} {s['x_best']:7.3f} {d / n:6.3f} "
                  f"{s['gain']:7.4f} {s['gain_max']:8.4f}")
    json.dump(dict(summary=out, rows=rows), open(os.path.join(ROOT, "results", "bf_best_response.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
