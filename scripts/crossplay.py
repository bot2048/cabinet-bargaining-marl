"""Cross-play between trained runs (spec Definition 3.12, RQ6).

X_AB = U(pi_A; pi_B) is the per-period payoff of a focal seat that plays run
A's policy while every other seat plays run B's.  The convention mismatch
CM_AB = X_AA - X_AB is the loss from playing with strangers: a large mismatch
together with low exploitability means the runs reached different equilibria.

    python scripts/crossplay.py --config Estar --runs mech_Estar mappo_Estar fsp_Estar
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from cabinet.policy import load_policy                    # noqa: E402
from cabinet.ppo import evaluate, seat_payoffs          # noqa: E402
from run_experiment import ROOT, get_config             # noqa: E402


def load(tag, seed):
    p = os.path.join(ROOT, "models", tag, f"seed{seed}.pt")
    if not os.path.exists(p):
        return None
    net = load_policy(p)
    return net.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="Estar")
    ap.add_argument("--runs", nargs="+", default=["mech_Estar", "mappo_Estar", "fsp_Estar"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = get_config(a.config)
    pols = {f"{t}/s{s}": load(t, s) for t in a.runs for s in a.seeds}
    pols = {k: v for k, v in pols.items() if v is not None}
    names = sorted(pols)
    X = np.full((len(names), len(names)), np.nan)
    for (i, A), (j, B) in itertools.product(enumerate(names), repeat=2):
        def focal(env, rng):
            seats = ["B"] * env.n
            seats[rng.integers(env.n)] = "A"
            return seats
        ctrl = {"A": pols[A], "B": pols[B]}
        _, ep = evaluate(cfg, ctrl, focal, episodes=a.episodes, seed=4242)
        X[i, j] = seat_payoffs(ep, "A")[0]
        print(f"it={i * len(names) + j + 1}/{len(names) ** 2} t=0s eta=0s X[{A} | {B}] = {X[i, j]:.4f}",
              flush=True)
    CM = np.diag(X)[:, None] - X
    algo = [n.split("/")[0] for n in names]
    same = [CM[i, j] for i in range(len(names)) for j in range(len(names))
            if i != j and algo[i] == algo[j]]
    diff = [CM[i, j] for i in range(len(names)) for j in range(len(names)) if algo[i] != algo[j]]
    out = dict(config=a.config, names=names, X=X.tolist(), CM=CM.tolist(),
               mismatch_same_algorithm=float(np.mean(same)) if same else None,
               mismatch_across_algorithms=float(np.mean(diff)) if diff else None)
    os.makedirs(os.path.join(ROOT, "results", "crossplay"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "crossplay", f"{a.config}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"mean mismatch, same algorithm: {out['mismatch_same_algorithm']}, "
          f"across algorithms: {out['mismatch_across_algorithms']}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
