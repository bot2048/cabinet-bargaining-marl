"""Value of proposal power (spec v0.5, Definition 1.8, Algorithm 4) of trained profiles.

For every seed of a model tag, estimate Psi_in, Psi_ex and their salience, taste,
policy and exit components by counterfactual replays (cabinet/proposal_power.py).
Evaluation only: no training. The benchmark E0 is probed cell by cell (n, delta)
and compared with the closed forms of Corollary 2.13.

    python scripts/proposal_power_eval.py --tag mech_BASE+M4_private
    python scripts/proposal_power_eval.py --tag e0
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from cabinet.config import E0  # noqa: E402
from cabinet.policy import load_policy  # noqa: E402
from cabinet.proposal_power import probe  # noqa: E402
from run_experiment import get_config  # noqa: E402

PREFIXES = ("mech_", "long_", "rec_", "explore_", "mappo_", "fsp_")


def config_of(tag):
    if tag == "e0":
        return E0
    if tag == "e1":
        return get_config("E1")
    if tag.startswith("entropy_"):
        return get_config("BASE")
    for p in PREFIXES:
        if tag.startswith(p):
            return get_config(tag[len(p):])
    raise ValueError(tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--states", type=int, default=2000)
    ap.add_argument("--rep", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")          # accepted for the queue runner; probes run on CPU
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = config_of(a.tag)
    paths = sorted(glob.glob(os.path.join(ROOT, "models", a.tag, "seed*.pt")),
                   key=lambda p: int(os.path.basename(p)[4:-3]))
    paths = [p for p in paths if os.path.basename(p) != "seed99.pt"]
    out = dict(tag=a.tag, config=cfg.name, states=a.states, rep=a.rep, seeds={})
    t0 = time.time()
    for k, p in enumerate(paths):
        seed = int(os.path.basename(p)[4:-3])
        net = load_policy(p).eval()
        if a.tag == "e0":
            cells = {}
            for n in cfg.n_choices:
                for d in cfg.delta_choices:
                    c = cfg.with_(n_choices=(n,), delta_choices=(d,))
                    r = probe(c, {"learn": net}, n_states=a.states // 4, n_rep=a.rep, seed=seed)
                    r["bf_in"] = 1 - d * (n + 1) / (2 * n)
                    r["bf_ex"] = 1 - d / 2
                    cells[f"{n},{d}"] = r
            out["seeds"][seed] = dict(cells=cells)
        else:
            out["seeds"][seed] = probe(cfg, {"learn": net}, n_states=a.states, n_rep=a.rep, seed=seed)
        el = time.time() - t0
        print(f"it={k + 1}/{len(paths)} t={el:.0f}s eta={el / (k + 1) * (len(paths) - k - 1):.0f}s "
              f"{a.tag} seed {seed} done", flush=True)
    os.makedirs(os.path.join(ROOT, "results", "psi"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "psi", f"{a.tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
