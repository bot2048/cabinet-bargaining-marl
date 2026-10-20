"""Re-score every saved model with the current evaluation code.

Training runs store their evaluation at the time they finished; when metrics
are added later, this script recomputes the ``overall`` block of every
results/<tag>/seed<k>.json from models/<tag>/seed<k>.pt so that all numbers
in the report come from the same code.

    python scripts/reevaluate.py [--episodes 3000] [--only <tag-substring>]
"""
import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from cabinet.metrics import summarise                     # noqa: E402
from cabinet.policy import PolicyNet                      # noqa: E402
from cabinet.ppo import evaluate                          # noqa: E402
from run_experiment import ALL, ROOT, get_config, jsonable  # noqa: E402


def config_for(tag):
    if tag == "e0":
        return get_config("E0")
    if tag == "e1":
        return get_config("E1")
    for prefix in ("mech_", "mappo_", "fsp_"):
        if tag.startswith(prefix):
            return get_config(tag[len(prefix):])
    if tag.startswith("entropy_"):
        return get_config(tag.split("_")[1])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--only", default="")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    for path in sorted(glob.glob(os.path.join(ROOT, "models", "*", "seed*.pt"))):
        tag = os.path.basename(os.path.dirname(path))
        seed = int(os.path.basename(path)[4:-3])
        res = os.path.join(ROOT, "results", tag, f"seed{seed}.json")
        cfg = config_for(tag)
        if a.only not in tag or cfg is None or not os.path.exists(res):
            continue
        net = PolicyNet()
        net.load_state_dict(torch.load(path))
        cab, ep = evaluate(cfg, {"learn": net}, ALL, episodes=a.episodes, seed=1000 + seed)
        with open(res) as f:
            data = json.load(f)
        data["overall"] = jsonable(summarise(cab, ep, cfg.m, cfg.divisible))
        with open(res, "w") as f:
            json.dump(data, f, indent=1)
        o = data["overall"]
        print(f"{tag:28s} s{seed}  phi~={o.get('phi_adj', float('nan')):+.3f}  "
              f"dg~={o.get('dg_adj', float('nan')):.3f}  phi_nu={o.get('phi_nu', float('nan'))}",
              flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
