"""Evaluate trained policies on real parliaments from ParlGov (spec RQ4, RQ7).

For every post-1945 Western European majority coalition in
data/parlgov/cabinets.json the real seat distribution becomes a fixed
parliament and the real prime minister's party is the first formateur.

1. Coalition prediction: the coalition the formateur proposes, against the
   coalition that actually governed (exact match and Jaccard overlap), compared
   with seat-based rules: Riker's minimal-seat coalition, a uniformly random
   minimal winning coalition, and the ideologically closest minimal winning
   coalition (ParlGov left-right scores, which the agents do not see).
2. Calibrated allocation: full formation episodes on the real parliaments,
   summarised with the statistics of spec Definition 1.7 and regression (1).

The parliaments were never used for training (spec R17).

    python scripts/parlgov_eval.py --tag mech_EstarWide --seed 0
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cabinet.config import estar_wide, estar_wide_public, real_parliament         # noqa: E402
from cabinet.env import CabinetEnv, PROPOSE                    # noqa: E402
from cabinet.metrics import summarise                          # noqa: E402
from cabinet.policy import load_policy, pad_obs, to_torch       # noqa: E402
from cabinet.ppo import _env_action, evaluate                  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def minimal_winning(w, q, f):
    """All minimal winning coalitions that contain f (as boolean masks)."""
    n = len(w)
    out = []
    for mask in range(1 << n):
        if not mask >> f & 1:
            continue
        C = np.array([(mask >> i) & 1 for i in range(n)], bool)
        tot = w[C].sum()
        if tot >= q and all(tot - w[j] < q for j in np.flatnonzero(C)):
            out.append(C)
    return out


def left_right():
    """ParlGov left-right score per (cabinet_id, party short name)."""
    path = os.path.join(ROOT, "data", "parlgov", "raw", "view_cabinet.csv")
    lr = {}
    if not os.path.exists(path):
        return lr
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                lr[(int(r["cabinet_id"]), r["party_name_short"])] = float(r["left_right"])
            except ValueError:
                pass
    return lr


def jaccard(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="mech_EstarWide")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first k cabinets (testing)")
    ap.add_argument("--ideology", action="store_true",
                    help="public types, with each party's ParlGov left-right score as its "
                         "position on the first policy dimension (calibrated environment)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scope", default="western", choices=["western", "all"],
                    help="western: 13 Western European countries (spec OD4); all: every "
                         "parliamentary democracy in ParlGov (out-of-sample extension)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    path = os.path.join(ROOT, "models", a.tag, f"seed{a.seed}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: train the model first")
    net = load_policy(path)
    net.eval()
    fname = "cabinets.json" if a.scope == "western" else "cabinets_all.json"
    with open(os.path.join(ROOT, "data", "parlgov", fname)) as f:
        cabinets = [c for c in json.load(f) if c["majority"]]
    if a.limit:
        cabinets = cabinets[:a.limit]
    lr = left_right()
    base = estar_wide_public() if a.ideology else estar_wide()
    acc = defaultdict(list)
    jac = defaultdict(list)
    per_cab, cab_log, ep_log = [], [], []
    for i, cab in enumerate(cabinets):
        pos = np.array([lr.get((cab["cabinet_id"], p), np.nan) for p in cab["parties"]])
        if a.ideology and np.isnan(pos).any():
            continue                        # needs every party's left-right score
        cfg = real_parliament(base, cab, left_right=pos.tolist() if a.ideology else None)
        actual = np.array(cab["coalition"], bool)
        w = np.array(cab["seats"])
        q, f = cab["quota"], cab["pm"]
        # ---- learned formateur: sample its first proposal
        env = CabinetEnv(cfg, seed=a.seed * 100000 + i)
        obs = []
        for _ in range(a.samples):
            env.reset()
            obs.append(env.observe(f, PROPOSE))
        with torch.no_grad():
            out, _, _, _ = net(to_torch(pad_obs(obs, len(w), cfg.m)))
        out = {k: v.numpy() for k, v in out.items()}
        props = [_env_action(o, out, b)[0] for b, o in enumerate(obs)]
        acc["learned"].append(np.mean([np.array_equal(C, actual) for C in props]))
        jac["learned"].append(np.mean([jaccard(C, actual) for C in props]))
        # ---- seat-based and ideological rules
        mwc = minimal_winning(w, q, f)
        tot = [w[C].sum() for C in mwc]
        riker = [C for C, t in zip(mwc, tot) if t == min(tot)]
        acc["min_seat"].append(np.mean([np.array_equal(C, actual) for C in riker]))
        jac["min_seat"].append(np.mean([jaccard(C, actual) for C in riker]))
        acc["random_mwc"].append(np.mean([np.array_equal(C, actual) for C in mwc]))
        jac["random_mwc"].append(np.mean([jaccard(C, actual) for C in mwc]))
        if not np.isnan(pos).any():
            spread = [pos[C].max() - pos[C].min() for C in mwc]
            close = [C for C, s in zip(mwc, spread) if s == min(spread)]
            acc["ideology_mwc"].append(np.mean([np.array_equal(C, actual) for C in close]))
            jac["ideology_mwc"].append(np.mean([jaccard(C, actual) for C in close]))
        # ---- full formation episodes on the real parliament
        c_log, e_log = evaluate(cfg, {"learn": net}, lambda env, rng: ["learn"] * env.n,
                                episodes=a.episodes, seed=a.seed * 1000 + i)
        cab_log += c_log
        ep_log += e_log
        per_cab.append(dict(country=cab["country"], start=cab["start"], name=cab["name"],
                            n=len(w), actual_size=int(actual.sum()), cabinet_id=cab["cabinet_id"],
                            learned_acc=acc["learned"][-1], min_seat_acc=acc["min_seat"][-1],
                            n_mwc=len(mwc)))
        if i % 20 == 0:
            print(f"it={i:4d}/{len(cabinets)} t=0s eta=0s {cab['country']} {cab['start']} "
                  f"learned={np.mean(acc['learned']):.3f} riker={np.mean(acc['min_seat']):.3f}",
                  flush=True)
    summary = {k: dict(exact=float(np.mean(v)), jaccard=float(np.mean(jac[k])), n=len(v))
               for k, v in acc.items()}
    alloc = summarise(cab_log, ep_log, base.m, base.divisible)
    out_tag = f"parlgov_{a.tag}" + ("" if a.scope == "western" else "_all") + (
        "_ideology" if a.ideology else "")
    os.makedirs(os.path.join(ROOT, "results", out_tag), exist_ok=True)
    with open(os.path.join(ROOT, "results", out_tag, f"seed{a.seed}.json"), "w") as f:
        json.dump(dict(prediction=summary, allocation={k: (None if isinstance(v, float) and np.isnan(v) else v)
                                                       for k, v in alloc.items()},
                       cabinets=per_cab), f, indent=1, default=float)
    for k, v in summary.items():
        print(f"{k:14s} exact={v['exact']:.3f} jaccard={v['jaccard']:.3f} (n={v['n']})")
    print({k: round(v, 3) for k, v in alloc.items() if isinstance(v, float)})
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
