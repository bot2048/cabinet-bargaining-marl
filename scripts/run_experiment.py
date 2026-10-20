"""Train and evaluate one experiment cell; results go to results/<tag>/seed<s>.json.

Examples
--------
python scripts/run_experiment.py --exp e0 --seed 0
python scripts/run_experiment.py --exp mech --config BASE+M1_repeat --seed 1
python scripts/run_experiment.py --exp exploit --config Estar --seed 0
python scripts/run_experiment.py --exp baselines
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cabinet import config as C                                  # noqa: E402
from cabinet.baselines import Scripted                           # noqa: E402
from cabinet.diagnostics import acceptance_curve, midpoint       # noqa: E402
from cabinet.metrics import cabinet_rows, summarise              # noqa: E402
from cabinet.policy import PolicyNet                             # noqa: E402
from cabinet.ppo import evaluate, exploitability, train_selfplay  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
ALL = lambda env, rng: ["learn"] * env.n                         # noqa: E731


def get_config(name):
    if name in C.LEVELS:
        return C.LEVELS[name]
    if name == "BASE":
        return C.BASE
    if name == "Estar":
        return C.full_estar()
    if name.startswith("BASE+"):
        return C.add_one_in(name[5:])
    if name.startswith("Estar-"):
        return C.leave_one_out(name[6:])
    if name.startswith("Estar_Lam0"):
        return C.full_estar().with_(name=name, Lam=0)
    raise KeyError(name)


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    return x


def save(tag, seed, payload):
    d = os.path.join(ROOT, "results", tag)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"seed{seed}.json"), "w") as f:
        json.dump(jsonable(payload), f, indent=1)


def model_path(tag, seed):
    d = os.path.join(ROOT, "models", tag)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"seed{seed}.pt")


def train(cfg, tag, seed, iters, hp=None):
    t0 = time.time()
    net, hist = train_selfplay(cfg, iters, seed=seed, hp=hp, log_every=10)
    torch.save(net.state_dict(), model_path(tag, seed))
    return net, hist, time.time() - t0


def cell_summary(cab, ep, cfg, key):
    """Summaries split by a cabinet-record key (e.g. n or (n, delta))."""
    groups_c, groups_e = defaultdict(list), defaultdict(list)
    for c in cab:
        groups_c[key(c)].append(c)
    for e in ep:
        groups_e[key(e)].append(e)
    return {str(k): summarise(groups_c[k], groups_e.get(k, []), cfg.m, cfg.divisible)
            for k in sorted(groups_c)}


# ------------------------------------------------------------------ experiments
def exp_e0(seed, iters):
    """RQ1 / S1: benchmark recovery in the symmetric Baron-Ferejohn game."""
    cfg = C.E0
    net, hist, secs = train(cfg, "e0", seed, iters)
    cab, ep = evaluate(cfg, {"learn": net}, ALL, episodes=4000, seed=1000 + seed)
    cells = cell_summary(cab, ep, cfg, lambda r: (r["n"], r["delta"]))
    offers = np.linspace(0, 0.6, 121)
    thr = {}
    for n in cfg.n_choices:
        for d in cfg.delta_choices:
            thr[f"({n}, {d})"] = float(midpoint(offers, acceptance_curve(net, cfg, n, d, offers)))
    save("e0", seed, dict(overall=summarise(cab, ep, cfg.m, True), cells=cells,
                          thresholds=thr, history=hist, seconds=secs))


def exp_e1(seed, iters):
    """Weighted parliaments: learned formateur share vs certified BF share."""
    cfg = C.E1
    net, hist, secs = train(cfg, "e1", seed, iters)
    cab, ep = evaluate(cfg, {"learn": net}, ALL, episodes=3000, seed=1000 + seed)
    rows, _, _ = cabinet_rows(cab, cfg.m, True)
    pairs = [(r["pf"], r["x_bf"], r["n"]) for r in rows if r["x_bf"] is not None]
    save("e1", seed, dict(overall=summarise(cab, ep, cfg.m, True), pairs=pairs,
                          history=hist, seconds=secs))


def exp_mech(config, seed, iters, hp=None, tag=None):
    """RQ2: one environment configuration (base, add-one-in, leave-one-out, E*)."""
    cfg = get_config(config)
    tag = tag or f"mech_{config}"
    net, hist, secs = train(cfg, tag, seed, iters, hp)
    cab, ep = evaluate(cfg, {"learn": net}, ALL, episodes=3000, seed=1000 + seed)
    _, rows, _ = cabinet_rows(cab, cfg.m, cfg.divisible)
    save(tag, seed, dict(config=config, overall=summarise(cab, ep, cfg.m, cfg.divisible),
                         rows=rows[:4000], history=hist, seconds=secs))


def exp_exploit(config, seed, iters, br_iters):
    """S2/S4: exploitability of a trained profile, plus scripted positive controls."""
    cfg = get_config(config)
    tag = f"mech_{config}"
    path = model_path(tag, seed)
    waited = 0
    while not os.path.exists(path):          # queued before its model finished
        if waited > 8 * 3600:
            raise FileNotFoundError(path)
        time.sleep(60)
        waited += 60
    time.sleep(5)
    net = PolicyNet()
    net.load_state_dict(torch.load(path))
    out = {}
    for init in ("warm", "scratch"):
        res = exploitability(cfg, net, br_iters, seed=seed + 50,
                             warm_start=net if init == "warm" else None)
        res.pop("br_history")
        out[init] = res
    gain = max(out["warm"]["gain"], out["scratch"]["gain"])
    controls = {}
    if seed == 0:
        for bot in ("gamson", "myopic"):
            r = exploitability(cfg, Scripted(bot, 7), br_iters, seed=77)
            r.pop("br_history")
            controls[bot] = r
    save(f"exploit_{config}", seed, dict(config=config, gain=gain, runs=out, controls=controls))


def exp_baselines():
    """Allocation statistics of the scripted agents (reference points)."""
    out = {}
    for cname in ("E0", "E1", "BASE", "Estar"):
        cfg = get_config(cname)
        for bot in ("bf", "gamson", "myopic", "random"):
            if bot == "bf" and not cfg.divisible:
                continue
            cab, ep = evaluate(cfg, {"b": Scripted(bot, 3)}, lambda env, rng: ["b"] * env.n,
                               episodes=1500, seed=5)
            out[f"{cname}/{bot}"] = summarise(cab, ep, cfg.m, cfg.divisible)
    save("baselines", 0, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True,
                    choices=["e0", "e1", "mech", "exploit", "baselines", "entropy"])
    ap.add_argument("--config", default="BASE")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--br-iters", type=int, default=150)
    ap.add_argument("--ent-end", type=float, default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--algo", default="ippo", choices=["ippo", "mappo", "fsp"])
    ap.add_argument("--device", default="cpu", help="cpu or cuda (PPO update only)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from cabinet import ppo
    ppo.DEFAULT_HP["device"] = a.device
    if a.exp == "e0":
        exp_e0(a.seed, a.iters)
    elif a.exp == "e1":
        exp_e1(a.seed, a.iters)
    elif a.exp == "mech" and a.algo != "ippo":     # RQ6: LA2 / LA3 instead of LA1
        exp_mech(a.config, a.seed, a.iters, hp=dict(algo=a.algo), tag=f"{a.algo}_{a.config}")
    elif a.exp == "mech":
        exp_mech(a.config, a.seed, a.iters)
    elif a.exp == "entropy":     # H9: non-vanishing entropy temperature
        exp_mech(a.config, a.seed, a.iters, hp=dict(ent_end=a.ent_end),
                 tag=f"entropy_{a.config}_{a.ent_end:g}")
    elif a.exp == "exploit":
        exp_exploit(a.config, a.seed, a.iters, a.br_iters)
    else:
        exp_baselines()
    print("DONE", flush=True)       # marker read by scripts/status.py


if __name__ == "__main__":
    main()
