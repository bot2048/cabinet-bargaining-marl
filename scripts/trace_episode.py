"""Print one episode of a trained profile step by step (teaching / live demo).

    python scripts/trace_episode.py --tag mech_BASE --seed 0 --episode-seed 3
    python scripts/trace_episode.py --tag explore_A452Demand --seed 0 --episode-seed 1
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from cabinet.env import CabinetEnv, PROPOSE, VOTE, GOVERN, DEMAND, CLOSE  # noqa: E402
from cabinet.policy import load_policy, pad_obs, to_torch  # noqa: E402
from cabinet.ppo import _env_action  # noqa: E402
from proposal_power_eval import config_of  # noqa: E402

KIND = {PROPOSE: "propose", VOTE: "vote", GOVERN: "stay/exit", DEMAND: "demand"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="mech_BASE")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episode-seed", type=int, default=3)
    a = ap.parse_args()
    torch.manual_seed(a.episode_seed)
    cfg = config_of(a.tag)
    net = load_policy(os.path.join(ROOT, "models", a.tag, f"seed{a.seed}.pt")).eval()
    env = CabinetEnv(cfg, seed=a.episode_seed).reset()
    N, M = cfg.n_max, 1 if cfg.divisible else cfg.m
    names = "ABCDEFGH"
    print(f"Parliament: " + ", ".join(f"{names[i]}={env.w[i]}" for i in range(env.n))
          + f" seats; quota {env.q}; delta {env.delta}; {'divisible prize' if cfg.divisible else f'{env.m} ministries'}")
    if cfg.xi > 0 and not cfg.divisible:
        for i in range(env.n):
            print(f"  {names[i]} values ministries: " + " ".join(f"{x:.2f}" for x in env.nu[i]))
    step = 0
    while not env.done and step < 60:
        pend = env.pending()
        obs = [env.observe(d.agent, d.kind) for d in pend]
        with torch.no_grad():
            out, _, _, _ = net(to_torch(pad_obs(obs, N, M)))
        out = {k: v.numpy() for k, v in out.items()}
        acts = {d.agent: _env_action(o, out, b) for b, (d, o) in enumerate(zip(pend, obs))}
        for d in pend:
            x = acts[d.agent]
            who = names[d.agent]
            if d.kind == PROPOSE:
                C, alloc = x
                members = "+".join(names[j] for j in np.flatnonzero(C))
                if cfg.divisible:
                    split = ", ".join(f"{names[j]} {alloc[j]:.2f}" for j in np.flatnonzero(C))
                else:
                    split = ", ".join(f"{names[j]} {int((np.asarray(alloc) == j).sum())}" for j in np.flatnonzero(C))
                print(f"[period {env.period}] formateur {who} proposes coalition {members}: {split}")
            elif d.kind == VOTE:
                print(f"    {who} votes {'ACCEPT' if x else 'REJECT'}")
            elif d.kind == GOVERN:
                print(f"    {who} {'stays' if x else 'EXITS'}")
            elif d.kind == DEMAND:
                print(f"    {who} {'CLOSES' if x == CLOSE else f'demands {x}%'}")
        env.act(acts)
        step += 1
    for rec in env.cabinet_log:
        C = np.flatnonzero(rec["C"])
        print("Cabinet formed: " + ", ".join(f"{names[j]} share {rec['p'][j]:.2f} (seat share in coalition "
                                              f"{env.w[j] / env.w[C].sum():.2f})" for j in C)
              + f"; formateur {names[rec['f']]}")
    print("Discounted payoffs: " + ", ".join(f"{names[i]} {env.ret[i]:.3f}" for i in range(env.n)))


if __name__ == "__main__":
    main()
