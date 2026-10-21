"""Predict how real coalition cabinets split their ministries (spec RQ4, S3).

Data: 195 majority coalition cabinets, 1988-2014, 31 countries, with each
party's share of the cabinet's partisan ministers (Seki-Williams, linked to
ParlGov; data/seki_williams/). The coalition and the prime minister's party
are taken as given; each model predicts the split:

* Gamson's law: shares proportional to seats within the coalition;
* Baron-Ferejohn: partners get delta x their stationary continuation value
  (weighted closed-rule bargaining, seat-proportional recognition), the
  formateur the rest; for surplus coalitions this is a heuristic extension;
* learned agents: the trained formateur policy, with the real coalition
  imposed and the real number of portfolios, averaged over sampled proposals.

Reported: the empirical Gamson regression, MAE per party, the paired MAE
difference to Gamson, all with country-clustered bootstrap 95% CIs.

    python scripts/seki_eval.py --tags mech_EstarWide --seeds 0 1 2
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
from cabinet.bf_solver import solve_bf                      # noqa: E402
from cabinet import config as C                              # noqa: E402
from cabinet.config import estar_wide                       # noqa: E402
from cabinet.env import CabinetEnv, PROPOSE                 # noqa: E402
from cabinet.metrics import banzhaf, gamson_regression, summarise  # noqa: E402
from cabinet.ppo import evaluate                              # noqa: E402
from cabinet.policy import load_policy, pad_obs, to_torch   # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
MAX_N = 8


def load_cabinets(clean_only=False):
    rows = defaultdict(list)
    with open(os.path.join(ROOT, "data", "seki_williams", "cabinets.csv"), newline="") as f:
        for r in csv.DictReader(f):
            rows[r["cabinet"]].append(r)
    cabs = []
    for name, rs in rows.items():
        if clean_only and rs[0]["clean_core"] != "1":
            continue
        seats = np.array([float(r["seats"]) for r in rs])
        order = np.argsort(-seats)
        rs = [rs[i] for i in order]
        seats = seats[order]
        member = np.array([r["in_cabinet"] == "1" for r in rs])
        pm = [i for i, r in enumerate(rs) if r["is_pm"] == "1"]
        if len(pm) != 1:
            continue
        cabs.append(dict(name=name, country=rs[0]["country"], seats=seats, house=seats.sum(),
                         member=member, pm=pm[0], share=np.array([float(r["portfolio_share"]) for r in rs]),
                         m=int(float(rs[0]["n_portfolios"])), clean=rs[0]["clean_core"] == "1"))
    return cabs


def bf_split(cab, delta=0.9):
    """BF over the 8 largest parties (the others still count in the quota)."""
    k = min(MAX_N, len(cab["seats"]))
    if cab["member"][k:].any():
        return None
    w = np.round(cab["seats"][:k]).astype(int)
    q = int(cab["house"] // 2 + 1)
    key = tuple(w)
    if key not in bf_split.cache:
        bf_split.cache[key] = solve_bf(w, q, w / w.sum(), delta)["v"]
    v = np.zeros(len(cab["seats"]))
    v[:k] = bf_split.cache[key]
    x = np.where(cab["member"], delta * v, 0.0)
    x[cab["pm"]] = 0.0
    if x.sum() >= 1:
        x = x / x.sum() * 0.99
    x[cab["pm"]] = 1.0 - x.sum()
    return x


bf_split.cache = {}


def agent_split(net, cab, samples, seed):
    """The trained formateur splits the real coalition's ministries."""
    keep = np.arange(len(cab["seats"]))[:MAX_N]
    if not cab["member"][keep].sum() == cab["member"].sum():
        return None                                # a member is outside the 8 largest
    seats = cab["seats"][keep]
    base = estar_wide()
    cfg = base.with_(name=cab["name"], seats=tuple(int(round(s)) for s in seats),
                     house=int(cab["house"]), n_choices=(len(keep),), seat_mode="fixed",
                     first_formateur=int(cab["pm"]), T=1.0, m=min(max(cab["m"], 1), 25))
    env = CabinetEnv(cfg, seed=seed)
    obs = []
    for _ in range(samples):
        env.reset()
        obs.append(env.observe(cab["pm"], PROPOSE))
    batch = pad_obs(obs, len(keep), cfg.m)
    force = np.full((samples, len(keep) - 1), -1, np.int64)
    for b, o in enumerate(obs):
        for t, j in enumerate(o["order"]):
            force[b, t] = int(cab["member"][j])
    batch["force_inc"] = force
    with torch.no_grad():
        out, _, _, _ = net(to_torch(batch))
    mu = out["mu"].numpy()[:, :cfg.m]
    shares = np.zeros(len(cab["seats"]))
    for b in range(samples):
        shares[keep] += np.bincount(mu[b], minlength=len(keep)) / cfg.m
    return shares / samples


def own_cabinets(net, cabs, base, episodes, seed):
    """Agents bargain freely in each real parliament (8 largest parties); their
    own cabinets are pooled and summarised like the real ones."""
    cab_log, ep_log = [], []
    for i, c in enumerate(cabs):
        k = min(MAX_N, len(c["seats"]))
        cfg = base.with_(name=c["name"], seats=tuple(int(round(x)) for x in c["seats"][:k]),
                         house=int(c["house"]), n_choices=(k,), seat_mode="fixed", T=1.0)
        cl, el = evaluate(cfg, {"learn": net}, lambda env, rng: ["learn"] * env.n,
                          episodes=episodes, seed=seed * 1000 + i)
        cab_log += cl
        ep_log += el
    return summarise(cab_log, ep_log, base.m, base.divisible)


def mae(pred, cab):
    mem = cab["member"]
    return float(np.mean(np.abs(pred[mem] - cab["share"][mem])))


def rows_of(pred, cab):
    mem = np.flatnonzero(cab["member"])
    s = cab["seats"][mem] / cab["seats"][mem].sum()
    return [(pred[j], s[k], float(j == cab["pm"])) for k, j in enumerate(mem)]


def cluster_boot(values_by_country, stat, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    countries = list(values_by_country)
    out = []
    for _ in range(B):
        pick = rng.choice(len(countries), len(countries))
        out.append(stat([v for i in pick for v in values_by_country[countries[i]]]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["mech_EstarWide"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--own", default=None,
                    help="config name: instead of conditional predictions, summarise the cabinets "
                         "the agents form themselves in the real parliaments")
    ap.add_argument("--episodes", type=int, default=64)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cabs = load_cabinets()
    res = dict(n_cabinets=len(cabs), n_countries=len(set(c["country"] for c in cabs)))

    # ---- empirical regression (what real cabinets do)
    emp = [r for c in cabs for r in rows_of(c["share"], c)]
    reg = gamson_regression(np.array(emp))
    by_c = defaultdict(list)
    for c in cabs:
        by_c[c["country"]] += rows_of(c["share"], c)
    lo, hi = cluster_boot(by_c, lambda rr: gamson_regression(np.array(rr))["gamma"])
    res["empirical"] = dict(**{k: float(v) for k, v in reg.items()}, gamma_ci=[lo, hi])
    bz_rows = []
    for c in cabs:
        k = min(MAX_N + 4, len(c["seats"]))      # the 12 largest parties (as in the companion data)
        b = np.zeros(len(c["seats"]))
        b[:k] = banzhaf(np.round(c["seats"][:k]).astype(int), int(c["house"] // 2 + 1))
        mem = np.flatnonzero(c["member"])
        bc = b[mem] / max(b[mem].sum(), 1e-12)
        bz_rows += [(c["share"][j], bc[k], float(j == c["pm"])) for k, j in enumerate(mem)]
    res["empirical_banzhaf"] = {k: float(v) for k, v in gamson_regression(np.array(bz_rows)).items()}
    print("empirical:", {k: round(v, 3) if isinstance(v, float) else v for k, v in res["empirical"].items()},
          "| banzhaf R2", round(res["empirical_banzhaf"]["r2"], 3), flush=True)

    if a.own:
        base = {"WideProposal": C.WIDE_DIV, "WideDemand": C.WIDE_DEMAND,
                "EstarWide": C.estar_wide()}[a.own]
        res["own"] = {}
        import time
        for tag in a.tags:
            for seed in a.seeds:
                p = os.path.join(ROOT, "models", tag, f"seed{seed}.pt")
                waited = 0
                while not os.path.exists(p) and waited < 8 * 3600:   # queued before training ended
                    time.sleep(60)
                    waited += 60
                if not os.path.exists(p):
                    continue
                time.sleep(5)
                o = own_cabinets(load_policy(p).eval(), cabs, base, a.episodes, seed)
                res["own"][f"{tag}/s{seed}"] = {k: float(v) for k, v in o.items()
                                               if isinstance(v, (int, float, np.floating)) and v == v}
                print(f"it=1/1 t=0s eta=0s own {tag}/s{seed}: " + " ".join(
                    f"{k}={o[k]:+.3f}" for k in ("reg_alpha", "reg_beta", "reg_gamma", "reg_r2",
                                                  "dem_gamma_close", "dem_gamma_first", "size")
                    if k in o and o[k] == o[k]), flush=True)
        os.makedirs(os.path.join(ROOT, "results", "seki"), exist_ok=True)
        with open(os.path.join(ROOT, "results", "seki", f"own_{a.own}.json"), "w") as f:
            json.dump(res, f, indent=1)
        print("DONE", flush=True)
        return

    # ---- predictions
    preds = {"gamson": {}, "bf": {}}
    for c in cabs:
        s = np.where(c["member"], c["seats"], 0.0)
        preds["gamson"][c["name"]] = s / s.sum()
        x = bf_split(c)
        if x is not None:
            preds["bf"][c["name"]] = x
    for tag in a.tags:
        for seed in a.seeds:
            p = os.path.join(ROOT, "models", tag, f"seed{seed}.pt")
            if not os.path.exists(p):
                continue
            net = load_policy(p).eval()
            name = f"{tag}/s{seed}"
            preds[name] = {}
            for i, c in enumerate(cabs):
                x = agent_split(net, c, a.samples, seed * 1000 + i)
                if x is not None:
                    preds[name][c["name"]] = x
            print(f"it=1/1 t=0s eta=0s {name}: {len(preds[name])} cabinets", flush=True)

    # ---- scores on the cabinets every model could predict
    common = [c for c in cabs if all(c["name"] in p for p in preds.values())]
    res["n_common"] = len(common)
    res["models"] = {}
    for name, p in preds.items():
        errs = defaultdict(list)
        diffs = defaultdict(list)
        rows = []
        for c in common:
            e = mae(p[c["name"]], c)
            errs[c["country"]].append(e)
            diffs[c["country"]].append(e - mae(preds["gamson"][c["name"]], c))
            rows += rows_of(p[c["name"]], c)
        allv = [v for vs in errs.values() for v in vs]
        r = gamson_regression(np.array(rows))
        res["models"][name] = dict(
            mae=float(np.mean(allv)), mae_ci=cluster_boot(errs, np.mean),
            dmae_vs_gamson=float(np.mean([v for vs in diffs.values() for v in vs])),
            dmae_ci=cluster_boot(diffs, np.mean) if name != "gamson" else [0.0, 0.0],
            gamma=float(r["gamma"]), beta=float(r["beta"]), alpha=float(r["alpha"]))
        m = res["models"][name]
        print(f"{name:24s} MAE={m['mae']:.3f} [{m['mae_ci'][0]:.3f},{m['mae_ci'][1]:.3f}] "
              f"dMAE={m['dmae_vs_gamson']:+.3f} [{m['dmae_ci'][0]:+.3f},{m['dmae_ci'][1]:+.3f}] "
              f"g={m['gamma']:+.3f} b={m['beta']:.2f}", flush=True)
    os.makedirs(os.path.join(ROOT, "results", "seki"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "seki", "summary.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
