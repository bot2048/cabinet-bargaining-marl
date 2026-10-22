"""Within-party analogue of Psi_in on real cabinets (spec v0.5, Section 4.4).

Compares a party's portfolio share when it holds the premiership with its share
when it is a junior partner, net of its seat share, with party fixed effects:

    p_ic = a_party + beta * s_ic + gamma * PM_ic + e_ic,

estimated on the parties that appear in the data both as prime minister's party
and as junior partner (others contribute nothing to gamma). Inference: wild
cluster bootstrap by country (Webb weights), as in scripts/real_robustness.py.

    python scripts/real_within_party.py
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import real_robustness as rr  # noqa: E402


def main():
    rows = defaultdict(list)
    with open(os.path.join(ROOT, "data", "seki_williams", "cabinets.csv"), newline="") as f:
        for r in csv.DictReader(f):
            rows[r["cabinet"]].append(r)
    obs = []
    for cab, rs in rows.items():
        mem = [r for r in rs if r["in_cabinet"] == "1"]
        if sum(r["is_pm"] == "1" for r in mem) != 1 or len(mem) < 2:
            continue
        tot = sum(float(r["seats"]) for r in mem)
        for r in mem:
            obs.append(dict(country=r["country"], party=f"{r['country']}:{r['party_id']}", cab=cab,
                            p=float(r["portfolio_share"]), s=float(r["seats"]) / tot, pm=float(r["is_pm"] == "1")))
    by = defaultdict(set)
    for o in obs:
        by[o["party"]].add(o["pm"])
    both = {p for p, v in by.items() if v == {0.0, 1.0}}
    sub = [o for o in obs if o["party"] in both]
    parties = sorted(both)
    idx = {p: k for k, p in enumerate(parties)}
    y = np.array([o["p"] for o in sub])
    D = np.zeros((len(sub), len(parties)))
    for k, o in enumerate(sub):
        D[k, idx[o["party"]]] = 1
    X = np.c_[np.array([o["s"] for o in sub]), np.array([o["pm"] for o in sub]), D]
    g = np.array([o["country"] for o in sub])
    rr.y_global = y
    gam, se, p, ci = rr.wild_cluster(X, y, g, 1, B=1999)
    b, _ = rr.ols(X, y)
    out = dict(parties=len(parties), observations=len(sub), cabinets=len({o["cab"] for o in sub}),
               countries=len(set(g)), gamma=gam, wild_ci=ci, wild_p=p, beta=float(b[0]),
               pm_observations=int(sum(o["pm"] for o in sub)))
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(ROOT, "results", "real_within_party.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
