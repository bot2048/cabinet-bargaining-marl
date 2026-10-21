"""The pre-registered tests of docs/PREREGISTRATION.md, from the saved results.

H1/H2: protocol effect on the formateur coefficient and the seat slope (exact
two-sample permutation test over seeds, one-sided; seed-bootstrap 95% CI of
the difference); H3: TOST equivalence |gamma(demand)| < 0.05. Holm across H1-H3.
P1-P3: fixed parliaments 45/35/20 and 40/30/25/5: within-cabinet seat slope
per seed, split by coalition, dummy inclusion; compared with the exact demand
equilibrium (cabinet/demand_solver.py) and Gamson.

    python scripts/confirmatory.py
"""
import glob
import itertools
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(__file__), "..")


def seeds(tag):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(ROOT, "results", tag, "seed*.json")))]


def perm_test(a, b):
    """One-sided exact permutation p-value for mean(a) - mean(b) > 0."""
    x, n = np.r_[a, b], len(a)
    obs = a.mean() - b.mean()
    hits = tot = 0
    for idx in itertools.combinations(range(len(x)), n):
        m = np.zeros(len(x), bool)
        m[list(idx)] = True
        tot += 1
        hits += x[m].mean() - x[~m].mean() >= obs - 1e-12
    return float(obs), hits / tot


def boot_ci(a, b, B=20000, seed=0):
    rng = np.random.default_rng(seed)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(B)]
    return [float(v) for v in np.percentile(d, [2.5, 97.5])]


def cabinets(rows):
    """Split the pooled party rows back into cabinets (seat shares sum to one)."""
    out, cur, tot = [], [], 0.0
    for r in rows:
        cur.append(r)
        tot += r[1]
        if abs(tot - 1) < 1e-6:
            out.append(np.array(cur))
            cur, tot = [], 0.0
    return out


def fixed_parliament(tag, seats):
    res = []
    for r in seeds(tag):
        R = np.array(r["rows"])
        cabs = cabinets(R)
        slope = float(np.polyfit(R[:, 1], R[:, 0], 1)[0])
        by = defaultdict(list)
        for c in cabs:
            ids = tuple(sorted(int(round(x)) for x in c[:, 3] * sum(seats)))[::-1]
            by[ids].append(c)
        pairs = {}
        for ids, cs in by.items():
            mean = {w: float(np.mean([x[0] for c in cs for x in c if round(x[3] * sum(seats)) == w])) for w in ids}
            form = [x[0] for c in cs for x in c if x[2] == 1]
            pairs["+".join(map(str, ids))] = dict(freq=len(cs) / len(cabs), shares=mean,
                                                  formateur_share=float(np.mean(form)))
        dummy = min(seats) if len(seats) == 4 else None
        incl = float(np.mean([any(round(x[3] * sum(seats)) == dummy for x in c) for c in cabs])) if dummy else None
        res.append(dict(slope=slope, gamma=r["overall"]["reg_gamma"], coalitions=pairs, dummy_included=incl))
    return res


def main():
    out = {}
    gp = np.array([r["overall"]["reg_gamma"] for r in seeds("mech_WideProposal")])
    gd = np.array([r["overall"]["reg_gamma"] for r in seeds("mech_WideDemand")])
    bp = np.array([r["overall"]["reg_beta"] for r in seeds("mech_WideProposal")])
    bd = np.array([r["overall"]["reg_beta"] for r in seeds("mech_WideDemand")])
    d1, p1 = perm_test(gp, gd)
    d2, p2 = perm_test(bd, bp)
    m, h = gd.mean(), stats.t.ppf(0.95, len(gd) - 1) * gd.std(ddof=1) / np.sqrt(len(gd))
    tost_p = float(max(stats.t.sf((m + 0.05) / (gd.std(ddof=1) / np.sqrt(len(gd))), len(gd) - 1),
                       stats.t.cdf((m - 0.05) / (gd.std(ddof=1) / np.sqrt(len(gd))), len(gd) - 1)))
    ps = sorted([("H1", p1), ("H2", p2), ("H3", tost_p)], key=lambda t: t[1])
    holm = {k: min(1.0, max(p * (len(ps) - i) for i2, (k2, p) in enumerate(ps[:i + 1])))
            for i, (k, _) in enumerate(ps)}
    out["H1"] = dict(proposal=gp.tolist(), demand=gd.tolist(), diff=d1, p=p1, ci=boot_ci(gp, gd), p_holm=holm["H1"])
    out["H2"] = dict(proposal=bp.tolist(), demand=bd.tolist(), diff=d2, p=p2, ci=boot_ci(bd, bp), p_holm=holm["H2"])
    out["H3"] = dict(mean=float(m), ci90=[float(m - h), float(m + h)], margin=0.05, p_tost=tost_p,
                     p_holm=holm["H3"], equivalent=bool(m - h > -0.05 and m + h < 0.05))
    for k in ("H1", "H2", "H3"):
        print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in out[k].items()
                  if not isinstance(b, list) or len(b) == 2})
    for tag, seats in (("explore_A452Demand", (45, 35, 20)), ("mech_A452Proposal", (45, 35, 20)),
                       ("explore_A4305Demand", (40, 30, 25, 5)), ("mech_A4305Proposal", (40, 30, 25, 5))):
        out[tag] = fixed_parliament(tag, seats)
        for s, r in enumerate(out[tag]):
            print(tag, f"s{s}", "slope", round(r["slope"], 2), "dummy", r["dummy_included"], " ".join(
                f"{k}:{v['freq']:.2f}:" + "/".join(f"{x:.2f}" for x in v["shares"].values()) +
                f"(F {v['formateur_share']:.2f})" for k, v in sorted(out[tag][s]["coalitions"].items())))
    sl = [r["slope"] for r in out["explore_A452Demand"]]
    h1 = stats.t.ppf(0.975, len(sl) - 1) * np.std(sl, ddof=1) / np.sqrt(len(sl))
    out["P1_ci"] = [float(np.mean(sl) - h1), float(np.mean(sl) + h1)]   # seed-level 95% t-interval
    out["P1"] = dict(slopes=sl, verdict="weights" if np.mean(sl) < 0.5 and max(sl) < 0.5 else
                     "seats" if np.mean(sl) > 0.5 and min(sl) > 0.5 else "mixed")
    out["P2"] = dict(dummy_included=[r["dummy_included"] for r in out["explore_A4305Demand"]])
    print("P1", out["P1"], "P2", out["P2"])
    json.dump(out, open(os.path.join(ROOT, "results", "confirmatory.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
