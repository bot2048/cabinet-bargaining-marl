"""Sharp real-data test: seats or voting weights?

Within real coalition cabinets, take every pair of members with the same
minimal integer voting weight but different seats. Gamson's law predicts that
their portfolio-share difference tracks their seat-share difference (slope 1);
voting-weight theories (demand bargaining, and our learned demand agents)
predict equal shares (slope 0). Country-clustered bootstrap over pairs.

    python scripts/real_weights_vs_seats.py
"""
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from real_robustness import min_integer_weights  # noqa: E402
from seki_eval import load_cabinets  # noqa: E402


def main():
    pairs = []
    for c in load_cabinets():
        top = min(12, len(c["seats"]))
        w, _ = min_integer_weights(np.round(c["seats"][:top]).astype(int), int(c["house"] // 2 + 1))
        wv = np.zeros(len(c["seats"]), int)
        wv[:top] = w
        mem = np.flatnonzero(c["member"])
        s = c["seats"][mem] / c["seats"][mem].sum()
        for a, b in itertools.combinations(range(len(mem)), 2):
            i, j = mem[a], mem[b]
            if wv[i] == wv[j] and wv[i] > 0 and abs(s[a] - s[b]) > 0.02:
                pairs.append(dict(country=c["country"], cab=c["name"], ds=float(s[a] - s[b]),
                                  dp=float(c["share"][i] - c["share"][j]),
                                  pm=int(i == c["pm"]) - int(j == c["pm"])))
    ds = np.array([p["ds"] for p in pairs])
    dp = np.array([p["dp"] for p in pairs])
    pm = np.array([p["pm"] for p in pairs], float)

    def fit(idx):
        X = np.c_[ds[idx], pm[idx]]          # through the origin: equal seats, no PM -> equal shares
        return np.linalg.lstsq(X, dp[idx], rcond=None)[0]

    b = fit(np.arange(len(pairs)))
    by = defaultdict(list)
    for k, p in enumerate(pairs):
        by[p["country"]].append(k)
    keys = list(by)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(2000):
        idx = np.concatenate([by[keys[r]] for r in rng.integers(len(keys), size=len(keys))])
        boot.append(fit(idx))
    lo, hi = np.percentile(np.array(boot), [2.5, 97.5], axis=0)
    no_pm = pm == 0
    b_nopm = np.linalg.lstsq(ds[no_pm, None], dp[no_pm], rcond=None)[0][0]
    out = dict(pairs=len(pairs), cabinets=len({p["cab"] for p in pairs}),
               countries=len(keys), seat_slope=float(b[0]), seat_slope_ci=[float(lo[0]), float(hi[0])],
               pm_coef=float(b[1]), pm_ci=[float(lo[1]), float(hi[1])],
               seat_slope_pairs_without_pm=float(b_nopm), pairs_without_pm=int(no_pm.sum()),
               larger_party_gets_more=float(np.mean(np.sign(dp[no_pm]) == np.sign(ds[no_pm]))))
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(ROOT, "results", "real_weights_vs_seats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
