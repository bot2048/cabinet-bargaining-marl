"""Level-1 validation table: learned play vs the symmetric BF equilibrium (E0).

For every cell (n, delta), from the saved E0 results (one file per training
seed; statistics of each seed come from 4000 evaluation episodes):

* formateur share: mean over seeds, the signed gap to x_BF = 1 - delta(n-1)/(2n),
  and a 95% interval over seeds (t-interval; seeds are the independent units);
* the voters' acceptance threshold (offer at which Pr(accept) crosses 1/2)
  against the equilibrium reservation value delta / n;
* the share of minimal winning coalitions (BF: all) and of failed rounds (BF: none).

Summary statistics are reported separately and named explicitly: the mean
signed gap (bias), the mean absolute gap (accuracy) and the number of cells
within the 0.02 tolerance of success criterion S1.

    python scripts/benchmark_table.py
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from cabinet.bf_solver import bf_symmetric_share  # noqa: E402


def main():
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(ROOT, "results", "e0", "seed*.json")))]
    cells = sorted(runs[0]["cells"], key=lambda k: eval(k))
    rows = []
    for key in cells:
        n, d = eval(key)
        bf = bf_symmetric_share(n, d)
        pf = np.array([r["cells"][key]["pf"] for r in runs])
        thr = np.array([r["thresholds"][key] for r in runs if r["thresholds"].get(key) is not None])
        mwc = np.array([r["cells"][key]["minimal_winning"] for r in runs])
        fail = np.array([r["cells"][key].get("fail_rate", np.nan) for r in runs])
        gap = pf - bf
        half = stats.t.ppf(0.975, len(gap) - 1) * gap.std(ddof=1) / np.sqrt(len(gap))
        rows.append(dict(n=n, delta=d, bf=bf, pf=float(pf.mean()), gap=float(gap.mean()),
                         gap_ci=[float(gap.mean() - half), float(gap.mean() + half)],
                         gap_seed_min=float(gap.min()), gap_seed_max=float(gap.max()),
                         threshold=float(thr.mean()), reservation=d / n,
                         minimal_winning=float(mwc.mean()), fail_rate=float(np.nanmean(fail)),
                         seeds=len(pf)))
    gaps = np.array([r["gap"] for r in rows])
    summary = dict(seeds=len(runs), cells=len(rows), mean_signed_gap=float(gaps.mean()),
                   mean_abs_gap=float(np.abs(gaps).mean()), max_abs_gap=float(np.abs(gaps).max()),
                   cells_within_002=int((np.abs(gaps) <= 0.02).sum()),
                   cells_ci_excludes_zero=int(sum(r["gap_ci"][0] > 0 or r["gap_ci"][1] < 0 for r in rows)))
    out = dict(summary=summary, cells=rows)
    with open(os.path.join(ROOT, "results", "e0_table.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"{'n':>2} {'delta':>5} {'BF':>6} {'learned':>8} {'gap [95% CI over seeds]':>28} "
          f"{'thresh':>7} {'d/n':>6} {'MWC':>5} {'fail':>5}")
    for r in rows:
        print(f"{r['n']:>2} {r['delta']:>5} {r['bf']:6.3f} {r['pf']:8.3f} {r['gap']:+7.3f} "
              f"[{r['gap_ci'][0]:+.3f},{r['gap_ci'][1]:+.3f}]      {r['threshold']:7.3f} "
              f"{r['reservation']:6.3f} {r['minimal_winning']:5.2f} {r['fail_rate']:5.3f}")
    print(summary)


if __name__ == "__main__":
    main()
