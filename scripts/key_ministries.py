"""Does the real formateur's premium hide in *which* ministries it holds?

Our agents keep counted shares proportional while the formateur takes the
ministries that matter most to it (valuations) or that set policy (lead
ministries).  This tests the real counterpart on the 195 majority coalition
cabinets of data/seki_williams/: is the prime minister's party's share of the
key ministries (finance, foreign affairs, interior, defence, justice) larger
than its share of all ministers and of the coalition's seats?

Ministers come from the Seki-Williams minister file (scripts/seki_download.sh);
every appointment of a partisan minister counts once, and junior posts
(deputy, state secretaries, ministers of state) are excluded.  Confidence
intervals: bootstrap over countries (2000 resamples).

    python scripts/key_ministries.py
"""
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "seki_williams", "raw")
KEY = {
    "finance": r"financ|treasury|exchequer",
    "foreign affairs": r"foreign affairs|external affairs|foreign minister",
    "interior": r"interior|home affairs|internal affairs",
    "defence": r"defen[cs]e|armed forces",
    "justice": r"justice",
}
JUNIOR = r"state secretary|secretary of state for|deputy minister|vice[- ]minister|minister of state|" \
         r"junior|parliamentary secretary|assistant|undersecretary|under-secretary"


def norm(s):
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def main():
    g = pd.read_stata(os.path.join(RAW, "governments_v2.dta"), convert_categoricals=False)
    m = pd.read_stata(os.path.join(RAW, "ministers_v2.dta"), convert_categoricals=False)
    cab = pd.read_csv(os.path.join(ROOT, "data", "seki_williams", "cabinets.csv"))
    rows, by_cat = [], defaultdict(list)
    for (country, seq), c in cab.groupby(["country", "sw_govtseq"]):
        gr = g[(g.country == country) & (g.govtseq == seq)]
        if gr.empty:
            continue
        gr = gr.iloc[0]
        mm = m[(m.ccode == gr.ccode) & (m.govtseq == seq)].copy()
        mm = mm[mm.party.notna() & ~mm.party.astype(str).str.upper().isin(["IND", "NONE", ""])]
        mm["port"] = mm.portfolio.astype(str).str.lower()
        mm = mm[~mm.port.str.contains(JUNIOR, regex=True)]
        if len(mm) < 4:
            continue
        pm_code = norm(gr.primeparty)
        mm["is_pm"] = mm.party.map(norm) == pm_code
        if not mm.is_pm.any() or mm.is_pm.all():
            continue
        mem = c[c.in_cabinet == 1]
        pm_row = mem[mem.is_pm == 1]
        seat_share = float(pm_row.seats.iloc[0] / mem.seats.sum())
        count_share = float(mm.is_pm.mean())
        key = mm[mm.port.str.contains("|".join(KEY.values()), regex=True)]
        if len(key) == 0:
            continue
        rows.append(dict(cabinet=f"{country}-{seq}", country=country, clean=int(c.clean_core.iloc[0]),
                         seat_share=seat_share, count_share=count_share,
                         sw_count_share=float(pm_row.portfolio_share.iloc[0]),
                         key_share=float(key.is_pm.mean()), n_parties=len(mem)))
        for name, pat in KEY.items():
            k = mm[mm.port.str.contains(pat, regex=True)]
            if len(k):
                by_cat[name].append((country, float(k.is_pm.mean()) - count_share))
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    countries = df.country.unique()

    def ci(col_fn, data=df):
        stats = []
        for _ in range(2000):
            pick = rng.choice(countries, len(countries))
            d = pd.concat([data[data.country == k] for k in pick])
            stats.append(col_fn(d))
        return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]

    out = dict(n_cabinets=len(df), n_countries=int(df.country.nunique()))
    for name, fn in (("key_minus_count", lambda d: (d.key_share - d.count_share).mean()),
                     ("key_minus_seat", lambda d: (d.key_share - d.seat_share).mean()),
                     ("count_minus_seat", lambda d: (d.count_share - d.seat_share).mean())):
        out[name] = dict(mean=float(fn(df)), ci=ci(fn))
        clean = df[df.clean == 1]
        out[name + "_clean"] = float(fn(clean))
    for col in ("seat_share", "count_share", "sw_count_share", "key_share"):
        out["mean_" + col] = float(df[col].mean())
    out["by_ministry"] = {k: dict(mean=float(np.mean([v for _, v in vals])), n=len(vals))
                          for k, vals in by_cat.items()}
    os.makedirs(os.path.join(ROOT, "results", "seki"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "seki", "key_ministries.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"{out['n_cabinets']} cabinets, {out['n_countries']} countries")
    print(f"PM party: seats {out['mean_seat_share']:.3f}, all ministers {out['mean_count_share']:.3f} "
          f"(Seki-Williams count {out['mean_sw_count_share']:.3f}), key ministries {out['mean_key_share']:.3f}")
    for k in ("key_minus_count", "key_minus_seat", "count_minus_seat"):
        print(f"{k:18s} {out[k]['mean']:+.3f}  95% CI [{out[k]['ci'][0]:+.3f}, {out[k]['ci'][1]:+.3f}]  "
              f"clean core {out[k + '_clean']:+.3f}")
    for k, v in out["by_ministry"].items():
        print(f"  {k:16s} PM share minus its count share {v['mean']:+.3f} (n={v['n']})")


if __name__ == "__main__":
    main()
