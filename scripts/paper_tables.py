"""Tables of the full paper (paper/tab_*.tex), generated from results/.

    python scripts/paper_tables.py
"""
import glob
import json
import os
import re

import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(__file__), "..")


def runs(tag):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(ROOT, "results", tag, "seed*.json")))]


def mci(v):
    """Mean and standard error over seeds (t-intervals with 2-3 seeds would be
    dominated by the t critical value, e.g. 12.7 with two seeds)."""
    v = np.array([x for x in v if x is not None and x == x], float)
    if len(v) == 0:
        return "--"
    if len(v) == 1:
        return f"{v[0]:+.3f}"
    se = v.std(ddof=1) / np.sqrt(len(v))
    return f"{v.mean():+.3f} $\\pm$ {se:.3f}"


def f3(x, sign=True):
    return "--" if x is None or x != x else (f"{x:+.3f}" if sign else f"{x:.3f}")


def e0_table():
    t = json.load(open(os.path.join(ROOT, "results", "e0_table.json")))
    br = json.load(open(os.path.join(ROOT, "results", "bf_best_response.json")))["summary"]
    L = [r"\begin{table}[t]\centering\footnotesize",
         r"\caption{Benchmark E0 (symmetric closed-rule Baron--Ferejohn game, divisible surplus; 10 seeds, "
         r"4000 evaluation episodes each). Gap: learned minus closed-form formateur share, with a 95\% "
         r"$t$-interval over seeds. Threshold: offer at which a learned voter accepts with probability 1/2. "
         r"BR gain: payoff a formateur would gain per formation by best-responding to the frozen learned "
         r"voters (mean over seeds).}\label{tab:e0}",
         r"\begin{tabular}{@{}rrcccccc@{}}\toprule",
         r"$n$ & $\delta$ & $x^{\mathrm{BF}}_f$ & learned & gap [95\% CI] & threshold & $\delta/n$ & BR gain\\\midrule"]
    for c in t["cells"]:
        g = br[f"{c['n']},{c['delta']},stochastic"]["gain"]
        L.append(f"{c['n']} & {c['delta']} & {c['bf']:.3f} & {c['pf']:.3f} & {c['gap']:+.3f} "
                 f"[{c['gap_ci'][0]:+.3f}, {c['gap_ci'][1]:+.3f}] & {c['threshold']:.3f} & "
                 f"{c['reservation']:.3f} & {g:.3f}\\\\")
    s = t["summary"]
    L += [r"\midrule", f"\\multicolumn{{8}}{{@{{}}l}}{{mean signed gap {s['mean_signed_gap']:+.3f}; "
          f"mean absolute gap {s['mean_abs_gap']:.3f}; {s['cells_within_002']}/9 cells within 0.02}}\\\\",
          r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(L)


MECH = [("mech_BASE", "base game (all mechanisms off)"), ("mech_BASE+M1_repeat", "+ repeated terms (M1)"),
        ("mech_BASE+M1_patient", "+ repeated, patient"), ("mech_BASE+M5_hetero", "+ heterogeneous public values (M5)"),
        ("mech_BASE+M4_private", "+ private values (M4+M5)"), ("mech_BASE+M8_exits", "+ exits (M8)"),
        ("mech_BASE+M9_policy", "+ ministerial policy (M9)"), ("mech_BASE+M10_seatrho", "+ seat recognition (M10)"),
        ("mech_Estar", "E$^\\star$ (all on)"), ("mech_Estar-M1_repeat", "E$^\\star$ $-$ repeated terms"),
        ("mech_Estar-M4_private", "E$^\\star$ $-$ private values"), ("mech_Estar-M5_hetero", "E$^\\star$ $-$ heterog. values"),
        ("mech_Estar-M8_exits", "E$^\\star$ $-$ exits"), ("mech_Estar-M9_policy", "E$^\\star$ $-$ ministerial policy"),
        ("mech_Estar-M10_seatrho", "E$^\\star$ $-$ seat recognition"),
        ("mappo_Estar", "E$^\\star$, MAPPO"), ("fsp_Estar", "E$^\\star$, fictitious self-play"),
        ("rec_Estar", "E$^\\star$, GRU memory"), ("mech_Estar_Lam0", "E$^\\star$, memoryless"),
        ("mech_Estar_m12", "E$^\\star$, 12 ministries"),
        ("long_BASE", "base, 4$\\times$ training"), ("long_BASE+M4_private", "+ private values, 4$\\times$"),
        ("long_BASE+M9_policy", "+ policy, 4$\\times$"), ("long_Estar", "E$^\\star$, 4$\\times$")]


def mech_table():
    L = [r"\begin{table}[t]\centering\footnotesize",
         r"\caption{Formateur coefficient $\hat\gamma$ of regression~\eqref{eq:reg} by configuration (mean "
         r"$\pm$ standard error over seeds). Counted: share of ministries. Value: share of the value each "
         r"party places on ministries. Lead: share of policy-setting ministries (only with M9). "
         r"$\tilde D_G$: floor-adjusted Gamson distance. Configurations with 2--3 seeds are exploratory; "
         r"the confirmatory tests are in Section~\ref{sec:protocol}.}\label{tab:mech}",
         r"\begin{tabular}{@{}lccccc@{}}\toprule",
         r"configuration & $\hat\gamma$ counted & $\hat\gamma$ value & $\hat\gamma$ lead & $\tilde D_G$ & seeds\\\midrule"]
    for tag, lab in MECH:
        R = runs(tag)
        if not R:
            continue
        o = [r["overall"] for r in R]
        lead = [x.get("reglead_gamma") for x in o]
        L.append(f"{lab} & {mci([x['reg_gamma'] for x in o])} & {mci([x.get('regnu_gamma') for x in o])} & "
                 f"{mci(lead) if any(v is not None for v in lead) else '--'} & "
                 f"{np.mean([x['dg_adj'] for x in o]):.3f} & {len(R)}\\\\")
        if tag in ("mech_BASE+M10_seatrho", "mech_Estar-M10_seatrho", "mech_Estar_m12"):
            L.append(r"\addlinespace")
    for tag, lab in (("psro_BASE", "base, PSRO"), ("psro_Estar", "E$^\\star$, PSRO")):
        R = runs(tag)
        if R:
            g = R[0].get("overall", R[0].get("final", {})).get("reg_gamma")
            L.append(f"{lab} & {f3(g)} & -- & -- & -- & 1\\\\")
    L.append(r"\bottomrule\end{tabular}\end{table}")
    return "\n".join(L)


def exploit_table():
    rows = [("exploit_Estar_br400", "E$^\\star$ (IPPO)", 400), ("exploit_mappo_Estar", "E$^\\star$ (MAPPO)", 150),
            ("exploit_Estar", "E$^\\star$ (IPPO, 3 seeds)", 150), ("exploit_BASE", "base game (3 seeds)", 150),
            ("exploit_e0_br300", "E0 (BF benchmark)", 300),
            ("exploit_mech_WideProposal_br400", "random parliaments, proposal", 400),
            ("exploit_mech_WideDemand_br400", "random parliaments, demand", 400),
            ("exploit_mech_E0demand_br300", "symmetric demand, default exploration", 300),
            ("exploit_explore_E0demand_br300", "symmetric demand, more exploration", 300)]
    L = [r"\begin{table}[t]\centering\footnotesize",
         r"\caption{Estimated exploitability: per-period gain of a best response trained by PPO for one random "
         r"seat against the frozen profile (larger of a warm and a from-scratch start), also relative to the "
         r"profile seat's payoff. Scripted positive controls (proposal protocol only) have no network weights to "
         r"start from, so their best response is trained from scratch only; with a weak budget its gain can be "
         r"slightly negative.}\label{tab:exploit}",
         r"\begin{tabular}{@{}lrcccc@{}}\toprule",
         r"profile & BR iters & gain & relative & Gamson ctrl. & BF-heur. ctrl.\\\midrule"]
    for tag, lab, it in rows:
        R = runs(tag)
        if not R:
            continue
        g = [r["gain"] for r in R]
        rel = []
        for r in R:
            if "runs" in r:
                b = max(r["runs"].values(), key=lambda v: v["gain"])
                rel.append(b["gain"] / b["u_base"])
        c = R[0].get("controls", {}) if "demand" not in tag.lower() else {}   # scripted agents cannot demand
        L.append(f"{lab} & {it} & {np.mean(g):.3f} & {(np.mean(rel) if rel else float('nan')):.0%} & "
                 f"{f3(c.get('gamson', {}).get('gain'), False)} & {f3(c.get('myopic', {}).get('gain'), False)}\\\\"
                 .replace("nan%", "--").replace("%", r"\%"))
    L.append(r"\bottomrule\end{tabular}\end{table}")
    return "\n".join(L)


def main():
    header = "% generated by scripts/paper_tables.py -- do not edit\n"
    for name, tex in (("tab_e0", e0_table()), ("tab_mech", mech_table()), ("tab_exploit", exploit_table())):
        tex = re.sub(r"(?<![\d-])-(?=\d)", r"\\ensuremath{-}", tex)   # typographic minus
        with open(os.path.join(ROOT, "paper", f"{name}.tex"), "w") as f:
            f.write(header + tex + "\n")
        print(tex)


if __name__ == "__main__":
    main()
