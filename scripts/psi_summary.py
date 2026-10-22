"""Summary of the value-of-proposal-power probes (results/psi/*.json).

Per configuration: Psi_in and its components (salience, taste, policy, exit),
Psi_ex, the on-path coverage of Algorithm 4 (mean +- standard error over seeds),
the observational statistics of the same models, and the region of spec v0.5
Problem 1.10. Tolerances are the thresholds of the spec's success criterion S2:
eps_G = 0.05 (floor-adjusted Gamson distance), eps_Phi = eps_Psi = eps_F = 0.02.

Writes results/psi_summary.json and the paper tables paper/tab_psi.tex and
paper/tab_psi_e0.tex.

    python scripts/psi_summary.py
"""
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
EPS_G, EPS_PHI, EPS_PSI, EPS_F = 0.05, 0.02, 0.02, 0.02
COMP = ("sigma", "nu", "pol", "e")

ROWS = [("mech_BASE", "base game"), ("mech_BASE+M1_repeat", "+ repeated terms"),
        ("mech_BASE+M1_patient", "+ repeated, patient"), ("rec_BASE+M1_patient", "+ repeated, patient, GRU"),
        ("mech_BASE+M5_hetero", "+ heterogeneous public values"), ("mech_BASE+M4_private", "+ private values"),
        ("mech_BASE+M8_exits", "+ exits"), ("mech_BASE+M9_policy", "+ ministerial policy"),
        ("mech_BASE+M10_seatrho", "+ seat recognition"),
        ("mech_Estar", "E$^\\star$ (all on)"), ("mech_Estar-M1_repeat", "E$^\\star$ $-$ repeated terms"),
        ("mech_Estar-M4_private", "E$^\\star$ $-$ private values"),
        ("mech_Estar-M5_hetero", "E$^\\star$ $-$ heterog. values"), ("mech_Estar-M8_exits", "E$^\\star$ $-$ exits"),
        ("mech_Estar-M9_policy", "E$^\\star$ $-$ ministerial policy"),
        ("mech_Estar-M10_seatrho", "E$^\\star$ $-$ seat recognition"),
        ("mappo_Estar", "E$^\\star$, MAPPO"), ("fsp_Estar", "E$^\\star$, fictitious self-play"),
        ("rec_Estar", "E$^\\star$, GRU"), ("mech_Estar_Lam0", "E$^\\star$, memoryless"),
        ("mech_Estar_m12", "E$^\\star$, 12 ministries"),
        ("long_BASE", "base, 4$\\times$ training"), ("long_BASE+M4_private", "+ private values, 4$\\times$"),
        ("long_BASE+M9_policy", "+ policy, 4$\\times$"), ("long_Estar", "E$^\\star$, 4$\\times$"),
        ("entropy_BASE_0.002", "base, entropy 0.002"), ("entropy_BASE_0.02", "base, entropy 0.02"),
        ("entropy_BASE_0.05", "base, entropy 0.05"),
        ("mech_EstarWide", "E$^\\star$, 3--8 parties"), ("mech_EstarWidePub", "E$^\\star$, 3--8, public types"),
        ("e1", "weighted parliaments (E1)"), ("mech_WideProposal", "proposal, 3--8 parties (divisible)"),
        ("mech_A452Proposal", "proposal, 45/35/20"), ("mech_A4305Proposal", "proposal, 40/30/25/5"),
        ("mech_WideDemand", "demand, 3--8 parties"), ("explore_WideDemand", "demand, more exploration"),
        ("explore_A452Demand", "demand, 45/35/20"), ("explore_A4305Demand", "demand, 40/30/25/5"),
        ("mech_E0demand", "demand, symmetric"), ("explore_E0demand", "demand, symmetric, more expl.")]


def ms(v):
    v = np.array([x for x in v if x is not None], float)
    if len(v) == 0:
        return None
    return dict(mean=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
                seeds=len(v), min=float(v.min()), max=float(v.max()))


def obs_stats(tag):
    runs = [json.load(open(p)) for p in glob.glob(os.path.join(ROOT, "results", tag, "seed*.json"))]
    o = [r.get("overall", {}) for r in runs]
    get = lambda k: ms([x.get(k) for x in o if x.get(k) is not None and x.get(k) == x.get(k)])
    return dict(phi_adj=get("phi_adj"), dg_adj=get("dg_adj"), gamma=get("reg_gamma"))


def region(psi, ob):
    if psi is None:
        return "--"
    obs_g = ob["dg_adj"] and ob["phi_adj"] and ob["dg_adj"]["mean"] <= EPS_G and abs(ob["phi_adj"]["mean"]) <= EPS_PHI
    if obs_g:
        return "Gamson" if abs(psi) <= EPS_PSI else "disguised"
    return "formateur" if psi >= EPS_F else "--"


def summarise_tag(tag):
    p = os.path.join(ROOT, "results", "psi", f"{tag}.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    seeds = list(r["seeds"].values())
    out = dict(tag=tag, n_seeds=len(seeds))
    for key in ("psi_in", "psi_in_all", "psi_ex"):
        vals = [s[key] for s in seeds if s.get(key)]
        out[key] = ms([v["total"] for v in vals])
        for c in COMP:
            out[f"{key}_{c}"] = ms([v[c] for v in vals])
    out["coverage"] = ms([s.get("on_path_coverage") for s in seeds])
    out["observational"] = obs_stats(tag)
    out["region"] = region(out["psi_in"]["mean"] if out["psi_in"] else None, out["observational"])
    return out


def h10_replication():
    """Spec v0.5 Table 14: H10 must replicate on seeds not used when it was formulated.
    The pilot runs behind H10 used seeds 0-2; seeds 3+ were trained later."""
    from scipy import stats
    out = {}
    for tag in ("mech_Estar", "mappo_Estar", "mech_BASE+M4_private", "mech_BASE+M9_policy"):
        p = os.path.join(ROOT, "results", "psi", f"{tag}.json")
        if not os.path.exists(p):
            continue
        seeds = json.load(open(p))["seeds"]
        new = [v for k, v in seeds.items() if int(k) >= 3 and v.get("psi_in")]
        if len(new) < 2:
            continue
        row = dict(seeds=sorted(int(k) for k in seeds if int(k) >= 3))
        for c in COMP:
            v = np.array([x["psi_in"][c] for x in new])
            h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            row[c] = dict(mean=float(v.mean()), ci=[float(v.mean() - h), float(v.mean() + h)])
        out[tag] = row
    return out


def e0_cells():
    p = os.path.join(ROOT, "results", "psi", "e0.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    cells = {}
    for s in r["seeds"].values():
        for k, c in s["cells"].items():
            cells.setdefault(k, []).append(c)
    out = {}
    for k, cs in cells.items():
        out[k] = dict(psi_in=ms([c["psi_in"]["total"] for c in cs if c["psi_in"]]),
                      psi_ex=ms([c["psi_ex"]["total"] for c in cs]), bf_in=cs[0]["bf_in"], bf_ex=cs[0]["bf_ex"],
                      coverage=ms([c["on_path_coverage"] for c in cs]))
    return out


def f(x, key="mean", sign=True):
    if x is None:
        return "--"
    return (f"{x[key]:+.3f}" if sign else f"{x[key]:.3f}")


def pm(x):
    return "--" if x is None else f"{x['mean']:+.3f} $\\pm$ {x['se']:.3f}"


def main():
    rows = {t: summarise_tag(t) for t, _ in ROWS}
    e0 = e0_cells()
    rep = h10_replication()
    json.dump(dict(rows=rows, e0=e0, h10_replication=rep, eps=dict(G=EPS_G, Phi=EPS_PHI, Psi=EPS_PSI, F=EPS_F)),
              open(os.path.join(ROOT, "results", "psi_summary.json"), "w"), indent=1)
    L = [r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}",
         r"\caption{Value of proposal power (spec v0.5, Definition~1.8), estimated by counterfactual replays "
         r"(Algorithm~4: 2000 probe states per seed, 4 replays per arm, common random numbers): $\Psi_{\mathrm{in}}$ "
         r"(mean $\pm$ standard error over seeds) and its components in salience ($\Psi_\sigma$, ministry counts), "
         r"taste ($\Psi_\nu$, which ministries), policy ($\Psi_{\mathrm{pol}}$) and exit bonuses ($\Psi_e$); "
         r"$\Psi_{\mathrm{ex}}$: value of being recognised; cov.: on-path coverage. $\tilde\Phi$: observational "
         r"floor-adjusted premium. Region (tolerances of S2): F formateur, D disguised premium, G Gamson. "
         r"Under the demand protocol only $\Psi_{\mathrm{ex}}$ is defined.}\label{tab:psi}",
         r"\begin{tabular}{@{}lccccccccc@{}}\toprule",
         r"configuration & $\Psi_{\mathrm{in}}$ & $\Psi_\sigma$ & $\Psi_\nu$ & $\Psi_{\mathrm{pol}}$ & $\Psi_e$ & "
         r"$\Psi_{\mathrm{ex}}$ & cov. & $\tilde\Phi$ & reg.\\\midrule"]
    breaks = {"mech_BASE+M10_seatrho", "mech_Estar-M10_seatrho", "mech_Estar_m12", "long_Estar",
              "entropy_BASE_0.05", "mech_A4305Proposal"}
    for t, lab in ROWS:
        r = rows[t]
        if r is None:
            continue
        reg = {"formateur": "F", "disguised": "D", "Gamson": "G"}.get(r["region"], "--")
        L.append(f"{lab} & {pm(r['psi_in'])} & {f(r['psi_in_sigma'])} & {f(r['psi_in_nu'])} & "
                 f"{f(r['psi_in_pol'])} & {f(r['psi_in_e'])} & {f(r['psi_ex'])} & "
                 f"{f(r['coverage'], sign=False)} & {f(r['observational']['phi_adj'])} & {reg}\\\\")
        if t in breaks:
            L.append(r"\addlinespace")
    L.append(r"\bottomrule\end{tabular}\end{table}")
    tex = "\n".join(L)
    if e0:
        E = [r"\begin{table}[t]\centering\footnotesize",
             r"\caption{Value of proposal power in the benchmark E0 (10 seeds, 500 probe states per cell and "
             r"seed): learned against the closed forms of Corollary~2.13, $\Psi_{\mathrm{in}}=1-\delta(n+1)/(2n)$ "
             r"and $\Psi_{\mathrm{ex}}=1-\delta/2$.}\label{tab:psie0}",
             r"\begin{tabular}{@{}rrcccccc@{}}\toprule",
             r"$n$ & $\delta$ & $\Psi_{\mathrm{in}}^{\mathrm{BF}}$ & learned $\Psi_{\mathrm{in}}$ & "
             r"$\Psi_{\mathrm{ex}}^{\mathrm{BF}}$ & learned $\Psi_{\mathrm{ex}}$ & cov.\\\midrule"]
        for k in sorted(e0, key=lambda s: tuple(map(float, s.split(",")))):
            c = e0[k]
            n, d = k.split(",")
            E.append(f"{n} & {d} & {c['bf_in']:.3f} & {pm(c['psi_in'])} & {c['bf_ex']:.3f} & {pm(c['psi_ex'])} & "
                     f"{f(c['coverage'], sign=False)}\\\\")
        E.append(r"\bottomrule\end{tabular}\end{table}")
        etex = "\n".join(E)
    else:
        etex = ""
    for name, body in (("tab_psi", tex), ("tab_psi_e0", etex)):
        body = re.sub(r"(?<![\d-])-(?=\d)", r"\\ensuremath{-}", body)
        with open(os.path.join(ROOT, "paper", f"{name}.tex"), "w") as fh:
            fh.write("% generated by scripts/psi_summary.py -- do not edit\n" + body + "\n")
    for t, v in rep.items():
        print("H10 replication", t, v["seeds"], {c: (round(v[c]["mean"], 3), [round(x, 3) for x in v[c]["ci"]])
                                                 for c in ("sigma", "nu", "pol")})
    for t, _ in ROWS:
        r = rows[t]
        if r and r["psi_in"]:
            print(f"{t:28s} in {r['psi_in']['mean']:+.3f}±{r['psi_in']['se']:.3f} "
                  + " ".join(f"{c}={r[f'psi_in_{c}']['mean']:+.3f}" for c in COMP)
                  + f" ex {r['psi_ex']['mean']:+.3f} cov {r['coverage']['mean']:.2f} phi~ "
                  + (f"{r['observational']['phi_adj']['mean']:+.3f}" if r['observational']['phi_adj'] else "--")
                  + f" {r['region']}")
        elif r:
            print(f"{t:28s} ex {r['psi_ex']['mean']:+.3f} (demand)")


if __name__ == "__main__":
    main()
