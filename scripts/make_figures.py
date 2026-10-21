"""Build the report figures and tables from results/*/seed*.json."""
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from cabinet.bf_solver import bf_symmetric_share  # noqa: E402

# categorical slots (validated reference palette), fixed order
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
    "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
    "font.family": "DejaVu Sans"})
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)


def load(tag):
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "results", tag, "seed*.json"))):
        with open(p) as f:
            out.append(json.load(f))
    return out


def mean_ci(vals):
    v = np.array([x for x in vals if x is not None], float)
    if len(v) == 0:
        return np.nan, np.nan
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return v.mean(), 1.96 * se


# ------------------------------------------------------------------ Fig. 1
def fig_e0():
    runs = load("e0")
    if not runs:
        return None
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ns = np.array([3, 5, 7])
    colors = {0.8: BLUE, 0.9: ORANGE, 0.95: AQUA}
    nn = np.linspace(3, 7, 50)
    ax.plot(nn, 2 / (nn + 1), ls="--", color=MUTED, lw=1.2, label="Gamson (equal split)")
    rows = []
    for d, col in colors.items():
        ax.plot(nn, [bf_symmetric_share(x, d) for x in nn], color=col, lw=1.2, alpha=0.55)
        m, ci = zip(*[mean_ci([r["cells"][f"({n}, {d})"]["pf"] for r in runs]) for n in ns])
        ax.errorbar(ns + (d - 0.9) * 2, m, yerr=ci, fmt="o", ms=4, color=col, capsize=2,
                    label=f"learned, $\\delta$={d}")
        for n, mm in zip(ns, m):
            th = np.mean([r["thresholds"][f"({n}, {d})"] for r in runs])
            rows.append((n, d, bf_symmetric_share(n, d), mm, th, d / n))
    ax.set_xlabel("number of symmetric parties $n$")
    ax.set_ylabel("formateur share $p_f$")
    ax.set_xticks(ns)
    ax.set_ylim(0.2, 0.8)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.set_title("E0: learned vs Baron-Ferejohn (lines)", loc="left")
    fig.savefig(os.path.join(FIG, "fig_e0_benchmark.png"))
    fig.savefig(os.path.join(FIG, "fig_e0_benchmark.pdf"))
    plt.close(fig)
    return rows


# ------------------------------------------------------------------ Fig. 2
def fig_e1():
    runs = load("e1")
    if not runs:
        return None
    pairs = np.array([p[:2] for r in runs for p in r["pairs"]])
    fig, ax = plt.subplots(figsize=(1.9, 1.9))
    ax.scatter(pairs[:, 1], pairs[:, 0], s=5, color=BLUE, alpha=0.25, linewidths=0)
    lo, hi = 0.4, 1.0
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("certified BF share $x_f$")
    ax.set_ylabel("learned $p_f$")
    ax.set_title("E1: weighted seats", loc="left")
    fig.savefig(os.path.join(FIG, "fig_e1_scatter.png"))
    fig.savefig(os.path.join(FIG, "fig_e1_scatter.pdf"))
    plt.close(fig)
    corr = np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]
    return dict(corr=corr, mae=np.mean(np.abs(pairs[:, 0] - pairs[:, 1])), n=len(pairs))


# ------------------------------------------------------------------ Fig. 3
MECH_ORDER = [("BASE", "base game (all off)"),
              ("BASE+M1_repeat", "+ repeated terms"),
              ("BASE+M1_patient", "+ repeated, patient"),
              ("BASE+M4_private", "+ private types"),
              ("BASE+M5_hetero", "+ heterog. values"),
              ("BASE+M8_exits", "+ exits"),
              ("BASE+M9_policy", "+ ministerial policy"),
              ("BASE+M10_seatrho", "+ seat recognition"),
              ("Estar", "E* (all on)"),
              ("Estar-M1_repeat", "E* - repeated terms"),
              ("Estar-M4_private", "E* - private types"),
              ("Estar-M5_hetero", "E* - heterog. values"),
              ("Estar-M8_exits", "E* - exits"),
              ("Estar-M9_policy", "E* - ministerial policy"),
              ("Estar-M10_seatrho", "E* - seat recognition"),
              ]


def fig_mech(baselines):
    found = [(c, lab, load(f"mech_{c}")) for c, lab in MECH_ORDER]
    found = [x for x in found if x[2]]
    if not found:
        return None
    fig, ax = plt.subplots(figsize=(3.4, 0.17 * len(found) + 0.75))
    y = np.arange(len(found))[::-1]
    table = []
    for k, (c, lab, runs) in enumerate(found):
        phi, phi_ci = mean_ci([r["overall"]["reg_gamma"] for r in runs])
        dg, dg_ci = mean_ci([r["overall"]["dg_adj"] for r in runs])
        col = MUTED if c in ("BASE", "Estar") else (BLUE if c.startswith("BASE+") else ORANGE)
        ax.barh(y[k], phi, height=0.62, color=col, xerr=phi_ci,
                error_kw=dict(ecolor=INK, lw=0.7, capsize=1.2))
        o = {key: np.mean([r["overall"].get(key) or 0 for r in runs]) for key in
             ("reg_beta", "reg_gamma", "fail_rate", "collapse_rate", "delay")}
        table.append((c, lab, phi, phi_ci, dg, dg_ci, len(runs), o))
    sep = [k for k, (c, _, _) in enumerate(found) if c == "Estar"]
    if sep:
        ax.axhline(y[sep[0]] + 0.5, color=GRID, lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab, _ in found])
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("formateur coefficient $\\hat\\gamma$ in counted shares (mean, 95% CI)")
    ax.set_title("Add-one-in (blue) and leave-one-out (orange)", loc="left")
    fig.savefig(os.path.join(FIG, "fig_mechanisms.png"))
    fig.savefig(os.path.join(FIG, "fig_mechanisms.pdf"))
    plt.close(fig)
    return table


# ------------------------------------------------------------------ Fig. 4
def fig_curves():
    series = [("e0", "E0 (benchmark)", "gap_bf", BLUE),
              ("mech_BASE", "BASE", "phi_adj", ORANGE),
              ("mech_Estar", "E*", "phi_adj", AQUA)]
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    any_ = False
    for tag, lab, key, col in series:
        runs = load(tag)
        if not runs:
            continue
        any_ = True
        L = min(len(r["history"]) for r in runs)
        st = np.array([[h["steps"] for h in r["history"][:L]] for r in runs]).mean(0) / 1e6
        val = np.array([[h.get(key) if h.get(key) is not None else np.nan
                         for h in r["history"][:L]] for r in runs])
        m = np.nanmean(val, 0)
        ax.plot(st, m, color=col, label=f"{lab}: " + ("$\\Delta_{BF}$" if key == "gap_bf" else "$\\tilde\\Phi$"))
        if len(runs) > 1:
            ax.fill_between(st, np.nanmin(val, 0), np.nanmax(val, 0), color=col, alpha=0.15, lw=0)
    if not any_:
        plt.close(fig)
        return
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("environment decisions (millions)")
    ax.set_ylabel("statistic")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Training curves (mean, band = min-max over seeds)", loc="left")
    fig.savefig(os.path.join(FIG, "fig_curves.png"))
    fig.savefig(os.path.join(FIG, "fig_curves.pdf"))
    plt.close(fig)


def exploit_table():
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "exploit_*"))):
        runs = load(os.path.basename(p))
        cfg = os.path.basename(p)[8:]
        g, ci = mean_ci([r["gain"] for r in runs])
        ctrl = {}
        for r in runs:
            for k, v in r.get("controls", {}).items():
                ctrl[k] = v["gain"]
        rows.append((cfg, g, ci, len(runs), ctrl))
    return rows


# ------------------------------------------------------------------ Fig. 5
def fig_parlgov():
    sets = [("parlgov_mech_EstarWide", "E*-wide"), ("parlgov_mech_EstarWidePub_ideology", "E*-wide, public\nleft-right")]
    rows = []
    for tag, lab in sets:
        runs = load(tag)
        if runs:
            m, ci = mean_ci([r["prediction"]["learned"]["exact"] for r in runs])
            rows.append((f"learned: {lab}", m, ci, BLUE))
    ref = load("parlgov_mech_EstarWide")
    if not ref:
        return None
    for key, lab in (("min_seat", "minimal-seat rule"), ("random_mwc", "random min. winning"),
                     ("ideology_mwc", "ideologically closest")):
        rows.append((lab, ref[0]["prediction"][key]["exact"], 0.0, MUTED))
    fig, ax = plt.subplots(figsize=(3.4, 1.45))
    y = np.arange(len(rows))[::-1]
    for k, (lab, m, ci, col) in enumerate(rows):
        ax.barh(y[k], m, height=0.62, color=col, xerr=ci if ci else None,
                error_kw=dict(ecolor=INK, lw=0.8, capsize=1.5))
        ax.text(m + 0.006, y[k], f"{m:.0%}", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("exact match with the coalition that governed")
    ax.set_xlim(0, max(r[1] for r in rows) * 1.25)
    ax.set_title("242 real coalitions (ParlGov, W. Europe 1945-)", loc="left")
    fig.savefig(os.path.join(FIG, "fig_parlgov.png"))
    fig.savefig(os.path.join(FIG, "fig_parlgov.pdf"))
    plt.close(fig)
    return rows


def fig_real():
    """Real cabinets: error of each model's predicted split (country-clustered CI)."""
    p = os.path.join(ROOT, "results", "seki", "summary.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        res = json.load(f)
    M = res["models"]
    groups = [("Gamson's law", ["gamson"], MUTED),
              ("agents, private types", [k for k in M if k.startswith("mech_EstarWide/")], BLUE),
              ("agents, public types", [k for k in M if k.startswith("mech_EstarWidePub/")], ORANGE),
              ("Baron-Ferejohn", ["bf"], MUTED)]
    fig, ax = plt.subplots(figsize=(3.4, 1.35))
    y = np.arange(len(groups))[::-1]
    for k, (lab, keys, col) in enumerate(groups):
        vals = [M[x]["mae"] for x in keys if x in M]
        if not vals:
            continue
        m = float(np.mean(vals))
        lo = min(M[x]["mae_ci"][0] for x in keys)
        hi = max(M[x]["mae_ci"][1] for x in keys)
        ax.barh(y[k], m, height=0.6, color=col, xerr=[[m - lo], [hi - m]],
                error_kw=dict(ecolor=INK, lw=0.8, capsize=1.5))
        ax.text(hi + 0.004, y[k], f"{m:.3f}", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([g[0] for g in groups])
    ax.set_xlabel("error of predicted portfolio shares (MAE)")
    ax.set_title(f"{res['n_common']} real cabinets (Seki-Williams, 31 countries)", loc="left")
    fig.savefig(os.path.join(FIG, "fig_real.png"))
    fig.savefig(os.path.join(FIG, "fig_real.pdf"))
    plt.close(fig)
    return res


def own_runs(cfg, explore=False):
    """Agents' own cabinets in the real parliaments, merged over the per-seed-group
    files of the final evaluation (own_<cfg>_a.json, _b, ...); falls back to the
    first run (own_<cfg>.json) when the final files are not there yet."""
    files = sorted(glob.glob(os.path.join(ROOT, "results", "seki", f"own_{cfg}_*.json")))
    files = [f for f in files if f.endswith("_explore.json") == explore]
    if not files and not explore:
        files = [os.path.join(ROOT, "results", "seki", f"own_{cfg}.json")]
    runs = {}
    for fp in files:
        if os.path.exists(fp):
            with open(fp) as f:
                runs.update(json.load(f)["own"])
    return list(runs.values())


def macros():
    """report/numbers.tex: every number quoted in the report, recomputed here."""
    M = {}

    def put(name, value, fmt="{:+.3f}"):
        M[name] = fmt.format(value) if value is not None and not (isinstance(value, float) and np.isnan(value)) else "--"

    def avg(tag, key, sub="overall"):
        runs = load(tag)
        v = [r[sub].get(key) for r in runs if r.get(sub) and r[sub].get(key) is not None]
        return (float(np.mean(v)), len(v)) if v else (np.nan, 0)

    e0 = load("e0")
    if e0:
        gaps, ok = [], 0
        for key in e0[0]["cells"]:
            n, d = eval(key)
            g = np.mean([r["cells"][key]["pf"] for r in e0]) - bf_symmetric_share(n, d)
            gaps.append(g)
            ok += abs(g) <= 0.02
        put("EzeroAbsGap", float(np.mean(np.abs(gaps))), "{:.3f}")
        put("EzeroGap", float(np.mean(gaps)))
        put("EzeroCellsOK", ok, "{:d}")
        put("EzeroSeeds", len(e0), "{:d}")
        put("EzeroFail", avg("e0", "fail_rate")[0], "{:.1%}")
    put("EoneGap", avg("e1", "gap_bf")[0])
    put("EoneAbsGap", avg("e1", "abs_gap_bf")[0], "{:.3f}")
    for tag, name in (("mech_BASE", "Base"), ("mech_Estar", "Estar"), ("mappo_Estar", "Mappo"),
                      ("fsp_Estar", "Fsp"), ("rec_Estar", "Rec"), ("mech_Estar_Lam0", "Lamzero"),
                      ("mech_BASE+M1_repeat", "Mone"), ("mech_BASE+M4_private", "Mfour"),
                      ("mech_BASE+M5_hetero", "Mfive"), ("mech_BASE+M8_exits", "Meight"),
                      ("mech_BASE+M9_policy", "Mnine"), ("mech_BASE+M10_seatrho", "Mten"),
                      ("mech_BASE+M1_patient", "Mpatient"), ("rec_BASE+M1_patient", "Mpatientrec"),
                      ("mech_Estar-M1_repeat", "LooMone"), ("mech_Estar-M4_private", "LooMfour"),
                      ("mech_Estar-M5_hetero", "LooMfive"), ("mech_Estar-M8_exits", "LooMeight"),
                      ("mech_Estar-M9_policy", "LooMnine"), ("mech_Estar-M10_seatrho", "LooMten"),
                      ("mech_EstarWidePub", "Pub"), ("mech_Estar_m12", "Mtwelve")):
        for key, short in (("phi_adj", "Phi"), ("dg_adj", "Dg"), ("reg_beta", "Beta"),
                           ("reg_gamma", "Gamma"), ("phi_nu", "PhiNu"), ("reg_r2", "Rtwo"),
                           ("regbz_r2", "RtwoBz"), ("fail_rate", "Fail"), ("collapse_rate", "Collapse")):
            v, k = avg(tag, key)
            put(f"{name}{short}", v, "{:.2f}" if short in ("Rtwo", "RtwoBz") else "{:+.3f}")
        put(f"{name}Seeds", avg(tag, "phi_adj")[1], "{:d}")
    pg = load("parlgov_mech_EstarWide")
    if pg:
        for key, name in (("learned", "PgLearned"), ("min_seat", "PgRiker"),
                          ("random_mwc", "PgRandom"), ("ideology_mwc", "PgIdeology")):
            put(name, float(np.mean([r["prediction"][key]["exact"] for r in pg])), "{:.1%}")
        for key, short in (("phi_adj", "Phi"), ("reg_beta", "Beta"), ("reg_gamma", "Gamma"),
                           ("reg_r2", "Rtwo"), ("regbz_r2", "RtwoBz"), ("phi_nu", "PhiNu")):
            put(f"Pg{short}", float(np.mean([r["allocation"][key] for r in pg])),
                "{:.2f}" if short in ("Rtwo", "RtwoBz") else "{:+.3f}")
    pga = load("parlgov_mech_EstarWide_all")
    if pga:
        for key, name in (("learned", "PgAllLearned"), ("min_seat", "PgAllRiker"),
                          ("random_mwc", "PgAllRandom"), ("ideology_mwc", "PgAllIdeology")):
            put(name, float(np.mean([r["prediction"][key]["exact"] for r in pga])), "{:.1%}")
        put("PgAllN", pga[0]["prediction"]["learned"]["n"], "{:d}")
    pgi = load("parlgov_mech_EstarWidePub_ideology")
    if pgi:
        put("PgIdeoLearned", float(np.mean([r["prediction"]["learned"]["exact"] for r in pgi])), "{:.1%}")
    for c in ("BASE", "Estar"):
        ex = load(f"exploit_{c}")
        if ex:
            put(f"Gain{c}", float(np.mean([r["gain"] for r in ex])), "{:+.3f}")
            put(f"Gain{c}Max", float(np.max([r["gain"] for r in ex])), "{:+.3f}")
            ctrl = [r["controls"] for r in ex if r.get("controls")]
            if ctrl:
                for b in ctrl[0]:
                    put(f"Gain{c}{b.capitalize()}", ctrl[0][b]["gain"], "{:+.3f}")
    for c, name in (("BASE", "Base"), ("Estar", "Estar")):
        ps = load(f"psro_{c}")
        if ps:
            put(f"Psro{name}Phi", float(np.mean([r["overall"]["phi_adj"] for r in ps])))
            put(f"Psro{name}Gain", float(np.mean([r["last_gain"] for r in ps])), "{:+.3f}")
    exl = load("exploit_Estar_br400")
    if exl:
        put("GainLong", float(np.mean([r["gain"] for r in exl])), "{:+.3f}")
        c0 = [r["controls"] for r in exl if r.get("controls")]
        if c0:
            put("GainLongGamson", c0[0]["gamson"]["gain"], "{:+.3f}")
            put("GainLongMyopic", c0[0]["myopic"]["gain"], "{:+.3f}")
    exd = load("exploit_mech_E0demand_br300")
    if exd:
        put("GainDemand", float(np.mean([r["gain"] for r in exd])), "{:+.2f}")
    exm = load("exploit_mappo_Estar")
    if exm:
        put("GainMappo", float(np.mean([r["gain"] for r in exm])), "{:+.3f}")
    for e in ("0.002", "0.02", "0.05"):
        put("Ent" + {"0.002": "Low", "0.02": "Mid", "0.05": "High"}[e],
            avg(f"entropy_BASE_{e}", "phi_adj")[0])
    cp = os.path.join(ROOT, "results", "crossplay", "Estar.json")
    if os.path.exists(cp):
        with open(cp) as f:
            c = json.load(f)
        put("CmSame", c.get("mismatch_same_algorithm"), "{:+.4f}")
        put("CmAcross", c.get("mismatch_across_algorithms"), "{:+.4f}")
    # real cabinets (Seki-Williams)
    sp = os.path.join(ROOT, "results", "seki", "summary.json")
    if os.path.exists(sp):
        with open(sp) as f:
            sk = json.load(f)
        e = sk["empirical"]
        put("RealA", e["alpha"], "{:.3f}")
        put("RealB", e["beta"], "{:.3f}")
        put("RealG", e["gamma"], "{:.3f}")
        put("RealGlo", e["gamma_ci"][0], "{:.3f}")
        put("RealGhi", e["gamma_ci"][1], "{:.3f}")
        put("RealRtwo", e["r2"], "{:.2f}")
        put("RealRtwoBz", sk["empirical_banzhaf"]["r2"], "{:.2f}")
        put("RealN", sk["n_cabinets"], "{:d}")
        put("RealNcommon", sk["n_common"], "{:d}")
        Mm = sk["models"]
        put("MaeGamson", Mm["gamson"]["mae"], "{:.3f}")
        put("MaeBF", Mm["bf"]["mae"], "{:.3f}")
        priv = [v for k, v in Mm.items() if k.startswith("mech_EstarWide/")]
        pub = [v for k, v in Mm.items() if k.startswith("mech_EstarWidePub/")]
        if priv:
            put("MaePrivLo", min(v["mae"] for v in priv), "{:.3f}")
            put("MaePrivHi", max(v["mae"] for v in priv), "{:.3f}")
            put("GcPrivLo", min(v["gamma"] for v in priv), "{:.2f}")
            put("GcPrivHi", max(v["gamma"] for v in priv), "{:.2f}")
            put("DmaePrivLo", min(v["dmae_ci"][0] for v in priv), "{:+.3f}")
        if pub:
            put("MaePubLo", min(v["mae"] for v in pub), "{:.3f}")
            put("MaePubHi", max(v["mae"] for v in pub), "{:.3f}")
            put("GcPubLo", min(v["gamma"] for v in pub), "{:.2f}")
            put("GcPubHi", max(v["gamma"] for v in pub), "{:.2f}")
        put("GcBF", Mm["bf"]["gamma"], "{:.2f}")
    # agents' own cabinets in the 195 real parliaments (ranges over seeds)
    for cfg, name in (("WideDemand", "OwnDem"), ("WideProposal", "OwnProp"), ("EstarWide", "OwnWide")):
        own = own_runs(cfg)
        if not own:
            continue
        for key, short, fmt in (("reg_alpha", "A", "{:.2f}"), ("reg_beta", "B", "{:.2f}"),
                                ("reg_gamma", "G", "{:+.2f}"), ("reg_r2", "Rtwo", "{:.2f}"),
                                ("phi_nu", "PhiNu", "{:+.2f}")):
            v = [o[key] for o in own if key in o]
            if v:
                lo, hi = fmt.format(min(v)), fmt.format(max(v))
                M[f"{name}{short}"] = f"{lo} to {hi}" if min(v) < 0 else lo + "--" + hi.lstrip("+")
    kp = os.path.join(ROOT, "results", "seki", "key_ministries.json")
    if os.path.exists(kp):
        with open(kp) as f:
            km = json.load(f)
        put("KeyN", km["n_cabinets"], "{:d}")
        put("KeyShare", km["mean_key_share"], "{:.3f}")
        put("KeyCount", km["mean_count_share"], "{:.3f}")
        put("KeySeat", km["mean_seat_share"], "{:.3f}")
        put("KeyDiff", km["key_minus_count"]["mean"], "{:+.3f}")
        put("KeyDiffLo", km["key_minus_count"]["ci"][0], "{:+.3f}")
        put("KeyDiffHi", km["key_minus_count"]["ci"][1], "{:+.3f}")
    # demand bargaining (M2) and its matched proposal control
    for tag, name in (("mech_WideDemand", "Dem"), ("mech_WideProposal", "Prop")):
        for key, short in (("reg_alpha", "A"), ("reg_beta", "B"), ("reg_gamma", "G"), ("reg_r2", "Rtwo"),
                           ("dem_gamma_first", "First"), ("dem_gamma_close", "Close")):
            v, k = avg(tag, key)
            put(f"{name}{short}", v, "{:.2f}" if short in ("B", "Rtwo") else "{:+.3f}")
        put(f"{name}Seeds", avg(tag, "reg_gamma")[1], "{:d}")
    rp = os.path.join(ROOT, "results", "seki", "e0demand_roles.json")
    if os.path.exists(rp):
        with open(rp) as f:
            roles = json.load(f)
        put("LockinCloserThree", roles["mech_E0demand"]["closer3"], "{:.2f}")
        put("LockinCloserFive", roles["mech_E0demand"]["closer5"], "{:.2f}")
        put("ExploreCloserThree", roles["explore_E0demand"]["closer3"], "{:.2f}")
        put("ExploreCloserFive", roles["explore_E0demand"]["closer5"], "{:.2f}")
    for key, short in (("dem_gamma_close", "Close"), ("dem_gamma_first", "First"), ("reg_beta", "B")):
        v, k = avg("explore_E0demand", key)
        put(f"Explore{short}", v, "{:.2f}" if short == "B" else "{:+.3f}")
        v0, _ = avg("mech_E0demand", key)
        put(f"Lockin{short}", v0, "{:.2f}" if short == "B" else "{:+.3f}")
    # long training (4x budget)
    for tag, name in (("long_BASE", "LongBase"), ("long_BASE+M4_private", "LongMfour"),
                      ("long_BASE+M9_policy", "LongMnine"), ("long_Estar", "LongEstar")):
        for key, short in (("reg_gamma", "G"), ("phi_nu", "PhiNu"), ("regnu_gamma", "Gnu"),
                           ("reglead_gamma", "Glead")):
            put(f"{name}{short}", avg(tag, key)[0])
    for tag, name in (("mech_BASE", "Base"), ("mech_BASE+M4_private", "Mfour"), ("mech_BASE+M5_hetero", "Mfive"),
                      ("mech_BASE+M9_policy", "Mnine"), ("mech_Estar", "Estar"), ("mech_EstarWide", "Wide"),
                      ("mech_EstarWidePub", "Pub"), ("mappo_Estar", "Mappo"), ("fsp_Estar", "Fsp"),
                      ("rec_Estar", "Rec"), ("mech_Estar_Lam0", "Lamzero"), ("mech_Estar_m12", "Mtwelve")):
        put(f"{name}Gnu", avg(tag, "regnu_gamma")[0])
        put(f"{name}Glead", avg(tag, "reglead_gamma")[0])
    # ---- added after the outside review (see docs/AUDIT.md section 6)
    cp = os.path.join(ROOT, "results", "confirmatory.json")
    if os.path.exists(cp):
        with open(cp) as f:
            cf = json.load(f)
        put("HoneDiff", cf["H1"]["diff"], "{:.2f}")
        put("HoneLo", cf["H1"]["ci"][0], "{:.2f}")
        put("HoneHi", cf["H1"]["ci"][1], "{:.2f}")
        put("HtwoDiff", cf["H2"]["diff"], "{:.2f}")
        put("HtwoLo", cf["H2"]["ci"][0], "{:.2f}")
        put("HtwoHi", cf["H2"]["ci"][1], "{:.2f}")
        e = int(np.floor(np.log10(max(cf["H1"]["p"], 1e-12))))
        M["HoneP"] = f"{cf['H1']['p'] / 10 ** e:.0f}\\times10^{{{e}}}"
        put("HthreeMean", cf["H3"]["mean"], "{:+.3f}")
        put("HthreeLo", cf["H3"]["ci90"][0], "{:+.3f}")
        put("HthreeHi", cf["H3"]["ci90"][1], "{:+.3f}")
        put("NprotSeeds", len(cf["H1"]["proposal"]), "{:d}")
        for tag, name in (("explore_A452Demand", "AfDem"), ("mech_A452Proposal", "AfProp"),
                          ("explore_A4305Demand", "AtDem"), ("mech_A4305Proposal", "AtProp")):
            runs = cf.get(tag, [])
            if not runs:
                continue
            sl = [r["slope"] for r in runs]
            M[f"{name}Slope"] = "{:+.2f}".format(min(sl)) + " to " + "{:+.2f}".format(max(sl))
            put(f"{name}Seeds", len(runs), "{:d}")
            fs = [v["formateur_share"] for r in runs for v in r["coalitions"].values() if v["freq"] > 0.05]
            put(f"{name}FormLo", min(fs), "{:.2f}")
            put(f"{name}FormHi", max(fs), "{:.2f}")
            # largest deviation from an even split in any coalition that forms
            dev = [max(abs(x - 0.5) for x in v["shares"].values()) for r in runs
                   for v in r["coalitions"].values() if v["freq"] > 0.05 and len(v["shares"]) == 2]
            put(f"{name}MaxDev", max(dev), "{:.2f}")
            d = [r["dummy_included"] for r in runs if r["dummy_included"] is not None]
            if d:
                put(f"{name}Dummy", max(d), "{:.1%}")
        # 40/30/25/5 under proposals: large-party formateurs (two-party cabinets) vs the
        # 5-seat dummy as formateur (it must buy two large partners)
        big = [v["formateur_share"] for r in cf.get("mech_A4305Proposal", [])
               for v in r["coalitions"].values() if v["freq"] > 0.05 and len(v["shares"]) == 2]
        dum = [v["formateur_share"] for r in cf.get("mech_A4305Proposal", [])
               for k, v in r["coalitions"].items() if v["freq"] > 0.01 and "5" in k.split("+")]
        if big:
            put("AtPropBigLo", min(big), "{:.2f}")
            put("AtPropBigHi", max(big), "{:.2f}")
        if dum:
            put("AtPropDumLo", min(dum), "{:.2f}")
            put("AtPropDumHi", max(dum), "{:.2f}")
        from cabinet.bf_solver import solve_bf
        wv = np.array([40, 30, 25, 5])
        v = solve_bf(wv, 51, wv / wv.sum(), 0.9)["v"]
        put("BfDummyForm", 1 - 0.9 * (v[1] + v[2]), "{:.2f}")
        put("BfBigForm", 1 - 0.9 * min(v[1], v[2]), "{:.2f}")
        if "P1_ci" in cf:
            put("PoneLo", cf["P1_ci"][0], "{:+.2f}")
            put("PoneHi", cf["P1_ci"][1], "{:+.2f}")
    dc = os.path.join(ROOT, "results", "demand_solver_check.json")
    if os.path.exists(dc):
        with open(dc) as f:
            chk = json.load(f)
        bf_rows = chk["brute_force"]
        put("SolverCompared", len(bf_rows), "{:d}")
        put("SolverMaxDiff", max(max(r["max_abs_diff_V"], r["max_abs_diff_onpath"]) for r in bf_rows), "{:.0e}")
        put("SolverEnvN", chk["env_consistency"]["random_histories"], "{:d}")
        put("SolverEnvBad", chk["env_consistency"]["mismatches"], "{:d}")
    bp = os.path.join(ROOT, "results", "bf_best_response.json")
    if os.path.exists(bp):
        with open(bp) as f:
            br = json.load(f)["summary"]
        for n, name in ((3, "Three"), (5, "Five"), (7, "Seven")):
            g = [v["gain"] for k, v in br.items() if k.startswith(f"{n},") and k.endswith("stochastic")]
            put(f"BrGain{name}", max(g), "{:.3f}")
        put("BrSevenXbest", br["7,0.8,stochastic"]["x_best"], "{:.3f}")
        put("BrSevenXlearned", br["7,0.8,stochastic"]["x_learned"], "{:.3f}")
        # ex-ante per-period gain of one seat (formateur with probability 1/n), mean over cells
        put("BrSeatGain", float(np.mean([v["gain"] / int(k.split(",")[0]) for k, v in br.items()
                                         if k.endswith("stochastic")])), "{:.3f}")
    rr = os.path.join(ROOT, "results", "real_robustness.json")
    if os.path.exists(rr):
        with open(rr) as f:
            rob = json.load(f)
        for key, name in (("baseline", "Base"), ("alpha_over_k", "Overk"), ("largest_party_control", "Largest"),
                          ("min_integer_weights", "Weights"), ("weights_and_seats", "Both")):
            put(f"Rob{name}G", rob[key]["gamma"], "{:+.3f}")
            put(f"Rob{name}Lo", rob[key]["wild_ci"][0], "{:+.3f}")
            put(f"Rob{name}Hi", rob[key]["wild_ci"][1], "{:+.3f}")
            put(f"Rob{name}Rtwo", rob[key]["r2"], "{:.2f}")
        put("RobPmNotLargest", rob["cabinets_pm_not_largest"], "{:d}")
    for cfg, name in (("WideDemand", "OwnDem"), ("WideProposal", "OwnProp"), ("EstarWide", "OwnWide")):
        own = own_runs(cfg)
        for key, short, fmt in (("regbz_gamma", "Gbz", "{:+.2f}"), ("regbz_beta", "Bbz", "{:.2f}"),
                                ("match_rate", "Match", "{:.0%}"), ("matched_reg_beta", "MatchB", "{:.2f}"),
                                ("matched_regbz_beta", "MatchBbz", "{:.2f}"),
                                ("matched_reg_gamma", "MatchG", "{:+.2f}"), ("regnu_gamma", "Gnu", "{:+.2f}")):
            v = [o[key] for o in own if key in o]
            if v:
                lo, hi = fmt.format(min(v)), fmt.format(max(v))
                M[f"{name}{short}"] = f"{lo} to {hi}" if min(v) < 0 else lo + "--" + hi.lstrip("+")
        put(f"{name}Seeds", len(own), "{:d}")
    sp = os.path.join(ROOT, "results", "seki", "summary.json")
    if os.path.exists(sp):
        with open(sp) as f:
            sk = json.load(f)
        put("RealGbz", sk["empirical_banzhaf"]["gamma"], "{:+.3f}")
        if "loco" in sk:
            put("LocoLo", sk["loco"]["empirical_gamma_range"][0], "{:+.3f}")
            put("LocoHi", sk["loco"]["empirical_gamma_range"][1], "{:+.3f}")
    # demand bargaining with more exploration of demands (same random parliaments)
    ex = load("explore_WideDemand")
    if ex:
        for key, short, fmt in (("reg_gamma", "G", "{:+.3f}"), ("reg_beta", "B", "{:.2f}")):
            v = [r["overall"][key] for r in ex]
            lo, hi = fmt.format(min(v)), fmt.format(max(v))
            M[f"ExpDem{short}"] = f"{lo} to {hi}" if min(v) < 0 else lo + "--" + hi.lstrip("+")
        put("ExpDemSeeds", len(ex), "{:d}")
    own_ex = own_runs("WideDemand", explore=True)
    for key, short, fmt in (("reg_beta", "B", "{:.2f}"), ("reg_gamma", "G", "{:+.2f}"),
                            ("regbz_gamma", "Gbz", "{:+.2f}"), ("reg_r2", "Rtwo", "{:.2f}")):
        v = [o[key] for o in own_ex if key in o]
        if v:
            lo, hi = fmt.format(min(v)), fmt.format(max(v))
            lo, hi = (x.replace("-0.00", "0.00").replace("+0.00", "0.00") if x[-1] == "0" and x.endswith("0.00")
                      else x for x in (lo, hi))
            M[f"OwnExp{short}"] = f"{lo} to {hi}" if min(v) < 0 else lo + "--" + hi.lstrip("+")
    e0x = load("exploit_e0_br300")
    if e0x:
        put("GainEzero", e0x[0]["gain"], "{:.3f}")
        best = max(e0x[0]["runs"].values(), key=lambda v: v["gain"])
        put("GainEzeroRel", best["gain"] / best["u_base"], "{:.0%}")
        c = e0x[0].get("controls", {})
        put("GainEzeroGamson", c.get("gamson", {}).get("gain", np.nan), "{:.3f}")
        put("GainEzeroMyopic", c.get("myopic", {}).get("gain", np.nan), "{:.3f}")
    wp = os.path.join(ROOT, "results", "real_weights_vs_seats.json")
    if os.path.exists(wp):
        with open(wp) as f:
            wv = json.load(f)
        put("WvsPairs", wv["pairs"], "{:d}")
        put("WvsSlope", wv["seat_slope"], "{:.2f}")
        put("WvsLo", wv["seat_slope_ci"][0], "{:.2f}")
        put("WvsHi", wv["seat_slope_ci"][1], "{:.2f}")
        put("WvsNoPm", wv["seat_slope_pairs_without_pm"], "{:.2f}")
        put("WvsNoPmN", wv["pairs_without_pm"], "{:d}")
        if "seat_slope_pairs_without_pm_ci" in wv:
            put("WvsNoPmLo", wv["seat_slope_pairs_without_pm_ci"][0], "{:.2f}")
            put("WvsNoPmHi", wv["seat_slope_pairs_without_pm_ci"][1], "{:.2f}")
            put("WvsNoPmCountries", wv["countries_without_pm"], "{:d}")
    # best-response gains relative to the payoff of a profile seat
    for tag, name in (("exploit_Estar_br400", "GainLongRel"), ("exploit_mech_WideDemand_br400", "GainDemRel"),
                      ("exploit_mech_WideProposal_br400", "GainPropRel")):
        rel = []
        for r in load(tag):
            best = max(r["runs"].values(), key=lambda v: v["gain"])
            rel.append(best["gain"] / best["u_base"])
        if rel:
            put(name, float(np.mean(rel)), "{:.0%}")
    # numbers whose experiments may not have finished yet: placeholder "--"
    for name in ("OwnDemA", "OwnDemB", "OwnDemG", "OwnDemRtwo", "OwnPropA", "OwnPropB", "OwnPropG",
                 "OwnPropRtwo", "OwnWideA", "OwnWideB", "OwnWideG", "OwnWideRtwo", "OwnWidePhiNu",
                 "CmSame", "CmAcross", "GainBASE", "GainEstar", "GainBASEGamson", "GainBASEMyopic",
                 "GainMappo", "GainDemand", "GainLong", "GainLongGamson", "GainLongMyopic", "GainBASEMax", "GainEstarMax", "PgAllLearned", "PgAllRiker", "PgAllRandom",
                 "PgAllIdeology", "PgAllN", "PsroBasePhi", "PsroEstarPhi", "PsroBaseGain", "PsroEstarGain",
                 "GainEstarGamson", "GainEstarMyopic", "PgIdeoLearned", "EntLow", "EntMid", "EntHigh"):
        M.setdefault(name, "--")
    os.makedirs(os.path.join(ROOT, "report"), exist_ok=True)
    with open(os.path.join(ROOT, "report", "numbers.tex"), "w") as f:
        f.write("% generated by scripts/make_figures.py -- do not edit\n")
        for k, v in sorted(M.items()):
            v = re.sub(r"(?<![\d-])-(?=\d)", r"\\ensuremath{-}", str(v))   # typographic minus
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n".replace("%", "\\%"))
    return M


def main():
    base = load("baselines")
    base = base[0] if base else {}
    out = {"e0": fig_e0(), "e1": fig_e1(), "mech": fig_mech(base), "exploit": exploit_table(),
           "baselines": base}
    fig_curves()
    out["parlgov"] = fig_parlgov()
    out["real"] = fig_real()
    out["macros"] = macros()
    with open(os.path.join(ROOT, "results", "summary.json"), "w") as f:
        json.dump(out, f, indent=1, default=lambda x: float(x) if hasattr(x, "item") else str(x))
    # human-readable summary
    if out["e0"]:
        print("E0  n delta  BF     learned  thr   delta/n")
        for n, d, bf, m, th, tv in out["e0"]:
            print(f"    {n} {d:<5} {bf:.3f}  {m:.3f}   {th:.3f} {tv:.3f}")
    if out["e1"]:
        print("E1 ", {k: round(float(v), 3) for k, v in out["e1"].items()})
    if out["mech"]:
        for c, lab, phi, pci, dg, dci, ns, o in out["mech"]:
            print(f"{lab:24s} phi~={phi:+.3f}+-{pci:.3f} dg~={dg:.3f}+-{dci:.3f} seeds={ns} "
                  + " ".join(f"{k}={v:.3f}" for k, v in o.items()))
    for row in out["exploit"]:
        print("exploit", row)


if __name__ == "__main__":
    main()
