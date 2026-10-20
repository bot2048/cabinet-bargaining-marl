"""Build the report figures and tables from results/*/seed*.json."""
import glob
import json
import os
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
    "savefig.dpi": 220, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
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
        phi, phi_ci = mean_ci([r["overall"]["phi_adj"] for r in runs])
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
    ax.set_xlabel("formateur premium $\\tilde\\Phi$ (mean, 95% CI over seeds)")
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
    # numbers whose experiments may not have finished yet: placeholder "--"
    for name in ("CmSame", "CmAcross", "GainBASE", "GainEstar", "GainBASEGamson", "GainBASEMyopic",
                 "GainMappo", "GainBASEMax", "GainEstarMax", "PgAllLearned", "PgAllRiker", "PgAllRandom",
                 "PgAllIdeology", "PgAllN", "PsroBasePhi", "PsroEstarPhi", "PsroBaseGain", "PsroEstarGain",
                 "GainEstarGamson", "GainEstarMyopic", "PgIdeoLearned", "EntLow", "EntMid", "EntHigh"):
        M.setdefault(name, "--")
    os.makedirs(os.path.join(ROOT, "report"), exist_ok=True)
    with open(os.path.join(ROOT, "report", "numbers.tex"), "w") as f:
        f.write("% generated by scripts/make_figures.py -- do not edit\n")
        for k, v in sorted(M.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n".replace("%", "\\%"))
    return M


def main():
    base = load("baselines")
    base = base[0] if base else {}
    out = {"e0": fig_e0(), "e1": fig_e1(), "mech": fig_mech(base), "exploit": exploit_table(),
           "baselines": base}
    fig_curves()
    out["parlgov"] = fig_parlgov()
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
