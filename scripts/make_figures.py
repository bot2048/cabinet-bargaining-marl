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
    ax.legend(frameon=False, ncol=1, loc="lower left")
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
MECH_ORDER = [("BASE", "base (all off)"),
              ("BASE+M1_repeat", "+M1 repeated terms"),
              ("BASE+M4_private", "+M4 private types"),
              ("BASE+M5_hetero", "+M5 heterog. values"),
              ("BASE+M8_exits", "+M8 exits"),
              ("BASE+M9_policy", "+M9 policy"),
              ("BASE+M10_seatrho", "+M10 seat recognition"),
              ("Estar", "E* (all on)"),
              ("Estar_Lam0", "E*, memoryless"),
              ]


def fig_mech(baselines):
    found = [(c, lab, load(f"mech_{c}")) for c, lab in MECH_ORDER]
    found = [x for x in found if x[2]]
    if not found:
        return None
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    y = np.arange(len(found))[::-1]
    table = []
    for k, (c, lab, runs) in enumerate(found):
        phi, phi_ci = mean_ci([r["overall"]["phi_adj"] for r in runs])
        dg, dg_ci = mean_ci([r["overall"]["dg_adj"] for r in runs])
        ax.barh(y[k] + 0.19, phi, height=0.36, color=BLUE, xerr=phi_ci,
                error_kw=dict(ecolor=INK, lw=0.8, capsize=1.5), label="premium $\\tilde\\Phi$" if k == 0 else None)
        ax.barh(y[k] - 0.19, dg, height=0.36, color=ORANGE, xerr=dg_ci,
                error_kw=dict(ecolor=INK, lw=0.8, capsize=1.5), label="Gamson dist. $\\tilde D_G$" if k == 0 else None)
        o = {key: np.mean([r["overall"].get(key) or 0 for r in runs]) for key in
             ("reg_beta", "reg_gamma", "fail_rate", "collapse_rate", "delay")}
        table.append((c, lab, phi, phi_ci, dg, dg_ci, len(runs), o))
    if baselines and "BASE/myopic" in baselines:
        ax.axvline(baselines["BASE/myopic"]["phi_adj"], color=MUTED, ls=":", lw=1)
        ax.text(baselines["BASE/myopic"]["phi_adj"], y[0] + 0.62, "myopic BF\nbaseline",
                fontsize=6, color=MUTED, ha="center", va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab, _ in found])
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("floor-adjusted statistic (mean, 95% CI over seeds)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Mechanisms: formateur premium vs proportionality", loc="left")
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
    ax.legend(frameon=False, loc="upper right")
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


def main():
    base = load("baselines")
    base = base[0] if base else {}
    out = {"e0": fig_e0(), "e1": fig_e1(), "mech": fig_mech(base), "exploit": exploit_table(),
           "baselines": base}
    fig_curves()
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
