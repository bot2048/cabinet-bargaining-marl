"""Report figure: value of proposal power Psi_in by reward channel (mean over seeds).

Reads results/psi_summary.json (written by scripts/psi_summary.py) and writes
figures/fig_psi.png.  Run from the repository root:

    python scripts/make_fig_psi.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = json.load(open(os.path.join(ROOT, "results", "psi_summary.json")))["rows"]
cfgs = [("mech_BASE", "base game"), ("mech_BASE+M1_repeat", "+ repeated terms"),
        ("mech_BASE+M8_exits", "+ exits"), ("mech_BASE+M10_seatrho", "+ seat recognition"),
        ("mech_BASE+M5_hetero", "+ heterog. values"), ("mech_BASE+M4_private", "+ private values"),
        ("mech_BASE+M9_policy", "+ ministerial policy"), ("mech_Estar", "E* (all on)"),
        ("mappo_Estar", "E*, MAPPO"), ("fsp_Estar", "E*, fict. self-play")]
ch = [("psi_in_sigma", "counts $\\Psi_\\sigma$", "#2b5d9b"),
      ("psi_in_nu", "which ministries $\\Psi_\\nu$", "#c0583c"),
      ("psi_in_pol", "policy $\\Psi_{pol}$", "#3d8f5a")]

plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(3.35, 2.15))
y = list(range(len(cfgs)))[::-1]
for yi, (tag, _) in zip(y, cfgs):
    r = rows[tag]
    pos = neg = 0.0
    for key, _, col in ch:          # stacked bars: positive channels right, negative left
        v = r[key]["mean"]
        if v >= 0:
            ax.barh(yi, v, left=pos, color=col, height=0.62)
            pos += v
        else:
            ax.barh(yi, v, left=neg, color=col, height=0.62)
            neg += v
    tot = r["psi_in"]
    ax.errorbar(tot["mean"], yi, xerr=tot["se"], fmt="D", ms=2.8, color="black", elinewidth=0.6, capsize=1.5)
ax.axvline(0, color="black", lw=0.5)
ax.axhline(2.5, color="grey", lw=0.5, ls="--")
ax.set_yticks(y)
ax.set_yticklabels([lab for _, lab in cfgs])
ax.set_xlabel("$\\Psi_{in}$ (per period, proposer minus partner)")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in ch] + [
    plt.Line2D([], [], marker="D", ls="", color="black", ms=3)]
fig.legend(handles, [l for _, l, _ in ch] + ["total $\\pm$ s.e."], fontsize=5.8, loc="upper center",
           ncol=4, frameon=False, handlelength=1.2, columnspacing=0.8)
ax.set_xlim(-0.06, 0.34)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93))
# PNG (600 dpi): the PDF backend needs fontTools DLLs that fail on long Windows paths
fig.savefig(os.path.join(ROOT, "figures", "fig_psi.png"), dpi=600)
print("figures/fig_psi.png written")
