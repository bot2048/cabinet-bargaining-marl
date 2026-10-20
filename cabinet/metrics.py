"""Allocation statistics of spec Definition 1.7 and the Gamson regression (1).

Every installed cabinet is logged by the environment.  As in empirical data a
cabinet's parties are its portfolio-holding members C+ and seat shares are
computed over C+; one-party cabinets are excluded and counted separately.
"""
from functools import lru_cache

import numpy as np

from .bf_solver import bf_symmetric_share, solve_bf


def hamilton_min(seat_share: np.ndarray, m: int, rng=None):
    """Most proportional assignment of m equal-salience ministries
    (spec Proposition 2.12).  Returns the shares under mu_min and D_G^min."""
    rng = rng or np.random.default_rng(0)
    quota = m * seat_share
    base = np.floor(quota + 1e-12).astype(int)
    rem = quota - base
    left = m - base.sum()
    order = np.lexsort((rng.random(len(rem)), -rem))
    base[order[:left]] += 1
    p = base / m
    return p, 0.5 * np.abs(p - seat_share).sum()


@lru_cache(maxsize=4096)
def _bf_cached(w: tuple, q: int, rho: tuple, delta: float):
    return solve_bf(np.array(w), q, np.array(rho), delta)


def bf_benchmark(rec: dict, divisible: bool):
    """Analytic formateur share for divisible levels with a stationary rule."""
    if not divisible or rec["rho"] == "largest":
        return None
    w = rec["w"]
    n = len(w)
    if np.all(w == 1) and n % 2 == 1 and rec["rho"] == "uniform":
        return bf_symmetric_share(n, rec["delta"])
    rho = np.full(n, 1 / n) if rec["rho"] == "uniform" else w / w.sum()
    sol = _bf_cached(tuple(int(x) for x in w), int(w.sum() // 2 + 1),
                     tuple(np.round(rho, 12)), float(rec["delta"]))
    return float(sol["x"][rec["f"]])


def cabinet_rows(log, m: int, divisible: bool, rng=None):
    """Per-cabinet statistics; also the party-level rows for regression (1)."""
    rng = rng or np.random.default_rng(0)
    cabs, rows = [], []
    single = 0
    for rec in log:
        p, w, f = rec["p"], rec["w"], rec["f"]
        Cp = p > 1e-9
        if Cp.sum() <= 1:
            single += 1
            continue
        s = np.where(Cp, w / w[Cp].sum(), 0.0)
        dg = 0.5 * np.abs(p - s)[Cp].sum()
        if divisible:
            pmin, dmin = s, 0.0
        else:
            pmin_c, dmin = hamilton_min(s[Cp], m, rng)
            pmin = np.zeros_like(p)
            pmin[Cp] = pmin_c
        x_bf = bf_benchmark(rec, divisible)
        cabs.append(dict(
            phi=p[f] - s[f], phi_adj=p[f] - pmin[f], dg=dg, dg_adj=dg - dmin,
            pf=p[f], x_bf=x_bf, size=int(Cp.sum()), size_named=int(rec["C"].sum()),
            n=rec["n"], delay=rec["delay"], duration=rec["duration"],
            collapsed=rec["collapsed"], delta=rec["delta"],
            minimal_winning=_is_minimal_winning(rec["C"], w),
            partner_offer=np.mean([p[j] for j in np.flatnonzero(rec["C"]) if j != f])
            if rec["C"].sum() > 1 else np.nan))
        for j in np.flatnonzero(Cp):
            rows.append((p[j], s[j], float(j == f), w[j] / w.sum()))
    return cabs, np.array(rows), single


def _is_minimal_winning(C, w):
    q = w.sum() // 2 + 1
    tot = w[C].sum()
    return bool(tot >= q and all(tot - w[j] < q for j in np.flatnonzero(C)))


def gamson_regression(rows: np.ndarray):
    """OLS of p_ic = alpha + beta s_ic + gamma F_ic (+ residual)."""
    if len(rows) < 4:
        return dict(alpha=np.nan, beta=np.nan, gamma=np.nan, r2=np.nan)
    y = rows[:, 0]
    X = np.column_stack([np.ones(len(rows)), rows[:, 1], rows[:, 2]])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    r2 = 1 - res.var() / max(y.var(), 1e-12)
    return dict(alpha=coef[0], beta=coef[1], gamma=coef[2], r2=r2)


def summarise(log, episodes, m, divisible):
    cabs, rows, single = cabinet_rows(log, m, divisible)
    out = dict(cabinets=len(cabs), single_party=single)
    if cabs:
        for key in ("phi", "phi_adj", "dg", "dg_adj", "pf", "size", "delay",
                    "minimal_winning"):
            out[key] = float(np.mean([c[key] for c in cabs]))
        offers = [c["partner_offer"] for c in cabs if not np.isnan(c["partner_offer"])]
        out["partner_offer"] = float(np.mean(offers)) if offers else np.nan
        gaps = [c["pf"] - c["x_bf"] for c in cabs if c["x_bf"] is not None]
        out["gap_bf"] = float(np.mean(gaps)) if gaps else np.nan
        out["abs_gap_bf"] = float(np.mean(np.abs(gaps))) if gaps else np.nan
        dur = [c["duration"] for c in cabs if c["duration"] not in (None, np.inf)]
        out["duration"] = float(np.mean(dur)) if dur else np.nan
        out["collapse_rate"] = float(np.mean([c["collapsed"] for c in cabs]))
    out.update({f"reg_{k}": v for k, v in gamson_regression(rows).items()})
    if episodes:
        out["fail_rate"] = float(np.mean([e["failed"] / max(e["rounds"], 1) for e in episodes]))
        out["governed_frac"] = float(np.mean([e["governed"] / max(e["periods"], 1)
                                              for e in episodes]))
    return out
