"""Environment parameters (spec Definition 2.1) and the nested levels E0-E6.

Every level is a restriction of one parameter set (spec principle P1), so the
code that is checked against Baron-Ferejohn in E0/E1 is the same code that
produces the mechanism results in E2-E6.
"""
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence


@dataclass(frozen=True)
class EnvConfig:
    name: str = "custom"
    # parliament
    n_choices: Sequence[int] = (3,)          # n drawn uniformly per episode
    seat_mode: str = "equal"                 # "equal" (w_i = 1) or "random"
    house_size: int = 100                    # W for random parliaments
    alpha_seat: float = 1.5                  # Dirichlet concentration of vote shares
    # good being divided
    divisible: bool = True                   # unit surplus (E0-E1) or ministries (E2+)
    m: int = 6                               # number of ministries
    xi: float = 0.0                          # valuation heterogeneity (M5)
    kappa_nu: float = 5.0
    # policy (M9)
    d: int = 2
    lam: float = 0.0
    y0: float = 0.5
    # formateur selection (M10)
    rho: str = "uniform"                     # uniform | seat | largest
    # patience
    delta_choices: Sequence[float] = (0.9,)
    # horizon. L=None: one formation episode with an infinite-horizon cabinet,
    # implemented through Proposition 2.6 (per-period shares paid once).
    L: Optional[int] = None
    T: float = 1.0                           # expected number of terms (M1)
    kappa_w: float = 200.0                   # electoral stability
    # governance and exits (M8)
    exits: bool = False
    ebar: float = 0.0
    # information (M4)
    public_types: bool = True
    # memory window (R9)
    Lam: int = 0
    max_rounds: int = 40                     # cap on failed rounds when L is None

    @property
    def beta_T(self) -> float:
        return 1.0 - 1.0 / self.T

    @property
    def n_max(self) -> int:
        return max(self.n_choices)

    def with_(self, **kw) -> "EnvConfig":
        return replace(self, **kw)


LAM_MAX = 4  # width reserved for the history window in observations

# ---------------------------------------------------------------------------
# Levels.  Sizes are scaled down from the spec (Table 4) so that every
# experiment runs on a CPU; see README "Scaling" for the mapping.
# ---------------------------------------------------------------------------
E0 = EnvConfig(name="E0", n_choices=(3, 5, 7), seat_mode="equal", divisible=True,
               rho="uniform", delta_choices=(0.8, 0.9, 0.95))
E1 = EnvConfig(name="E1", n_choices=(3, 4, 5), seat_mode="random", divisible=True,
               rho="uniform", delta_choices=(0.8, 0.9, 0.95))
E2 = E1.with_(name="E2", divisible=False, m=6, xi=0.0)
E3 = E2.with_(name="E3", L=6, exits=True, ebar=0.05)
E4 = E3.with_(name="E4", T=3.0, Lam=4)
E5 = E4.with_(name="E5", xi=0.5, public_types=False)
E6 = E5.with_(name="E6", lam=0.5, rho="seat")          # main environment E*

LEVELS = {c.name: c for c in (E0, E1, E2, E3, E4, E5, E6)}

# Base configuration for add-one-in tests: every mechanism switched off.
# Indivisible ministries are kept on (M9 and M4 need them) and the horizon
# is one formation episode.
BASE = E2.with_(name="BASE", delta_choices=(0.9,))

# Mechanism switches (spec Table 6).  Term length L is owned by M1 here,
# because terms need a finite length; M8 owns only the exit stage.  This
# resolves the double ownership of L noted in our review of spec v0.4.
MECHANISMS = {
    "M1_repeat":  dict(L=6, T=3.0, Lam=4),
    "M4_private": dict(public_types=False),
    "M5_hetero":  dict(xi=0.5),
    "M8_exits":   dict(L=6, exits=True, ebar=0.05),
    "M9_policy":  dict(lam=0.5),
    "M10_seatrho": dict(rho="seat"),
}
# Private types matter only if types differ (spec Table 6 caption), so the
# add-one-in test of M4 switches M5 on with it.
NEEDS = {"M4_private": ("M5_hetero",)}


def _merge(names) -> dict:
    kw = {}
    for k in names:
        kw.update(MECHANISMS[k])
    return kw


def add_one_in(mech: str) -> EnvConfig:
    return BASE.with_(name=f"BASE+{mech}", **_merge(NEEDS.get(mech, ()) + (mech,)))


def full_estar() -> EnvConfig:
    return BASE.with_(name="Estar", **_merge(MECHANISMS))


def leave_one_out(mech: str) -> EnvConfig:
    kw = _merge(k for k in MECHANISMS if k != mech)
    if mech == "M1_repeat":
        kw.update(T=1.0, Lam=0)        # L stays finite: exits need a term
    return BASE.with_(name=f"Estar-{mech}", **kw)
