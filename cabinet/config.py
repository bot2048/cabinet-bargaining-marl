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
    alpha_seat_choices: Optional[Sequence[float]] = None   # drawn per episode if given
    # a fixed (e.g. real) parliament: seats of the modelled parties and the
    # full house size that sets the quota; the first formateur may be fixed
    seats: Optional[Sequence[int]] = None
    house: Optional[int] = None
    first_formateur: Optional[int] = None
    # known policy positions (e.g. ParlGov left-right / 10) for the first
    # dimensions of each party's ideal point; the rest stay random
    ideal_points: Optional[Sequence[Sequence[float]]] = None
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
    # bargaining protocol (M2): "proposal" (closed rule) or "demand" (Morelli 1999;
    # divisible surplus only)
    protocol: str = "proposal"
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


# Variants that are not mechanism switches (so they never enter E*).
# M1_patient: repetition with agents patient across terms.  With delta = 0.9 and
# 6-period terms the discount between consecutive formations is about
# beta_T * delta^L = 0.67 * 0.53 = 0.35; here it is 0.83 * 0.98^3 = 0.78.
EXTRA = {"M1_patient": dict(L=3, T=6.0, Lam=4, delta_choices=(0.98,))}


def add_one_in(mech: str) -> EnvConfig:
    if mech in EXTRA:
        return BASE.with_(name=f"BASE+{mech}", **EXTRA[mech])
    return BASE.with_(name=f"BASE+{mech}", **_merge(NEEDS.get(mech, ()) + (mech,)))


def full_estar() -> EnvConfig:
    return BASE.with_(name="Estar", **_merge(MECHANISMS))


# Divisible game over 3-8 parties with varied seat distributions, under the two
# protocols (matched comparison for M2); used on real parliaments as well.
WIDE_DIV = E1.with_(name="WideProposal", n_choices=(3, 4, 5, 6, 7, 8),
                    alpha_seat_choices=(0.5, 1.0, 2.0, 3.0), house_size=150, rho="seat",
                    delta_choices=(0.9,))
WIDE_DEMAND = WIDE_DIV.with_(name="WideDemand", protocol="demand")
E0_DEMAND = E0.with_(name="E0demand", protocol="demand", n_choices=(3, 5), delta_choices=(0.9,))


def fixed_divisible(seats, protocol, name):
    """A fixed asymmetric parliament with a divisible surplus (mechanism test:
    do learned shares follow seats, voting weights or proposal power?)."""
    return WIDE_DIV.with_(name=name, seats=tuple(seats), house=int(sum(seats)),
                          n_choices=(len(seats),), seat_mode="fixed", protocol=protocol)


ASYM = {"A452": (45, 35, 20), "A4305": (40, 30, 25, 5)}


def estar_wide() -> EnvConfig:
    """E* over 3-8 parties with more varied seat distributions, for transfer
    to real parliaments (spec RQ7); real parliaments are held out."""
    return full_estar().with_(name="EstarWide", n_choices=(3, 4, 5, 6, 7, 8),
                              alpha_seat_choices=(0.5, 1.0, 2.0, 3.0), house_size=150)


def estar_wide_public() -> EnvConfig:
    """EstarWide with public policy positions, so formateurs can see ideology
    (used with real left-right positions on ParlGov parliaments)."""
    return estar_wide().with_(name="EstarWidePub", public_types=True)


def real_parliament(base: EnvConfig, cab: dict, left_right=None) -> EnvConfig:
    """One ParlGov cabinet as a fixed parliament with the PM party as formateur.
    ``left_right`` (0-10 per party) sets the first policy dimension."""
    ideal = None if left_right is None else tuple((x / 10.0,) for x in left_right)
    return base.with_(name=f"{cab['country']}-{cab['start']}", seats=tuple(cab["seats"]),
                      house=cab["house"], n_choices=(len(cab["seats"]),), seat_mode="fixed",
                      first_formateur=cab["pm"], T=1.0, ideal_points=ideal)


def leave_one_out(mech: str) -> EnvConfig:
    kw = _merge(k for k in MECHANISMS if k != mech)
    if mech == "M1_repeat":
        kw.update(T=1.0, Lam=0)        # L stays finite: exits need a term
    return BASE.with_(name=f"Estar-{mech}", **kw)
