# Pre-specified tests (written 21 October 2026, before the remaining runs finished)

Written and committed while these runs were still training or queued:

| Runs | Done when this file was written |
|---|---|
| `mech_WideProposal` | seeds 0-2 (seeds 3-9 running or queued) |
| `mech_WideDemand` | seeds 0-4 (seeds 5-9 running or queued) |
| `explore_A452Demand`, `explore_A4305Demand`, `mech_A452Proposal`, `mech_A4305Proposal` | none |

The commit time of this file is the record. Anything not listed here is exploratory.

## Exact benchmarks (computed before the runs)

`cabinet/demand_solver.py` solves our demand protocol exactly (backward induction on the demand grid, every
move order, stationary failure values, δ = 0.9). Its results are the same for both tie-breaking rules and
both grids (1% and 2%; 2.5% and 4% for five parties).

| Parliament | Equilibrium split |
|---|---|
| 3 equal parties | first two movers 0.50 / 0.50 |
| 5 equal parties | first three movers about 1/3 each |
| 45/35/20 | 0.50 / 0.50 in whichever pair forms, so seats do not matter |
| 40/30/25/5 | 0.50 / 0.50 between two of the three large parties; the 5-seat party (a dummy) gets 0 |

These are the same predictions as Morelli's proportionality to minimal-integer weights, which here are
(1,1,1) and (1,1,1,0). Gamson's law instead predicts seat-proportional splits (for example 0.56 / 0.44 for
the pair 45+35). For the proposal protocol, the benchmark is the weighted Baron-Ferejohn solution (`cabinet/bf_solver.py`).

## Confirmatory family (Holm correction, α = 0.05)

Seeds are the units. Each seed's coefficients come from its own evaluation cabinets (regression (1):
share = α + β·seat share in coalition + γ·formateur).

* **H1, protocol effect on the premium.** γ(WideProposal) > γ(WideDemand). Test: exact two-sample
  permutation test on the 10 + 10 seed-level γ, one-sided. Also reported: the difference in means with a
  seed-bootstrap 95% CI.
* **H2, protocol effect on the seat slope.** β(WideDemand) > β(WideProposal). Same test.
* **H3, no premium under demand (equivalence).** |γ(WideDemand)| < 0.05. Test: TOST, i.e. the 90%
  t-interval over seeds lies inside (−0.05, 0.05).

## Pre-specified predictions with 3 seeds (descriptive, no p-values)

* **P1, weights vs seats (45/35/20, demand, with exploration).** In each seed, regress within-cabinet
  shares on within-cabinet seat shares. Equilibrium predicts slope 0 (even pairs); Gamson predicts slope 1.
  We will say learned play "follows the equilibrium (voting weights)" if the mean slope is below 0.5 and
  every seed is below 0.5. We will say it "follows seats" if the mean slope is above 0.5 and every seed is
  above 0.5. Otherwise the result is mixed.
* **P2, dummy party (40/30/25/5, demand).** Equilibrium: the 5-seat party is never in the cabinet. Gamson
  only says it gets 5/(coalition seats) when included. We report its inclusion rate and mean share.
* **P3, the same parliaments under the proposal protocol.** Prediction: a formateur premium (the formateur
  share exceeds the partner share in the same pair), as the weighted Baron-Ferejohn solution predicts.
* **P4, exploitability scale.** Every best-response gain is reported both in per-period units and relative
  to the mean per-period payoff of a profile seat.
