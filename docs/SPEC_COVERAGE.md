# Coverage of the project specification

This repository implements the specification *Bargaining over Cabinets: A Multi-Agent Reinforcement
Learning Study of Portfolio Allocation in Coalition Governments* (v0.4, 15 October 2026, and the
additions of v0.5, 22 October 2026, listed at the end). This file maps
every identifier of the specification to the code, test or result that addresses it, and lists every
deviation. Status: **done**, **partial** (implemented with a stated simplification), **not done**.

## Scale

The specification budgets about 230-285 GPU-hours (its Table 15). We ran on one laptop (32 CPU threads,
an RTX 4060, 15 GB RAM) within a week, so sizes are scaled down: n ≤ 5 parties in the mechanism
experiments (3-8 for transfer to real parliaments), m = 6 ministries, L = 6 periods per term, an
expected 3 terms, 500-700 PPO iterations per run. The parameters are in `cabinet/config.py`; the
mapping to the specification's Table 4 is in the docstring there.

## Environment and theory (spec §2)

| ID | What | Where | Status |
|---|---|---|---|
| Def 2.1-2.4 | parameters, game, rewards, observations | `cabinet/config.py`, `cabinet/env.py` | done |
| Def 2.5, E0-E6 | nested levels of one class | `cabinet/config.py` (`LEVELS`) | done |
| Prop 2.6, Thm 2.7 | BF embedding, symmetric closed form | `bf_solver.bf_symmetric_share`; `tests/test_core.py::test_symmetric_bf_closed_form`, `test_bf_agent_recovers_benchmark_in_env` | done |
| Fact 2.8, Rem 2.9 | weighted BF payoffs, checked solver | `bf_solver.solve_bf` (homotopy root finder + LP residual check) | done |
| Prop 2.10 | ex-post formateur advantage | checked numerically in `test_weighted_solver_certificate` | done |
| Prop 2.11 | welfare identity | `test_welfare_identity` | done |
| Prop 2.12-2.13 | Hamilton floor; floor-adjusted statistics | `metrics.hamilton_min`, `test_hamilton_floor_bound` | done (equal salience; the integer program for unequal salience is not needed because σ is uniform) |
| Lemma 2.21 | masked autoregressive proposals | `policy._propose`; `test_masks_always_feasible` | done |
| Lemma 2.22, 3.2 | permutation equivariance | `test_permutation_equivariance_of_value`; decoding order by seats with random ties | done |
| Lemma 2.23 | potential-based shaping | not used (the spec makes it optional and off by default) | n/a |
| Def 2.24, 3.10 | exploitability in office units | `ppo.exploitability`, `run_experiment.py --exp exploit` | done |
| Rem 2.25 | infinite horizons | L = ∞ levels use the embedding of Prop 2.6 (shares paid once as per-period equivalents) instead of stochastic termination; same expected returns, lower variance | partial |

## Requirements

| ID | Requirement | Status |
|---|---|---|
| R1 | levels as restrictions of one class | done |
| R2 | automated tests of the E0 closed form and E1 solver | done (`tests/`) |
| R3 | masks, infeasible actions impossible | done (env asserts feasibility) |
| R4 | any n in 3-12, m in 1-25, equivariant observations | done (no size constants; tested up to n = 8) |
| R5 | per-agent observation and global state | done (`observe(..., global_view=True)` for the centralised critic) |
| R6 | transition discount | done (agent-level interval discounts) |
| R7 | deterministic given a seed; vectorised | done (`test_determinism_given_seed`; batched runner) |
| R8 | PettingZoo AEC interface (SHOULD) | **not done**: a `pending()/act()` decision interface, which handles simultaneous votes directly |
| R9 | last Λ cabinets in observations, Λ = 0 supported | done (last votes/offers also belong to the window) |
| R10 | ParlGov seat vectors; cabinet log in the format of (1) | done (`real_parliament`, `cabinet_log`) |
| R11 | policy of Def 3.1 with memory switches | done: GRU core (`--rec`, stored-state recurrence: no back-propagation through earlier decisions), history window only ("-rec", default), memoryless ("-mem", `Estar_Lam0`) |
| R12 | stored masks, exact ratios | done: masks are recomputed from the stored observation and actions, which is exact because masks depend only on them (Lemma 3.3); `test_teacher_forced_logp_matches_sampling` |
| R13 | advantages on agent-level sequences | done (`ppo.gae`) |
| R14 | training logs | partial: Φ, D_G, floor-adjusted versions, Δ_BF, single-party cabinets, delay, collapse, entropy, KL, value loss; returns by role are not logged |
| R15 | checkpoints with metadata | done (`models/<experiment>/seed<k>.pt`; resumable checkpoints in `runs/ckpt/`) |
| R16 | exploitability algorithm and cross-play | done (`exploitability`, `scripts/crossplay.py`) |
| R17 | held-out real parliaments | done (never used for training) |
| R18 | record steps, time | done (in each result's `history`) |
| R19 | reproducible from a configuration | done (`run_experiment.py` arguments) |
| R20 | finite support | done (valuations, ideal points and bonuses rounded) |

## Learning framework (spec §3)

| ID | Status |
|---|---|
| LA1 shared-parameter IPPO self-play | done (default) |
| LA2 MAPPO with a centralised critic | done (`--algo mappo`) |
| LA3 fictitious self-play | done (`--algo fsp`; checkpoint pool, no separate average policy) |
| LA4 symmetric PSRO | done (`--exp psro`, `cabinet/psro.py`; pairwise surrogate for the restricted game, as the spec allows) |
| LA5 opponent shaping (exploratory in the spec) | **not done** |
| BL1-BL4 scripted baselines | done (`cabinet/baselines.py`) |
| Curriculum K0-K6 | **not done**: every level is trained from scratch (the spec's from-scratch arm, OD8) |
| Remark 3.8 / H9 entropy temperature | done (`--exp entropy`) |
| Def 3.12 cross-play | done |

Deviation found during tuning: a Dirichlet head for divisible shares collapsed before learning; offers
use a masked 1% grid instead (finer than the spec's 0.02 tolerance).

## Research questions, hypotheses and success criteria

| ID | Experiment | Result file(s) |
|---|---|---|
| RQ1, H1, S1 | E0 benchmark, n ∈ {3,5,7}, δ ∈ {0.8,0.9,0.95}, 10 seeds; E1 weighted parliaments | `results/e0`, `results/e1` |
| RQ2, M1-M10 | add-one-in from the base configuration, leave-one-out from E*, 3-5 seeds | `results/mech_*` |
| RQ3, H4 | regression on seat shares vs Banzhaf power shares | `reg_*`, `regbz_*` in every result |
| RQ4, S3 | real cabinets: Seki-Williams ministers per party linked to ParlGov (195 cabinets, 31 countries); empirical regression, conditional split prediction against Gamson and BF, the agents' own cabinets in the real parliaments, key-ministry test; ParlGov coalition choice | `results/seki/`, `results/parlgov_*`, `scripts/seki_eval.py`, `scripts/key_ministries.py` |
| RQ5 | delay, failure, collapse per configuration | `fail_rate`, `delay`, `collapse_rate` in every result |
| RQ6, H7 | IPPO vs MAPPO vs FSP on E*, cross-play; memory variants | `results/mech_Estar`, `mappo_Estar`, `fsp_Estar`, `rec_Estar`, `Estar_Lam0`, `results/crossplay` |
| RQ7, H8, S4 | transfer to held-out real parliaments | `results/parlgov_mech_EstarWide*` |
| S2 | Gamson region with exploitability | `results/exploit_*` with scripted positive controls |
| M2 demand protocol | Morelli (1999), divisible levels: `protocol="demand"` in `cabinet/config.py`, `CLOSE` action in `cabinet/env.py`; matched comparison WideDemand vs WideProposal | done (`results/mech_WideDemand`, `mech_WideProposal`, `seki/own_*`) |
| M3 pre-electoral pacts | marked PLANNED in the spec itself | not done |
| M6 measuring party strength | evaluation only (RQ3) | done |
| M7 bounded rationality | entropy temperature (H9) | done |

## Additions of specification v0.5 (22 October 2026)

v0.5 makes the value of proposal power the primary measure of a formateur premium.

| ID | What | Where | Status |
|---|---|---|---|
| Def 1.8 | within-coalition value of proposal power Psi_in and ex-ante value Psi_ex | `cabinet/proposal_power.py` | done |
| Def 2.4 | payoff to the end of the term in per-period units, split into salience, taste, policy and exit | `env.comp` in `cabinet/env.py`; `test_reward_components_add_up` | done |
| Algorithm 4 | proposal-power probe: saved formation states, overridden formateur, forced coalition, common random numbers, on-path filter | `cabinet/proposal_power.py::probe`; `scripts/proposal_power_eval.py` | done (probe states are the first formation state of each episode; N_probe = 2000, N_rep = 4, eta_min = 0.02 estimated from 64 sampled proposals) |
| Lemma 3.13 | coalition forcing exact on path | forcing reuses the policy's teacher-forced inclusion decisions (`force_inc`); primary Psi_in keeps only on-path pairs | done |
| Prop 2.12, Cor 2.13 | closed forms of Psi in E0 and E1, Psi_in = 0 for the Gamson agent | `test_proposal_power_closed_forms` (exact) | done |
| R21 | save/restore formation states with random streams, override the formateur, force coalition decisions | states are deep-copied with their generators; `CabinetEnv.override_recognition`; `force_inc` | done |
| R22 | rewards logged per component; Algorithm 4; unit tests against Cor 2.13, Prop 2.12 and BL2 | as above | done |
| R23 | premium claims rest on Psi or on observational statistics; own-valuation shares descriptive only | paper and report: the currency shift is stated with Psi components; own-valuation regressions are reported as descriptive only | done |
| Problem 1.10 regions | observational Gamson, Gamson, disguised-premium and formateur regions | `scripts/psi_summary.py` (tolerances: the thresholds of S2, eps_G = 0.05, eps_Phi = eps_Psi = eps_F = 0.02, because the spec's Section 4 does not yet fix them) | done |
| H10 | private values or policy drive Phi to zero while Psi_in stays positive, in Psi_nu or Psi_pol | `results/psi_summary.json` (`h10_replication`: seeds trained after the pilot runs) | done |
| 4.4 (planned) | within-party analogue of Psi_in on real data | `scripts/real_within_party.py` | done |
| M2 analogue of Psi_in | the spec defers it to Section 4 | only Psi_ex reported for demand runs, as the spec states | n/a |
