# Scientific audit (21 October 2026)

A critical review of our own results before the final write-up: what each claim rests on, what was wrong
or overstated, what we changed, and what remains open. Numbers come from the files listed in
[`RESULTS_AUDIT.md`](RESULTS_AUDIT.md).

## 1. Novelty: what is and is not new

Closest prior work (all verified):

| Work | What it establishes | Consequence for us |
|---|---|---|
| Baron & Ferejohn (1989, APSR 83(4)) | proposal bargaining gives the proposer a premium | our theoretical benchmark |
| Gamson (1961); Browne & Franklin (1973) | portfolio shares follow seat shares | the empirical regularity |
| Warwick & Druckman (2006, EJPR 45) | salience-weighted payoffs also mirror seat contributions | real proportionality is not a count artefact; agrees with our key-ministry test |
| Ansolabehere, Snyder, Strauss & Ting (2005, AJPS 49(3)) | measured by voting weights, a formateur bonus appears | supports proposal models; our real-data regression uses seats (see §5) |
| Morelli (1999, APSR 93(4)); Montero & Vidal-Puga (2007, APSR 101(4) comment) | demand bargaining: no proposer premium, shares follow bargaining power; a counterexample to proportionality | demand protocol is an existing theory, not ours |
| Fréchette, Kagel & Morelli (2005, Econometrica 73(6)) | lab behaviour is less sensitive to the protocol than theory predicts | our protocol effect must be shown to be robust, not assumed |
| Carroll & Cox (2007, AJPS 51(2)) | pre-electoral pacts produce Gamsonian splits | alternative mechanism we do not model |
| Cox (2021, APSR 115(3)) | with no structural bargaining advantage, allocations are Gamsonian with a small-party bias | closest theoretical counterpart of our demand-protocol result |
| Kamm & Siegenthaler (2024, Experimental Economics 27) | lab: proportional payoffs when coalitions are committed before the split, pivotality otherwise | closest experimental counterpart of the protocol effect |
| Bachrach et al. (2020, Artificial Intelligence 288) | deep RL agents negotiating teams in weighted voting games reach outcomes near cooperative solutions | closest RL work |

Not novel: multi-agent RL itself, the Gamson-BF puzzle, demand bargaining, or the idea that the protocol
or commitment structure matters. We found no prior work training multi-agent RL agents in the
Baron-Ferejohn cabinet game and validating them against its exact stationary equilibrium.

Narrowest defensible contribution: a reproducible computational study that (i) checks learned play
against the exact BF solution, (ii) compares protocols, information and policy motives under matched
conditions while measuring the formateur premium in counts, in the formateur's own valuation and in
policy-setting ministries, and (iii) scores the resulting policies out of sample on historical cabinets
against Gamson and BF. Our evidence is a negative result about proposal bargaining plus a replication,
with learning agents, of the importance of the protocol.

## 2. Findings that were wrong or overstated, and what changed

| Earlier claim | Problem | Now |
|---|---|---|
| "Private information removes the formateur premium" | measured in value or policy currencies the premium remains | reported as a change of currency |
| Raw premium p_f - s_f as the headline | under seat-proportional recognition it falls mechanically | seat-controlled coefficient gamma throughout |
| "Real cabinets have no premium in key ministries" | the CI (-0.021, +0.056) does not exclude a modest premium | "no detectable premium" |
| "Learned play recovers BF" | shares match on average, but learned voters accept below delta/n for n >= 5 | outcome match, not equilibrium play; tested by best responses (§3) |
| One 0.017 figure summarising validation | mean absolute gap (0.017), mean signed gap (+0.001) and cells within 0.02 (6/9) are different statistics | named separately, per-cell table with seed CIs |
| Demand protocol "reproduces Gamson's law" | 3 seeds, default exploration, own cabinets only | 10 seeds, exploration robustness, exploitability, conditional prediction on real cabinets |
| 183 vs 195 cabinets unexplained | different filters | trail: 195 in the data; 12 have a coalition member outside the 8 largest parties; 183 scored by every model |

## 3. Equilibrium and exploitability: what the evaluation can and cannot show

Protocol (`cabinet/ppo.py::exploitability`): one seat per episode is controlled by a best-response
network trained with PPO (same algorithm and hyperparameters as the learners) while all other seats play
the frozen profile; two initialisations (warm start from the profile, from scratch) are trained and the
larger estimated gain is kept. Gain = per-period payoff of the best-response seat minus the per-period
payoff of a profile seat, each from separate evaluation episodes (3000). Payoffs are per period (one unit
= the whole office flow of a period; one-shot formation games pay per-period equivalents).

* A learned best response is weaker than the true best response, so the estimate is a lower bound on
  exploitability up to evaluation noise; keeping the maximum of two runs adds a small upward bias.
* Positive controls (scripted Gamson and BF-heuristic profiles) show whether the test can detect
  exploitable profiles at the given budget; they exist only for the proposal protocol.
* Low gains therefore mean "no exploit found at this budget", not "equilibrium".

## 4. Statistics

* Independent units are training seeds; evaluation episodes of one model are not independent replicates.
* Condition differences are reported with bootstrap intervals over seeds of the difference.
* Historical data: country-clustered bootstrap (2000 resamples); paired differences on identical cabinets;
  leave-one-country-out sensitivity.
* Confirmatory (fixed before the run): protocol comparison (formateur coefficient and seat slope, demand vs
  proposal, 10 seeds each). Everything else is exploratory.

## 5. Historical evaluation

* Seki-Williams ministers per party for majority coalition cabinets (1988-2014), linked to ParlGov by our
  team's companion analysis; the prime minister's party proxies the formateur (standard, but imperfect).
* Historical outcomes were never used for training or hyperparameter selection. They did influence which
  experiments we ran after the first results (exploratory), which we disclose.
* The real regression uses seat shares; with Banzhaf voting power the fit is worse (R^2 0.87 vs 0.93),
  consistent with Warwick & Druckman, while Ansolabehere et al. report a formateur bonus relative to voting
  weights. We do not adjudicate that measurement debate.

## 6. External review (21 October): what we accepted, rejected and did

We asked several outside reviewers (other AI systems) for critique. Their claims were checked against
our code and data before acting on them.

| Claim | Verdict | Action |
|---|---|---|
| Our demand protocol is not Morelli's game, so its equilibrium is unknown | correct | exact solver (`cabinet/demand_solver.py`): 3 equal 50/50, 5 equal ~1/3 each, 45/35/20 50/50 in any pair, 40/30/25/5 dummy gets 0 |
| 3 equal parties: first mover 0.70, closer 0.30 (one reviewer) | wrong: ignores undercutting under the cheapest-set rule | solver gives 0.50/0.50 |
| Demand theory predicts voting weights, not seats; the seat slope may be a learning bias | testable | fixed parliaments 45/35/20 and 40/30/25/5 moved to the front of the queue; predictions pre-registered (`PREREGISTRATION.md`) |
| CLOSE is 1 of 102 actions in a flat demand head, so exploration starves closing | correct (code) | explains the default-exploration lock-in; exploration runs are the robustness check; stated as a limitation (no retraining) |
| A non-pivotal party can close and take the residual | correct (code) | solver models the same rule; the dummy still gets 0 in equilibrium |
| Lenient BF voters: the formateur may best-respond to noisy voters (no exploration failure) | rejected by test | `scripts/bf_best_response.py`: against the frozen voters a best response gains 0.01 (n=3) to 0.15 (n=7) per formation, so learned play is outcome-equivalent, not an equilibrium |
| Shares sum to one within a cabinet, so a constant intercept is misspecified | correct | alpha/k specification: gamma 0.019 (was 0.011) |
| PM party is usually the largest; 31 clusters need a wild bootstrap; weights vs seats | correct | `scripts/real_robustness.py`: gamma +0.011 vs seats, +0.059 with a largest-party control (20 cabinets identify it), +0.074 vs minimal integer weights (wild-cluster CIs exclude 0); seats fit better (R^2 0.93 vs 0.88) |
| Imposed-coalition evaluation of demand agents removes outside options (off-distribution; demands exceed the pie in 12-22% of cabinets) | correct | reported as off-distribution; the in-distribution test is agents' own cabinets, now also restricted to the real coalition (matched-coalition design) |
| Report exploitability relative to payoffs | correct | gains reported per period and relative to the mean seat payoff |
| Need >= 20 seeds per protocol | not needed | seed ranges do not overlap; the exact permutation test with 10 + 10 seeds reaches p ~ 1e-5 |
| Train on pre-2000 cabinets, predict post-2000 | rests on a false premise | real cabinets are never used for training; the whole real-data test is out of sample |
| Cross-protocol tournament; party-specific manifesto salience (MARPOR) | not feasible or not meaningful here | protocols are institutions, not agent types; MARPOR needs registration and portfolio coding; listed as future work |

## 7. Open

See the end of [`RESULTS_AUDIT.md`](RESULTS_AUDIT.md) for the experiments not run and why.
