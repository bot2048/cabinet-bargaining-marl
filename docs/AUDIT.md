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

## 6. Open

See the end of [`RESULTS_AUDIT.md`](RESULTS_AUDIT.md) for the experiments not run and why.
