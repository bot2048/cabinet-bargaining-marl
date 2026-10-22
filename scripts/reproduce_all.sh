#!/usr/bin/env bash
# Every command that produced the results in results/, in the order they were run.
# Runs sequentially; on one laptop this is several days of CPU time. Add "--device cuda" to
# scripts/run_experiment.py commands to run the PPO updates on a GPU, and "--threads N" to set
# the number of CPU threads. Each command skips nothing: delete results/<tag>/ to recompute.
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- data (ParlGov and Seki-Williams; see the NOTICE files in data/)
bash scripts/parlgov_download.sh
python scripts/parlgov_prepare.py
bash scripts/seki_download.sh

# ---- training, evaluation and probes
python scripts/run_experiment.py --exp mech --config BASE --iters 500 --seed 0
python scripts/run_experiment.py --exp mech --config BASE+M1_repeat --iters 500 --seed 0
python scripts/run_experiment.py --exp e0 --iters 700 --seed 0
python scripts/run_experiment.py --exp mech --config Estar --iters 500 --seed 0
python scripts/run_experiment.py --exp mech --config BASE+M4_private --iters 500 --seed 0
python scripts/run_experiment.py --exp mech --config BASE+M8_exits --iters 500 --seed 0
python scripts/run_experiment.py --exp mech --config BASE+M5_hetero --iters 500 --seed 0
python scripts/run_experiment.py --exp mech --config BASE+M9_policy --iters 500 --seed 0
python scripts/run_experiment.py --exp mech --config BASE+M10_seatrho --iters 500 --seed 0
python scripts/run_experiment.py --exp e1 --iters 700 --seed 0
python scripts/run_experiment.py --exp e0 --iters 700 --seed 1
python scripts/run_experiment.py --exp mech --config Estar_Lam0 --iters 500 --seed 0
python scripts/run_experiment.py --exp mech --config BASE --iters 500 --seed 1
python scripts/run_experiment.py --exp mech --config BASE+M4_private --iters 500 --seed 1
python scripts/run_experiment.py --exp mech --config Estar --iters 500 --seed 1
python scripts/run_experiment.py --exp mech --config BASE+M1_repeat --iters 500 --seed 1
python scripts/run_experiment.py --exp mech --config BASE+M5_hetero --iters 500 --seed 1
python scripts/run_experiment.py --exp mech --config BASE+M8_exits --iters 500 --seed 1
python scripts/run_experiment.py --exp mech --config BASE+M10_seatrho --iters 500 --seed 1
python scripts/run_experiment.py --exp e1 --iters 700 --seed 1
python scripts/run_experiment.py --exp mech --config BASE+M9_policy --iters 500 --seed 1
python scripts/run_experiment.py --exp mech --config Estar_Lam0 --iters 500 --seed 1
python scripts/run_experiment.py --exp e0 --iters 700 --seed 2
python scripts/run_experiment.py --exp mech --config BASE --iters 500 --seed 2
python scripts/run_experiment.py --exp mech --config Estar --iters 500 --seed 2
python scripts/run_experiment.py --exp mech --config BASE+M1_repeat --iters 500 --seed 2
python scripts/run_experiment.py --exp mech --config BASE+M4_private --iters 500 --seed 2
python scripts/run_experiment.py --exp mech --config BASE+M5_hetero --iters 500 --seed 2
python scripts/run_experiment.py --exp mech --config BASE+M8_exits --iters 500 --seed 2
python scripts/run_experiment.py --exp mech --config BASE+M10_seatrho --iters 500 --seed 2
python scripts/run_experiment.py --exp e1 --iters 700 --seed 2
python scripts/run_experiment.py --exp mech --config BASE+M9_policy --iters 500 --seed 2
python scripts/run_experiment.py --exp mech --config Estar_Lam0 --iters 500 --seed 2
python scripts/run_experiment.py --exp mech --algo mappo --config Estar --iters 500 --seed 0
python scripts/run_experiment.py --exp mech --algo mappo --config Estar --iters 500 --seed 1
python scripts/run_experiment.py --exp mech --algo mappo --config Estar --iters 500 --seed 2
python scripts/run_experiment.py --exp mech --config Estar-M1_repeat --iters 500 --seed 0
python scripts/run_experiment.py --exp mech --config Estar-M4_private --iters 500 --seed 0
python scripts/run_experiment.py --exp mech --algo fsp --config Estar --iters 500 --seed 1
python scripts/run_experiment.py --exp mech --config Estar-M5_hetero --iters 500 --seed 0
python scripts/run_experiment.py --exp mech --algo fsp --config Estar --iters 500 --seed 0
python scripts/run_experiment.py --exp mech --config Estar-M8_exits --iters 500 --seed 0
python scripts/run_experiment.py --exp mech --algo fsp --config Estar --iters 500 --seed 2
python scripts/run_experiment.py --exp mech --config EstarWide --iters 600 --seed 0
python scripts/run_experiment.py --exp mech --config Estar-M1_repeat --iters 500 --seed 1
python scripts/run_experiment.py --exp mech --config Estar-M9_policy --iters 500 --seed 0
python scripts/run_experiment.py --exp mech --config Estar-M10_seatrho --iters 500 --seed 0
python scripts/run_experiment.py --exp mech --config Estar-M4_private --iters 500 --seed 1
python scripts/run_experiment.py --exp mech --config Estar-M5_hetero --iters 500 --seed 1
python scripts/run_experiment.py --exp mech --config Estar-M8_exits --iters 500 --seed 1
python scripts/parlgov_eval.py --tag mech_EstarWide --seed 1
python scripts/run_experiment.py --exp entropy --config BASE --ent-end 0.002 --iters 500 --seed 0
python scripts/run_experiment.py --exp mech --config EstarWide --iters 600 --seed 2
python scripts/parlgov_eval.py --tag mech_EstarWide --seed 2
python scripts/parlgov_eval.py --tag mech_EstarWide --seed 0
python scripts/run_experiment.py --exp mech --config Estar-M9_policy --iters 500 --seed 1
python scripts/run_experiment.py --exp mech --config Estar-M10_seatrho --iters 500 --seed 1
python scripts/run_experiment.py --exp entropy --config BASE --ent-end 0.02 --iters 500 --seed 0
python scripts/run_experiment.py --exp entropy --config BASE --ent-end 0.05 --iters 500 --seed 1
python scripts/run_experiment.py --exp entropy --config BASE --ent-end 0.05 --iters 500 --seed 0
python scripts/run_experiment.py --exp entropy --config BASE --ent-end 0.02 --iters 500 --seed 1
python scripts/run_experiment.py --exp entropy --config BASE --ent-end 0.002 --iters 500 --seed 1
python scripts/run_experiment.py --exp mech --config EstarWidePub --iters 600 --seed 1
python scripts/parlgov_eval.py --tag mech_EstarWidePub --ideology --seed 1
python scripts/parlgov_eval.py --tag mech_EstarWidePub --ideology --seed 0
python scripts/run_experiment.py --exp mech --config EstarWidePub --iters 600 --seed 0
python scripts/run_experiment.py --exp baselines --seed 0
python scripts/run_experiment.py --exp exploit --config BASE --br-iters 150 --seed 1
python scripts/parlgov_eval.py --tag mech_EstarWide --scope all --seed 0
python scripts/parlgov_eval.py --tag mech_EstarWide --scope all --seed 1
python scripts/parlgov_eval.py --tag mech_EstarWide --scope all --seed 2
python scripts/run_experiment.py --exp exploit --config BASE --br-iters 150 --seed 2
python scripts/run_experiment.py --exp mech --config Estar --iters 500 --seed 3
python scripts/run_experiment.py --exp mech --config EstarWidePub --iters 600 --seed 2
python scripts/parlgov_eval.py --tag mech_EstarWidePub --ideology --seed 2
python scripts/run_experiment.py --exp exploit --config BASE --br-iters 150 --seed 0
python scripts/run_experiment.py --exp exploit --config Estar --br-iters 150 --seed 1
python scripts/run_experiment.py --exp mech --config Estar --iters 500 --seed 4
python scripts/run_experiment.py --exp mech --algo mappo --config Estar --iters 500 --seed 3
python scripts/run_experiment.py --exp exploit --config Estar --br-iters 150 --seed 2
python scripts/run_experiment.py --exp e0 --iters 700 --seed 4
python scripts/run_experiment.py --exp e0 --iters 700 --seed 3
python scripts/run_experiment.py --exp e0 --iters 700 --seed 6
python scripts/run_experiment.py --exp exploit --config Estar --br-iters 150 --seed 0
python scripts/run_experiment.py --exp mech --config Estar --iters 500 --seed 6
python scripts/run_experiment.py --exp mech --algo mappo --config Estar --iters 500 --seed 4
python scripts/run_experiment.py --exp mech --config Estar --iters 500 --seed 5
python scripts/run_experiment.py --exp e0 --iters 700 --seed 5
python scripts/run_experiment.py --exp e0 --iters 700 --seed 8
python scripts/run_experiment.py --exp mech --config Estar --iters 500 --seed 8
python scripts/run_experiment.py --exp mech --algo mappo --config Estar --iters 500 --seed 7
python scripts/run_experiment.py --exp mech --config Estar --iters 500 --seed 7
python scripts/run_experiment.py --exp e0 --iters 700 --seed 7
python scripts/run_experiment.py --exp mech --algo mappo --config Estar --iters 500 --seed 5
python scripts/run_experiment.py --exp mech --algo mappo --config Estar --iters 500 --seed 6
python scripts/run_experiment.py --exp mech --config Estar --iters 500 --seed 9
python scripts/run_experiment.py --exp e0 --iters 700 --seed 9
python scripts/run_experiment.py --exp mech --config BASE+M4_private --iters 500 --seed 3
python scripts/run_experiment.py --exp mech --config BASE --iters 500 --seed 3
python scripts/run_experiment.py --exp mech --algo mappo --config Estar --iters 500 --seed 9
python scripts/run_experiment.py --exp mech --algo mappo --config Estar --iters 500 --seed 8
python scripts/run_experiment.py --exp mech --config BASE --iters 500 --seed 4
python scripts/run_experiment.py --exp mech --config BASE+M10_seatrho --iters 500 --seed 3
python scripts/run_experiment.py --exp mech --config BASE+M4_private --iters 500 --seed 4
python scripts/run_experiment.py --exp mech --config BASE+M9_policy --iters 500 --seed 3
python scripts/run_experiment.py --exp mech --config BASE+M9_policy --iters 500 --seed 4
python scripts/run_experiment.py --exp mech --rec --config Estar --iters 500 --seed 0
python scripts/run_experiment.py --exp mech --config BASE+M10_seatrho --iters 500 --seed 4
python scripts/run_experiment.py --exp mech --config BASE+M1_patient --iters 500 --seed 0
python scripts/run_experiment.py --exp mech --rec --config Estar --iters 500 --seed 1
python scripts/run_experiment.py --exp mech --rec --config Estar --iters 500 --seed 2
python scripts/crossplay.py --config Estar --runs mech_Estar mappo_Estar fsp_Estar --seeds 0 1 2
python scripts/run_experiment.py --exp mech --config BASE+M1_patient --iters 500 --seed 1
python scripts/run_experiment.py --exp mech --rec --config BASE+M1_patient --iters 500 --seed 0
python scripts/run_experiment.py --exp mech --rec --config BASE+M1_patient --iters 500 --seed 1
python scripts/run_experiment.py --exp mech --config BASE+M1_patient --iters 500 --seed 2
python scripts/run_experiment.py --exp mech --config Estar_m12 --iters 500 --seed 0
python scripts/run_experiment.py --exp mech --config Estar-M4_private --iters 500 --seed 2
python scripts/run_experiment.py --exp mech --config Estar-M8_exits --iters 500 --seed 2
python scripts/run_experiment.py --exp mech --config Estar-M1_repeat --iters 500 --seed 2
python scripts/run_experiment.py --exp mech --config Estar-M10_seatrho --iters 500 --seed 2
python scripts/run_experiment.py --exp mech --config Estar-M5_hetero --iters 500 --seed 2
python scripts/run_experiment.py --exp mech --config Estar-M9_policy --iters 500 --seed 2
python scripts/run_experiment.py --exp mech --config Estar_m12 --iters 500 --seed 1
python scripts/run_experiment.py --exp mech --rec --config BASE+M1_patient --iters 500 --seed 2
python scripts/run_experiment.py --exp mech --config Estar_m12 --iters 500 --seed 2
python scripts/run_experiment.py --exp exploit --config Estar --model-tag mappo_Estar --br-iters 150 --seed 0
python scripts/run_experiment.py --exp psro --config BASE --iters 6 --br-iters 150 --seed 0
python scripts/run_experiment.py --exp psro --config Estar --iters 6 --br-iters 150 --seed 0
python scripts/run_experiment.py --exp exploit --config Estar --br-iters 400 --seed 0
python scripts/reevaluate.py --episodes 3000
python scripts/run_experiment.py --exp mech --config E0demand --iters 500 --seed 0
python scripts/run_experiment.py --exp mech --config E0demand --iters 500 --seed 1
python scripts/run_experiment.py --exp mech --config E0demand --iters 500 --seed 2
python scripts/run_experiment.py --exp mech --config WideDemand --iters 600 --seed 0
python scripts/run_experiment.py --exp mech --config WideDemand --iters 600 --seed 1
python scripts/run_experiment.py --exp mech --config BASE --iters 2000 --tag long_BASE --seed 0
python scripts/run_experiment.py --exp mech --config BASE --iters 2000 --tag long_BASE --seed 1
python scripts/run_experiment.py --exp mech --config BASE+M4_private --iters 2000 --tag long_BASE+M4_private --seed 0
python scripts/run_experiment.py --exp mech --config WideProposal --iters 600 --seed 0
python scripts/run_experiment.py --exp mech --config BASE+M4_private --iters 2000 --tag long_BASE+M4_private --seed 1
python scripts/run_experiment.py --exp mech --config BASE+M9_policy --iters 2000 --tag long_BASE+M9_policy --seed 0
python scripts/run_experiment.py --exp mech --config E0demand --iters 1000 --ent-demand-mult 5 --tag explore_E0demand --seed 1
python scripts/run_experiment.py --exp mech --config Estar --iters 2000 --tag long_Estar --seed 0
python scripts/run_experiment.py --exp mech --config Estar --iters 2000 --tag long_Estar --seed 1
python scripts/run_experiment.py --exp mech --config E0demand --iters 1000 --ent-demand-mult 5 --tag explore_E0demand --seed 0
python scripts/run_experiment.py --exp mech --config WideDemand --iters 600 --seed 2
python scripts/run_experiment.py --exp mech --config WideProposal --iters 600 --seed 1
python scripts/run_experiment.py --exp mech --config WideProposal --iters 600 --seed 2
python scripts/run_experiment.py --exp mech --config BASE+M9_policy --iters 2000 --tag long_BASE+M9_policy --seed 1
python scripts/run_experiment.py --exp exploit --config E0demand --model-tag mech_E0demand --br-iters 300 --seed 0
python scripts/reevaluate.py --episodes 3000 --types-only --missing-only --only Wide
python scripts/reevaluate.py --episodes 3000 --types-only --missing-only --only Estar
python scripts/run_experiment.py --exp mech --config WideDemand --iters 600 --seed 3
python scripts/run_experiment.py --exp mech --config WideDemand --iters 600 --seed 4
python scripts/run_experiment.py --exp mech --config WideProposal --iters 600 --seed 4
python scripts/run_experiment.py --exp mech --config WideProposal --iters 600 --seed 3
python scripts/run_experiment.py --exp mech --config WideDemand --iters 600 --seed 5
python scripts/run_experiment.py --exp mech --config WideDemand --iters 600 --seed 6
python scripts/run_experiment.py --exp mech --config WideProposal --iters 600 --seed 5
python scripts/run_experiment.py --exp mech --config WideDemand --iters 600 --seed 7
python scripts/run_experiment.py --exp mech --config WideDemand --iters 600 --seed 8
python scripts/run_experiment.py --exp mech --config WideProposal --iters 600 --seed 6
python scripts/run_experiment.py --exp exploit --config E0demand --model-tag explore_E0demand --br-iters 300 --seed 0
python scripts/run_experiment.py --exp mech --config A452Demand --iters 600 --ent-demand-mult 5 --tag explore_A452Demand --seed 0
python scripts/run_experiment.py --exp mech --config WideDemand --iters 600 --seed 9
python scripts/run_experiment.py --exp mech --config A452Proposal --iters 600 --seed 0
python scripts/run_experiment.py --exp mech --config WideProposal --iters 600 --seed 7
python scripts/run_experiment.py --exp mech --config A4305Demand --iters 600 --ent-demand-mult 5 --tag explore_A4305Demand --seed 0
python scripts/run_experiment.py --exp mech --config A4305Proposal --iters 600 --seed 0
python scripts/run_experiment.py --exp mech --config WideProposal --iters 600 --seed 8
python scripts/run_experiment.py --exp mech --config A452Demand --iters 600 --ent-demand-mult 5 --tag explore_A452Demand --seed 1
python scripts/run_experiment.py --exp mech --config WideProposal --iters 600 --seed 9
python scripts/run_experiment.py --exp exploit --config WideDemand --model-tag mech_WideDemand --br-iters 400 --seed 1
python scripts/run_experiment.py --exp mech --config A4305Demand --iters 600 --ent-demand-mult 5 --tag explore_A4305Demand --seed 1
python scripts/run_experiment.py --exp mech --config A452Proposal --iters 600 --seed 1
python scripts/run_experiment.py --exp exploit --config WideDemand --model-tag mech_WideDemand --br-iters 400 --seed 0
python scripts/run_experiment.py --exp mech --config A452Demand --iters 600 --ent-demand-mult 5 --tag explore_A452Demand --seed 2
python scripts/run_experiment.py --exp mech --config A4305Proposal --iters 600 --seed 1
python scripts/run_experiment.py --exp mech --config A452Proposal --iters 600 --seed 2
python scripts/run_experiment.py --exp mech --config A4305Proposal --iters 600 --seed 2
python scripts/run_experiment.py --exp mech --config A4305Demand --iters 600 --ent-demand-mult 5 --tag explore_A4305Demand --seed 2
python scripts/run_experiment.py --exp mech --config WideDemand --iters 600 --ent-demand-mult 5 --tag explore_WideDemand --seed 0
python scripts/seki_eval.py --own WideDemand --tags mech_WideDemand --seeds 0 1 2 3 4 --episodes 64 --suffix _a
python scripts/seki_eval.py --own WideProposal --tags mech_WideProposal --seeds 0 1 2 3 4 --episodes 64 --suffix _a
python scripts/seki_eval.py --own WideDemand --tags mech_WideDemand --seeds 5 6 7 8 9 --episodes 64 --suffix _b
python scripts/seki_eval.py --own EstarWide --tags mech_EstarWide --seeds 0 1 2 --episodes 64 --suffix _final
python scripts/seki_eval.py --own WideProposal --tags mech_WideProposal --seeds 5 6 7 8 9 --episodes 64 --suffix _b
python scripts/seki_eval.py --tags mech_EstarWide mech_EstarWidePub --demand-tags mech_WideDemand --seeds 0 1 2 --samples 32
python scripts/run_experiment.py --exp mech --config WideDemand --iters 600 --ent-demand-mult 5 --tag explore_WideDemand --seed 2
python scripts/run_experiment.py --exp exploit --config E0 --model-tag e0 --br-iters 300 --seed 0
python scripts/run_experiment.py --exp mech --config WideDemand --iters 600 --ent-demand-mult 5 --tag explore_WideDemand --seed 1
python scripts/seki_eval.py --own WideDemand --tags explore_WideDemand --seeds 0 1 2 --episodes 64 --suffix _explore
python scripts/run_experiment.py --exp exploit --config WideProposal --model-tag mech_WideProposal --br-iters 400 --seed 1
python scripts/proposal_power_eval.py --tag explore_A4305Demand --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag explore_A452Demand --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag mech_WideDemand --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag entropy_BASE_0.05 --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag e1 --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag entropy_BASE_0.02 --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag explore_E0demand --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag entropy_BASE_0.002 --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag explore_WideDemand --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag long_BASE+M4_private --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag long_BASE --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag long_BASE+M9_policy --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag mech_A4305Proposal --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag mech_A452Proposal --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag long_Estar --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag fsp_Estar --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag mech_EstarWide --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag mech_BASE+M5_hetero --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag mech_BASE+M1_patient --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag mech_BASE+M1_repeat --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag mech_BASE+M10_seatrho --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag mech_E0demand --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag mech_WideProposal --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag mech_EstarWidePub --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag mech_BASE+M4_private --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag mech_BASE --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag mech_BASE+M8_exits --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag mech_Estar-M8_exits --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag mech_Estar --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag mappo_Estar --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag mech_Estar-M1_repeat --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag mech_Estar-M10_seatrho --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag mech_Estar-M4_private --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag mech_BASE+M9_policy --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag rec_BASE+M1_patient --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag mech_Estar-M9_policy --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag mech_Estar_Lam0 --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag mech_Estar-M5_hetero --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag mech_Estar_m12 --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag rec_Estar --states 2000 --rep 4
python scripts/proposal_power_eval.py --tag e0 --states 2000 --rep 4

# ---- analyses, figures, report numbers and paper tables
python scripts/benchmark_table.py
python scripts/bf_best_response.py
python scripts/confirmatory.py
python scripts/demand_solver_check.py
python scripts/key_ministries.py
python scripts/real_robustness.py
python scripts/real_weights_vs_seats.py
python scripts/real_within_party.py
python scripts/psi_summary.py
python scripts/make_figures.py
python scripts/paper_tables.py
