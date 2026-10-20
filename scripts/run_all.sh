#!/usr/bin/env bash
# Queue every experiment of the report and start the queue runner.
# Jobs whose result already exists (or that are already queued) are skipped,
# so this is safe to re-run.  Progress: bash scripts/status.sh
#
#   SEEDS="0 1 2" bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )
SEEDS=${SEEDS:-"0 1 2"}
ITERS_E0=${ITERS_E0:-700}
ITERS_MECH=${ITERS_MECH:-500}
EST_E0=${EST_E0:-60}        # estimated minutes per job (used for the ETA only)
EST_MECH=${EST_MECH:-55}
EST_EXPLOIT=${EST_EXPLOIT:-70}

q() {  # q <tag> <est> <run_experiment args...>
  local tag=$1 est=$2; shift 2
  local seed=${@: -1}
  local res=${RES:-$tag}
  [ -f "results/$res/seed$seed.json" ] && return 0
  $PY scripts/jobs.py add "${tag}_s$seed" "$est" -- "$@" > /dev/null
}

for s in $SEEDS; do
  # RQ1: benchmark recovery
  q e0 "$EST_E0" --exp e0 --iters "$ITERS_E0" --seed "$s"
  q e1 "$EST_E0" --exp e1 --iters "$ITERS_E0" --seed "$s"
  # RQ2: add-one-in from the base configuration, full E*, memoryless E*
  for c in BASE Estar BASE+M1_repeat BASE+M4_private BASE+M5_hetero BASE+M8_exits \
           BASE+M9_policy BASE+M10_seatrho Estar_Lam0; do
    q "mech_$c" "$EST_MECH" --exp mech --config "$c" --iters "$ITERS_MECH" --seed "$s"
  done
done
# RQ6: learning algorithms on E* (LA2 MAPPO, LA3 fictitious self-play)
for s in $SEEDS; do
  for a in mappo fsp; do
    q "${a}_Estar" "$EST_MECH" --exp mech --algo "$a" --config Estar --iters "$ITERS_MECH" --seed "$s"
  done
done
# RQ2 necessity: leave-one-out from E* (two seeds)
for s in 0 1; do
  for m in M1_repeat M4_private M5_hetero M8_exits M9_policy M10_seatrho; do
    q "mech_Estar-$m" "$EST_MECH" --exp mech --config "Estar-$m" --iters "$ITERS_MECH" --seed "$s"
  done
done
# H9: entropy temperature (bounded rationality)
for s in 0 1; do
  for e in 0.002 0.02 0.05; do
    q "entropy_BASE_$e" "$EST_MECH" --exp entropy --config BASE --ent-end "$e" --iters "$ITERS_MECH" --seed "$s"
  done
done
# exploitability (waits for the trained model) and scripted baselines
for c in BASE Estar; do
  for s in $SEEDS; do
    q "exploit_$c" "$EST_EXPLOIT" --exp exploit --config "$c" --br-iters 150 --seed "$s"
  done
done
q baselines 10 --exp baselines --seed 0

$PY scripts/jobs.py list | tail -1
$PY scripts/jobs.py start
