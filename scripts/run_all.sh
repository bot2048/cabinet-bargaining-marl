#!/usr/bin/env bash
# Run every experiment of the report.  Jobs run in parallel (JOBS at a time,
# THREADS PyTorch threads each); each job writes results/<tag>/seed<k>.json
# and models/<tag>/seed<k>.pt.  Re-running skips cells whose result exists.
#
#   JOBS=6 THREADS=2 SEEDS="0 1 2" bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python )}
JOBS=${JOBS:-6}
THREADS=${THREADS:-2}
SEEDS=${SEEDS:-"0 1 2"}
ITERS_E0=${ITERS_E0:-700}
ITERS_MECH=${ITERS_MECH:-500}
mkdir -p runs/logs

jobs_file=$(mktemp)
add() {  # add <result tag> <args...>
  local tag=$1; shift
  local seed=${@: -1}
  [ -f "results/$tag/seed$seed.json" ] && return 0
  echo "$PY scripts/run_experiment.py $* --threads $THREADS > runs/logs/${tag}_s${seed}.log 2>&1" >> "$jobs_file"
}

for s in $SEEDS; do
  # RQ1: benchmark recovery
  add e0 --exp e0 --iters "$ITERS_E0" --seed "$s"
  add e1 --exp e1 --iters "$ITERS_E0" --seed "$s"
  # RQ2: add-one-in from the base configuration, full E*, memoryless E*
  for c in BASE BASE+M1_repeat BASE+M4_private BASE+M5_hetero BASE+M8_exits \
           BASE+M9_policy BASE+M10_seatrho Estar Estar_Lam0; do
    add "mech_$c" --exp mech --config "$c" --iters "$ITERS_MECH" --seed "$s"
  done
done
# RQ2 necessity: leave-one-out from E* (two seeds)
for s in 0 1; do
  for m in M1_repeat M4_private M5_hetero M8_exits M9_policy M10_seatrho; do
    add "mech_Estar-$m" --exp mech --config "Estar-$m" --iters "$ITERS_MECH" --seed "$s"
  done
done
# H9: entropy temperature (bounded rationality)
for s in 0 1; do
  for e in 0.002 0.02 0.05; do
    add "entropy_BASE_$e" --exp entropy --config BASE --ent-end "$e" --iters "$ITERS_MECH" --seed "$s"
  done
done
# scripted baselines
[ -f results/baselines/seed0.json ] || echo "$PY scripts/run_experiment.py --exp baselines > runs/logs/baselines.log 2>&1" >> "$jobs_file"

echo "$(wc -l < "$jobs_file") jobs, $JOBS in parallel"
xargs -P "$JOBS" -I CMD bash -c CMD < "$jobs_file"

# exploitability needs the trained models
ex_file=$(mktemp)
for c in BASE Estar; do
  for s in 0 1 2; do
    [ -f "results/exploit_$c/seed$s.json" ] || \
      echo "$PY scripts/run_experiment.py --exp exploit --config $c --seed $s --br-iters 150 --threads $THREADS > runs/logs/exploit_${c}_s$s.log 2>&1" >> "$ex_file"
  done
done
xargs -P "$JOBS" -I CMD bash -c CMD < "$ex_file"
rm -f "$jobs_file" "$ex_file"
$PY scripts/make_figures.py
