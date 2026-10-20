#!/usr/bin/env bash
# Download the ParlGov tables used by scripts/parlgov_prepare.py (CC-BY-SA 4.0).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/parlgov/raw
for t in view_cabinet view_election; do
  curl -sSL --max-time 120 -o "data/parlgov/raw/$t.csv" "https://parlgov.org/data/parlgov-development_csv-utf-8/$t.csv"
done
ls -la data/parlgov/raw
