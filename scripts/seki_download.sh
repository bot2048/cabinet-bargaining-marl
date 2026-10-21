#!/usr/bin/env bash
# Download the Seki-Williams Government and Ministers Data v2.0 (Harvard
# Dataverse, doi:10.7910/DVN/0UNUAM, CC0 1.0) used by scripts/key_ministries.py.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/seki_williams/raw
curl -sSL -o data/seki_williams/raw/governments_v2.dta "https://dataverse.harvard.edu/api/access/datafile/2886779?format=original"
curl -sSL -o data/seki_williams/raw/ministers_v2.dta "https://dataverse.harvard.edu/api/access/datafile/2886956?format=original"
sha256sum data/seki_williams/raw/*.dta
# expected: f46edd67...a5c (governments_v2.dta), dcb29421...a5c (ministers_v2.dta); see data/seki_williams/NOTICE.md
