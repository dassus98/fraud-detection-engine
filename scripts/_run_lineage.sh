#!/usr/bin/env bash
set -euo pipefail
cd /home/dchit/projects/fraud-detection-engine
uv run python -m pytest tests/lineage --no-cov -v > logs/lineage_verify.log 2>&1
echo "EXIT=$?" >> logs/lineage_verify.log
