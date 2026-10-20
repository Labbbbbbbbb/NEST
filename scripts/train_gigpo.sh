#!/usr/bin/env bash
# NEST entry point; the original GiGPO implementation remains unchanged.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ENV_NAME="${1:-alfworld}"
if [[ $# -gt 0 ]]; then shift; fi
case "$ENV_NAME" in
  alfworld|webshop|sokoban) ;;
  *) echo "Supported wrapper environments: alfworld, webshop, sokoban" >&2; exit 2 ;;
esac
exec bash -e "examples/gigpo_trainer/run_${ENV_NAME}.sh" vllm \
  trainer.project_name=nest \
  trainer.experiment_name="gigpo_${ENV_NAME}" \
  'trainer.logger=[console]' "$@"
