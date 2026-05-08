#!/usr/bin/env bash
# Run post-training conformal prediction evaluation for TensorFlow models.
# Requires trained model artifacts to already exist.
#
# Default behavior:
#   - TensorFlow: variants {baseline, multi}
#
# Usage examples:
#   ./test_conformal_metrics.sh
#   LEVELS=0.9,0.95 ./test_conformal_metrics.sh
#   ./test_conformal_metrics.sh --limit 200
#   BONE_AGE_DATA_ROOT=/path/to/rsna_training ./test_conformal_metrics.sh

set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE"

LIMIT_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --limit requires a numeric value." >&2
        exit 2
      fi
      LIMIT_ARG="--limit $2"
      shift 2
      ;;
    -h|--help)
      sed -n '1,15p' "$0"
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      echo "Try: $0 --help" >&2
      exit 2
      ;;
  esac
done

CONDA_ENV="${CONDA_ENV:-boneageuq}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-42}"
LEVELS="${LEVELS:-0.90,0.95}"
SPLITS="${SPLITS:-test}"

TF_DIR="$WORKSPACE/tf-pediatric-bone-age"
TF_OUT="${BONE_AGE_TF_OUTPUT_DIR:-$TF_DIR/outputs/rsna_boneage_models}"
LOG_DIR="$WORKSPACE/_logs"
mkdir -p "$LOG_DIR" "$TF_OUT"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is required but not on PATH." >&2
  exit 1
fi

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: cannot locate conda.sh for activation." >&2
  exit 1
fi

conda activate "$CONDA_ENV"

run_tf() {
  local variant="$1"
  local ts log
  ts="$(date +%Y%m%d_%H%M%S)"
  log="$LOG_DIR/conformal_tf_${variant}_${ts}.log"
  echo "[TF][$variant] Running conformal inference (log: $log)"
  (
    cd "$TF_DIR"
    python scripts/conformal_inference.py \
      --variant "$variant" \
      --batch-size "$BATCH_SIZE" \
      --seed "$SEED" \
      --levels "$LEVELS" \
      --splits "$SPLITS" \
      --output-dir "$TF_OUT" \
      $LIMIT_ARG
  ) 2>&1 | tee "$log"
}

echo "Starting post-training conformal evaluation..."
echo "  conda env : $CONDA_ENV"
echo "  levels    : $LEVELS"
echo "  splits    : $SPLITS"
echo "  tf output : $TF_OUT"

run_tf baseline
run_tf multi

echo
echo "Done. Expected summary CSVs:"
echo "  $TF_OUT/conformal_metrics_baseline.csv"
echo "  $TF_OUT/conformal_metrics_multi.csv"
