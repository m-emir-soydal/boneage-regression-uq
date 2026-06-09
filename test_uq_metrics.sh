#!/usr/bin/env bash
# Run post-training uncertainty quantification (MC-dropout) for the TensorFlow project.
# Requires trained model artifacts to already exist.
#
# Default behavior:
#   - TensorFlow: variants {baseline, multi}
#
# Usage examples:
#   ./test_uq_metrics.sh
#   N_SAMPLES=100 ./test_uq_metrics.sh
#   ./test_uq_metrics.sh --limit 200
#   BONE_AGE_DATA_ROOT=/path/to/rsna_training ./test_uq_metrics.sh

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
N_SAMPLES="${N_SAMPLES:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-42}"

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
elif command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "ERROR: cannot locate conda.sh for activation." >&2
  exit 1
fi

conda activate "$CONDA_ENV"

run_tf() {
  local variant="$1"
  local model_name
  if [[ "$variant" == "baseline" ]]; then
    model_name="baseline_final.keras"
  else
    model_name="multi_input_final.keras"
  fi
  local model_path="$TF_OUT/$model_name"
  if [[ ! -f "$model_path" ]]; then
    echo "[TF][$variant] SKIP: missing model file: $model_path"
    return 0
  fi

  local ts log
  ts="$(date +%Y%m%d_%H%M%S)"
  log="$LOG_DIR/uq_tf_${variant}_${ts}.log"
  echo "[TF][$variant] Running MC-dropout inference (log: $log)"
  (
    cd "$TF_DIR"
    python scripts/mc_dropout_inference.py \
      --variant "$variant" \
      --n-samples "$N_SAMPLES" \
      --batch-size "$BATCH_SIZE" \
      --seed "$SEED" \
      --output-dir "$TF_OUT" \
      $LIMIT_ARG
  ) 2>&1 | tee "$log"
}

echo "Starting post-training UQ evaluation..."
echo "  conda env : $CONDA_ENV"
echo "  n_samples : $N_SAMPLES"
echo "  tf output : $TF_OUT"

run_tf baseline
run_tf multi

echo
echo "Done. Expected summary CSVs:"
echo "  $TF_OUT/uq_metrics_baseline.csv"
echo "  $TF_OUT/uq_metrics_multi.csv"
