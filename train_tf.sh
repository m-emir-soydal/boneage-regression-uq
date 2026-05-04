#!/usr/bin/env bash
# Train the TensorFlow bone-age model by executing the notebook.
#
# Usage:
#   ./train_tf.sh
#   TF_GPUS=2,3 CONDA_ENV=boneageuq ./train_tf.sh

set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE"

CONDA_ENV="${CONDA_ENV:-boneageuq}"
TF_GPUS="${TF_GPUS:-2,3}"
BONE_AGE_DATA_ROOT="${BONE_AGE_DATA_ROOT:-$WORKSPACE/data/rsna_training}"
TF_DIR="$WORKSPACE/tf-pediatric-bone-age"
TF_NOTEBOOK="$TF_DIR/notebooks/RSNA_Pediatric_Bone_Age_Estimation.ipynb"
TF_OUT="${BONE_AGE_TF_OUTPUT_DIR:-$TF_DIR/outputs/rsna_boneage_models}"
LOG_DIR="$WORKSPACE/_logs"

mkdir -p "$LOG_DIR" "$TF_OUT"

if ! command -v conda >/dev/null; then
  echo "ERROR: conda is required but not on PATH." >&2
  exit 1
fi

if [[ ! -d "$BONE_AGE_DATA_ROOT" ]]; then
  echo "ERROR: BONE_AGE_DATA_ROOT does not exist: $BONE_AGE_DATA_ROOT" >&2
  exit 1
fi

if [[ ! -f "$TF_NOTEBOOK" ]]; then
  echo "ERROR: TensorFlow notebook does not exist: $TF_NOTEBOOK" >&2
  exit 1
fi

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
  # shellcheck disable=SC1091
  source "$(dirname "$(dirname "$CONDA_PREFIX")")/etc/profile.d/conda.sh"
else
  echo "ERROR: cannot locate conda.sh" >&2
  exit 1
fi

conda activate "$CONDA_ENV"

export BONE_AGE_DATA_ROOT="$BONE_AGE_DATA_ROOT"
export BONE_AGE_OUTPUT_DIR="$TF_OUT"
export CUDA_VISIBLE_DEVICES="$TF_GPUS"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2

cd "$TF_DIR"
echo "[TensorFlow] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES OUTPUT_DIR=$BONE_AGE_OUTPUT_DIR"
ts="$(date +%Y%m%d_%H%M%S)"
log="$LOG_DIR/train_tf_${ts}.log"
ln -sf "$log" "$LOG_DIR/train_tf.log"
exec_nb="$TF_OUT/RSNA_Pediatric_Bone_Age_Estimation.executed.ipynb"
#region agent log
python -c "import json,time; p='/home/spacing/Emir/Boneage UQ/.cursor/debug-395fbc.log'; open(p,'a',encoding='utf-8').write(json.dumps({'sessionId':'395fbc','runId':'pre-fix','hypothesisId':'H3_H5','location':'train_tf.sh:before_nbconvert','message':'Starting TF nbconvert execution','data':{'tf_notebook':'$TF_NOTEBOOK','exec_nb':'$exec_nb','cuda_visible_devices':'$TF_GPUS','conda_env':'$CONDA_ENV'},'timestamp':int(time.time()*1000)}) + '\n')"
#endregion
jupyter nbconvert --to notebook --execute --output "$exec_nb" --ExecutePreprocessor.timeout=-1 "$TF_NOTEBOOK" 2>&1 | tee "$log"
exit_code="${PIPESTATUS[0]}"
#region agent log
python -c "import json,time,os; p='/home/spacing/Emir/Boneage UQ/.cursor/debug-395fbc.log'; open(p,'a',encoding='utf-8').write(json.dumps({'sessionId':'395fbc','runId':'pre-fix','hypothesisId':'H3_H5','location':'train_tf.sh:after_nbconvert','message':'Finished TF nbconvert execution','data':{'exit_code':$exit_code,'exec_nb_exists':os.path.isfile('$exec_nb'),'log_path':'$log'},'timestamp':int(time.time()*1000)}) + '\n')"
#endregion
echo "[TensorFlow] finished with exit=$exit_code (executed nb: $exec_nb)"
exit "$exit_code"
