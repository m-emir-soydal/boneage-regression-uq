# Pediatric Bone Age Estimation with Uncertainty Quantification

This project implements a deep learning-based system to estimate pediatric bone age from hand X-ray images using the RSNA dataset. It features both a regression model for age prediction and a Monte-Carlo (MC) Dropout mechanism for Uncertainty Quantification (UQ).

## Project Structure

- `tf-pediatric-bone-age/`: Core TensorFlow implementation.
  - `notebooks/`: Contains the training logic in Jupyter notebooks.
  - `scripts/`: Python scripts for inference and UQ metrics.
  - `outputs/`: Directory where trained models and evaluation results are saved.
- `data/`: Expected location for the RSNA dataset.
- `train_tf.sh`: Shell script to automate the training process.
- `test_uq_metrics.sh`: Shell script to run MC Dropout inference and generate UQ metrics.
- `_logs/`: Execution logs for training and evaluation.

## Getting Started

### Prerequisites

The project uses a Conda environment. Ensure you have Conda installed.
```bash
# Recommended environment name: boneageuq
conda create -n boneageuq python=3.10
conda activate boneageuq
pip install -r tf-pediatric-bone-age/requirements.txt
```

### Data Setup (Manual Step)

**Note: The `data/` directory is not included in this repository.** To run the training and evaluation, you must manually download the [RSNA Bone Age Dataset](https://www.kaggle.com/datasets/kmader/rsna-bone-age) and place it in the `data/rsna_training` directory.

The directory should contain:
- `boneage-training-dataset/`: Folder with X-ray images.
- `train.csv`: CSV file with `id`, `boneage`, and `male` (boolean) columns.

## Workflow

### 1. Training the Model

The model is trained using TensorFlow and EfficientNet-B3. Two variants are supported:
- **Baseline**: Image only.
- **Multi-Input**: Image + Patient sex (improves accuracy).

To train the model, run:
```bash
./train_tf.sh
```
This script executes the training notebook and saves the final models as `.keras` files in `tf-pediatric-bone-age/outputs/rsna_boneage_models/`.

### 2. Uncertainty Quantification (MC Dropout)

After training, the project uses **Monte-Carlo Dropout** to estimate the uncertainty of the predictions.

#### How it works:
Standard dropout is usually only active during training. In this project, we keep dropout active during inference (`training=True`). By performing multiple forward passes ($T=50$ by default) for the same image, we obtain a distribution of predictions.
- **Mean Prediction**: The average of all stochastic passes, used as the final age estimate.
- **Standard Deviation**: Represents the model's uncertainty for that specific image.

#### Cooperation between TF and MC scripts:
1. The **Training Notebook** saves a standard Keras model that includes Dropout layers.
2. The **MC Dropout Script** (`mc_dropout_inference.py`) loads this model.
3. It uses a custom prediction loop that calls `model(inputs, training=True)` to ensure dropout layers remain stochastic.
4. It calculates UQ metrics like **PICP** (how often the true age falls within the predicted interval) and **MPIW** (average width of the uncertainty interval).

### 3. Running UQ Evaluation

To run the MC Dropout inference and generate metrics:
```bash
./test_uq_metrics.sh
```
This will produce:
- `mc_dropout_<variant>_<split>.csv`: Per-image predictions and uncertainty.
- `uq_metrics_<variant>.csv`: Summary metrics (PICP, MPIW, ECE).

## Results

The Multi-Input model typically achieves:
- **MAE**: ~3.8 months
- **RMSE**: ~5.0 months
- **R²**: ~0.92

UQ evaluation helps identify images where the model is less confident, which is crucial for clinical applications.

---
*Refactored for clarity and ease of use.*
