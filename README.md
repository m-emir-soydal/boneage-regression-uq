# RSNA Pediatric Bone Age Estimation with Uncertainty Quantification

**Reference Original Repository:** [Jayaashri/pediatric-bone-age-estimation](https://github.com/Jayaashri/pediatric-bone-age-estimation)

This repository contains a deep learning pipeline for estimating pediatric bone age from hand radiographs using the [RSNA Bone Age dataset](https://www.kaggle.com/datasets/kmader/rsna-bone-age). The project focuses on building robust regression models and quantifying their prediction uncertainty using Monte Carlo (MC) Dropout.

## Project Overview

Bone age assessment is a critical task in pediatric radiology, used to diagnose growth disorders and estimate skeletal maturity. This project implements a state-of-the-art regression pipeline using TensorFlow and EfficientNet, featuring:
- A **Baseline** model utilizing only image data.
- A **Multi-Input** model that combines image features with patient sex for improved accuracy.
- **Uncertainty Quantification (UQ)** to provide confidence intervals for every prediction, enhancing clinical reliability.

## Data Preprocessing

The preprocessing logic is detailed in the `RSNA_Pediatric_Bone_Age_Estimation.ipynb` Jupyter notebook. Key steps include:

1.  **Exploratory Data Analysis (EDA):**
    - Analysis of bone age distribution (ranging from newborns to 228 months), identifying a slight right skew.
    - Verification of sex distribution, which is approximately balanced (Male/Female).
    - Visualization of age-by-sex distributions using histograms and boxplots.
2.  **Feature Engineering:**
    - **Sex Encoding:** The `Sex` column is binary-encoded (0/1) to be used as a numerical feature.
    - **Target Normalization:** Bone age values are normalized for stable regression training.
3.  **Image Pipeline:**
    - Images are resized to **300x300** pixels.
    - **EfficientNet-B3 Preprocessing:** Specialized input scaling and normalization are applied to match the requirements of the pre-trained backbone.
    - Data augmentation (rotation, flipping, zooming) is used during training to improve generalization.

## Model Architectures

The project evaluates two primary model architectures:

-   **Baseline Model:** Uses an **EfficientNet-B3** backbone pre-trained on ImageNet. The final global average pooling layer is followed by dense layers to regress a single bone age value.
-   **Multi-Input Model:** This architecture takes two inputs:
    1.  The radiograph (processed via EfficientNet-B3).
    2.  The patient's sex (as a binary feature).
    The image features and the sex feature are concatenated before the final regression layers. This allows the model to leverage the physiological differences in bone maturation rates between sexes.

## Monte Carlo Dropout & Uncertainty Quantification

In clinical settings, knowing *how confident* a model is can be as important as the prediction itself. This project implements **Monte Carlo (MC) Dropout** as proposed by [Gal & Ghahramani (2016)](https://arxiv.org/abs/1506.02142).

### Implementation
While standard Dropout is only active during training, MC Dropout keeps it active during inference. In this project, the `mc_dropout_inference.py` script performs $T=50$ stochastic forward passes for each image by setting `training=True` during the model call.

### Usage in this Project
For every test image, the model generates a distribution of predictions rather than a single point estimate:
-   **Predictive Mean:** The average of the stochastic passes is used as the final bone age estimate.
-   **Predictive Standard Deviation:** The variance among the passes represents the model's **epistemic uncertainty**. Higher standard deviation indicates images that the model finds difficult or ambiguous.

### Evaluation Metrics
The project evaluates the UQ performance using:
-   **PICP (Prediction Interval Coverage Probability):** The percentage of true values that fall within the predicted 95% confidence interval.
-   **MPIW (Mean Prediction Interval Width):** The average width of the confidence intervals.
-   **ECE (Expected Calibration Error):** Measures how well the predicted uncertainty aligns with the actual error.

## Reproduction Guide

Follow these steps to reproduce the results:

### 1. Environment Setup
Ensure you have Conda installed, then create the environment:
```bash
conda create -n boneageuq python=3.10
conda activate boneageuq
pip install -r tf-pediatric-bone-age/requirements.txt
```

### 2. Data Preparation
Download the RSNA dataset and place it in `data/rsna_training`. Ensure the structure includes `train.csv` and the `boneage-training-dataset` folder containing the images.

### 3. Training
To train the models (this will execute the Jupyter notebook and save the final `.keras` files):
```bash
./train_tf.sh
```

### 4. Inference & UQ Evaluation
To run the MC Dropout inference and generate the UQ metrics:
```bash
./test_uq_metrics.sh
```
The results will be saved in `tf-pediatric-bone-age/outputs/rsna_boneage_models/` as CSV files.
