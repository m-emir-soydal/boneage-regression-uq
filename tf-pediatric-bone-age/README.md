# Pediatric Bone Age Estimation (RSNA Dataset)

Deep learning-based regression system to predict pediatric bone age (in months) from hand X-ray images using EfficientNet and multi-input modeling.

---

## Project Overview

Bone age assessment is critical for diagnosing pediatric growth disorders, but traditional methods are:
- Subjective (high inter-observer variability)
- Time-consuming (10–15 minutes per image)

This project builds an automated, deep learning-based system to estimate bone age from radiographs, improving consistency and efficiency.

---

## Key Results

| Model                  | MAE (months) | RMSE | R²   |
|-----------------------|-------------|------|------|
| Baseline (Image Only) | 4.47        | 5.78 | 0.90 |
| Multi-Input (+ Sex)   |   3.82      | 4.98 | 0.917 |

- ✅ ~15% reduction in error using multi-input model  
- ✅ ~80% predictions within ±6 months  
- ✅ ~98% predictions within ±12 months  

---
## Visual Results

### Training Curves
![MAE Curve](outputs/training_mae_curve.png)
![MSE Curve](outputs/training_mse_curve.png)

### Error Analysis
![Baseline Error](outputs/baseline_error_histogram.png)
![Multi Input Error](outputs/multi_input_error_histogram.png)
![Cumulative Error](outputs/cumulative_error_curve.png)

### Sample Predictions
![Predictions](outputs/sample_predictions.png)
---

##  Model Architecture

###  Baseline Model
- EfficientNet-B3 (ImageNet pretrained)
- Global Average Pooling
- Dense (256) + Dropout
- Output: Bone age regression

###  Multi-Input Model (Improved)
- Image input (EfficientNet-B3)
- + Patient sex as auxiliary input
- Feature fusion via concatenation
- Dense layers for regression

Insight: Incorporating sex helps model learn sex-specific growth patterns, improving accuracy.

---

## Data Pipeline

Built using TensorFlow tf.data for efficient training:

- Image resizing: `300 x 300`
- Normalization + EfficientNet preprocessing
- Data augmentation:
  - Rotation (±20°)
  - Translation (±10%)
  - Zoom (±10%)
  - Brightness adjustment
- Optimizations:
  - Caching
  - Prefetching
  - Parallel mapping

---

## Dataset

- RSNA Pediatric Bone Age Dataset (Kaggle)
- 12,611 labeled X-ray images
- Labels:
  - Bone age (months)
  - Patient sex

- Age range: 1–228 months
- Distribution: ~Gaussian centered ~11 years
---

## Training Details

- Loss: Mean Absolute Error (MAE)
- Optimizer: Adam
- Batch size: 32
- Early stopping + LR scheduling
- Model checkpointing

---

## Error Analysis

- Error distribution ~Gaussian centered near 0 
- Majority of predictions fall within clinically acceptable range  
- Higher errors at:
  - Very young ages
  - Older adolescents
  - Noisy / cropped images

---

##  Key Insights

- CNNs can learn fine-grained ossification patterns from raw X-rays  
- Adding structured metadata (sex) significantly improves performance  
- EfficientNet provides strong performance with manageable compute cost  

---

##  Limitations

- No segmentation → background noise affects predictions  
- Limited metadata (only sex used)  
- Performance drops at extreme age ranges  

---

##  Future Work

- Add segmentation model (U-Net) to isolate hand region  
- Incorporate more metadata (height, weight, ethnicity)  
- Explore CNN + Transformer hybrid models  
- Deploy as API (FastAPI) for real-world use

---

##  Tech Stack

- Python
- TensorFlow / Keras
- EfficientNet
- NumPy, Pandas, Matplotlib

---

##  References

- RSNA Bone Age Dataset (Kaggle)
- TensorFlow Image Models

---

##  Author

Jayaashri Chezhian  
MS Computer Science
