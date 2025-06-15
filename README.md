# 🔍 Hyperparameter Tuning Job on AWS SageMaker (Sklearn + XGBoost)

This repository contains a simple setup to run a **Hyperparameter Tuning (HPT) job** on **Amazon SageMaker**, using custom container and the `HyperparameterTuner` class from the SageMaker SDK.  
The goal is to tune an **XGBoost** classifier using F1-score as the optimization metric.

---

## 🚀 Features

- Uses a custom container in SageMaker  
- Tunes two hyperparameters: `n_estimators` and `max_depth`  
- Custom training logic defined in `train.py`  
- Evaluation uses `f1_score` as the optimization objective  
- Results automatically saved to **S3**  
- Compatible with `ml.m5.large` instances (adjustable)

---

## ⚙️ How It Works

### 1. `train.py`

- Reads CLI arguments from SageMaker (`year`, `month`, `n_estimators`, `max_depth`)
- (Simplified) Prints `f1_score: 0.77` to simulate evaluation
- SageMaker captures the metric using regex and logs it for tuning

> You can replace this with a real training pipeline reading datasets from S3 and training a model with `xgboost.XGBClassifier`.

---

### 2. `launch_hpt_job_aws.py`

This script:

- Sets up the estimator using SageMaker's official `scikit-learn` container (`framework_version="1.2-1"`)
- Defines static and tunable hyperparameters
- Uses `HyperparameterTuner` to explore combinations of:
  - `n_estimators` ∈ [2, 10]
  - `max_depth` ∈ [2, 10]
- Launches the HPT job with up to 4 trials and 2 running in parallel
- Stores outputs in S3 under:  
  `s3://proyecto-1-ml/output/`

Run it with:

```bash
python launch_hpt_job_aws.py
```

---
