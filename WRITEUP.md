# Customer Churn Prediction - Technical Writeup

## Scope
This writeup reflects the current implementation in:
- `train.py`
- `src/feature_engineering.py`
- `src/model_training.py`
- `src/predict.py`

It documents the current process and the latest recorded results already present in the repository.

## Current End-to-End Process

### 1. Data loading and validation
- Entry point: `train.py`
- Data source: `data/Churn_Modelling.csv`
- Validation and ingestion handled by `src/data_ingestion.py`.

### 2. Feature engineering
- Implemented in `src/feature_engineering.py`.
- Engineered features include:
  - `AgeGroup`
  - `TenureGroup`
  - `HasBalance`
  - `HighBalance`
  - `ProductEngagement`
  - `ActivityIndex`
  - `CreditScoreCategory`
  - `SalaryToBalanceRatio`
  - `ProductsPerTenure`
- Non-predictive columns removed: `RowNumber`, `CustomerId`, `Surname`.

### 3. Preprocessing (leakage-safe)
- Preprocessing is fit on train split only in `train_pipeline`.
- Numeric features: `StandardScaler`
- Categorical features: `OrdinalEncoder`
- Transform is then applied consistently to train/validation/test splits.

### 4. Data split
- Implemented in `src/model_training.py` via `split_data_flexible`.
- Supports:
  - Train/Test (default path when `use_validation=False`)
  - Train/Validation/Test (when `use_validation=True`)
- Uses stratified splitting.

### 5. Class imbalance handling
Supported strategies:
- `class_weight`
- `smote`
- `undersampling`
- `combined`

Current `train.py` run configuration uses:
- `imbalance_strategy='undersampling'`

### 6. Model training
Three models are trained:
- Logistic Regression
- Random Forest
- XGBoost

Optional capabilities in pipeline:
- Cross-validation (`use_cross_validation=True`)
- Grid search (`use_grid_search=True`)

Current `train.py` run configuration uses:
- `use_cross_validation=True`
- `use_grid_search=False`

### 7. Model selection and final evaluation
- Winner selected by highest ROC-AUC from configured evaluation step.
- Final test metrics are printed via `test_model()`.

### 8. Fit diagnosis
After test evaluation, the pipeline now prints:
- Best-model train metrics
- Fit diagnosis: `overfitting`, `underfitting`, or `good fit`
- Gaps:
  - ROC-AUC gap (Train - Validation/Test)
  - F1 gap (Train - Validation/Test)

This is implemented in:
- `evaluate_split`
- `assess_fit_quality`
- `train_pipeline` Step 8

### 9. Artifacts
Saved artifacts:
- `models/best_model_<model_name>.pkl`
- `models/preprocessor.pkl`

## Latest Recorded Results

### Best model summary
- Best model: XGBoost
- Fit status: good fit

### Train vs Test
| Metric | Train | Test |
|---|---:|---:|
| ROC-AUC | 0.8983 | 0.8608 |
| F1-Score | 0.6375 | 0.5858 |
| Precision | 0.5228 | 0.4722 |
| Recall | 0.8166 | 0.7715 |
| Accuracy | 0.8107 | 0.7780 |

Gap summary:
- ROC-AUC gap (Train - Test): 0.0375
- F1 gap (Train - Test): 0.0516

### Test confusion matrix (2,000 samples)
- TN: 1242
- FP: 351
- FN: 93
- TP: 314

### Cross-validation comparison (5-fold)
| Model | CV ROC-AUC | Std |
|---|---:|---:|
| Logistic Regression | 0.7582 | 0.0126 |
| Random Forest | 0.8363 | 0.0118 |
| XGBoost | 0.8616 | 0.0059 |

Recorded best XGBoost params (from repo notes):
- `max_depth=4`
- `learning_rate=0.1`
- `n_estimators=100`
- `reg_alpha=1`
- `subsample=0.8`
- `colsample_bytree=0.6`

## Reproducibility
Run:

```bash
pip install -r requirements.txt
python train.py
```

Outputs to verify:
- Split, transform, imbalance strategy, model training logs
- Final test metrics
- Train metrics + fit diagnosis
- Saved model and preprocessor artifacts
