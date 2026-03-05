# Customer Churn Prediction — Production-Ready ML Pipeline

End-to-end machine learning solution for predicting customer churn with production-ready code, proper data leakage prevention, class imbalance handling, and flexible evaluation strategies.

**Dataset:** ~10,000 bank customers from Kaggle (20.4% churned, 79.6% active)  
**Best Model:** XGBoost · **Test ROC-AUC: 0.8608** · **Fit Status: ✅ Good fit**

---

## Latest Pipeline Results

| Metric | Train | Test |
|--------|-------|------|
| ROC-AUC | 0.8983 | **0.8608** |
| F1-Score | 0.6375 | 0.5858 |
| Precision | 0.5228 | 0.4722 |
| Recall | 0.8166 | 0.7715 |
| Accuracy | 0.8107 | 0.7780 |

ROC-AUC gap (Train − Test): **0.0375** · F1 gap: **0.0516** → Train and test performance are reasonably close.

### Confusion Matrix (Test Set — 2,000 samples)

```
                  Predicted: Active   Predicted: Churn
Actual: Active         1242               351
Actual: Churn            93               314
```

### Model Comparison (5-Fold Cross-Validation)

| Model | CV ROC-AUC | Std | Best Params |
|-------|-----------|-----|-------------|
| Logistic Regression | 0.7582 | ±0.0126 | C=0.1, solver=liblinear |
| Random Forest | 0.8363 | ±0.0118 | max_depth=5, min_samples_split=4, n_estimators=10 |
| **XGBoost ✅** | **0.8616** | **±0.0059** | max_depth=4, lr=0.1, n_estimators=100, reg_alpha=1, subsample=0.8, colsample_bytree=0.6 |

Winner selected by highest CV ROC-AUC. XGBoost grid search evaluated 2,880 candidates across 14,400 fits.

---

## Quick Start

### 1. Set up
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```

**Expected output:**
```
CHURN PREDICTION PIPELINE
- Data loaded (10,000 samples)
- Features engineered (9 new features)
- Data split: 8,000 train, 2,000 test
- Preprocessor fitted on train data only
- Class imbalance handled: random undersampling
- Best Model: xgboost
- Test ROC-AUC: 0.8608
- Models saved to models/
```

### 3. Make predictions
```bash
# Single prediction
python src/predict.py --input data/customer_sample.json

# Batch predictions
python src/predict.py --input data/customers_batch.csv --output predictions.csv

# With explanation
python src/predict.py --input data/customer.json --explain
```

### 4. Explore data (optional)
```bash
jupyter notebook notebooks/eda.ipynb
```

---

## Key Capabilities

### Class Imbalance Handling

The dataset has an 80/20 split — a naive model predicting "all active" scores 79.6% accuracy while catching **zero** churners. Four strategies are available:

| Strategy | Use Case | Performance | Speed | Requires imblearn |
|----------|----------|-------------|-------|-------------------|
| `class_weight` (DEFAULT) | General use | ROC-AUC ~0.85 | Fast | No |
| `smote` | Small datasets | ROC-AUC ~0.86 | Slow | Yes |
| `undersampling` ✅ (this run) | Large majority class | ROC-AUC ~0.84 | Fast | No |
| `combined` | Balanced approach | ROC-AUC ~0.86 | Slow | Yes |

This run used **random undersampling**: training set balanced from `{active: 6370, churned: 1630}` → `{active: 1630, churned: 1630}`.

```python
# Example: switch strategy
trainer, results = train_pipeline(X, y, imbalance_strategy='smote')
```

### Flexible Data Splitting

```python
# Option 1: Train/Test Only (80/20) — DEFAULT
trainer, results = train_pipeline(X, y, use_validation=False)

# Option 2: Train/Val/Test (60/20/20)
trainer, results = train_pipeline(X, y, use_validation=True)
```

### Cross-Validation

```python
# 5-fold cross-validation
trainer, results = train_pipeline(X, y, use_cross_validation=True, cv_folds=5)
```

Benefits: better performance estimate, uses more data for training, robust to random splits.

### Hyperparameter Grid Search

```python
# Default grid search
trainer, results = train_pipeline(X, y, use_grid_search=True, cv_folds=3)

# Custom grid
trainer, results = train_pipeline(
    X, y,
    use_grid_search=True,
    grid_search_params={
        'xgboost': {
            'max_depth': [5, 7, 9],
            'learning_rate': [0.05, 0.1, 0.15]
        }
    }
)
```

---

## Project Structure

```
MainFolder/
├── train.py                             # Run this to train
├── README.md                            # This file
├── WRITEUP.md                           # Detailed technical explanation
├── CHANGES.md                           # Changelog
├── preprocessing_strategy.md            # Deep dive: preprocessing & leakage prevention
├── requirements.txt
│
├── data/
│   ├── Churn_Modelling.csv              # Main dataset (~10K rows)
│   ├── customer_sample.json             # Example for single prediction
│   └── customers_batch.csv             # Example for batch predictions
│
├── src/
│   ├── data_ingestion.py                # Load, validate, handle missing values
│   ├── feature_engineering.py           # Create 9 engineered features
│   ├── model_training.py                # Train, evaluate, compare models
│   └── predict.py                       # Inference on new data
│
├── notebooks/
│   └── eda.ipynb                        # Exploratory data analysis
│
└── models/
    ├── best_model_xgboost.pkl           # Trained model (this run)
    └── preprocessor.pkl                 # Fitted scaler + encoder
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda

```bash
# Install dependencies
pip install -r requirements.txt

# Verify
python -c "import pandas, sklearn, xgboost; print('All dependencies OK')"

# Check data is in place
ls data/
# Should show: Churn_Modelling.csv, customer_sample.json, customers_batch.csv
```

---

## Usage

### Option 1: Default Pipeline (Recommended)
```bash
python train.py
```

Default config: 80/20 train-test split · `class_weight='balanced'` · fixed hyperparameters · ~30s training time.

### Option 2: Advanced Configuration

```python
trainer, results = train_pipeline(
    X, y,
    use_validation=True,            # Train/val/test split
    imbalance_strategy='smote',     # Or 'undersampling', 'combined'
    use_cross_validation=False,     # Set True for k-fold CV
    use_grid_search=False,          # Set True for hyperparameter tuning
    cv_folds=5
)
```

### Option 3: Predictions on New Data

```bash
python src/predict.py --input data/customer_sample.json          # single
python src/predict.py --input data/customers_batch.csv \
                      --output predictions.csv                    # batch
```

---

## Methodology

### Data Leakage Prevention

Fitting preprocessing before splitting leaks test statistics into the scaler, producing overly optimistic results. This pipeline enforces strict ordering:

```
Raw Data (10,000)
       ↓
Feature Engineering        ← no transformations applied
       ↓
Split: Train (8,000) / Test (2,000)   ← SPLIT FIRST
       ↓
Fit Preprocessor on Train ONLY        ← test data never seen
       ↓
Transform Train and Test consistently
       ↓
Balance Training Data                  ← after split, train only
       ↓
Train Models → Evaluate on Test
```

See `preprocessing_strategy.md` for full details.

### Data Split Options

| Option | Split | Best For |
|--------|-------|----------|
| Train/Test (default) | 80 / 20 | Large datasets, fast iteration |
| Train/Val/Test | 60 / 20 / 20 | Hyperparameter tuning with held-out test |
| K-Fold CV | 80 train (k folds) / 20 test | Small datasets, robust estimates |

### Feature Engineering

9 domain-informed features created from the original 14 columns (final shape: 10,000 × 19):

| Feature | Type | Rationale |
|---------|------|-----------|
| AgeGroup | Categorical | Age is non-linear; grouping captures lifecycle stages |
| TenureGroup | Categorical | Captures loyalty milestones |
| HasBalance | Binary | Savings behaviour indicator |
| HighBalance | Binary | Top 25% balance flag |
| ProductEngagement | Numeric | Products + HasCard + IsActiveMember |
| ActivityIndex | Numeric | TenureYears × IsActiveMember |
| CreditScoreCategory | Categorical | Ordinal binning |
| SalaryToBalanceRatio | Numeric | Financial health proxy |
| ProductsPerTenure | Numeric | Product adoption rate |

### Evaluation Metrics

ROC-AUC is the primary metric — it evaluates across all thresholds and is invariant to class imbalance (0.5 = random, 1.0 = perfect).

| Metric | Meaning | Note |
|--------|---------|------|
| **ROC-AUC** ← primary | Area under ROC curve | Best for imbalanced data |
| Recall | Of actual churners, % caught | Minimise missed retention |
| Precision | Of predicted churners, % correct | Minimise wasted discounts |
| F1-Score | Harmonic mean of P & R | Balanced trade-off |
| Accuracy | % correct overall | Misleading — "all active" = 79.6% accuracy, 0 churners caught |

---

## Complete Parameter Reference

```python
train_pipeline(
    X,                                   # Features (DataFrame)
    y,                                   # Target array
    model_dir='models',                  # Where to save artifacts
    engineer=None,                       # FeatureEngineer instance
    use_validation=False,                # False: 80/20 | True: 60/20/20
    imbalance_strategy='class_weight',   # 'smote' | 'undersampling' | 'combined'
    use_cross_validation=False,          # True: k-fold CV
    cv_folds=5,
    use_grid_search=False,               # True: GridSearchCV
    grid_search_params=None              # Custom param grid dict
)
```

### Imbalance Strategy Reference

| Strategy | Modifies Data | Speed | Requires imblearn |
|----------|--------------|-------|-------------------|
| `class_weight` | No | Fast | No |
| `smote` | Yes (adds synthetic rows) | Slow | Yes |
| `undersampling` | Yes (removes rows) | Fast | No |
| `combined` | Yes (both) | Slow | Yes |

---

## Data Quality & Robustness

### Missing Values
```python
handler.handle_missing_values(strategy='best', gap=10)
# gap <= 10%: use mean | gap > 10%: use median | categorical: mode
```

This run had **zero missing values** across all 14 columns and 10,000 records.

### Rare Categories
```python
handler.handle_rare_categories(threshold=5, ignore_cols=['CustomerID'])
# Groups categories with fewer than 5 occurrences
```

This run found no rare categories in `Geography` or `Gender`.

### Leakage Checklist

| Step | Timing | Rule |
|------|--------|------|
| Data Split | Before preprocessing | No test data allowed |
| Fit Preprocessor | On train only | Never see test statistics |
| Transform | Post-fit | Apply same transform to both splits |
| Balance Data | After split, train only | Never synthesise test samples |

---

## Advanced Usage

### Accessing Trained Models

```python
trainer, test_results = train_pipeline(X, y)

best_model      = trainer.best_model
best_model_name = trainer.best_model_name
all_models      = trainer.models
preprocessor    = trainer.preprocessor
```

### Ensemble Predictions

```python
probs_lr  = trainer.models['logistic_regression'].predict_proba(X_new)[:, 1]
probs_rf  = trainer.models['random_forest'].predict_proba(X_new)[:, 1]
probs_xgb = trainer.models['xgboost'].predict_proba(X_new)[:, 1]

# Weighted ensemble
ensemble_prob = 0.25 * probs_lr + 0.35 * probs_rf + 0.40 * probs_xgb
```

### Custom Feature Engineering

```python
# src/feature_engineering.py
def engineer_features(self, df):
    df_engineered = df.copy()
    df_engineered['MyFeature'] = df['Feature1'] * df['Feature2']
    return df_engineered
```

---

## Production Deployment

### Option A: Batch scoring
```bash
python src/predict.py --input customers.csv --output predictions.csv
```

### Option B: REST API (FastAPI)
```python
from fastapi import FastAPI
import pickle

app = FastAPI()
model        = pickle.load(open('models/best_model_xgboost.pkl', 'rb'))
preprocessor = pickle.load(open('models/preprocessor.pkl', 'rb'))

@app.post("/predict")
def predict(customer_data: dict):
    X    = preprocessor.transform([customer_data])
    prob = model.predict_proba(X)[0, 1]
    return {"churn_probability": prob}
```
```bash
uvicorn app:app --port 8000
```

### Option C: Scheduled retraining
```bash
# Retrain daily at 2 AM
0 2 * * * cd /path/to/project && python train.py
```

---

## Reproducibility

```bash
pip install -r requirements.txt
python train.py
```

Expected test results: ROC-AUC `0.86 ± 0.01` · F1 `0.59 ± 0.02` · Recall `0.77 ± 0.02`

Reproducibility guaranteed by fixed `random_state` in all algorithms, stratified splits, and deterministic feature engineering.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: xgboost` | `pip install xgboost` |
| `FileNotFoundError: Churn_Modelling.csv` | Download from [Kaggle](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction), place in `data/` |
| `ImportError: imblearn` | `pip install imbalanced-learn` |
| Out of memory | Use `imbalance_strategy='class_weight'` |
| Grid search too slow | Reduce `cv_folds=2` or narrow the param grid |
| Training very slow | Set `use_grid_search=False`, `use_cross_validation=False` |

---

## Documentation Reference

| File | Purpose |
|------|---------|
| `README.md` | Overview, results, usage (this file) |
| `WRITEUP.md` | Detailed technical explanation |
| `CHANGES.md` | Changelog and migration guide |
| `preprocessing_strategy.md` | Data leakage prevention deep dive |
| `examples.py` | 7 ready-to-run usage examples |
