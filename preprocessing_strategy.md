# Preprocessing Strategy and Rationale

## Overview
This document explains the preprocessing decisions and reasoning for the customer churn prediction model. All preprocessing is designed to prevent data leakage and ensure valid model evaluation.

---

## 1. Data Ingestion Phase

### Missing Value Handling
**Location:** `src/data_ingestion.py` - `handle_missing_values()`

**Strategies Available:**
- **drop**: Remove rows with missing values (only if missing % is acceptable)
- **mean**: Fill numeric features with mean (good for normally distributed data)
- **median**: Fill numeric features with median (robust to outliers)
- **best**: Automatic strategy selection based on data distribution:
  - If mean-median gap ≤ gap_threshold: Use mean
  - If mean-median gap > gap_threshold: Use median (data likely has outliers)
  - For categorical: Use mode

**Why:** Different distributions require different imputation. The "best" strategy automatically detects skewness and outliers.

### Rare Category Handling
**Location:** `src/data_ingestion.py` - `handle_rare_categories()`

**Approach:** Categories with count < threshold (default: 5) are grouped as 'Other'

**Why:** 
- Prevents overfitting on sparse categories
- Reduces feature dimensionality
- Handles unseen categories gracefully at inference time
- Improves model generalization

**Parameters:**
- `threshold`: Minimum occurrences (default: 5)
- `ignore_cols`: Columns to skip (e.g., IDs, keys)

---

## 2. Feature Engineering Phase

**Location:** `src/feature_engineering.py` - `engineer_features()`

### Features Created (9 new features)

| Feature | Type | Rationale |
|---------|------|-----------|
| **AgeGroup** | Categorical | Age is non-linear predictor; grouping captures lifecycle stages |
| **TenureGroup** | Categorical | Loyalty patterns differ by tenure buckets |
| **HasBalance** | Binary | Binary indicator is simpler than continuous balance |
| **HighBalance** | Binary | Captures high-net-worth customers (top 25%) |
| **ProductEngagement** | Numeric | Composite: NumProducts + HasCard + IsActive = engagement proxy |
| **ActivityIndex** | Numeric | IsActive × Tenure captures long-term engagement |
| **CreditScoreCategory** | Categorical | Ordinal grouping of credit risk levels |
| **SalaryToBalanceRatio** | Numeric | Financial health indicator |
| **ProductsPerTenure** | Numeric | Product adoption rate over tenure |

**Why Engineered Features?**
- Capture domain knowledge (business intuition)
- Convert continuous to categorical when nonlinear
- Reduce feature dimensionality with composites
- Improve model interpretability

---

## 3. Preprocessing Pipeline (Data Leakage Prevention)

**Location:** `src/model_training.py` - `train_pipeline()`

### Critical Ordering to Prevent Leakage

```
DATA FLOW:
1. Raw Data
   ↓
2. Feature Engineering (no scaling/encoding yet)
   ↓
3. SPLIT into Train/Test (FIRST - before any transformation)
   ↓
4. FIT Preprocessor (Scaler + Encoder) on TRAIN only
   ↓
5. TRANSFORM all splits using train-fitted preprocessor
   ↓
6. Train models on transformed data
```

### Why This Order Matters

**❌ WRONG (causes leakage):**
```
Raw Data → Engineer Features → Scale/Encode → Split → Train
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           Stats computed on ALL data!
```
Problem: Test/validation data statistics influencing scaling parameters

**✅ RIGHT (prevents leakage):**
```
Raw Data → Engineer Features → Split → Fit on Train → Transform All → Train
           ^^^^^^^^^^^^^^^^^^         ^^^^^^^^^^^^^^^^
           No stats yet              Stats from train ONLY
```
Benefit: Preprocessor has never "seen" test data

### Preprocessor Components

**StandardScaler (Numeric Features)**
- Formula: `(X - μ_train) / σ_train`
- Why: LinearModels need normalized scale; tree-based models benefit from it
- Fitted on: Training data only

**OrdinalEncoder (Categorical Features)**
- Formula: Maps each category to integer code (0, 1, 2, ...)
- Why: Tree models accept integer codes; preserves ordinality for ordinal features
- Fitted on: Training data only
- Unknown handling: If test has unseen category, OrdinalEncoder will error (this is intentional - alerts you to data drift)

---

## 4. Class Imbalance Handling

**Location:** `src/model_training.py` - `handle_class_imbalance()`

### Data Distribution
```
Class 0 (Active):  7,963 customers (79.6%)
Class 1 (Churned): 2,037 customers (20.4%)
Imbalance Ratio:   ~4:1
```

### Strategies Available

**Strategy 1: class_weight='balanced'** (DEFAULT)
- Models weight minority class higher during training
- No data modification
- Pros: Simple, prevents overfitting, works with all models
- When to use: When you want probabilistic predictions (for ranking)

**Strategy 2: SMOTE (Oversampling)**
- Creates synthetic minority samples using k-NN interpolation
- Applied to: Training data only (AFTER split)
- Pros: Better for small datasets, no information loss
- When to use: When you have enough features and want to balance classes
- Requires: `pip install imbalanced-learn`

**Strategy 3: Random Undersampling**
- Randomly removes majority class samples
- Applied to: Training data only
- Pros: Fast, reduces training time
- When to use: When you have large majority class and can afford to discard data
- Con: Loses information about majority class

**Strategy 4: Combined (SMOTE + Undersampling)**
- First oversample minority, then undersample majority
- Applied to: Training data only
- Pros: Balanced dataset without losing too much information
- When to use: When you want moderate class balance

### Why Applied After Split?

```
WRONG: Imbalance handling BEFORE split
Split → Train has synthetic data leak into test
                       ↓
Test set has "seen" synthetic minority patterns

RIGHT: Imbalance handling AFTER split
Split → Train gets balanced (no test data leakage)
Test stays original distribution (realistic evaluation)
```

---

## 5. Model Selection After Preprocessing

**Location:** `src/model_training.py` - Training methods

### Models Trained
1. **Logistic Regression**: Linear baseline, interpretable
2. **Random Forest**: Non-linear, captures interactions
3. **XGBoost**: Gradient boosting, usually best performance

### Evaluation Approaches

**Option 1: Validation Set (DEFAULT)**
- Split: 60% train, 20% validation, 20% test
- Use case: Large datasets (10K+ samples)
- Pros: Fast, easy to parallelize
- Cons: Validation set is smaller

**Option 2: Cross-Validation (use_cross_validation=True)**
- Split: 80% train, 20% test; use k-fold CV on train
- How: Train on k-1 folds (80%), evaluate on 1 fold
- Use case: Small datasets, robust evaluation needed
- Pros: Uses more data for training, robust estimate of model performance
- Cons: Slower, harder to parallelize

**Option 3: Grid Search (use_grid_search=True)**
- Searches over hyperparameter grid using cross-validation
- How: For each hyperparameter combo, run cross-validation
- Use case: Need to find optimal hyperparameters
- Pros: Systematic, finds best parameters
- Cons: Computationally expensive

---

## 6. Final Evaluation

**Location:** `src/model_training.py` - `test_model()`

### Test Set Metrics
- **Accuracy**: Overall correctness (can be misleading with imbalanced data)
- **Precision**: Of predicted churners, how many actually churn (important for targeting costs)
- **Recall**: Of actual churners, how many we detect (important for retention impact)
- **F1-Score**: Harmonic mean of precision & recall (balanced metric)
- **ROC-AUC**: Probability ranking metric (best for imbalanced classification)

**Why ROC-AUC is Primary Metric:**
- Evaluates across all threshold values
- Insensitive to class imbalance
- 0.5 = random classifier, 1.0 = perfect classifier

---

## 7. Usage Examples

### Default (Class Weights, Train/Test Split, No CV)
```python
from model_training import train_pipeline

trainer, results = train_pipeline(X, y)
# Uses: class_weight='balanced', train/test split (80/20)
```

### With Class Imbalance (SMOTE)
```python
trainer, results = train_pipeline(X, y, imbalance_strategy='smote')
# Uses: SMOTE oversampling on training data
```

### With Cross-Validation
```python
trainer, results = train_pipeline(X, y, use_cross_validation=True, cv_folds=5)
# Uses: 5-fold cross-validation instead of validation set
```

### With Validation Set + Grid Search
```python
trainer, results = train_pipeline(
    X, y, 
    use_validation=True,
    use_grid_search=True,
    cv_folds=3
)
# Uses: Train/val/test split + grid search with 3-fold CV
```

### Full Control
```python
trainer, results = train_pipeline(
    X, y,
    use_validation=True,           # Use validation set
    imbalance_strategy='combined',  # SMOTE + undersampling
    use_cross_validation=False,     # Don't use cross-validation
    use_grid_search=True,          # Perform grid search
    cv_folds=3,                    # 3-fold CV for grid search
    grid_search_params={
        'xgboost': {'max_depth': [5, 7, 9], 'learning_rate': [0.01, 0.1, 0.2]}
    }
)
```

---

## 8. Key Takeaways

✅ **DO:**
- Always split BEFORE applying any transformation
- Fit preprocessor on train data ONLY
- Use appropriate imbalance strategy for your use case
- Choose validation strategy based on dataset size
- Use ROC-AUC for imbalanced classification

❌ **DON'T:**
- Apply scaling/encoding before split
- Use test metrics to tune hyperparameters
- Oversample before split (causes leakage)
- Ignore class imbalance (model will be overconfident)
- Use accuracy alone for imbalanced data

---

## 9. Files and Functions

| File | Function | Purpose |
|------|----------|---------|
| `data_ingestion.py` | `load_and_validate()` | Load and validate raw data |
| `data_ingestion.py` | `handle_missing_values()` | Intelligent missing value imputation |
| `data_ingestion.py` | `handle_rare_categories()` | Group rare categories |
| `feature_engineering.py` | `engineer_features()` | Create domain-informed features |
| `feature_engineering.py` | `build_preprocessor()` | Fit scaler + encoder on data |
| `model_training.py` | `split_data_flexible()` | Split with optional validation |
| `model_training.py` | `handle_class_imbalance()` | Apply imbalance strategy |
| `model_training.py` | `cross_validate_model()` | k-fold cross-validation |
| `model_training.py` | `grid_search_hyperparameters()` | Hyperparameter tuning |
| `model_training.py` | `train_pipeline()` | Orchestrate full pipeline |

---

**Last Updated:** 2026-03-05
**Pipeline Version:** 2.1 (with imbalance handling and flexible validation)
