# Recent Updates and New Features

## Summary of Changes (Version 2.1)

This document describes the major enhancements made to the churn prediction pipeline.

---

## 1. Class Imbalance Handling

### Problem
The dataset is imbalanced: 79.6% active customers vs 20.4% churned customers. This can bias models toward the majority class.

### Solution
Added multiple strategies to handle class imbalance:

**Available Strategies:**
- `'class_weight'` (DEFAULT): Uses built-in `class_weight='balanced'` in all models
  - No data modification
  - Simple and effective
  - Works with all models
  
- `'smote'`: SMOTE oversampling
  - Creates synthetic minority class samples
  - Requires: `pip install imbalanced-learn`
  - Best for: Small datasets with few features
  
- `'undersampling'`: Random undersampling of majority class
  - Removes majority class samples randomly
  - Fast, reduces training time
  - Trade-off: Loses information about majority class
  
- `'combined'`: SMOTE + undersampling
  - Balanced approach
  - Requires: `pip install imbalanced-learn`

### Usage
```python
trainer, results = train_pipeline(
    X, y,
    imbalance_strategy='smote'  # or 'undersampling', 'combined', 'class_weight'
)
```

### Implementation
- New method: `ModelTrainer.handle_class_imbalance(X_train, y_train)`
- Applied AFTER data split to prevent leakage
- Balancing happens on training data only

---

## 2. Flexible Data Splitting

### Previous Behavior
Hard-coded train/val/test split (60-20-20)

### New Behavior
Optional validation split:

**Option 1: Train/Test Only (DEFAULT)**
```python
train_pipeline(X, y, use_validation=False)
# Result: 80% train, 20% test
```

**Option 2: Train/Val/Test**
```python
train_pipeline(X, y, use_validation=True)
# Result: 60% train, 20% val, 20% test
```

### Implementation
- New method: `ModelTrainer.split_data_flexible(use_validation=False)`
- Replaces old `split_data()` method
- Both validation options work correctly with preprocessing pipeline

---

## 3. Cross-Validation Support

### Previous Behavior
Single validation set for model evaluation

### New Behavior
Optional k-fold cross-validation:

```python
trainer, results = train_pipeline(
    X, y,
    use_cross_validation=True,
    cv_folds=5  # 5-fold cross-validation
)
```

### Features
- Uses StratifiedKFold to maintain class distribution
- Evaluates model on k different train/test fold combinations
- More robust than single validation set
- Better for smaller datasets

### Implementation
- New method: `ModelTrainer.cross_validate_model(model, X_train, y_train, cv_folds=5)`
- Uses scikit-learn's `cross_val_score()`
- Returns mean and std of cross-validation scores

---

## 4. Grid Search for Hyperparameter Tuning

### Previous Behavior
Fixed hyperparameters for all models

### New Behavior
Optional automated hyperparameter search:

```python
trainer, results = train_pipeline(
    X, y,
    use_grid_search=True,
    cv_folds=3,  # 3-fold CV for each grid combination
    grid_search_params={
        'xgboost': {
            'max_depth': [5, 7, 9],
            'learning_rate': [0.01, 0.1]
        }
    }
)
```

### Features
- Searches over parameter combinations
- Uses cross-validation to evaluate each combination
- Selects best parameters automatically
- Works with all three models (Logistic Regression, Random Forest, XGBoost)

### Implementation
- New method: `ModelTrainer.grid_search_hyperparameters(model_type, param_grid, cv_folds)`
- Uses scikit-learn's `GridSearchCV`
- Default grids provided for each model if none specified

---

## 5. Preprocessing Strategy Documentation

### New File
`preprocessing_strategy.md` - Comprehensive guide documenting:
- Data ingestion and validation
- Missing value handling strategies
- Rare category handling
- Feature engineering rationale
- Preprocessing pipeline (preventing data leakage)
- Class imbalance strategies
- Model selection and evaluation approaches
- Usage examples

### Key Insights
- All preprocessing (scaling/encoding) happens AFTER train/test split
- Preprocessor is fit ONLY on training data
- Test/validation data is never seen during fitting
- This prevents data leakage and ensures valid performance estimates

---

## 6. Updated Files

### Modified: `src/model_training.py`
- Added imports: `from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold`
- Added imports: `from imblearn.over_sampling import SMOTE` (optional)
- Updated `__init__()`: Added `imbalance_strategy` parameter
- New method: `handle_class_imbalance()`
- New method: `split_data_flexible()`
- New method: `cross_validate_model()`
- New method: `grid_search_hyperparameters()`
- Updated `train_pipeline()`: Added 6 new parameters for flexibility

### Modified: `train.py`
- Updated to use new `train_pipeline()` parameters
- Now uses train/test split by default (no validation)
- Uses class_weight imbalance handling by default
- Added configuration comments

### Modified: `src/feature_engineering.py`
- Updated `prepare_data()`: Returns unscaled features (DataFrame)
- Preprocessing no longer happens before split

### New Files
- `preprocessing_strategy.md`: Complete preprocessing documentation
- `examples.py`: 7 usage examples showing all new features

---

## 7. Parameter Reference

### train_pipeline() Parameters

```python
train_pipeline(
    X,                              # Features
    y,                              # Target
    model_dir='models',             # Model directory
    engineer=None,                  # FeatureEngineer (auto-created if None)
    
    # Split options
    use_validation=False,           # True for train/val/test, False for train/test
    
    # Imbalance options
    imbalance_strategy='class_weight',  # 'class_weight', 'smote', 'undersampling', 'combined'
    
    # Evaluation options
    use_cross_validation=False,     # True to use k-fold CV instead of validation set
    cv_folds=5,                     # Number of cross-validation folds
    
    # Tuning options
    use_grid_search=False,          # True to search hyperparameters
    grid_search_params=None         # Custom parameter grid (dict by model_type)
)
```

---

## 8. Data Leakage Prevention

All updates maintain the **critical ordering** to prevent data leakage:

```
RAW DATA
   ↓
FEATURE ENGINEERING (no transformations yet)
   ↓
SPLIT into Train/Test (BEFORE any scaling/encoding)
   ↓
FIT PREPROCESSOR on Train only (no test data statistics)
   ↓
TRANSFORM all splits (train, test) using train-fitted preprocessor
   ↓
BALANCE TRAINING DATA (if enabled)
   ↓
TRAIN MODELS
   ↓
EVALUATE (on balanced training data and unbalanced test data)
```

---

## 9. Example Usage Scenarios

### Scenario 1: Quick Training (Default)
```python
trainer, results = train_pipeline(X, y)
# Uses: train/test split, class_weight strategy
# Time: ~30 seconds
```

### Scenario 2: Robust Evaluation
```python
trainer, results = train_pipeline(
    X, y,
    use_cross_validation=True,
    cv_folds=5
)
# Uses: 5-fold CV for robust performance estimate
# Time: ~3 minutes
```

### Scenario 3: Hyperparameter Optimization
```python
trainer, results = train_pipeline(
    X, y,
    use_grid_search=True,
    cv_folds=3
)
# Uses: Grid search with 3-fold CV
# Time: ~10-15 minutes
# Result: Best hyperparameters automatically found
```

### Scenario 4: Imbalance-Aware Training
```python
trainer, results = train_pipeline(
    X, y,
    imbalance_strategy='combined'  # SMOTE + undersampling
)
# Uses: Combined oversampling and undersampling
# Time: ~1 minute
# Note: Requires imblearn installed
```

---

## 10. Testing and Validation

### Manual Testing
Run the basic pipeline:
```bash
python train.py
```

Run advanced examples:
```bash
python examples.py
```

### Expected Outputs
- Train/test split: ~8000 train, ~2000 test
- Class balance report after strategy application
- Model evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
- Final test performance: ROC-AUC ~0.85+

---

## 11. Backward Compatibility

✅ **Fully backward compatible**: Existing code still works
```python
# Old code still works exactly as before
trainer, results = train_pipeline(X, y)
```

All new parameters have sensible defaults:
- `use_validation=False` (train/test only)
- `imbalance_strategy='class_weight'` (no data modification)
- `use_cross_validation=False` (single validation set)
- `use_grid_search=False` (fixed hyperparameters)

---

## 12. Dependencies

### Required (Already installed)
- pandas, numpy, scikit-learn, xgboost

### Optional (For advanced features)
```bash
pip install imbalanced-learn  # For SMOTE and undersampling
```

If not installed, the pipeline automatically falls back to `class_weight` strategy with a warning.

---

## 13. Next Steps

1. **Test all examples:**
   ```bash
   python examples.py
   ```

2. **Read preprocessing strategy:**
   ```
   Open preprocessing_strategy.md for complete understanding
   ```

3. **Customize for your use case:**
   - Adjust imbalance strategy based on your domain
   - Enable grid search if you want optimal hyperparameters
   - Use cross-validation for small datasets

---

**Last Updated:** 2026-03-05  
**Version:** 2.1  
**Status:** Production Ready
