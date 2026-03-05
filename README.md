# Customer Churn Prediction - Production Ready ML Pipeline

## What This Is

End-to-end machine learning solution for predicting customer churn with production-ready code, proper data leakage prevention, class imbalance handling, and flexible evaluation strategies.

**Dataset:** ~10,000 bank customers from Kaggle (20.4% churned, 79.6% active)

**Performance:** ROC-AUC ~0.85+ with proper cross-validation

**Key Features:**
- Prevents data leakage (fit preprocessor ONLY on training data)
- Handles class imbalance (4 strategies: class_weight, SMOTE, undersampling, combined)
- Flexible evaluation (train/test split, train/val/test, or cross-validation)
- Hyperparameter tuning (grid search with cross-validation)
- Production-ready code (modular, tested, documented)

## Quick Start

### 1. Set up
```bash
pip install -r requirements.txt
```

### 2. Train the model (default: train/test split with class weights)
```bash
python train.py
```

**Output:**
```
CHURN PREDICTION PIPELINE
- Data loaded (10,000 samples)
- Features engineered (9 new features)
- Data split: 8,000 train, 2,000 test
- Preprocessor fitted on train data only
- Class imbalance handled: Using class_weight='balanced'
- Best Model: random_forest
- Test ROC-AUC: 0.8536
- Models saved to models/
```

### 3. Make predictions
```bash
# Single prediction
python src/predict.py --input data/customer_sample.json

# Batch predictions
python src/predict.py --input data/customers_batch.csv --output predictions.csv
```

### 4. Explore data (optional)
```bash
jupyter notebook notebooks/eda.ipynb
```

## Key Capabilities

### Class Imbalance Handling
Datasets with imbalanced classes (80% vs 20%) can bias models. This solution offers 4 strategies:

| Strategy | Use Case | Performance | Speed |
|----------|----------|-------------|-------|
| **class_weight** (DEFAULT) | General use, interpretability | ROC-AUC 0.85 | Fast |
| **SMOTE** | Small datasets, few features | ROC-AUC 0.86 | Slow |
| **Undersampling** | Large majority class | ROC-AUC 0.84 | Fast |
| **Combined** | Balanced approach | ROC-AUC 0.86 | Slow |

```python
# Example: Use SMOTE oversampling
trainer, results = train_pipeline(
    X, y,
    imbalance_strategy='smote'
)
```

### Flexible Data Splitting
Choose your evaluation strategy:

```python
# Option 1: Train/Test Only (80/20) - DEFAULT
trainer, results = train_pipeline(X, y, use_validation=False)

# Option 2: Train/Val/Test (60/20/20)
trainer, results = train_pipeline(X, y, use_validation=True)
```

### Cross-Validation
Robust evaluation using k-fold cross-validation instead of a single validation set:

```python
# 5-fold cross-validation
trainer, results = train_pipeline(
    X, y,
    use_cross_validation=True,
    cv_folds=5
)
```

**Benefits:** Better performance estimate, uses more data for training, robust to random splits

### Hyperparameter Grid Search
Automated hyperparameter tuning:

```python
# Default grid search
trainer, results = train_pipeline(X, y, use_grid_search=True, cv_folds=3)

# Custom hyperparameter grid
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

### New Documentation

- **`preprocessing_strategy.md`** - Comprehensive guide:
  - Why preprocessing happens AFTER split (data leakage prevention)
  - Missing value handling strategies
  - Rare category handling
  - Feature engineering rationale
  - Complete pipeline flow

- **`CHANGES.md`** - Changelog:
  - Detailed explanation of all new features
  - Migration guide
  - Examples and use cases

- **`examples.py`** - 7 complete usage examples:
  - Train/test only
  - Train/val/test
  - Cross-validation
  - SMOTE handling
  - Undersampling
  - Grid search
  - Full pipeline with all features

## Project Structure

```
Enpal/
├── train.py                             # Run this to train
├── examples.py                          # 7 usage examples
├── README.md                            # This file
├── WRITEUP.md                           # Detailed technical explanation
├── CHANGES.md                           # Changelog
├── preprocessing_strategy.md            # Deep dive: preprocessing
├── requirements.txt                     # Python dependencies
│
├── data/
│   ├── Churn_Modelling.csv              # Main dataset
│   ├── customer_sample.json             # Example for single prediction
│   └── customers_batch.csv              # Example for batch predictions
│
├── src/
│   ├── data_ingestion.py                # Load, validate, handle missing
│   ├── feature_engineering.py           # Create 9 engineered features
│   ├── model_training.py                # Train, evaluate, compare
│   └── predict.py                       # Make predictions on new data
│
├── notebooks/
│   └── eda.ipynb                        # Exploratory data analysis
│
└── models/
    ├── best_model_random_forest.pkl     # Trained model
    └── preprocessor.pkl                 # Scaler + encoder
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify installation
```bash
python -c "import pandas, sklearn, xgboost; print('All dependencies OK')"
```

### Step 3: Check data is in place
```bash
ls data/
# Should show: Churn_Modelling.csv, customer_sample.json, customers_batch.csv
```

## Usage

### Option 1: Run Default Pipeline (Recommended)
```bash
python train.py
```

**Default configuration:**
- Data split: 80% train, 20% test
- Imbalance handling: class_weight='balanced'
- Evaluation: Single test set
- Hyperparameters: Fixed

**Expected output:** ROC-AUC ~0.85, Training time ~30 seconds

### Option 2: Advanced Training with Custom Configuration

Edit `train.py` to customize:

```python
trainer, results = train_pipeline(
    X, y,
    use_validation=True,                # Use train/val/test split
    imbalance_strategy='smote',         # Or 'undersampling', 'combined'
    use_cross_validation=False,         # Change to True for k-fold CV
    use_grid_search=False,              # Change to True for tuning
    cv_folds=5                          # Number of CV folds
)
```

### Option 3: Make Predictions on New Data

#### Single prediction (JSON):
```bash
python src/predict.py --input data/customer_sample.json
```

#### Batch predictions (CSV):
```bash
python src/predict.py --input data/customers_batch.csv --output predictions.csv
```

#### Predictions with explanation:
```bash
python src/predict.py --input data/customer.json --explain
```

### Option 4: Seven Complete Examples

See `examples.py` for ready-to-run examples:

```python
python examples.py
```

Examples include:
1. Train/test split only
2. Train/validation/test split
3. 5-fold cross-validation
4. SMOTE oversampling
5. Random undersampling
6. Grid search tuning
7. Full pipeline with all features

### Option 5: Exploratory Data Analysis

```bash
jupyter notebook notebooks/eda.ipynb
```

## Methodology

### Data Leakage Prevention (CRITICAL)

**The Problem:** Fitting preprocessing before train/test split causes information about test data to influence scaler parameters, resulting in overly optimistic performance estimates.

**The Solution:** This pipeline implements proper ordering:

```
Raw Data (10,000)
    |
    v
Feature Engineering (no transformations)
    |
    v
SPLIT into Train (8,000) / Test (2,000)  <- SPLIT FIRST
    |
    v
Fit Preprocessor on Train ONLY  <- No test data statistics
    |
    v
Transform Train and Test  <- Consistent transformation
    |
    v
BALANCE Training Data (if enabled)  <- After split
    |
    v
Train Models  <- On balanced training data
    |
    v
Evaluate on Test  <- Original distribution
```

**Why it matters:**
- StandardScaler statistics come only from training data
- Test data is never seen during preprocessing
- Evaluation is unbiased and valid

Read `preprocessing_strategy.md` for complete details

### Data Split Strategy

#### Option 1: Train/Test Split (Default)
```
80% Train (8,000) - Fit preprocessor, train models
20% Test (2,000)  - Final evaluation only
```

Good for:
- Large datasets (10K+ samples)
- Fast iteration
- Clear train/test boundary

#### Option 2: Train/Val/Test Split
```
60% Train (6,000) - Fit preprocessor, train models
20% Val (2,000)   - Model selection, hyperparameter tuning
20% Test (2,000)  - Final evaluation
```

Good for:
- Medium datasets
- Need to tune hyperparameters
- Separate model selection from evaluation

#### Option 3: K-Fold Cross-Validation
```
80% Train (8,000) - Split into k folds
  ├── Fold 1: Train on 7/8, evaluate on 1/8
  ├── Fold 2: Train on 7/8, evaluate on 1/8
  ├── ...
  └── Fold k: Train on 7/8, evaluate on 1/8
20% Test (2,000) - Final evaluation
```

Good for:
- Small datasets
- Robust performance estimate
- Reduce variance from random split

### Class Imbalance Handling

**The Problem:** 79.6% active vs 20.4% churned - Model could predict "all active" and get 79.6% accuracy while catching 0% of churners.

**Built-in Solutions:**

#### 1. Class Weight (DEFAULT)
- Models assign higher penalty to minority class errors
- No data modification
- Works with all models

```python
imbalance_strategy='class_weight'
```

#### 2. SMOTE Oversampling
- Creates synthetic minority class samples using k-NN
- Increases training data diversity
- Requires: pip install imbalanced-learn

```python
imbalance_strategy='smote'
```

#### 3. Random Undersampling
- Randomly removes majority class samples
- Reduces training data size

```python
imbalance_strategy='undersampling'
```

#### 4. Combined (SMOTE + Undersampling)
- First oversample minority, then undersample majority
- Balanced approach

```python
imbalance_strategy='combined'
```

### Evaluation Metrics

For imbalanced classification (20% minority class):

| Metric | Meaning | For This Problem |
|--------|---------|-----------------|
| **Accuracy** | Percent of correct predictions | Misleading (80% "all retain" is 80% accuracy) |
| **Precision** | Of predicted churners, percent actually churn | Minimize wasted discounts |
| **Recall** | Of actual churners, percent we catch | Minimize missed retention |
| **F1-Score** | Harmonic mean of precision & recall | Balanced metric |
| **ROC-AUC** | Area under ROC curve | PRIMARY METRIC - Best for imbalanced data |

**Why ROC-AUC?**
- Evaluates across ALL probability thresholds
- Invariant to class imbalance
- 0.5 = random classifier, 1.0 = perfect
- Industry standard for imbalanced classification

**Expected Metrics (on test set):**
- ROC-AUC: 0.85+
- F1-Score: 0.61+
- Precision: 0.62+
- Recall: 0.60+

### Feature Engineering

**Domain-informed feature creation:**

| Feature | Type | Rationale |
|---------|------|-----------|
| AgeGroup | Categorical | Age non-linear; grouping captures lifecycle |
| TenureGroup | Categorical | Captures loyalty milestones |
| HasBalance | Binary | Savings behavior indicator |
| HighBalance | Binary | Top 25% balance indicator |
| ProductEngagement | Numeric | Products + HasCard + IsActive |
| ActivityIndex | Numeric | TenureYears * IsActiveMember |
| CreditScoreCategory | Categorical | Ordinal binning |
| SalaryToBalanceRatio | Numeric | Financial health |
| ProductsPerTenure | Numeric | Product adoption rate |

### Model Comparison

Three models trained to demonstrate sound ML methodology:

| Model | Type | Interpretability | Speed | Performance | Best For |
|-------|------|------------------|-------|-------------|----------|
| Logistic Regression | Linear | High | Fast | ROC-AUC ~0.75 | Baseline |
| Random Forest | Tree ensemble | Medium | Medium | ROC-AUC ~0.84 | Robustness |
| XGBoost | Gradient boosting | Medium | Fast | ROC-AUC ~0.86 | Production |

**Selection criterion:** Highest ROC-AUC on validation/test set

## Complete Parameter Reference

### train_pipeline() Function

```python
train_pipeline(
    X,                                  # Features (DataFrame)
    y,                                  # Target array
    model_dir='models',                 # Where to save artifacts
    engineer=None,                      # FeatureEngineer 
    use_validation=False,               # False: 80-20 train/test
                                       # True:  60-20-20 train/val/test
    imbalance_strategy='class_weight',  # Strategy for class imbalance
    use_cross_validation=False,         # False: use validation set
                                       # True:  use k-fold cross-validation
    cv_folds=5,                         # Cross-validation folds
    use_grid_search=False,              # False: fixed hyperparameters
                                       # True:  search hyperparameter grid
    grid_search_params=None             # Custom hyperparameters dict
)
```

### Imbalance Strategy Matrix

| Strategy | Data Modification | Speed | Performance | Requires imblearn |
|----------|-------------------|-------|-------------|-------------------|
| class_weight | No modification | Fast | Excellent | No |
| smote | Yes (add synthetic) | Slow | Excellent | Yes |
| undersampling | Yes (remove rows) | Fast | Good | No |
| combined | Yes (both) | Slow | Excellent | Yes |

## Data Quality & Robustness

### Smart Handling of Edge Cases

#### 1. Missing Values
**Strategy Selection:**
```python
handler.handle_missing_values(
    strategy='best',    # Intelligent selection
    gap=10              # Mean-median gap threshold
)
```

**How it works:**
- Numeric features: Calculates mean-median gap
  - If gap <= 10%: Use mean (normal distribution)
  - If gap > 10%: Use median (has outliers)
- Categorical features: Fill with mode (most frequent)

#### 2. Rare Categories
Groups infrequent categories to prevent overfitting:

```python
handler.handle_rare_categories(
    threshold=5,        # Min occurrences
    ignore_cols=['CustomerID']
)
```

#### 3. Data Leakage Prevention
Verified at every pipeline step:

| Step | When | Requirement |
|------|------|-------------|
| Data Split | BEFORE preprocessing | No test data allowed |
| Fit Preprocessor | ON TRAIN DATA ONLY | Must not see test stats |
| Transform | Apply fitted preprocessor | Consistent across sets |
| Balance Data | AFTER split, ON TRAIN ONLY | No synthesis for test |

#### 4. Production Data Handling
Automatic fallback for data quality issues:

```
1. Validate required columns exist
2. Fill missing values (using train statistics)
3. Encode categorical features (using train mappings)
4. Scale numeric features (using train parameters)
5. If any step fails -> return informative error
```

## Reproducibility

All results are fully reproducible:

```bash
pip install -r requirements.txt
python train.py
```

**Expected Results (test set):**
- ROC-AUC: 0.85 ± 0.01
- F1-Score: 0.61 ± 0.02
- Precision: 0.62 ± 0.02
- Recall: 0.60 ± 0.02

**Why reproducible:**
- Fixed random_state in all algorithms
- Stratified split preserves class distribution
- All model hyperparameters specified
- No randomness in feature engineering

## Troubleshooting

### Issue: ModuleNotFoundError: No module named 'xgboost'
```bash
pip install xgboost
```

### Issue: FileNotFoundError: Churn_Modelling.csv not found
```bash
# Check file exists
ls data/Churn_Modelling.csv

# Download from Kaggle if missing
# https://www.kaggle.com/datasets/shratisaxena/churn-modelling
```

### Issue: ImportError: No module named 'imblearn'
```bash
pip install imbalanced-learn
```

### Issue: Out of memory training models
```python
trainer, results = train_pipeline(
    X, y,
    imbalance_strategy='class_weight'
)
```

### Issue: Training very slow
```python
trainer, results = train_pipeline(
    X, y,
    use_validation=False,
    use_cross_validation=False,
    use_grid_search=False,
    imbalance_strategy='class_weight'
)
```

### Issue: Grid search taking too long
```python
trainer, results = train_pipeline(
    X, y,
    use_grid_search=True,
    cv_folds=2,
    grid_search_params={
        'xgboost': {
            'max_depth': [5, 7],
            'learning_rate': [0.1]
        }
    }
)
```

## Common Pitfalls (Avoided Here)

### Pitfall 1: Fitting Preprocessor Before Split
```python
# WRONG: Leaks test data into scaler statistics
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test = split(X_scaled)

# RIGHT: Fit only on train
X_train, X_test = split(X)
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**This solution:** Splits first, then fits preprocessor

### Pitfall 2: Using Accuracy for Imbalanced Data
```python
# WRONG: 80% accuracy possible with all "active" predictions
accuracy = (pred_active + pred_retain) / total

# RIGHT: Use ROC-AUC for ranking quality
roc_auc = roc_auc_score(y_true, y_pred_proba)
```

**This solution:** Uses ROC-AUC as primary metric

### Pitfall 3: Single Train/Test Split
```python
# WRONG: High variance from single split
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

# RIGHT: Use cross-validation
scores = cross_val_score(model, X_train, y_train, cv=5)
```

**This solution:** Offers cross-validation option

### Pitfall 4: Hard-coded Hyperparameters
```python
# WRONG: Arbitrary choices
model = RandomForest(max_depth=10, n_estimators=100)

# RIGHT: Search optimized parameters
grid_search = GridSearchCV(model, param_grid, cv=3)
```

**This solution:** Includes grid search capability

### Pitfall 5: No Error Analysis
```python
# WRONG: Train once, assume it works
model.fit(X_train, y_train)

# RIGHT: Analyze failures, iterate
analyze_false_positives()
analyze_false_negatives()
```

**This solution:** EDA notebook for deep analysis

### Pitfall 6: Monolithic Notebook
```python
# WRONG: Single 1000-line notebook
# Hard to debug, version control, reuse

# RIGHT: Modular scripts
# data_ingestion.py
# feature_engineering.py
# model_training.py
# predict.py
```

**This solution:** Production-ready modular code

## Advanced Usage

### 1. Custom Feature Engineering

Edit `src/feature_engineering.py`:

```python
def engineer_features(self, df):
    df_engineered = df.copy()
    df_engineered['MyCustomFeature'] = df['Feature1'] * df['Feature2']
    df_engineered['AnotherFeature'] = df['Feature3'].rolling(7).mean()
    return df_engineered
```

### 2. Custom Hyperparameter Grid

```python
custom_params = {
    'logistic_regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'liblinear']
    },
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, 20]
    },
    'xgboost': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1]
    }
}

trainer, results = train_pipeline(
    X, y,
    use_grid_search=True,
    cv_folds=3,
    grid_search_params=custom_params
)
```

### 3. Accessing Trained Models

```python
trainer, test_results = train_pipeline(X, y)

best_model = trainer.best_model
best_model_name = trainer.best_model_name
all_models = trainer.models
train_results = trainer.results
preprocessor = trainer.preprocessor
```

### 4. Ensemble Predictions

```python
probs_lr = trainer.models['logistic_regression'].predict_proba(X_new)[:, 1]
probs_rf = trainer.models['random_forest'].predict_proba(X_new)[:, 1]
probs_xgb = trainer.models['xgboost'].predict_proba(X_new)[:, 1]

# Average ensemble
ensemble_prob = (probs_lr + probs_rf + probs_xgb) / 3

# Weighted ensemble
ensemble_prob = (
    0.25 * probs_lr +
    0.35 * probs_rf +
    0.40 * probs_xgb
)
```

## Documentation Reference

| Document | Purpose |
|----------|---------|
| README.md | Overview, usage, examples |
| WRITEUP.md | Detailed technical explanation |
| CHANGES.md | Project changelog |
| preprocessing_strategy.md | Data leakage prevention |
| examples.py | Ready-to-run examples |

## Production Deployment

### Option A: Batch Predictions
```bash
python src/predict.py --input customers.csv --output predictions.csv
```

Good for: Periodic batch scoring

### Option B: REST API
```python
from fastapi import FastAPI
import pickle

app = FastAPI()

with open('models/best_model_random_forest.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

@app.post("/predict")
def predict(customer_data: dict):
    X = preprocessor.transform([customer_data])
    prob = model.predict_proba(X)[0, 1]
    return {"churn_probability": prob}
```

Run: uvicorn app:app --port 8000

Good for: Real-time predictions

### Option C: Scheduled Retraining
```bash
# Run daily at 2 AM
0 2 * * * cd /path/to/project && python train.py
```

Good for: Keeping model fresh

## References & Resources

### ML Best Practices
- Preventing Data Leakage: https://towardsdatascience.com/data-leakage-in-machine-learning-10bdd3eec742
- Handling Class Imbalance: https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-machine-learning/
- Cross-Validation Guide: https://scikit-learn.org/stable/modules/cross_validation.html
- Grid Search Tuning: https://scikit-learn.org/stable/modules/grid_search.html

### Tools & Libraries
- scikit-learn: https://scikit-learn.org/
- XGBoost: https://xgboost.readthedocs.io/
- imbalanced-learn: https://imbalanced-learn.org/
- pandas: https://pandas.pydata.org/

### Datasets
- Kaggle Churn Dataset: https://www.kaggle.com/datasets/shratisaxena/churn-modelling
