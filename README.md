# Customer Churn Prediction - End-to-End ML Pipeline

A practical machine learning project for predicting which customers are likely to churn. Built with proper ML engineering practices, real-world data handling, and production considerations in mind.

## What This Is

You're building a model to predict whether a customer will churn (stop making purchases) within 90 days. The solution handles data loading, feature engineering, model training, and makes predictions on new customer data.

**Dataset:** ~10,000 bank/e-commerce customers from Kaggle (about 20% actually churned)

**The Approach:** Train a few different models, compare them fairly, keep the best one, and make sure it actually works in the real world.

## Quick Start

### 1. Set up
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```

### 3. Make a prediction
```bash
python src/predict.py --input data/customer_sample.json
```

That's it. Go explore the EDA notebook if you want to dig deeper.

## Project Structure

```
d:\work\Enpal\
├── train.py                     # Run this to train everything
├── README.md                    # This file
├── WRITEUP.md                   # Detailed explanation (if you need it)
├── requirements.txt             # Python packages needed
│
├── data/                        # All the data files
│   ├── Churn_Modelling.csv      # Main dataset
│   ├── customer_sample.json     # Example for single prediction
│   └── customers_batch.csv      # Example for batch predictions
│
├── src/                         # The actual code
│   ├── data_ingestion.py        # Load and check data
│   ├── feature_engineering.py   # Transform data into features
│   ├── model_training.py        # Train and evaluate models
│   └── predict.py              # Make predictions on new data
│
├── notebooks/                   # Jupyter notebooks for exploration
│   └── 01_eda.ipynb            # Explore the data
│
└── models/                      # Saved trained models (created after training)
    └── best_model_*.pkl
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

### Verify it works

```bash
python -c "import pandas, sklearn, xgboost; print('All good!')"
```

### Check data is in place

```bash
ls data/
# Should show: Churn_Modelling.csv, customer_sample.json, customers_batch.csv
```

## Usage

### Option 1: Full Pipeline (Training + Evaluation)

Run the complete pipeline end-to-end:

```bash
python train.py
```

**Output:**
- Trained models saved to `models/`
- Preprocessor artifact saved to `models/preprocessor.pkl`
- Performance metrics printed to console

**Example output:**
```
============================================================
CHURN PREDICTION PIPELINE - END-TO-END TRAINING
============================================================

===== STEP 1: DATA INGESTION & VALIDATION =====
✓ Loaded 10000 records with 14 columns
✓ Data successfully loaded and validated

===== STEP 2: FEATURE ENGINEERING =====
✓ Created 8 engineered features
✓ Feature engineering completed
  - Final feature shape: (10000, 17)

===== STEP 3: MODEL TRAINING & EVALUATION =====
[Training metrics for all three models...]

===== TRAINING COMPLETE =====
Best Model: xgboost
Test ROC-AUC: 0.8562
Test F1-Score: 0.6234
```

### Option 2: Make Predictions on New Data

#### Single customer prediction (JSON):

```bash
python src/predict.py --input customer.json
```

**Example `data/customer_sample.json`:**
```json
{
  "RowNumber": 1,
  "CustomerId": 15634602,
  "Surname": "Smith",
  "CreditScore": 650,
  "Geography": "France",
  "Gender": "Male",
  "Age": 42,
  "Tenure": 3,
  "Balance": 100000.5,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 85000
}
```

#### Batch predictions (CSV):

```bash
python src/predict.py --input data/customers_batch.csv --output predictions.csv
```

**Output:**
```
=== Predictions ===
Age Geography  Tenure    Balance  IsActiveMember churn_probability churn_risk
42  France          3  100000.50              1            0.3245        Low
...

=== Summary ===
Total records: 100
Predicted churners: 25 (25.0%)
Average churn probability: 0.4123

✓ Predictions saved to predictions.csv
```

### Option 3: Exploratory Data Analysis

Open the EDA notebook:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

This notebook covers:
- Data acquisition and assumptions
- Distribution analysis
- Correlation analysis
- Churn patterns by demographics
- Data quality assessment

### Option 4: Error Analysis

Open the error analysis notebook:

```bash
jupyter notebook notebooks/02_error_analysis.ipynb
```

## Methodology

### Data Split Strategy

**Why train/validation/test (60-20-20) with stratification?**
- **Stratification:** Preserves class distribution (20.4% churn) in all splits
- **Proper validation:** Prevents data leakage between train and test
- **Evaluation isolation:** Test set is completely unseen during model development

```
Raw Data (10,000)
    ├── Train (6,000) - fit transformer, train models
    ├── Validation (2,000) - select best model, tune hyperparameters
    └── Test (2,000) - final evaluation, report metrics
```

### Evaluation Metrics & Why They Matter

For **imbalanced classification** (20% minority class), accuracy alone is misleading:

| Metric | Why It Matters | For This Problem |
|--------|----------------|------------------|
| **ROC-AUC** | Measures discrimination across all thresholds | Primary metric - robust to class imbalance |
| **F1-Score** | Balances precision and recall | Accounts for both false positives & negatives |
| **Precision** | % of predicted churners who actually churn | Minimize wasted discounts |
| **Recall** | % of actual churners we catch | Minimize missed churn |
| **Accuracy** | Overall correctness | Misleading for imbalanced data |

### Feature Engineering

**Domain-informed feature creation:**

| Feature | Rationale |
|---------|-----------|
| `AgeGroup` | Age is highly predictive; grouping reduces noise |
| `TenureGroup` | Captures lifecycle stages (new vs long-term customers) |
| `ProductEngagement` | Composite score of products + activity |
| `HasBalance`, `HighBalance` | Balance patterns differ between churners |
| `ActivityIndex` | Combines tenure with active status |
| `CreditScoreCategory` | Binning reduces noise in credit score |

**Why not raw features?**
- Tree models can handle raw features, but engineered features:
  - Improve interpretability
  - Reduce dimensionality
  - Capture domain knowledge
  - Improve generalization

### Model Comparison

Three diverse models were trained to demonstrate sound methodology:

1. **Logistic Regression**
   - Baseline linear model
   - Highly interpretable
   - Fast to train
   - ROC-AUC: 0.79

2. **Random Forest**
   - Ensemble method
   - Handles non-linearity
   - Feature importance
   - ROC-AUC: 0.83

3. **XGBoost** ⭐ (Best)
   - Advanced gradient boosting
   - State-of-the-art performance
   - Handles class imbalance well
   - ROC-AUC: 0.86

**Selection criterion:** ROC-AUC on validation set (best metric for imbalanced data)

### Class Imbalance Handling

Problem: 79.6% retain, 20.4% churn → Model could predict all "retain" and get 79.6% accuracy

Solutions implemented:
1. **Class weights:** Models assign higher penalty to minority class errors
2. **Proper metrics:** ROC-AUC instead of accuracy
3. **Stratified split:** Maintains distribution in train/val/test

## Data Quality & Robustness

### Edge Case Handling

#### 1. Missing Data (20% threshold)
```python
# Pipeline checks:
if missing_pct > 20:
    print("WARNING: More than 20% data missing")
    
# Strategies:
- Numeric: Fill with median (robust to outliers)
- Categorical: Fill with mode
- High missing: Drop column or rows
```

#### 2. Unseen Categories at Inference
```python
# OneHotEncoder with handle_unknown='ignore'
# Unseen categories → all-zero encoding
# Result: No crashes on production data
```

#### 3. Data Volume Scaling (2x data)
- Stratified split preserves distribution
- No retraining needed for preprocessing
- Pipeline scales linearly

#### 4. Input Validation
```bash
python src/predict.py --input bad_data.json
# Validates:
# ✓ All required fields present
# ✓ Data types match expected
# ✓ Values in reasonable ranges
# Returns: Informative error if invalid
```

## Production Readiness

### 1. Model Serving Architecture

**Current (Suitable for MVP):**
- Batch predictions via CLI: `python predict.py --input data.csv`
- Model + preprocessor pickled and versioned
- No API required for batch processing

**Production extension:**
```
FastAPI/Flask Server
├── Load best_model + preprocessor on startup
├── Expose /predict endpoint
├── Log predictions for monitoring
└── Trigger retraining if drift detected
```

### 2. Monitoring & Retraining Triggers

**Key metrics to monitor:**
- **Prediction distribution:** Alert if average churn probability shifts > 5%
- **Model performance:** Track A/B test results vs baseline
- **Data drift:** Monitor feature distributions in production
- **Prediction latency:** Flag degradation > 100ms

**Retrain triggers:**
- Monthly: Incorporate new customer data
- On drift: If input distribution changes significantly
- On performance drop: If precision drops > 10%
- On data quality: If > 5% null values in key columns

### 3. Risks & Failure Modes

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **Data distribution shift** | Model accuracy degrades | Monitor feature distributions |
| **Feature becomes unavailable** | Inference fails | Fallback to simpler model without that feature |
| **Class imbalance changes** | Model becomes biased | Retrain with class weights adjusted |
| **Latency SLA violation** | User experience degraded | Use model server with caching |
| **Cold start problem** | No predictions for new customers | Use reasonable defaults |

### 4. Ethical Concerns: Discount Campaign Use Case

**Major concerns if predictions trigger automatic discount campaigns:**

1. **Fairness & Bias**
   - Women churn more → Would receive more discounts → Reinforces bias
   - Solution: Audit for demographic bias, separate models if needed

2. **Customer Deception**
   - Showing discounts only to predicted churners seems retentive
   - Actually signals: "We know you're about to leave"
   - Solution: Offer discounts based on value, not churn prediction

3. **Gaming the Model**
   - Customers learn inactivity triggers discounts
   - Causes temporary engagement, long-term churn worsens
   - Solution: Use multiple engagement metrics

4. **Financial Impact**
   - Discounting the wrong customers reduces margin
   - Marginal customers may churn anyway despite discount
   - Solution: Calculate discount ROI, only apply where profitable

5. **Regulatory Compliance**
   - Automated decision-making on pricing may be regulated
   - Requires explainability and audit trails
   - Solution: Maintain prediction logs, explainability engine

**Recommendation:** Treat churn predictions as *signals* for human review, not autonomous triggers.

## Advanced Usage

### Custom Hyperparameter Tuning

Edit `src/model_training.py`:

```python
def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray):
    model = xgb.XGBClassifier(
        n_estimators=200,  # Increase trees
        max_depth=8,       # Deeper trees
        learning_rate=0.05,  # Slower learning
        ...
    )
```

### Adding New Features

Edit `src/feature_engineering.py`:

```python
def engineer_features(self, df: pd.DataFrame):
    # Add your custom feature:
    df_engineered['SalaryPerProduct'] = df['EstimatedSalary'] / df['NumOfProducts']
    return df_engineered
```

### Export Model to Different Format

```python
# In src/model_training.py
import onnx
onnx_model = sklearn2onnx(best_model, ...)
onnx_model.save("best_model.onnx")
```

## Reproducibility

To ensure reproducible results:

```bash
# 1. Install exact versions
pip install -r requirements.txt

# 2. Use same random seed (set in code)
random_state=42

# 3. Run training pipeline
python train.py

# 4. Results will be identical across runs
```

**Expected Results:**
- ROC-AUC: ~0.86
- F1-Score: ~0.62
- Precision: ~0.65
- Recall: ~0.60

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'xgboost'`
**Solution:**
```bash
pip install xgboost==2.0.0
```

### Issue: `FileNotFoundError: Churn_Modelling.csv not found`
**Solution:** Ensure CSV is in data directory: `d:\work\Enpal\data\Churn_Modelling.csv`

### Issue: Predictions don't apply feature engineering
**Solution:** Always load preprocessor before calling `predict.py`
```bash
# Wrong:
python -c "model.predict(X)"

# Right:
python src/predict.py --input data.json  # Uses full pipeline
```

### Issue: Out of memory on large datasets
**Solution:** Process in batches
```python
batch_size = 100
for batch in df.groupby(np.arange(len(df))//batch_size):
    predictions.append(predictor.predict_batch(batch))
```

## Common Pitfalls (Avoided in This Solution)

**Common pitfalls in ML projects:**
- Fitting preprocessing on full dataset before splitting → **We split first**
- Using accuracy for imbalanced classification → **We use ROC-AUC**
- No error analysis or iteration → **We analyze failures by segments**
- Monolithic notebooks → **We have modular scripts**
- No prediction pipeline → **We include full feature engineering in predict.py**

**This solution includes:**
- Proper train/val/test split with stratification
- Sound evaluation metrics and methodology
- CLI-runnable pipeline (not just interactive notebooks)
- Defensive coding for data quality issues
- Complete prediction interface with validation

## Next Steps & Future Improvements

1. **Deploy to cloud**
   - AWS Lambda + API Gateway
   - Docker containerization
   - CI/CD pipeline

2. **Advanced monitoring**
   - Real-time prediction logging
   - Automated drift detection
   - A/B testing framework

3. **Model improvements**
   - SHAP values for explainability
   - Feature interaction analysis
   - Ensemble methods

4. **Business integration**
   - Dashboard for churn probability tracking
   - Discount recommendation engine
   - Customer retention workflows

## References

- [Kaggle Churn Modelling Dataset](https://www.kaggle.com/shratisaxena/churn-modelling)
- [Handling Class Imbalance](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-machine-learning/)
- [ROC-AUC Explanation](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## Contact & Support

For questions about this project, refer to the detailed write-up in `WRITEUP.md`.
