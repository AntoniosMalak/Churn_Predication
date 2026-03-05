"""
Main training pipeline script.
Runs the entire ML pipeline from data ingestion to model training.

Usage:
    python train.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_ingestion import load_and_validate
from feature_engineering import prepare_data
from model_training import train_pipeline
import pandas as pd


def main():
    """Run the complete training pipeline."""
    
    print("="*60)
    print("CHURN PREDICTION PIPELINE - END-TO-END TRAINING")
    print("="*60)
    
    # Configuration
    data_path = Path(__file__).parent / "data" / "Churn_Modelling.csv"
    model_dir = Path(__file__).parent / "models"
    
    # Step 1: Data Ingestion
    print("\n" + "="*60)
    print("STEP 1: DATA INGESTION & VALIDATION")
    print("="*60)
    
    df = load_and_validate(str(data_path))
    print(f"\n✓ Data successfully loaded and validated")
    
    # Step 2: Feature Engineering
    print("\n" + "="*60)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*60)
    
    X, y, engineer = prepare_data(df, model_dir=str(model_dir), fit=True)
    print(f"\n✓ Feature engineering completed")
    print(f"  - Final feature shape: {X.shape}")
    print(f"  - Target distribution: Churned={y.sum()}, Active={len(y)-y.sum()}")
    print(f"  - Class balance: {y.sum()/len(y)*100:.1f}% churn rate")
    
    # Step 3: Model Training
    print("\n" + "="*60)
    print("STEP 3: MODEL TRAINING & EVALUATION")
    print("="*60)
    
    trainer, test_results = train_pipeline(X, y, model_dir=str(model_dir))
    
    # Step 4: Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nBest Model: {test_results['model_name']}")
    print(f"Test ROC-AUC: {test_results['roc_auc']:.4f}")
    print(f"Test F1-Score: {test_results['f1']:.4f}")
    print(f"\nModel saved to: {model_dir}")
    print(f"To make predictions: python src/predict.py --input <customer_data.json>")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
