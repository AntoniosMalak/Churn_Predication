"""
Train the churn prediction model end-to-end.
Run this to load data, build features, and train models.
"""

import sys
from pathlib import Path

# Add the src folder to the path, so we can import everything properly
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_ingestion import load_and_validate
from feature_engineering import prepare_data
from model_training import train_pipeline
import pandas as pd


def main():
    """Run the complete pipeline."""
    
    print("\n" + "="*60)
    print("🚀 CHURN PREDICTION PIPELINE - LET'S GO")
    print("="*60)
    
    # Set up paths
    data_path = Path(__file__).parent / "data" / "Churn_Modelling.csv"
    model_dir = Path(__file__).parent / "models"
    
    # Step 1: Load and validate data
    print("\n" + "="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)
    
    df = load_and_validate(str(data_path))
    print(f"\n✓ Data loaded successfully")
    
    # Step 2: Engineer features
    print("\n" + "="*60)
    print("STEP 2: ENGINEERING FEATURES")
    print("="*60)
    
    X, y, engineer = prepare_data(df, model_dir=str(model_dir), fit=True)
    print(f"\n✓ Features ready to go")
    print(f"  📊 Shape: {X.shape}")
    print(f"  🔍 Churn: {y.sum()} out of {len(y)} customers")
    
    # Step 3: Train and evaluate models
    print("\n" + "="*60)
    print("STEP 3: TRAINING MODELS")
    print("="*60)
    
    trainer, test_results = train_pipeline(X, y, model_dir=str(model_dir))
    
    # Pipeline complete
    print("\n" + "="*60)
    print("✅ Pipeline completed successfully")
    print("="*60)
    print(f"\n📈 Best Model: {test_results['model_name']}")
    print(f"🎯 ROC-AUC Score: {test_results['roc_auc']:.4f}")
    print(f"📊 F1-Score: {test_results['f1']:.4f}")
    print(f"\n💾 Models saved to: {model_dir}")
    print(f"🔮 Make predictions: python src/predict.py --input data/customer_sample.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
