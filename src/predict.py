"""
Make predictions with the trained churn model.
Load a customer (or bunch of customers), get a risk score.

Usage:
    python predict.py --input customer.json
    python predict.py --input customers.csv --output predictions.csv
"""

import json
import pickle
import argparse
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_ingestion import DataIngestion
from feature_engineering import FeatureEngineer


class ChurnPredictor:
    """Predicts if a customer will churn. Uses the trained model + full feature pipeline."""
    
    def __init__(self, model_dir: str = "models"):
        """Load the trained model and preprocessor.
        
        Args:
            model_dir: Where the model files are stored
        """
        self.model_dir = model_dir
        self.model = None
        self.feature_engineer = None
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Grab the model and preprocessor from disk."""
        # Load preprocessor
        preprocessor_path = os.path.join(self.model_dir, "preprocessor.pkl")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
        
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        self.feature_engineer = FeatureEngineer(model_dir=self.model_dir)
        self.feature_engineer.preprocessor = preprocessor
        
        # Try to find whichever model was trained
        # (Ideally we trained XGBoost, but could be any of these)
        model_names = ['best_model_xgboost.pkl', 'best_model_random_forest.pkl', 'best_model_logistic_regression.pkl']
        model_path = None
        
        for name in model_names:
            path = os.path.join(self.model_dir, name)
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError(f"No model found in {self.model_dir}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"✓ Loaded model from {model_path}")
        print(f"✓ Loaded preprocessor from {preprocessor_path}")
    
    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Score one customer. Is this person likely to churn?
        
        Args:
            customer_data: Customer info (age, tenure, balance, etc.)
            
        Returns:
            Dict with prediction and confidence
        """
        # Make sure we have all the required info
        required_fields = ['Age', 'CreditScore', 'Geography', 'Gender', 'Tenure', 
                          'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
        
        missing_fields = [f for f in required_fields if f not in customer_data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Convert to dataframe
        df = pd.DataFrame([customer_data])
        
        # Fill in any optional fields that might be missing
        if 'RowNumber' not in df.columns:
            df['RowNumber'] = 1
        if 'CustomerId' not in df.columns:
            df['CustomerId'] = 0
        if 'Surname' not in df.columns:
            df['Surname'] = 'Unknown'
        
        # Apply feature engineering (age groups, balance checks, etc.)
        df_engineered = self.feature_engineer.engineer_features(df)
        
        # Remove columns we don't need for prediction
        df_engineered = self.feature_engineer.drop_non_predictive_columns(df_engineered)
        
        # Remove target column if it happens to be there
        if 'Exited' in df_engineered.columns:
            df_engineered = df_engineered.drop(columns=['Exited'])
        
        # Scale/encode the features
        try:
            X_transformed = self.feature_engineer.transform_features(df_engineered)
        except Exception as e:
            print(f"⚠ Error during feature transformation: {e}")
            raise
        
        # Run through the model
        churn_probability = self.model.predict_proba(X_transformed)[0, 1]
        predicted_churn = (churn_probability >= 0.5).astype(int)
        
        result = {
            'churn_probability': float(churn_probability),
            'predicted_churn': int(predicted_churn),
            'churn_risk': 'High ⚠️' if predicted_churn == 1 else 'Low ✓',
            'confidence': float(max(churn_probability, 1 - churn_probability))
        }
        
        return result
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score multiple customers at once.
        
        Args:
            df: DataFrame with customer data (multiple rows)
            
        Returns:
            Same DataFrame but with prediction columns added
        """
        predictions = []
        
        for idx, row in df.iterrows():
            try:
                pred = self.predict_single(row.to_dict())
                predictions.append(pred)
            except Exception as e:
                print(f"⚠ Error predicting for row {idx}: {e}")
                # Still add something so the row doesn't get lost
                predictions.append({
                    'churn_probability': np.nan,
                    'predicted_churn': np.nan,
                    'churn_risk': 'Error',
                    'confidence': np.nan
                })
        
        # Add predictions to the dataframe
        pred_df = pd.DataFrame(predictions)
        result_df = pd.concat([df.reset_index(drop=True), pred_df], axis=1)
        
        return result_df


def load_input_data(input_path: str) -> pd.DataFrame:
    """Load customer data from either JSON or CSV file.
    
    Args:
        input_path: Path to the file (JSON or CSV)
        
    Returns:
        DataFrame with customer data
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    
    if input_path.endswith('.json'):
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Handle both single object and list of objects
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError("JSON must be an object or array of objects")
    
    elif input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    
    else:
        raise ValueError("File must be .json or .csv")
    
    return df


def main():
    """CLI for making predictions. Run with --help to see options."""
    parser = argparse.ArgumentParser(
        description='🔮 Predict which customers might churn'
    )
    parser.add_argument('--input', required=True, help='Input file (JSON or CSV)')
    parser.add_argument('--output', default=None, help='Save predictions here (optional)')
    parser.add_argument('--model_dir', default='models', help='Where the model lives')
    
    args = parser.parse_args()
    
    try:
        # Load the data
        print(f"📂 Loading data from {args.input}...")
        df = load_input_data(args.input)
        print(f"✓ Loaded {len(df)} customer records\n")
        
        # Set up the predictor
        print(f"🔧 Initializing model from {args.model_dir}...")
        predictor = ChurnPredictor(model_dir=args.model_dir)
        
        # Make predictions
        print(f"\n🎯 Making predictions...\n")
        predictions_df = predictor.predict_batch(df)
        
        # Show results
        print("=" * 70)
        print("🎯 PREDICTIONS")
        print("=" * 70)
        display_cols = ['Age', 'Geography', 'Tenure', 'Balance', 'IsActiveMember', 
                       'churn_probability', 'churn_risk']
        available_cols = [col for col in display_cols if col in predictions_df.columns]
        print(predictions_df[available_cols].to_string())
        
        # Summary
        print("\n" + "=" * 70)
        print("📊 SUMMARY")
        print("=" * 70)
        churn_count = (predictions_df['predicted_churn'] == 1).sum()
        print(f"Total customers: {len(predictions_df)}")
        print(f"Predicted to churn: {churn_count} ({churn_count/len(predictions_df)*100:.1f}%)")
        print(f"Avg churn risk: {predictions_df['churn_probability'].mean():.1%}")
        
        # Save if user wants
        if args.output:
            predictions_df.to_csv(args.output, index=False)
            print(f"\n✓ Saved to {args.output}")
        
        return 0
    
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    exit(main())
