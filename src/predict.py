"""
Prediction script for churn model.
Provides a command-line interface to make predictions on new customer data.

Usage:
    python predict.py --input customer.json
    python predict.py --input customer.csv
"""

import json
import pickle
import argparse
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_ingestion import DataIngestion
from feature_engineering import FeatureEngineer


class ChurnPredictor:
    """
    Wrapper for making predictions with the trained churn model and full feature pipeline.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize predictor with trained model and preprocessor.
        
        Args:
            model_dir: Directory containing trained model and preprocessor
        """
        self.model_dir = model_dir
        self.model = None
        self.feature_engineer = None
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load trained model and preprocessor from disk."""
        # Load preprocessor
        preprocessor_path = os.path.join(self.model_dir, "preprocessor.pkl")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
        
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        self.feature_engineer = FeatureEngineer(model_dir=self.model_dir)
        self.feature_engineer.preprocessor = preprocessor
        
        # Load model (try different model names)
        model_names = ['best_model_xgboost.pkl', 'best_model_random_forest.pkl', 'best_model_logistic_regression.pkl']
        model_path = None
        
        for name in model_names:
            path = os.path.join(self.model_dir, name)
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError(f"No trained model found in {self.model_dir}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"✓ Loaded model from {model_path}")
        print(f"✓ Loaded preprocessor from {preprocessor_path}")
    
    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction for a single customer.
        
        Args:
            customer_data: Dictionary with customer features
            
        Returns:
            Dictionary with prediction and probability
        """
        # Validate input
        required_fields = ['Age', 'CreditScore', 'Geography', 'Gender', 'Tenure', 
                          'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
        
        missing_fields = [f for f in required_fields if f not in customer_data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Create DataFrame with customer data
        df = pd.DataFrame([customer_data])
        
        # Add missing optional columns
        if 'RowNumber' not in df.columns:
            df['RowNumber'] = 1
        if 'CustomerId' not in df.columns:
            df['CustomerId'] = 0
        if 'Surname' not in df.columns:
            df['Surname'] = 'Unknown'
        
        # Engineer features
        df_engineered = self.feature_engineer.engineer_features(df)
        
        # Drop non-predictive columns
        df_engineered = self.feature_engineer.drop_non_predictive_columns(df_engineered)
        
        # Remove target if present
        if 'Exited' in df_engineered.columns:
            df_engineered = df_engineered.drop(columns=['Exited'])
        
        # Transform features
        try:
            X_transformed = self.feature_engineer.transform_features(df_engineered)
        except Exception as e:
            print(f"⚠ Error during feature transformation: {e}")
            raise
        
        # Make prediction
        churn_probability = self.model.predict_proba(X_transformed)[0, 1]
        predicted_churn = (churn_probability >= 0.5).astype(int)
        
        result = {
            'churn_probability': float(churn_probability),
            'predicted_churn': int(predicted_churn),
            'churn_risk': 'High' if predicted_churn == 1 else 'Low',
            'confidence': float(max(churn_probability, 1 - churn_probability))
        }
        
        return result
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for multiple customers.
        
        Args:
            df: DataFrame with customer data
            
        Returns:
            DataFrame with predictions added
        """
        predictions = []
        
        for idx, row in df.iterrows():
            try:
                pred = self.predict_single(row.to_dict())
                predictions.append(pred)
            except Exception as e:
                print(f"⚠ Error predicting for row {idx}: {e}")
                predictions.append({
                    'churn_probability': np.nan,
                    'predicted_churn': np.nan,
                    'churn_risk': 'Error',
                    'confidence': np.nan
                })
        
        pred_df = pd.DataFrame(predictions)
        result_df = pd.concat([df.reset_index(drop=True), pred_df], axis=1)
        
        return result_df


def load_input_data(input_path: str) -> pd.DataFrame:
    """
    Load customer data from JSON or CSV file.
    
    Args:
        input_path: Path to input file (JSON or CSV)
        
    Returns:
        DataFrame with customer data
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if input_path.endswith('.json'):
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError("Invalid JSON format. Expected dict or list of dicts.")
    
    elif input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    
    else:
        raise ValueError("Input file must be JSON or CSV")
    
    return df


def main():
    """Command-line interface for predictions."""
    parser = argparse.ArgumentParser(
        description='Make churn predictions for customer data'
    )
    parser.add_argument('--input', required=True, help='Input file (JSON or CSV)')
    parser.add_argument('--output', default=None, help='Output file for predictions (optional)')
    parser.add_argument('--model_dir', default='models', help='Directory containing trained model')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for churn prediction')
    
    args = parser.parse_args()
    
    try:
        # Load input data
        print(f"Loading data from {args.input}...")
        df = load_input_data(args.input)
        print(f"✓ Loaded {len(df)} customer records")
        
        # Load predictor
        print(f"\nInitializing predictor from {args.model_dir}...")
        predictor = ChurnPredictor(model_dir=args.model_dir)
        
        # Make predictions
        print(f"\nMaking predictions...")
        predictions_df = predictor.predict_batch(df)
        
        # Display results
        print("\n=== Predictions ===")
        display_cols = ['Age', 'Geography', 'Tenure', 'Balance', 'IsActiveMember', 
                       'churn_probability', 'churn_risk']
        available_cols = [col for col in display_cols if col in predictions_df.columns]
        print(predictions_df[available_cols].to_string())
        
        # Summary statistics
        print(f"\n=== Summary ===")
        churn_count = (predictions_df['predicted_churn'] == 1).sum()
        print(f"Total records: {len(predictions_df)}")
        print(f"Predicted churners: {churn_count} ({churn_count/len(predictions_df)*100:.1f}%)")
        print(f"Average churn probability: {predictions_df['churn_probability'].mean():.4f}")
        
        # Save output if specified
        if args.output:
            predictions_df.to_csv(args.output, index=False)
            print(f"\n✓ Predictions saved to {args.output}")
        
        return 0
    
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
