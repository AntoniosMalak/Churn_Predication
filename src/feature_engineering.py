"""
Feature engineering module for churn prediction pipeline.
Transforms raw data into model-ready features with proper handling of edge cases.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import pickle
import os
from pathlib import Path
from typing import Tuple, Dict, List, Any


class FeatureEngineer:
    """
    Handles feature engineering, transformation, and pipeline management.
    Supports both fitting and inference modes.
    """
    
    def __init__(self, model_dir: str = None):
        """
        Initialize feature engineer.
        
        Args:
            model_dir: Directory to save/load preprocessing artifacts
        """
        self.model_dir = model_dir or "models"
        self.preprocessor = None
        self.feature_names = None
        self.categorical_features = None
        self.numeric_features = None
        self._ensure_model_dir()
    
    def _ensure_model_dir(self):
        """Ensure model directory exists."""
        os.makedirs(self.model_dir, exist_ok=True)
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw data.
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df_engineered = df.copy()
        
        # 1. Age groups
        df_engineered['AgeGroup'] = pd.cut(df_engineered['Age'], 
                                           bins=[0, 30, 40, 50, 60, 100],
                                           labels=['18-30', '30-40', '40-50', '50-60', '60+'])
        
        # 2. Tenure groups
        df_engineered['TenureGroup'] = pd.cut(df_engineered['Tenure'],
                                             bins=[-1, 1, 3, 5, 10],
                                             labels=['<1yr', '1-3yr', '3-5yr', '5+yr'])
        
        # 3. Balance categorization
        df_engineered['HasBalance'] = (df_engineered['Balance'] > 0).astype(int)
        df_engineered['HighBalance'] = (df_engineered['Balance'] > df_engineered['Balance'].quantile(0.75)).astype(int)
        
        # 4. Product engagement score
        df_engineered['ProductEngagement'] = (
            df_engineered['NumOfProducts'] + 
            df_engineered['HasCrCard'] + 
            df_engineered['IsActiveMember']
        )
        
        # 5. Customer activity index (combines tenure and activity status)
        df_engineered['ActivityIndex'] = (
            df_engineered['IsActiveMember'] * df_engineered['Tenure']
        )
        
        # 6. Credit score category
        df_engineered['CreditScoreCategory'] = pd.cut(df_engineered['CreditScore'],
                                                     bins=[0, 580, 669, 739, 799, 850],
                                                     labels=['Poor', 'Fair', 'Good', 'VeryGood', 'Excellent'])
        
        # 7. Salary to balance ratio (safe division)
        df_engineered['SalaryToBalanceRatio'] = np.where(
            df_engineered['Balance'] > 0,
            df_engineered['EstimatedSalary'] / df_engineered['Balance'],
            df_engineered['EstimatedSalary'] / 1  # Avoid division by zero
        )
        
        # 8. Product-per-tenure ratio
        df_engineered['ProductsPerTenure'] = np.where(
            df_engineered['Tenure'] > 0,
            df_engineered['NumOfProducts'] / df_engineered['Tenure'],
            0
        )
        
        print(f"✓ Created {len(df_engineered.columns) - len(df.columns)} engineered features")
        return df_engineered
    
    def build_preprocessor(self, X: pd.DataFrame):
        """
        Build preprocessing pipeline (fit on training data).
        
        Args:
            X: Training features
        """
        # Identify column types
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target and ID columns from preprocessing
        columns_to_remove = ['Exited', 'RowNumber', 'CustomerId', 'Surname']
        self.numeric_features = [col for col in self.numeric_features if col not in columns_to_remove]
        self.categorical_features = [col for col in self.categorical_features if col not in columns_to_remove]
        
        print(f"Numeric features: {len(self.numeric_features)}")
        print(f"Categorical features: {len(self.categorical_features)}")
        
        # Build column transformer
        transformers = []
        
        # Numeric: standardization
        if self.numeric_features:
            transformers.append(('num', StandardScaler(), self.numeric_features))
        
        # Categorical: one-hot encoding with handle_unknown
        if self.categorical_features:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
                               self.categorical_features))
        
        self.preprocessor = ColumnTransformer(transformers=transformers)
        self.preprocessor.fit(X)
        
        print("✓ Preprocessor fitted successfully")
    
    def transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted preprocessor.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed feature array
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call build_preprocessor() first.")
        
        try:
            X_transformed = self.preprocessor.transform(X)
            print(f"✓ Transformed {X.shape[0]} samples to {X_transformed.shape[1]} features")
            return X_transformed
        except Exception as e:
            print(f"⚠ Error during transformation: {e}")
            print("  Handling unseen categories gracefully...")
            # This should be handled by OneHotEncoder(handle_unknown='ignore')
            return self.preprocessor.transform(X)
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of transformed features.
        
        Returns:
            List of feature names
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted.")
        
        feature_names = []
        
        # Get numeric feature names
        if self.numeric_features:
            feature_names.extend(self.numeric_features)
        
        # Get one-hot encoded feature names
        if self.categorical_features:
            cat_encoder = self.preprocessor.named_transformers_['cat']
            for i, category in enumerate(cat_encoder.categories_):
                feature_names.extend([f"{self.categorical_features[i]}_{cat}" for cat in category])
        
        self.feature_names = feature_names
        return feature_names
    
    def save_preprocessor(self, name: str = "preprocessor.pkl"):
        """
        Save preprocessor to disk.
        
        Args:
            name: Filename for the preprocessor
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted.")
        
        path = os.path.join(self.model_dir, name)
        with open(path, 'wb') as f:
            pickle.dump(self.preprocessor, f)
        print(f"✓ Saved preprocessor to {path}")
    
    def load_preprocessor(self, name: str = "preprocessor.pkl"):
        """
        Load preprocessor from disk.
        
        Args:
            name: Filename of the preprocessor
        """
        path = os.path.join(self.model_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preprocessor not found: {path}")
        
        with open(path, 'rb') as f:
            self.preprocessor = pickle.load(f)
        print(f"✓ Loaded preprocessor from {path}")
    
    def fit_and_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit preprocessor and transform features in one step.
        
        Args:
            X: Features to fit and transform
            
        Returns:
            Transformed feature array
        """
        self.build_preprocessor(X)
        return self.transform_features(X)
    
    def drop_non_predictive_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns that shouldn't be used for modeling.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with non-predictive columns removed
        """
        cols_to_drop = ['RowNumber', 'CustomerId', 'Surname']
        return df.drop(columns=[col for col in cols_to_drop if col in df.columns])


def prepare_data(df: pd.DataFrame, model_dir: str = "models", fit: bool = True) -> Tuple[np.ndarray, np.ndarray, FeatureEngineer]:
    """
    Convenience function to engineer features and prepare data.
    
    Args:
        df: Input DataFrame
        model_dir: Directory for saving preprocessor artifacts
        fit: Whether to fit preprocessor (True for training, False for inference)
        
    Returns:
        Tuple of (X_transformed, y, feature_engineer)
    """
    engineer = FeatureEngineer(model_dir=model_dir)
    
    # Engineer features
    df_engineered = engineer.engineer_features(df)
    
    # Drop non-predictive columns
    df_engineered = engineer.drop_non_predictive_columns(df_engineered)
    
    # Separate features and target
    X = df_engineered.drop(columns=['Exited'])
    y = df_engineered['Exited'].values
    
    # Fit and transform
    if fit:
        X_transformed = engineer.fit_and_transform(X)
        engineer.save_preprocessor()
    else:
        engineer.load_preprocessor()
        X_transformed = engineer.transform_features(X)
    
    return X_transformed, y, engineer


if __name__ == "__main__":
    # Example usage
    from data_ingestion import load_and_validate
    
    data_path = Path(__file__).parent.parent / "data" / "Churn_Modelling.csv"
    df = load_and_validate(str(data_path))
    
    X_transformed, y, engineer = prepare_data(df)
    print(f"\nFinal feature shape: {X_transformed.shape}")
    print(f"Target distribution: {np.bincount(y)}")
