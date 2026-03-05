"""
Feature engineering for the churn model.
Turn raw customer data into features the model can actually learn from.
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
    """Takes messy customer data and turns it into something useful for the model."""
    
    def __init__(self, model_dir: str = None):
        """Set up the feature engineer.
        
        Args:
            model_dir: Where to save the preprocessor for later use
        """
        self.model_dir = model_dir or "models"
        self.preprocessor = None
        self.feature_names = None
        self.categorical_features = None
        self.numeric_features = None
        self._ensure_model_dir()
    
    def _ensure_model_dir(self):
        """Make sure the models folder exists."""
        os.makedirs(self.model_dir, exist_ok=True)
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create smart features from the raw data. These help the model learn better.
        
        Args:
            df: Raw customer data
            
        Returns:
            Data with new features added
        """
        df_engineered = df.copy()
        
        # Group ages into buckets (instead of using raw age directly)
        df_engineered['AgeGroup'] = pd.cut(df_engineered['Age'], 
                                           bins=[0, 30, 40, 50, 60, 100],
                                           labels=['18-30', '30-40', '40-50', '50-60', '60+'])
        
        # Group tenure too (loyalty matters!)
        df_engineered['TenureGroup'] = pd.cut(df_engineered['Tenure'],
                                             bins=[-1, 1, 3, 5, 10],
                                             labels=['<1yr', '1-3yr', '3-5yr', '5+yr'])
        
        # Does this customer have money in the bank?
        df_engineered['HasBalance'] = (df_engineered['Balance'] > 0).astype(int)
        df_engineered['HighBalance'] = (df_engineered['Balance'] > df_engineered['Balance'].quantile(0.75)).astype(int)
        
        # How engaged is this customer? (number of products + has card + is active)
        df_engineered['ProductEngagement'] = (
            df_engineered['NumOfProducts'] + 
            df_engineered['HasCrCard'] + 
            df_engineered['IsActiveMember']
        )
        
        # Activity over time (stays active for years = loyal)
        df_engineered['ActivityIndex'] = (
            df_engineered['IsActiveMember'] * df_engineered['Tenure']
        )
        
        # Credit score quality
        df_engineered['CreditScoreCategory'] = pd.cut(df_engineered['CreditScore'],
                                                     bins=[0, 580, 669, 739, 799, 850],
                                                     labels=['Poor', 'Fair', 'Good', 'VeryGood', 'Excellent'])
        
        # Financial health (income vs balance)
        df_engineered['SalaryToBalanceRatio'] = np.where(
            df_engineered['Balance'] > 0,
            df_engineered['EstimatedSalary'] / df_engineered['Balance'],
            df_engineered['EstimatedSalary'] / 1
        )
        
        # How many products per year of tenure
        df_engineered['ProductsPerTenure'] = np.where(
            df_engineered['Tenure'] > 0,
            df_engineered['NumOfProducts'] / df_engineered['Tenure'],
            0
        )
        
        new_features = len(df_engineered.columns) - len(df.columns)
        print(f"✓ Created {new_features} new features")
        return df_engineered
    
    def build_preprocessor(self, X: pd.DataFrame):
        """Build the pipeline that'll transform data consistently every time.
        
        This is trained ONLY on the training data - important to avoid cheating!
        
        Args:
            X: Training features
        """
        # Split features into numeric and categorical
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove customer ID columns (not useful for prediction)
        columns_to_remove = ['Exited', 'RowNumber', 'CustomerId', 'Surname']
        self.numeric_features = [col for col in self.numeric_features if col not in columns_to_remove]
        self.categorical_features = [col for col in self.categorical_features if col not in columns_to_remove]
        
        print(f"📊 {len(self.numeric_features)} numeric features")
        print(f"🏷️  {len(self.categorical_features)} categorical features")
        
        # Build the transformer
        transformers = []
        
        # Numeric: standardize (scale to mean 0, std 1)
        if self.numeric_features:
            transformers.append(('num', StandardScaler(), self.numeric_features))
        
        # Categorical: one-hot encoding (convert to 0/1 columns)
        if self.categorical_features:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
                               self.categorical_features))
        
        self.preprocessor = ColumnTransformer(transformers=transformers)
        self.preprocessor.fit(X)
        
        print("✓ Ready to transform data!")
    
    def transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """Apply the preprocessor to new data.
        
        Args:
            X: New features to transform
            
        Returns:
            Transformed data ready for the model
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted yet. Call build_preprocessor() first.")
        
        try:
            X_transformed = self.preprocessor.transform(X)
            print(f"✓ Transformed {X.shape[0]} samples")
            return X_transformed
        except Exception as e:
            print(f"⚠️  Transform error: {e}")
            raise
    
    def get_feature_names(self) -> List[str]:
        """Get the names of all transformed features.
        
        Returns:
            List of feature names
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted.")
        
        feature_names = []
        
        if self.numeric_features:
            feature_names.extend(self.numeric_features)
        
        if self.categorical_features:
            cat_encoder = self.preprocessor.named_transformers_['cat']
            for i, category in enumerate(cat_encoder.categories_):
                feature_names.extend([f"{self.categorical_features[i]}_{cat}" for cat in category])
        
        self.feature_names = feature_names
        return feature_names
    
    def save_preprocessor(self, name: str = "preprocessor.pkl"):
        """Save the preprocessor so we can use it later.
        
        Args:
            name: Filename to save as
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted.")
        
        path = os.path.join(self.model_dir, name)
        with open(path, 'wb') as f:
            pickle.dump(self.preprocessor, f)
        print(f"✓ Saved to {path}")
    
    def load_preprocessor(self, name: str = "preprocessor.pkl"):
        """Load a previously saved preprocessor.
        
        Args:
            name: Filename to load
        """
        path = os.path.join(self.model_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Not found: {path}")
        
        with open(path, 'rb') as f:
            self.preprocessor = pickle.load(f)
        print(f"✓ Loaded from {path}")
    
    def fit_and_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one shot.
        
        Args:
            X: Features to fit and transform
            
        Returns:
            Transformed data
        """
        self.build_preprocessor(X)
        return self.transform_features(X)
    
    def drop_non_predictive_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns that aren't useful for predictions (like customer name).
        
        Args:
            df: Input data
            
        Returns:
            Data without the junk columns
        """
        cols_to_drop = ['RowNumber', 'CustomerId', 'Surname']
        return df.drop(columns=[col for col in cols_to_drop if col in df.columns])


def prepare_data(df: pd.DataFrame, model_dir: str = "models", fit: bool = True) -> Tuple[np.ndarray, np.ndarray, FeatureEngineer]:
    """One function to handle all feature preparation.
    
    Args:
        df: Raw data
        model_dir: Where to save/load the preprocessor
        fit: whether to fit (training) or just transform (inference)
        
    Returns:
        Tuple of (transformed features, target, engineer object)
    """
    engineer = FeatureEngineer(model_dir=model_dir)
    
    # Create features
    df_engineered = engineer.engineer_features(df)
    df_engineered = engineer.drop_non_predictive_columns(df_engineered)
    
    # Split into features and target
    X = df_engineered.drop(columns=['Exited'])
    y = df_engineered['Exited'].values
    
    # Transform
    if fit:
        X_transformed = engineer.fit_and_transform(X)
        engineer.save_preprocessor()
    else:
        engineer.load_preprocessor()
        X_transformed = engineer.transform_features(X)
    
    return X_transformed, y, engineer


if __name__ == "__main__":
    # Quick test
    from data_ingestion import load_and_validate
    
    data_path = Path(__file__).parent.parent / "data" / "Churn_Modelling.csv"
    df = load_and_validate(str(data_path))
    
    X_transformed, y, engineer = prepare_data(df)
    print(f"\n✓ Final shape: {X_transformed.shape}")
    print(f"✓ Churn: {y.sum()} out of {len(y)}")
