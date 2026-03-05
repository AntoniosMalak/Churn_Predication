"""
Data loading and validation for the churn prediction pipeline.
Basically: Load CSV, check if it looks reasonable, give you a heads up if something's off.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Dict, Any


class DataIngestion:
    """Simple class to load and validate customer data."""
    
    def __init__(self, data_path: str):
        """Set up the data loader.
        
        Args:
            data_path: Path to your CSV file with customer data
        """
        self.data_path = data_path
        self.df = None
        self.data_quality_report = {}
    
    def load_data(self) -> pd.DataFrame:
        """Load the CSV and make sure it's not broken.
        
        Returns:
            Your data as a DataFrame
            
        Raises:
            FileNotFoundError: Oops, file doesn't exist
            ValueError: File is empty or something's wrong
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Can't find data file: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        
        if self.df.empty:
            raise ValueError("Data file is empty")
        
        print(f"✓ Loaded {len(self.df)} records with {len(self.df.columns)} columns")
        return self.df
    
    def validate_data(self) -> Dict[str, Any]:
        """
        Perform data validation checks.
        
        Returns:
            Dictionary with validation results and warnings
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        report = {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "duplicates": self.df.duplicated().sum(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "target_distribution": self._check_target_distribution()
        }
        
        self.data_quality_report = report
        
        # Print validation summary
        print("\n=== Data Validation Report ===")
        print(f"Total records: {report['total_rows']}")
        print(f"Duplicate rows: {report['duplicates']}")
        print(f"Missing values:\n{pd.Series(report['missing_values']).to_string()}")
        print(f"\nTarget distribution:\n{report['target_distribution']}")
        
        # Warn about high missing rates
        for col, missing_count in report['missing_values'].items():
            missing_pct = (missing_count / len(self.df)) * 100
            if missing_pct > 20:
                print(f"WARNING: Column '{col}' has {missing_pct:.1f}% missing values")
        
        return report
    
    def _check_target_distribution(self) -> Dict[str, float]:
        """Check the distribution of the target variable (Exited)."""
        if 'Exited' in self.df.columns:
            dist = self.df['Exited'].value_counts().to_dict()
            pct = (self.df['Exited'].value_counts(normalize=True) * 100).to_dict()
            print(f"  - Churned (1): {dist.get(1, 0)} ({pct.get(1, 0):.1f}%)")
            print(f"  - Active (0): {dist.get(0, 0)} ({pct.get(0, 0):.1f}%)")
            return dist
        return {}
    
    def get_data_summary(self) -> pd.DataFrame:
        """
        Get descriptive statistics of the data.
        
        Returns:
            DataFrame with summary statistics
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return self.df.describe()
    
    def get_column_info(self) -> pd.DataFrame:
        """
        Get information about columns (names, types, non-null counts).
        
        Returns:
            DataFrame with column information
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return pd.DataFrame({
            'column': self.df.columns,
            'dtype': self.df.dtypes.values,
            'non_null_count': self.df.count().values,
            'null_count': self.df.isnull().sum().values
        })
    
    def handle_missing_values(self, strategy: str = 'drop') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            strategy: 'drop' to remove rows with missing values,
                     'mean' to fill numerical columns with mean,
                     'median' to fill numerical columns with median
        
        Returns:
            DataFrame with missing values handled
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        df_copy = self.df.copy()
        
        if strategy == 'drop':
            initial_rows = len(df_copy)
            df_copy = df_copy.dropna()
            dropped_rows = initial_rows - len(df_copy)
            print(f"Dropped {dropped_rows} rows with missing values")
        
        elif strategy == 'mean':
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].mean())
            print(f"Filled numeric columns with mean values")
        
        elif strategy == 'median':
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].median())
            print(f"Filled numeric columns with median values")
        
        self.df = df_copy
        return self.df
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the current dataframe.
        
        Returns:
            Current DataFrame
        """
        return self.df


def load_and_validate(data_path: str) -> pd.DataFrame:
    """
    Convenience function to load and validate data in one step.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        Validated DataFrame
    """
    ingestion = DataIngestion(data_path)
    ingestion.load_data()
    ingestion.validate_data()
    ingestion.handle_missing_values(strategy='mean')
    
    return ingestion.get_data()


if __name__ == "__main__":
    # Example usage
    data_path = Path(__file__).parent.parent / "data" / "Churn_Modelling.csv"
    ingestion = DataIngestion(str(data_path))
    ingestion.load_data()
    ingestion.validate_data()
    print("\n" + ingestion.get_column_info().to_string())
    print("\n" + ingestion.get_data_summary().to_string())
