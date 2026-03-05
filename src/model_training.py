"""
Model training module for churn prediction pipeline.
Trains, evaluates, and compares multiple model types with proper methodology.
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from typing import Tuple, Dict, Any, List

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score, precision_score, recall_score,
    accuracy_score, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


class ModelTrainer:
    """
    Handles model training, evaluation, and comparison.
    """
    
    def __init__(self, model_dir: str = "models", random_state: int = 42):
        """
        Initialize model trainer.
        
        Args:
            model_dir: Directory to save trained models
            random_state: Random seed for reproducibility
        """
        self.model_dir = model_dir
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        os.makedirs(model_dir, exist_ok=True)
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, val_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets using stratified splitting.
        Ensures class distribution is preserved across splits and prevents data leakage.
        
        Args:
            X: Feature array
            y: Target array
            test_size: Proportion of data for test set
            val_size: Proportion of remaining data for validation
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: train + val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=self.random_state, stratify=y_temp
        )
        
        print(f"✓ Data split:")
        print(f"  - Train: {len(X_train)} samples ({y_train.sum()} churned, {(y_train.sum()/len(y_train)*100):.1f}%)")
        print(f"  - Val:   {len(X_val)} samples ({y_val.sum()} churned, {(y_val.sum()/len(y_val)*100):.1f}%)")
        print(f"  - Test:  {len(X_test)} samples ({y_test.sum()} churned, {(y_test.sum()/len(y_test)*100):.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
        """
        Train Logistic Regression model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        print("\n=== Training Logistic Regression ===")
        model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced'  # Handle class imbalance
        )
        model.fit(X_train, y_train)
        print("✓ Logistic Regression trained")
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """
        Train Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        print("\n=== Training Random Forest ===")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=self.random_state,
            class_weight='balanced',  # Handle class imbalance
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        print("✓ Random Forest trained")
        return model
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBClassifier:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        print("\n=== Training XGBoost ===")
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            random_state=self.random_state,
            scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),  # Handle class imbalance
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        print("✓ XGBoost trained")
        return model
    
    def evaluate_model(self, model: Any, X_val: np.ndarray, y_val: np.ndarray, 
                      model_name: str, threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate model on validation set.
        
        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation target
            model_name: Name of the model
            threshold: Probability threshold for classification
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_val, y_pred_proba),
        }
        
        print(f"\n{model_name} Validation Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def train_and_evaluate_all(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Train and evaluate all models.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Dictionary with results for each model
        """
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)
        
        # Train models
        self.models['logistic_regression'] = self.train_logistic_regression(X_train, y_train)
        self.models['random_forest'] = self.train_random_forest(X_train, y_train)
        self.models['xgboost'] = self.train_xgboost(X_train, y_train)
        
        # Evaluate models
        print("\n" + "="*50)
        print("EVALUATING MODELS")
        print("="*50)
        
        self.results['logistic_regression'] = self.evaluate_model(
            self.models['logistic_regression'], X_val, y_val, "Logistic Regression"
        )
        self.results['random_forest'] = self.evaluate_model(
            self.models['random_forest'], X_val, y_val, "Random Forest"
        )
        self.results['xgboost'] = self.evaluate_model(
            self.models['xgboost'], X_val, y_val, "XGBoost"
        )
        
        # Select best model based on ROC-AUC (best metric for imbalanced data)
        best_roc_auc = -1
        for model_name, metrics in self.results.items():
            if metrics['roc_auc'] > best_roc_auc:
                best_roc_auc = metrics['roc_auc']
                self.best_model_name = model_name
                self.best_model = self.models[model_name]
        
        print(f"\n✓ Best model: {self.best_model_name} (ROC-AUC: {best_roc_auc:.4f})")
        
        return self.results
    
    def test_model(self, model_name: str = None) -> Dict[str, Any]:
        """
        Evaluate best model on test set.
        
        Args:
            model_name: Name of model to test (uses best if not specified)
            
        Returns:
            Dictionary with test results
        """
        if not hasattr(self, 'X_test') or not hasattr(self, 'y_test'):
            raise ValueError("Test data not available. Must call split_data first and store X_test, y_test.")
        
        if model_name is None:
            model = self.best_model
            name = self.best_model_name
        else:
            model = self.models[model_name]
            name = model_name
        
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        test_results = {
            'model_name': name,
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
        }
        
        print(f"\n=== Test Set Results ({name}) ===")
        print(f"Accuracy:  {test_results['accuracy']:.4f}")
        print(f"Precision: {test_results['precision']:.4f}")
        print(f"Recall:    {test_results['recall']:.4f}")
        print(f"F1-Score:  {test_results['f1']:.4f}")
        print(f"ROC-AUC:   {test_results['roc_auc']:.4f}")
        print(f"\nConfusion Matrix:\n{np.array(test_results['confusion_matrix'])}")
        
        return test_results
    
    def save_best_model(self):
        """Save the best trained model to disk."""
        if self.best_model is None:
            raise ValueError("No best model to save. Train models first.")
        
        path = os.path.join(self.model_dir, f"best_model_{self.best_model_name}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"✓ Saved best model to {path}")
    
    def load_best_model(self, model_name: str = None):
        """Load a trained model from disk."""
        if model_name is None:
            model_name = self.best_model_name
        
        path = os.path.join(self.model_dir, f"best_model_{model_name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        
        with open(path, 'rb') as f:
            self.best_model = pickle.load(f)
        print(f"✓ Loaded model from {path}")
    
    def get_feature_importance(self, n_features: int = 20) -> pd.DataFrame:
        """
        Get feature importance from tree-based models.
        
        Args:
            n_features: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.best_model_name not in ['random_forest', 'xgboost']:
            print(f"⚠ Feature importance not available for {self.best_model_name}")
            return None
        
        importances = self.best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:n_features]
        
        importance_df = pd.DataFrame({
            'feature_index': indices,
            'importance': importances[indices]
        })
        
        return importance_df


def train_pipeline(X: np.ndarray, y: np.ndarray, model_dir: str = "models") -> Tuple[ModelTrainer, Dict[str, Any]]:
    """
    Convenience function to run full training pipeline.
    
    Args:
        X: Feature array
        y: Target array
        model_dir: Directory for saving models
        
    Returns:
        Tuple of (trainer, test_results)
    """
    trainer = ModelTrainer(model_dir=model_dir)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X, y)
    trainer.X_test = X_test
    trainer.y_test = y_test
    
    # Train and evaluate
    trainer.train_and_evaluate_all(X_train, y_train, X_val, y_val)
    
    # Test
    test_results = trainer.test_model()
    
    # Save
    trainer.save_best_model()
    
    return trainer, test_results


if __name__ == "__main__":
    # Example usage
    from data_ingestion import load_and_validate
    from feature_engineering import prepare_data
    
    data_path = Path(__file__).parent.parent / "data" / "Churn_Modelling.csv"
    df = load_and_validate(str(data_path))
    X, y, engineer = prepare_data(df)
    
    trainer, results = train_pipeline(X, y)
