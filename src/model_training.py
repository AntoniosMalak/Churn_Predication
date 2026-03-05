"""
Training and evaluating models for customer churn prediction.
Train a few different models, see which one wins, use that one.
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from typing import Tuple, Dict, Any, List

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


class ModelTrainer:
    """Trains and compares different models. Picks the best one."""
    
    def __init__(self, model_dir: str = "models", random_state: int = 42):
        """Set up the model trainer.
        
        Args:
            model_dir: Where to save trained models
            random_state: Keep this fixed so results are reproducible
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
        """Split data into train/val/test while keeping class balance the same everywhere.
        
        This is important: if we don't stratify, we might train on different ratios
        of churners vs non-churners, which would mess up our evaluation.
        
        Args:
            X: Features
            y: Target (0/1)
            test_size: How much to save for final testing (usually 20%)
            val_size: How much of the remaining to use for validation
            
        Returns:
            Train, val, and test splits for both features and target
        """
        # First split: set aside test data (untouchable until the very end)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: split remaining into train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=self.random_state, stratify=y_temp
        )
        
        print(f"\n📊 Data Split:")
        print(f"  Train: {len(X_train):,} samples ({y_train.sum():,} churned, {(y_train.sum()/len(y_train)*100):.1f}%)")
        print(f"  Valid: {len(X_val):,} samples ({y_val.sum():,} churned, {(y_val.sum()/len(y_val)*100):.1f}%)")
        print(f"  Test:  {len(X_test):,} samples ({y_test.sum():,} churned, {(y_test.sum()/len(y_test)*100):.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
        """Train a simple linear model. Good baseline, easy to understand.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained model
        """
        print("\n🚂 Training Logistic Regression...")
        model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced'  # Make it care about the 20% minority class
        )
        model.fit(X_train, y_train)
        print("✓ Done")
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """Train a forest of decision trees. Can catch complex patterns.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained model
        """
        print("🌲 Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=100,        # 100 trees
            max_depth=15,            # Prevent overfitting
            min_samples_split=10,    # Need at least 10 samples to split a node
            min_samples_leaf=4,      # Leaves need at least 4 samples
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1  # Use all CPU cores
        )
        model.fit(X_train, y_train)
        print("✓ Done")
        return model
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBClassifier:
        """Train XGBoost. Usually the most accurate, bit more complex to tune.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained model
        """
        print("🚀 Training XGBoost...")
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            random_state=self.random_state,
            scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),  # Weight minority class
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        print("✓ Done")
        return model
    
    def evaluate_model(self, model: Any, X_val: np.ndarray, y_val: np.ndarray, 
                      model_name: str, threshold: float = 0.5) -> Dict[str, float]:
        """Score the model on validation data.
        
        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation targets
            model_name: Name (for printing)
            threshold: Probability threshold (default 0.5)
            
        Returns:
            Dict with all the metrics
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
        
        print(f"\n📈 {model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f} ⭐")
        
        return metrics
    
    def train_and_evaluate_all(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Train all three models and see which one's best.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Results dict for all models
        """
        print("\n" + "="*60)
        print("🎓 TRAINING MODELS")
        print("="*60)
        
        # Train all three
        self.models['logistic_regression'] = self.train_logistic_regression(X_train, y_train)
        self.models['random_forest'] = self.train_random_forest(X_train, y_train)
        self.models['xgboost'] = self.train_xgboost(X_train, y_train)
        
        # Evaluate all three
        print("\n" + "="*60)
        print("🏆 EVALUATING MODELS")
        print("="*60)
        
        self.results['logistic_regression'] = self.evaluate_model(
            self.models['logistic_regression'], X_val, y_val, "Logistic Regression"
        )
        self.results['random_forest'] = self.evaluate_model(
            self.models['random_forest'], X_val, y_val, "Random Forest"
        )
        self.results['xgboost'] = self.evaluate_model(
            self.models['xgboost'], X_val, y_val, "XGBoost"
        )
        
        # Pick the best one (highest ROC-AUC)
        best_score = -1
        for name, metrics in self.results.items():
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                self.best_model_name = name
                self.best_model = self.models[name]
        
        print(f"\n🎯 Winner: {self.best_model_name} (ROC-AUC: {best_score:.4f})")
        
        return self.results
    
    def test_model(self, model_name: str = None) -> Dict[str, Any]:
        """Final evaluation on test set (only use once, at the very end).
        
        Args:
            model_name: Which model to test (defaults to best one)
            
        Returns:
            Test results
        """
        if not hasattr(self, 'X_test') or not hasattr(self, 'y_test'):
            raise ValueError("No test data set. Need to call split_data first and store X_test, y_test.")
        
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
        
        print(f"\n🧪 TEST RESULTS ({name}):")
        print(f"  Accuracy:  {test_results['accuracy']:.4f}")
        print(f"  Precision: {test_results['precision']:.4f}")
        print(f"  Recall:    {test_results['recall']:.4f}")
        print(f"  F1-Score:  {test_results['f1']:.4f}")
        print(f"  ROC-AUC:   {test_results['roc_auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  {np.array(test_results['confusion_matrix'])}")
        
        return test_results
    
    def save_best_model(self):
        """Save the best model so we can use it later for predictions."""
        if self.best_model is None:
            raise ValueError("No best model to save.")
        
        path = os.path.join(self.model_dir, f"best_model_{self.best_model_name}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"✓ Model saved to {path}")
    
    def load_best_model(self, model_name: str = None):
        """Load a previously saved model.
        
        Args:
            model_name: Which model to load
        """
        if model_name is None:
            model_name = self.best_model_name
        
        path = os.path.join(self.model_dir, f"best_model_{model_name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        
        with open(path, 'rb') as f:
            self.best_model = pickle.load(f)
        print(f"✓ Loaded from {path}")
    
    def get_feature_importance(self, n_features: int = 20) -> pd.DataFrame:
        """For tree-based models, show which features matter most.
        
        Args:
            n_features: Top N features to show
            
        Returns:
            DataFrame with feature importance
        """
        if self.best_model_name not in ['random_forest', 'xgboost']:
            print(f"❌ Feature importance not available for {self.best_model_name}")
            return None
        
        importances = self.best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:n_features]
        
        importance_df = pd.DataFrame({
            'feature_index': indices,
            'importance': importances[indices]
        })
        
        return importance_df


def train_pipeline(X: np.ndarray, y: np.ndarray, model_dir: str = "models") -> Tuple[ModelTrainer, Dict[str, Any]]:
    """One function to handle all model training and evaluation.
    
    Args:
        X: Features
        y: Target
        model_dir: Where to save models
        
    Returns:
        Tuple of (trainer object, test results)
    """
    trainer = ModelTrainer(model_dir=model_dir)
    
    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X, y)
    trainer.X_test = X_test
    trainer.y_test = y_test
    
    # Train and compare
    trainer.train_and_evaluate_all(X_train, y_train, X_val, y_val)
    
    # Final test
    test_results = trainer.test_model()
    
    # Save the winner
    trainer.save_best_model()
    
    return trainer, test_results


if __name__ == "__main__":
    # Quick test
    from data_ingestion import load_and_validate
    from feature_engineering import prepare_data
    
    data_path = Path(__file__).parent.parent / "data" / "Churn_Modelling.csv"
    df = load_and_validate(str(data_path))
    X, y, engineer = prepare_data(df)
    
    trainer, results = train_pipeline(X, y)
