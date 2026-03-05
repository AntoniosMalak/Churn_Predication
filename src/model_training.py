"""
Training and evaluating models for customer churn prediction.
Train a few different models, see which one wins, use that one.
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


class ModelTrainer:
    """Trains and compares different models. Picks the best one."""
    
    def __init__(self, model_dir: str = "models", random_state: int = 42, imbalance_strategy: str = "class_weight"):
        """Set up the model trainer.
        
        Args:
            model_dir: Where to save trained models
            random_state: Keep this fixed so results are reproducible
            imbalance_strategy: How to handle class imbalance:
                - 'class_weight': Use class_weight='balanced' (default, built into models)
                - 'smote': Use SMOTE oversampling (requires imblearn)
                - 'undersampling': Randomly undersample majority class
                - 'combined': SMOTE + undersampling
        """
        self.model_dir = model_dir
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.preprocessor = None
        self.imbalance_strategy = imbalance_strategy
        self.X_train_balanced = None
        self.y_train_balanced = None
        os.makedirs(model_dir, exist_ok=True)

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Compute standard classification metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
        }

    def evaluate_split(self, model: Any, X_data: np.ndarray, y_data: np.ndarray,
                       split_name: str, threshold: float = 0.5) -> Dict[str, float]:
        """Evaluate a model on one split and print metrics."""
        y_pred_proba = model.predict_proba(X_data)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        metrics = self._calculate_metrics(y_data, y_pred, y_pred_proba)

        print(f"\n{split_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        return metrics

    def assess_fit_quality(self, train_metrics: Dict[str, float], ref_metrics: Dict[str, float],
                           ref_name: str) -> Dict[str, Any]:
        """Assess whether model appears overfit, underfit, or reasonably fit."""
        roc_gap = train_metrics['roc_auc'] - ref_metrics['roc_auc']
        f1_gap = train_metrics['f1'] - ref_metrics['f1']

        if train_metrics['roc_auc'] < 0.70 and ref_metrics['roc_auc'] < 0.70:
            status = "underfitting"
            reason = "Both train and evaluation ROC-AUC are low."
        elif roc_gap > 0.05 or f1_gap > 0.08:
            status = "overfitting"
            reason = f"Train performance is noticeably higher than {ref_name.lower()}."
        else:
            status = "good fit"
            reason = f"Train and {ref_name.lower()} performance are reasonably close."

        assessment = {
            'status': status,
            'reason': reason,
            'roc_auc_gap': roc_gap,
            'f1_gap': f1_gap
        }

        print("\nFIT DIAGNOSIS:")
        print(f"  Status: {status}")
        print(f"  Reason: {reason}")
        print(f"  ROC-AUC gap (Train - {ref_name}): {roc_gap:.4f}")
        print(f"  F1 gap (Train - {ref_name}): {f1_gap:.4f}")
        return assessment
    
    def handle_class_imbalance(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply imbalance handling strategy to training data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Balanced (X_train, y_train)
        """
        class_dist = np.bincount(y_train)
        imbalance_ratio = class_dist[1] / class_dist[0]
        
        print(f"\nClass distribution: {dict(zip([0, 1], class_dist))}")
        print(f"Imbalance ratio: {imbalance_ratio:.2%} (minority/majority)")
        
        if self.imbalance_strategy == "class_weight":
            print("Strategy: Using class_weight='balanced' in models")
            return X_train, y_train
        
        
        if self.imbalance_strategy == "smote":
            print("Strategy: SMOTE oversampling (creating synthetic minority samples)")
            smote = SMOTE(random_state=self.random_state)
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        
        elif self.imbalance_strategy == "undersampling":
            print("Strategy: Random undersampling (removing majority class samples)")
            undersampler = RandomUnderSampler(random_state=self.random_state)
            X_balanced, y_balanced = undersampler.fit_resample(X_train, y_train)
        
        elif self.imbalance_strategy == "combined":
            print("Strategy: Combined SMOTE + undersampling")
            pipeline = ImbPipeline([
                ('smote', SMOTE(random_state=self.random_state)),
                ('undersampler', RandomUnderSampler(random_state=self.random_state))
            ])
            X_balanced, y_balanced = pipeline.fit_resample(X_train, y_train)
        
        else:
            print(f"Unknown strategy: {self.imbalance_strategy}. Using original data.")
            return X_train, y_train
        
        new_dist = np.bincount(y_balanced)
        print(f"After balancing: {dict(zip([0, 1], new_dist))}")
        
        self.X_train_balanced = X_balanced
        self.y_train_balanced = y_balanced
        
        return X_balanced, y_balanced
    
    def split_data_flexible(self, X: np.ndarray, y: np.ndarray, 
                           test_size: float = 0.2, use_validation: bool = False) -> Tuple:
        """Split data into train/test (optionally with validation).
        
        IMPORTANT: Scaling and encoding happens AFTER this split to prevent data leakage.
        The preprocessor is fit ONLY on train data, then applied to all splits.
        
        Args:
            X: Features
            y: Target (0/1)
            test_size: How much to save for final testing (usually 20%)
            use_validation: If True, split into train/val/test; if False, only train/test
            
        Returns:
            If use_validation=False: (X_train, X_test, y_train, y_test)
            If use_validation=True: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: set aside test data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        if not use_validation:
            print(f"\nData Split (BEFORE scaling/encoding) - Train/Test Only:")
            print(f"  Train: {len(X_temp):,} samples ({y_temp.sum():,} churned, {(y_temp.sum()/len(y_temp)*100):.1f}%)")
            print(f"  Test:  {len(X_test):,} samples ({y_test.sum():,} churned, {(y_test.sum()/len(y_test)*100):.1f}%)")
            return X_temp, X_test, y_temp, y_test
        
        else:
            # Second split: split remaining into train and validation
            val_size = 0.2 / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size, 
                random_state=self.random_state, stratify=y_temp
            )
            
            print(f"\nData Split (BEFORE scaling/encoding) - Train/Val/Test:")
            print(f"  Train: {len(X_train):,} samples ({y_train.sum():,} churned, {(y_train.sum()/len(y_train)*100):.1f}%)")
            print(f"  Valid: {len(X_val):,} samples ({y_val.sum():,} churned, {(y_val.sum()/len(y_val)*100):.1f}%)")
            print(f"  Test:  {len(X_test):,} samples ({y_test.sum():,} churned, {(y_test.sum()/len(y_test)*100):.1f}%)")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, val_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/val/test while keeping class balance the same everywhere.
        
        IMPORTANT: Scaling and encoding happens AFTER this split to prevent data leakage.
        The preprocessor is fit ONLY on train data, then applied to all splits.
        
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
        
        print(f"\nData Split (BEFORE scaling/encoding):")
        print(f"  Train: {len(X_train):,} samples ({y_train.sum():,} churned, {(y_train.sum()/len(y_train)*100):.1f}%)")
        print(f"  Valid: {len(X_val):,} samples ({y_val.sum():,} churned, {(y_val.sum()/len(y_val)*100):.1f}%)")
        print(f"  Test:  {len(X_test):,} samples ({y_test.sum():,} churned, {(y_test.sum()/len(y_test)*100):.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def cross_validate_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray, 
                            cv_folds: int = 5, scoring: str = 'roc_auc') -> Dict[str, Any]:
        """Evaluate model using k-fold cross-validation.
        
        Args:
            model: Trained or untrained model
            X_train: Training features
            y_train: Training targets
            cv_folds: Number of cross-validation folds
            scoring: Metric to use ('roc_auc', 'f1', 'accuracy', etc.)
            
        Returns:
            Dictionary with CV results
        """
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        print(f"Running {cv_folds}-fold cross-validation...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        
        cv_results = {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'all_scores': cv_scores.tolist(),
            'n_folds': cv_folds
        }
        
        print(f"  Mean {scoring}: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
        
        return cv_results
    
    def grid_search_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, 
                                   model_type: str = 'xgboost', 
                                   param_grid: Optional[Dict] = None,
                                   cv_folds: int = 3) -> Dict[str, Any]:
        """Search for best hyperparameters using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training targets
            model_type: 'logistic_regression', 'random_forest', or 'xgboost'
            param_grid: Parameter grid to search. If None, uses defaults.
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with best parameters and model
        """
        # Define models
        if model_type == 'logistic_regression':
            base_model = LogisticRegression(random_state=self.random_state, class_weight='balanced', max_iter=1000)
            default_grid = {
                'C': [0.1, 1, 10],
                'solver': ['lbfgs', 'liblinear']
            }
        
        elif model_type == 'random_forest':
            base_model = RandomForestClassifier(random_state=self.random_state, class_weight='balanced', n_jobs=-1)
            default_grid = {
                'n_estimators': [3, 5, 10],
                'max_depth': [3, 4, 5],
                'min_samples_split': [3, 4, 5]
            }
        
        elif model_type == 'xgboost':
            base_model = xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss')
            default_grid = {
                'n_estimators': [5, 10, 15, 50, 100],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.1, 0.001],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                'reg_alpha': [0, 0.01, 0.1, 1], # L1
                # 'reg_lambda': [0.1, 1, 5, 10], # L2
            }
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        param_grid = param_grid or default_grid
        
        print(f"\nPerforming grid search for {model_type}...")
        print(f"Parameter grid: {param_grid}")
        
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv_folds, 
            scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return {
            'best_model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
        """Train a simple linear model. Good baseline, easy to understand.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained model
        """
        print("\nTraining Logistic Regression...")
        model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced'  # Make it care about the 20% minority class
        )
        model.fit(X_train, y_train)
        print("Done")
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """Train a forest of decision trees. Can catch complex patterns.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained model
        """
        print("Training Random Forest...")
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
        print("Done")
        return model
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBClassifier:
        """Train XGBoost. Usually the most accurate, bit more complex to tune.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained model
        """
        print("Training XGBoost...")
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            random_state=self.random_state,
            scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),  # Weight minority class
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        print("Done")
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
        return self.evaluate_split(model, X_val, y_val, model_name, threshold=threshold)
    
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
        print("TRAINING MODELS")
        print("="*60)
        
        # Train all three
        self.models['logistic_regression'] = self.train_logistic_regression(X_train, y_train)
        self.models['random_forest'] = self.train_random_forest(X_train, y_train)
        self.models['xgboost'] = self.train_xgboost(X_train, y_train)
        
        # Evaluate all three
        print("\n" + "="*60)
        print("EVALUATING MODELS")
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
        
        print(f"\nWinner: {self.best_model_name} (ROC-AUC: {best_score:.4f})")
        
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
        
        print(f"\nTEST RESULTS ({name}):")
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
        print(f"Model saved to {path}")
    
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
        print(f"Loaded from {path}")
    
    def get_feature_importance(self, n_features: int = 20) -> pd.DataFrame:
        """For tree-based models, show which features matter most.
        
        Args:
            n_features: Top N features to show
            
        Returns:
            DataFrame with feature importance
        """
        if self.best_model_name not in ['random_forest', 'xgboost']:
            print(f"ERROR: Feature importance not available for {self.best_model_name}")
            return None
        
        importances = self.best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:n_features]
        
        importance_df = pd.DataFrame({
            'feature_index': indices,
            'importance': importances[indices]
        })
        
        return importance_df
    
    def fit_preprocessor_on_train(self, X_train: np.ndarray, engineer) -> None:
        """Fit preprocessor (scaler + encoder) on train data ONLY to prevent data leakage.
        
        Args:
            X_train: Training features (must be DataFrame with column names)
            engineer: FeatureEngineer object
        """
        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train must be a DataFrame with column names for preprocessing")
        
        print(f"Fitting preprocessor on {len(X_train):,} training samples...")
        engineer.build_preprocessor(X_train)
        self.preprocessor = engineer.preprocessor
        print("Preprocessor fitted on train data only")
    
    def transform_all_splits(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, engineer) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform all splits using preprocessor fitted on train data.
        
        This prevents data leakage: scaler statistics come only from train data.
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            engineer: FeatureEngineer object with fitted preprocessor
            
        Returns:
            Tuple of (X_train_scaled, X_val_scaled, X_test_scaled)
        """
        print(f"\nTransforming train data ({len(X_train):,} samples)...")
        X_train_scaled = engineer.transform_features(X_train)
        
        print(f"Transforming validation data ({len(X_val):,} samples)...")
        X_val_scaled = engineer.transform_features(X_val)
        
        print(f"Transforming test data ({len(X_test):,} samples)...")
        X_test_scaled = engineer.transform_features(X_test)
        
        print(f"All splits transformed consistently")
        return X_train_scaled, X_val_scaled, X_test_scaled


def train_pipeline(X: np.ndarray, y: np.ndarray, model_dir: str = "models", engineer = None, 
                   use_validation: bool = False, imbalance_strategy: str = "class_weight",
                   use_cross_validation: bool = False, cv_folds: int = 5,
                   use_grid_search: bool = False, grid_search_params: Optional[Dict] = None) -> Tuple[ModelTrainer, Dict[str, Any]]:
    """One function to handle all model training and evaluation with proper data leakage prevention.
    
    Pipeline order:
    1. Split data into train/test (optionally with validation)
    2. Fit preprocessor (scaler + encoder) on TRAIN data ONLY
    3. Transform all splits using the fitted preprocessor
    4. Handle class imbalance on training data
    5. Train models on transformed data
    6. Evaluate using cross-validation or validation set
    7. Optionally perform grid search for hyperparameter tuning
    
    Args:
        X: Features (as DataFrame for preprocessing)
        y: Target
        model_dir: Where to save models
        engineer: FeatureEngineer object (optional, will be created if not provided)
        use_validation: If True, use train/val/test split; if False, use train/test only
        imbalance_strategy: How to handle class imbalance:
            - 'class_weight': Use class_weight='balanced' (default)
            - 'smote': Use SMOTE oversampling
            - 'undersampling': Random undersampling
            - 'combined': SMOTE + undersampling
        use_cross_validation: If True, use k-fold cross-validation instead of validation set
        cv_folds: Number of cross-validation folds
        use_grid_search: If True, perform grid search for hyperparameter tuning
        grid_search_params: Custom parameters for grid search by model type
        
    Returns:
        Tuple of (trainer object, test results)
    """
    from feature_engineering import FeatureEngineer
    
    trainer = ModelTrainer(model_dir=model_dir, imbalance_strategy=imbalance_strategy)
    
    # Step 1: Split data FIRST (before any scaling/encoding)
    print("\n" + "="*60)
    print("STEP 1: SPLITTING DATA")
    print("="*60)
    
    if use_validation:
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data_flexible(X, y, use_validation=True)
    else:
        X_train, X_test, y_train, y_test = trainer.split_data_flexible(X, y, use_validation=False)
        X_val, y_val = None, None
    
    trainer.X_test = X_test
    trainer.y_test = y_test
    
    # Step 2: Fit preprocessor on train data ONLY (prevent data leakage!)
    print("\n" + "="*60)
    print("STEP 2: FITTING PREPROCESSOR ON TRAIN DATA")
    print("="*60)
    if engineer is None:
        engineer = FeatureEngineer(model_dir=model_dir)
    trainer.fit_preprocessor_on_train(X_train, engineer)
    
    # Step 3: Transform all splits using train-fitted preprocessor
    print("\n" + "="*60)
    print("STEP 3: TRANSFORMING ALL SPLITS")
    print("="*60)
    
    if use_validation:
        X_train_scaled, X_val_scaled, X_test_scaled = trainer.transform_all_splits(X_train, X_val, X_test, engineer)
    else:
        print(f"\nTransforming train data ({len(X_train):,} samples)...")
        X_train_scaled = engineer.transform_features(X_train)
        
        print(f"Transforming test data ({len(X_test):,} samples)...")
        X_test_scaled = engineer.transform_features(X_test)
        
        print(f"All splits transformed consistently")
        X_val_scaled = None
    
    trainer.X_test = X_test_scaled  # Replace with scaled version
    
    # Step 4: Handle class imbalance on training data
    print("\n" + "="*60)
    print("STEP 4: HANDLING CLASS IMBALANCE")
    print("="*60)
    X_train_balanced, y_train_balanced = trainer.handle_class_imbalance(X_train_scaled, y_train)
    
    # Step 5: Train and compare models
    print("\n" + "="*60)
    print("STEP 5: TRAINING MODELS")
    print("="*60)
    
    if use_grid_search:
        print("\nGrid search enabled - tuning hyperparameters...\n")
        for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
            print(f"\nSearching for {model_name}...")
            gs_result = trainer.grid_search_hyperparameters(
                X_train_balanced, y_train_balanced, 
                model_type=model_name,
                param_grid=grid_search_params.get(model_name) if grid_search_params else None,
                cv_folds=cv_folds
            )
            trainer.models[model_name] = gs_result['best_model']
    else:
        trainer.models['logistic_regression'] = trainer.train_logistic_regression(X_train_balanced, y_train_balanced)
        trainer.models['random_forest'] = trainer.train_random_forest(X_train_balanced, y_train_balanced)
        trainer.models['xgboost'] = trainer.train_xgboost(X_train_balanced, y_train_balanced)
    
    # Step 6: Evaluate models
    print("\n" + "="*60)
    if use_cross_validation:
        print(f"STEP 6: CROSS-VALIDATION EVALUATION ({cv_folds}-Fold)")
    else:
        print("STEP 6: VALIDATION SET EVALUATION")
    print("="*60)
    
    if use_cross_validation:
        print("\nEvaluating models using cross-validation...")
        for model_name, model in trainer.models.items():
            print(f"\n{model_name.upper()}:")
            cv_result = trainer.cross_validate_model(model, X_train_balanced, y_train_balanced, cv_folds=cv_folds)
            trainer.results[model_name] = cv_result
        
        # Pick best model based on CV mean score
        best_score = -1
        for name, result in trainer.results.items():
            if result['mean_score'] > best_score:
                best_score = result['mean_score']
                trainer.best_model_name = name
                trainer.best_model = trainer.models[name]
        
        print(f"\nWinner (by CV score): {trainer.best_model_name} (Mean ROC-AUC: {best_score:.4f})")
    
    else:
        if use_validation:
            print("\nEvaluating models on validation set...")
            print("="*60)
            # Manually evaluate each model on validation set
            for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
                trainer.results[model_name] = trainer.evaluate_model(
                    trainer.models[model_name], X_val_scaled, y_val, model_name.replace('_', ' ').title()
                )
            
            # Pick best model based on ROC-AUC
            best_score = -1
            for name, result in trainer.results.items():
                if result['roc_auc'] > best_score:
                    best_score = result['roc_auc']
                    trainer.best_model_name = name
                    trainer.best_model = trainer.models[name]
            
            print(f"\nWinner: {trainer.best_model_name} (ROC-AUC: {best_score:.4f})")
        else:
            print("\nNo validation set provided - using training data for model selection")
            for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
                trainer.results[model_name] = trainer.evaluate_model(
                    trainer.models[model_name], X_train_balanced, y_train_balanced, model_name.replace('_', ' ').title()
                )
            
            # Pick best model
            best_score = -1
            for name, result in trainer.results.items():
                if result['roc_auc'] > best_score:
                    best_score = result['roc_auc']
                    trainer.best_model_name = name
                    trainer.best_model = trainer.models[name]
            
            print(f"\nWinner: {trainer.best_model_name} (ROC-AUC: {best_score:.4f})")
    
    
    # Step 7: Final test
    print("\n" + "="*60)
    print("STEP 7: FINAL TEST EVALUATION")
    print("="*60)
    test_results = trainer.test_model()

    # Step 8: Best model train-vs-eval diagnostics
    print("\n" + "="*60)
    print("STEP 8: FIT CHECK (OVERFITTING / UNDERFITTING)")
    print("="*60)
    train_results = trainer.evaluate_split(
        trainer.best_model, X_train_scaled, y_train, "TRAIN RESULTS (BEST MODEL)"
    )

    if use_validation and (X_val_scaled is not None) and (y_val is not None):
        reference_results = trainer.evaluate_split(
            trainer.best_model, X_val_scaled, y_val, "VALIDATION RESULTS (BEST MODEL)"
        )
        fit_assessment = trainer.assess_fit_quality(train_results, reference_results, "Validation")
    else:
        reference_results = {
            'accuracy': test_results['accuracy'],
            'precision': test_results['precision'],
            'recall': test_results['recall'],
            'f1': test_results['f1'],
            'roc_auc': test_results['roc_auc'],
        }
        fit_assessment = trainer.assess_fit_quality(train_results, reference_results, "Test")

    test_results['train_results'] = train_results
    test_results['fit_assessment'] = fit_assessment
    
    # Save the winner and preprocessor
    print("\n" + "="*60)
    print("SAVING ARTIFACTS")
    print("="*60)
    trainer.save_best_model()
    engineer.save_preprocessor()
    
    return trainer, test_results


if __name__ == "__main__":
    from data_ingestion import load_and_validate
    from feature_engineering import prepare_data
    
    data_path = Path(__file__).parent.parent / "data" / "Churn_Modelling.csv"
    df = load_and_validate(str(data_path))
    X, y, engineer = prepare_data(df)
    
    trainer, results = train_pipeline(X, y)
