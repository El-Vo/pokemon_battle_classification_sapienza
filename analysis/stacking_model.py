import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import (
    RandomForestClassifier, 
    StackingClassifier, 
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
import numpy as np
from typing import Optional, Dict, List, Tuple

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline


from analysis.accuracy_results_logger import AccuracyResultsLogger
from analysis.model_performance import ModelPerformanceReport, summarize_model_performance


class RunStackingModel:

    def __init__(self, train_df: pd.DataFrame, feature_names):
        self.train_df = train_df
        self.train_feature_names = feature_names
        self.model: Optional[StackingClassifier] = None
        self._results_logger = AccuracyResultsLogger()
    
    def train_stacking_model(self, log_result: bool = False) -> StackingClassifier:
        """Train a Stacking Classifier with Logistic Regression, Random Forest, and KNN as base learners."""
        
        # --- Step 1: Generate a synthetic binary classification dataset ---
        
        X = self.train_df[self.train_feature_names]

        y = self.train_df['player_won']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Calculate correlations between base models
        #print("\nAnalyzing base model correlations...")
        #self.calculate_model_correlations(X_train, y_train)
        
        # --- Base learners ---
        estimators = [
            #('lr', self.train_logistic_regression_model(X_train, y_train)),
            ('rf', self.train_random_forest_model(X_train, y_train)),
            ('knn', self.train_knn_model(X_train, y_train)),
            ('gb', self.train_gradient_boosting_model(X_train, y_train))
        ]

        # --- Meta-learner ---
        meta_learner = self.train_logistic_regression_model(X_train, y_train)

        # --- Define Stacking ensemble ---
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5  # cross-validation for base model predictions
        )

        # --- Train and evaluate ---
        print("\n=== Stacking Classifier Training ===")
        stacking_clf.fit(X_train, y_train)
        
        # Get training accuracy
        y_train_pred = stacking_clf.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        
        # Get test accuracy
        y_test_pred = stacking_clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        print(f"Stacking Classifier Training Accuracy: {train_acc:.3f}")
        print(f"Stacking Classifier Test Accuracy: {test_acc:.3f}")
        
        self.model = stacking_clf
        if log_result:
            self._log_training_accuracy(X_train, y_train)
        return self.model
    
    def train_random_forest_model(self, X_train, y_train) -> RandomForestClassifier:
        """Train a Random Forest model with hyperparameter tuning using GridSearchCV."""
        # Define the parameter grid to search
        param_grid = {
            'randomforestclassifier__max_depth': [10, 20, 30, None]
        }

        # Create a pipeline with RandomForest
        pipeline = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(random_state=42, min_samples_split=2, ccp_alpha=0.01, n_estimators=300)
        )

        # Use GridSearchCV to find the best combination of parameters
        grid_rf = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring='roc_auc',
            n_jobs=-1,  # Use all available cores
            cv=5,
            refit=True,
            return_train_score=True
        )

        grid_rf.fit(X_train, y_train)
        
        # Print the best parameters and scores
        print("\n=== Random Forest Results ===")
        print("Best Parameters:", grid_rf.best_params_)
        print(f"Best Cross-Validation ROC-AUC: {grid_rf.best_score_:.3f}")
        
        # Get training and test accuracy
        best_rf = grid_rf.best_estimator_
        train_acc = accuracy_score(y_train, best_rf.predict(X_train))
        print(f"Training Accuracy: {train_acc:.3f}")
        
        return grid_rf.best_estimator_

    def train_logistic_regression_model(self, X_train, y_train) -> SklearnLogisticRegression:
        """Train a logistic regression model with hyperparameter tuning using GridSearchCV."""
        # Define two separate parameter grids for different solvers
        param_grid = [
            # Grid 1: L1 penalty with liblinear solver
            {
                'logisticregression__C': [0.01, 0.1, 1, 10],
                'logisticregression__penalty': ['l1'],
                'logisticregression__solver': ['liblinear']
            },
            # Grid 2: L2 penalty with lbfgs solver
            {
                'logisticregression__C': [0.01, 0.1, 1, 10],
                'logisticregression__penalty': ['l2'],
                'logisticregression__solver': ['lbfgs']
            }
        ]

        # Use a pipeline: first standardize the data, then apply logistic regression.
        pipeline = make_pipeline(
            StandardScaler(),
            SklearnLogisticRegression(
                random_state=42,
                max_iter=10000
            )
        )

        # use GridSearchCV to find the best combination of parameters, use 5-fold cross-validation
        grid_logreg = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring='roc_auc',
            n_jobs=4,        
            cv=5,        
            refit=True,      
            return_train_score=True
        )

        grid_logreg.fit(X_train, y_train)

        # Print the best parameters and scores
        print("\n=== Logistic Regression Results ===")
        print("Best Parameters:", grid_logreg.best_params_)
        print(f"Best Cross-Validation ROC-AUC: {grid_logreg.best_score_:.3f}")
        
        # Get training and test accuracy
        best_logreg = grid_logreg.best_estimator_
        train_acc = accuracy_score(y_train, best_logreg.predict(X_train))
        print(f"Training Accuracy: {train_acc:.3f}")
        
        return grid_logreg.best_estimator_ # Use the best model with the best combination of parameters

    def train_gradient_boosting_model(self, X_train, y_train) -> GradientBoostingClassifier:
        """Train a Gradient Boosting model with hyperparameter tuning using GridSearchCV."""
        # Define the parameter grid to search
        param_grid = {
            'gradientboostingclassifier__n_estimators': [100, 200, 300],
            'gradientboostingclassifier__learning_rate': [0.01, 0.1, 0.3],
            'gradientboostingclassifier__max_depth': [3, 4, 5]
        }

        # Create a pipeline with GradientBoosting
        pipeline = make_pipeline(
            StandardScaler(),
            GradientBoostingClassifier(random_state=42)
        )

        # Use GridSearchCV to find the best combination of parameters
        grid_gb = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring='roc_auc',
            n_jobs=-1,  # Use all available cores
            cv=5,
            refit=True,
            return_train_score=True
        )

        grid_gb.fit(X_train, y_train)
        
        # Print the best parameters and scores
        print("\n=== Gradient Boosting Results ===")
        print("Best Parameters:", grid_gb.best_params_)
        print(f"Best Cross-Validation ROC-AUC: {grid_gb.best_score_:.3f}")
        
        # Get training accuracy
        best_gb = grid_gb.best_estimator_
        train_acc = accuracy_score(y_train, best_gb.predict(X_train))
        print(f"Training Accuracy: {train_acc:.3f}")
        
        return grid_gb.best_estimator_

    def train_knn_model(self, X_train, y_train) -> KNeighborsClassifier:
        """Train a K-Nearest Neighbors model with hyperparameter tuning using GridSearchCV."""
        # Define the parameter grid to search
        param_grid = {
            'kneighborsclassifier__n_neighbors': [10,15,20,25,30],
            'kneighborsclassifier__p': [1, 2]  # p=1 for manhattan, p=2 for euclidean
        }

        # Create a pipeline with KNN
        pipeline = make_pipeline(
            StandardScaler(),  # Scale features for better distance calculations
            KNeighborsClassifier(weights='uniform')
        )

        # Use GridSearchCV to find the best combination of parameters
        grid_knn = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring='roc_auc',
            n_jobs=-1,  # Use all available cores
            cv=5,
            refit=True,
            return_train_score=True
        )

        grid_knn.fit(X_train, y_train)
        
        # Print the best parameters and scores
        print("\n=== K-Nearest Neighbors Results ===")
        print("Best Parameters:", grid_knn.best_params_)
        print(f"Best Cross-Validation ROC-AUC: {grid_knn.best_score_:.3f}")
        
        # Get training and test accuracy
        best_knn = grid_knn.best_estimator_
        train_acc = accuracy_score(y_train, best_knn.predict(X_train))
        print(f"Training Accuracy: {train_acc:.3f}")
        
        return grid_knn.best_estimator_
    
    def _log_training_accuracy(self, X_train, y_train) -> None:
        """Log the training accuracy to a CSV file."""
        train_preds = self.model.predict(X_train)
        acc = accuracy_score(y_train, train_preds)
        self._results_logger.append_result(accuracy=acc, features=self.train_feature_names)
        print(f"Training accuracy appended to {self._results_logger.csv_path}")

    def calculate_model_correlations(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Calculate and print the correlation between base model predictions.
        This helps identify if models are too similar or provide unique perspectives.
        
        Args:
            X: Feature matrix
            y: True labels
        """
        print("\n=== Base Model Prediction Correlations ===")
        
        # Create and fit individual models
        models = {
            #'LogisticRegression': self.train_logistic_regression_model(X, y),
            'RandomForest': self.train_random_forest_model(X, y),
            'KNN': self.train_knn_model(X, y),
            'GradientBoosting': self.train_gradient_boosting_model(X, y)
        }
        
        # Get predictions from each model
        predictions = {}
        for name, model in models.items():
            pred = model.predict_proba(X)[:, 1]  # Get probability of class 1
            predictions[name] = pred
        
        # Calculate correlations between model predictions
        model_names = list(predictions.keys())
        correlation_matrix = np.zeros((len(model_names), len(model_names)))
        
        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names):
                correlation = np.corrcoef(predictions[name1], predictions[name2])[0, 1]
                correlation_matrix[i, j] = correlation
        
        # Print correlation matrix
        print("\nPrediction Correlation Matrix:")
        print("=" * 60)
        print(f"{'':20}", end="")
        for name in model_names:
            print(f"{name:>15}", end="")
        print("\n" + "-" * 60)
        
        for i, name1 in enumerate(model_names):
            print(f"{name1:20}", end="")
            for j in range(len(model_names)):
                print(f"{correlation_matrix[i,j]:15.3f}", end="")
            print()
        
        # Print interpretation
        print("\nInterpretation:")
        print("- Correlation close to 1: Models make very similar predictions")
        print("- Correlation close to 0: Models make independent predictions")
        print("- Lower correlations between models are better for stacking")
        
        # Print recommendations
        high_correlation_threshold = 0.8
        high_correlations = []
        
        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names[i+1:], i+1):
                if correlation_matrix[i,j] > high_correlation_threshold:
                    high_correlations.append((name1, name2, correlation_matrix[i,j]))
        
        if high_correlations:
            print("\nRecommendations:")
            print("The following model pairs have high correlation (>0.8) and might be redundant:")
            for name1, name2, corr in high_correlations:
                print(f"- {name1} and {name2}: {corr:.3f}")
            print("Consider replacing one of each highly correlated pair with a different model type.")

    def evaluate_training_performance(self) -> ModelPerformanceReport:
        if self.model is None:
            raise ValueError("Model must be trained before evaluation.")

        X_train = self.train_df[self.train_feature_names]
        y_train = self.train_df['player_won']
        return summarize_model_performance(
            self.model,
            X_train,
            y_train,
            feature_names=self.train_feature_names,
        )
