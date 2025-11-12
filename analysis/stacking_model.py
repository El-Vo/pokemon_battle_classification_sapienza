import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import (
    RandomForestClassifier, 
    StackingClassifier, 
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
import numpy as np
from typing import Optional

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
        
        # --- Base learners ---
        estimators = [
            ('rf', self.train_random_forest_model(X_train, y_train)),
            ('knn', self.train_knn_model(X_train, y_train)),
            ('gb', self.train_gradient_boosting_model(X_train, y_train)),
            ('svm', self.train_svm_model(X_train, y_train))
        ]

        # Calculate correlations between base models
        print("\nAnalyzing base model correlations...")
        self.calculate_model_correlations(X_train, y_train, estimators)

        # --- Stacking classifier ---
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=SklearnLogisticRegression(max_iter=10000, random_state=42),
            passthrough=True,
            cv=5
        )

        # Define two separate parameter grids for different solvers
        param_grid = [
            # Grid 1: L1 penalty with liblinear solver
            {
                'final_estimator__C': [0.01, 0.1, 1, 10],
                'final_estimator__penalty': ['l1'],
                'final_estimator__solver': ['liblinear']
            },
            # Grid 2: L2 penalty with lbfgs solver
            {
                'final_estimator__C': [0.01, 0.1, 1, 10],
                'final_estimator__penalty': ['l2'],
                'final_estimator__solver': ['lbfgs']
            }
        ]

        grid = GridSearchCV(stacking_clf, param_grid=param_grid, cv=5, scoring='roc_auc')
        grid.fit(X_train, y_train)
        
        # Get training accuracy
        y_train_pred = grid.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        
        # Get test accuracy
        y_test_pred = grid.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        print(f"Stacking Classifier Training Accuracy: {train_acc:.3f}")
        print(f"Stacking Classifier Test Accuracy: {test_acc:.3f}")
        
        self.model = grid
        if log_result:
            self._log_training_accuracy(X_train, y_train)
        return self.model
    
    def train_random_forest_model(self, X_train, y_train) -> RandomForestClassifier:
        """Train a Random Forest model with hyperparameter tuning using GridSearchCV."""
        # Define the parameter grid to search
        param_grid = {
            'randomforestclassifier__n_estimators': [100, 200, 300],
            'randomforestclassifier__max_depth': [6, 8, 10],
            'randomforestclassifier__min_samples_split': [5, 10, 20],
            'randomforestclassifier__min_samples_leaf': [2, 4, 6],
            'randomforestclassifier__max_features': ['sqrt', 'log2', 0.5],
            'randomforestclassifier__ccp_alpha': [0.0, 0.005, 0.01]
        }

        # Create a pipeline with RandomForest
        pipeline = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(random_state=42)
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
        
        return grid_logreg.best_estimator_ # Use the best model with the best combination of parameters
    
    def train_gradient_boosting_model(self, X_train, y_train) -> GradientBoostingClassifier:
        """Train a Gradient Boosting model with hyperparameter tuning using GridSearchCV."""
        # Define the parameter grid to search
        param_grid = {
            'gradientboostingclassifier__n_estimators': [100, 200, 300],
            'gradientboostingclassifier__learning_rate': [0.01, 0.05, 0.1],
            'gradientboostingclassifier__max_depth': [3, 5, 7],
            'gradientboostingclassifier__subsample': [0.7, 0.8, 1.0]
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
            'kneighborsclassifier__n_neighbors': [5, 10, 15, 20],
            'kneighborsclassifier__weights': ['uniform', 'distance'],
            'kneighborsclassifier__metric': ['minkowski', 'euclidean', 'manhattan'],
            'kneighborsclassifier__p': [1, 2],
            'kneighborsclassifier__leaf_size': [20, 30, 40]
        }

        # Create a pipeline with KNN
        pipeline = make_pipeline(
            StandardScaler(),  # Scale features for better distance calculations
            KNeighborsClassifier()
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

    def train_svm_model(self, X_train, y_train) -> SVC:
        """Train an SVM model with hyperparameter tuning using GridSearchCV."""
        # Define the parameter grid to search
        param_grid = {
            'svc__C': [0.1, 1, 10],
            'svc__kernel': ['rbf'],
            'svc__gamma': ['scale', 'auto'],
            'svc__class_weight': [None, 'balanced']
        }


        # Create a pipeline with SVM
        pipeline = make_pipeline(
            StandardScaler(),
            SVC(random_state=42, probability=True)  # probability=True needed for voting classifier
        )

        # Use GridSearchCV to find the best combination of parameters
        grid_svm = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring='roc_auc',
            n_jobs=-1,
            cv=5,
            refit=True,
            return_train_score=True
        )

        grid_svm.fit(X_train, y_train)
        
        # Print the best parameters and scores
        print("\n=== Support Vector Machine Results ===")
        print("Best Parameters:", grid_svm.best_params_)
        print(f"Best Cross-Validation ROC-AUC: {grid_svm.best_score_:.3f}")
        
        # Get training accuracy
        best_svm = grid_svm.best_estimator_
        train_acc = accuracy_score(y_train, best_svm.predict(X_train))
        print(f"Training Accuracy: {train_acc:.3f}")
        
        return grid_svm.best_estimator_
    
    def calculate_model_correlations(self, X: pd.DataFrame, y: pd.Series, models) -> None:
        """
        Calculate and print the correlation between base model predictions.
        This helps identify if models are too similar or provide unique perspectives.
        """
        print("\n=== Base Model Prediction Correlations ===")
        
        # Get predictions from each model
        predictions = {}
        for name, model in models:
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

    
    def _log_training_accuracy(self, X_train, y_train) -> None:
        """Log the training accuracy to a CSV file."""
        train_preds = self.model.predict(X_train)
        acc = accuracy_score(y_train, train_preds)
        self._results_logger.append_result(accuracy=acc, features=self.train_feature_names)
        print(f"Training accuracy appended to {self._results_logger.csv_path}")

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
