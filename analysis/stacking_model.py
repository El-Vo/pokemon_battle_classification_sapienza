import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from typing import Optional

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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
            ('lr', self.train_logistic_regression_model(X_train, y_train)),
            ('rf', self.train_random_forest_model(X_train, y_train)),
            ('knn', self.train_knn_model(X_train, y_train))
        ]

        # --- Meta-learner ---
        final_estimator = SklearnLogisticRegression()  # combines base model predictions

        # --- Define Stacking ensemble ---
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5  # cross-validation for base model predictions
        )

        # --- Train and evaluate ---
        stacking_clf.fit(X_train, y_train)
        y_pred = stacking_clf.predict(X_test)

        print(f"Stacking Classifier Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        self.model = stacking_clf
        if log_result:
            self._log_training_accuracy(X_train, y_train)
        return self.model
    
    def train_random_forest_model(self, X_train, y_train) -> RandomForestClassifier:
        """Train a Random Forest model with hyperparameter tuning using GridSearchCV."""
        # Define the parameter grid to search
        param_grid = {
            'randomforestclassifier__n_estimators': [100, 200, 300],
            'randomforestclassifier__max_depth': [10, 20, 30, None],
            'randomforestclassifier__min_samples_split': [2, 5, 10],
            'randomforestclassifier__min_samples_leaf': [1, 2, 4]
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
            scoring='accuracy',
            n_jobs=-1,  # Use all available cores
            cv=5,
            refit=True,
            return_train_score=True
        )

        grid_rf.fit(X_train, y_train)
        
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

        return grid_logreg.best_estimator_ # Use the best model with the best combination of parameters

    def train_knn_model(self, X_train, y_train) -> KNeighborsClassifier:
        """Train a K-Nearest Neighbors model with hyperparameter tuning using GridSearchCV."""
        # Define the parameter grid to search
        param_grid = {
            'kneighborsclassifier__n_neighbors': [3, 5, 7, 9, 11],
            'kneighborsclassifier__weights': ['uniform', 'distance'],
            'kneighborsclassifier__metric': ['euclidean', 'manhattan', 'minkowski'],
            'kneighborsclassifier__p': [1, 2]  # p=1 for manhattan, p=2 for euclidean
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
            scoring='accuracy',
            n_jobs=-1,  # Use all available cores
            cv=5,
            refit=True,
            return_train_score=True
        )

        grid_knn.fit(X_train, y_train)
        
        # Print the best parameters and score
        print("Best KNN Parameters:", grid_knn.best_params_)
        print("Best CrossValidation Score:", grid_knn.best_score_)
        
        return grid_knn.best_estimator_
    
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
