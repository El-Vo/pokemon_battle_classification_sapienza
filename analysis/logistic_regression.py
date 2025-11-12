import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score
from typing import Optional

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline

from analysis.accuracy_results_logger import AccuracyResultsLogger
from analysis.model_performance import (
    ModelPerformanceReport,
    summarize_model_performance,
)


class RunLogisticRegression:

    def __init__(self, train_df: pd.DataFrame, feature_names):
        self.train_df = train_df
        self.train_feature_names = feature_names
        self.model: Optional[Pipeline] = None
        self._results_logger = AccuracyResultsLogger()

    def train_model(self, log_result: bool = False) -> Pipeline:
        # Define our features (X) and target (y)
        X_train = self.train_df[self.train_feature_names]
        y_train = self.train_df["player_won"]

        # Initialize and train the model
        print("Training a simple Logistic Regression model...")

        # Define the parameter grid to search
        param_grid = [
            {
                "logisticregression__C": [0.01, 0.1, 1, 10],
                "logisticregression__penalty": ["l1"],
                "logisticregression__solver": ["liblinear"],
            },
            {
                "logisticregression__C": [0.01, 0.1, 1, 10],
                "logisticregression__penalty": ["l2"],
                "logisticregression__solver": ["liblinear", "lbfgs"],
            },
        ]

        # Use a pipeline: first standardize the data, then apply logistic regression.
        pipeline = make_pipeline(
            StandardScaler(), SklearnLogisticRegression(random_state=42, max_iter=10000)
        )

        # use GridSearchCV to find the best combination of parameters, use 5-fold cross-validation
        grid_logreg = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            n_jobs=4,
            cv=5,
            refit=True,
            return_train_score=True,
        )

        grid_logreg.fit(X_train, y_train)

        cv_results_df = pd.DataFrame(grid_logreg.cv_results_)
        self.model = (
            grid_logreg.best_estimator_
        )  # Use the best model with the best combination of parameters

        """ self._log_gridsearch_res(
            cv_results_df, grid_logreg
        )  """  # Display the result of the gridcvsearch

        if log_result:
            self._log_training_accuracy(X_train, y_train)
        return self.model

    def _log_training_accuracy(self, X_train, y_train) -> None:
        train_preds = self.model.predict(X_train)
        acc = accuracy_score(y_train, train_preds)
        self._results_logger.append_result(
            accuracy=acc, features=self.train_feature_names
        )
        print(f"Training accuracy appended to {self._results_logger.csv_path}")

    def _log_gridsearch_res(self, cv_results_df, grid_logreg):
        # Print the full results table
        print(cv_results_df)

        # Print all tested hyperparameter combinations
        column = cv_results_df["params"].apply(lambda d: list(d.values()))
        print("Tested parameters: \n", column)

        # Print the parameters of the best performing model:
        print("Best model parameters: ", grid_logreg.best_params_)

        best_score = grid_logreg.best_score_
        print("Best ROC_AUC score:", best_score)

    def evaluate_training_performance(self) -> ModelPerformanceReport:
        if self.model is None:
            raise ValueError("Model must be trained before evaluation.")

        X_train = self.train_df[self.train_feature_names]
        y_train = self.train_df["player_won"]
        return summarize_model_performance(
            self.model,
            X_train,
            y_train,
            feature_names=self.train_feature_names,
        )

    def evaluate_dataset(
        self,
        data_frame: pd.DataFrame,
        *,
        target_column: str = "player_won",
    ) -> ModelPerformanceReport:
        if self.model is None:
            raise ValueError("Model must be trained before evaluation.")

        X = data_frame[self.train_feature_names]
        y = data_frame[target_column]
        return summarize_model_performance(
            self.model,
            X,
            y,
            feature_names=self.train_feature_names,
        )
