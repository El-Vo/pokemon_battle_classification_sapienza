import pandas as pd
from typing import Optional, cast
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

from analysis.accuracy_results_logger import AccuracyResultsLogger
from analysis.model_performance import (
    ModelPerformanceReport,
    summarize_model_performance,
)


class RunGradientBoosting:

    def __init__(self, train_df: pd.DataFrame, feature_names):
        self.train_df = train_df
        self.train_feature_names = feature_names
        self.model: Optional[HistGradientBoostingClassifier] = None
        self._results_logger = AccuracyResultsLogger()

    def train_model(self, log_result: bool = False) -> HistGradientBoostingClassifier:
        X_train = self.train_df[self.train_feature_names]
        y_train = self.train_df["player_won"]

        param_distributions = {
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [3, 5, None],
            "max_bins": [32, 64, 128],
            "min_samples_leaf": [10, 20, 50],
            "l2_regularization": [0.0, 0.1, 1.0],
        }

        search = RandomizedSearchCV(
            estimator=HistGradientBoostingClassifier(
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
            ),
            param_distributions=param_distributions,
            n_iter=20,
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
            random_state=42,
        )

        search.fit(X_train, y_train)
        self.model = cast(HistGradientBoostingClassifier, search.best_estimator_)

        if log_result:
            self._log_training_accuracy(X_train, y_train)

        return self.model

    def _log_training_accuracy(self, X_train, y_train) -> None:
        if self.model is None:
            return

        train_preds = self.model.predict(X_train)
        acc = float(accuracy_score(y_train, train_preds))
        self._results_logger.append_result(
            accuracy=acc, features=self.train_feature_names
        )
        print(f"Training accuracy appended to {self._results_logger.csv_path}")

    def evaluate_training_performance(self) -> ModelPerformanceReport:
        if self.model is None:
            raise ValueError("Model must be trained before evaluation.")

        X_train = self.train_df[self.train_feature_names]
        y_train = self.train_df["player_won"]
        return summarize_model_performance(
            self.model,
            X_train,
            y_train.tolist(),
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
            y.tolist(),
            feature_names=self.train_feature_names,
        )
