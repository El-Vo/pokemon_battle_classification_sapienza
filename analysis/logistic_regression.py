import pandas as pd
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score
from typing import Optional

from analysis.accuracy_results_logger import AccuracyResultsLogger
from analysis.model_performance import ModelPerformanceReport, summarize_model_performance


class RunLogisticRegression:

    def __init__(self, train_df: pd.DataFrame, feature_names):
        self.train_df = train_df
        self.train_feature_names = feature_names
        self.model: Optional[SklearnLogisticRegression] = None
        self._results_logger = AccuracyResultsLogger()

    def train_model(self, log_result: bool = False) -> SklearnLogisticRegression:
        # Define our features (X) and target (y)
        X_train = self.train_df[self.train_feature_names]
        y_train = self.train_df['player_won']

        # Initialize and train the model
        print("Training a simple Logistic Regression model...")
        model = SklearnLogisticRegression(
            random_state=42,
            max_iter=10000,
            solver='lbfgs',
            penalty=None
        )
        model.fit(X_train, y_train)
        self.model = model

        # Compute and print training accuracy
        train_preds = model.predict(X_train)
        acc = accuracy_score(y_train, train_preds)

        if log_result:
            self._log_training_accuracy(X_train, y_train)

        return model

    def _log_training_accuracy(self, X_train, y_train) -> None:
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

    def evaluate_dataset(
        self,
        data_frame: pd.DataFrame,
        *,
        target_column: str = 'player_won',
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


