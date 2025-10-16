from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from typing import Optional


class RunLogisticRegression:

    def __init__(self, train_df: pd.DataFrame, features):
        self.train_df = train_df
        self.features = features
        self.model: Optional[SklearnLogisticRegression] = None

    def train_model(self) -> SklearnLogisticRegression:
        # Define our features (X) and target (y)
        X_train = self.train_df[self.features]
        y_train = self.train_df['player_won']

        # Initialize and train the model
        print("Training a simple Logistic Regression model...")
        model = SklearnLogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        self.model = model

        # Compute and print training accuracy
        train_preds = model.predict(X_train)
        acc = accuracy_score(y_train, train_preds)
        print(f"Model training complete. Training accuracy: {acc:.4f}")

        return model


