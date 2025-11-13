import pandas as pd
from datetime import datetime
import os


class TestModel:

    def __init__(self, model: any, test_df: pd.DataFrame, features):
        self.model = model
        self.test_df = test_df
        self.test_features = self.test_df[features]

    def test(self, submission_filepath: str):
        # Make predictions on the test data
        print("Generating predictions on the test set...")
        test_predictions = self.model.predict(self.test_features)

        # Create the submission DataFrame
        submission_df = pd.DataFrame(
            {"battle_id": self.test_df["battle_id"], "player_won": test_predictions}
        )

        # Ensure results directory exists and save the DataFrame to a timestamped .csv file
        os.makedirs("results", exist_ok=True)
        now = datetime.now()
        filename = f"{submission_filepath}/submission_{now:%Y%m%d_%H%M%S}.csv"
        submission_df.to_csv(filename, index=False)

        print(f"\nSubmission file created: {filename}")
