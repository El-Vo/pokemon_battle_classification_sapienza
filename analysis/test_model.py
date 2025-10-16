import pandas as pd

class TestModel:

    def __init__(self, model: any, test_df: pd.DataFrame, features):
        self.model = model
        self.test_df = test_df
        self.test_features = self.test_df[features]

    def test(self):
        # Make predictions on the test data
        print("Generating predictions on the test set...")
        test_predictions = self.model.predict(self.test_features)

        # Create the submission DataFrame
        submission_df = pd.DataFrame({
            'battle_id': self.test_df['battle_id'],
            'player_won': test_predictions
        })

        # Save the DataFrame to a .csv file
        submission_df.to_csv('results/submission.csv', index=False)

        print("\n'submission.csv' file created successfully!")