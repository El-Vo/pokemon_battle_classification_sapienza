from typing import Any, Optional, Sequence

from pandas import DataFrame


class SingleBattlePredictor:
    """Handle single-battle predictions and reporting for a trained model."""

    def __init__(self, model: Any, features: Sequence[str]):
        if not features:
            raise ValueError("Feature list must not be empty.")
        self.model = model
        self.features = list(features)

    def predict(
        self, battles_df: DataFrame, random_state: Optional[int] = None
    ) -> None:
        if battles_df.empty:
            print("No battle data provided – unable to run prediction.")
            return

        sample_df = (
            battles_df
            if len(battles_df) == 1
            else battles_df.sample(n=1, random_state=random_state)
        )

        battle_row = sample_df.iloc[0]
        battle_features = sample_df[self.features]

        test_predictions = self.model.predict(battle_features)
        test_probabilities = self.model.predict_proba(battle_features)

        battle_identifier = battle_row.get("battle_id", 1)

        print("\n=== PREDICTION ANALYSIS ===")
        print(f"Battle {battle_identifier} was chosen.")
        print(
            f"\nPrediction: {'Player 1 wins' if test_predictions[0] == 1 else 'Player 1 loses'}"
        )
        print(f"Probability of loss: {test_probabilities[0][0]:.2%}")
        print(f"Probability of win: {test_probabilities[0][1]:.2%}")

        print("\n=== FEATURE VALUES ===")
        feature_values = battle_row[self.features]
        for feature, value in feature_values.items():
            print(f"{feature}: {value}")

        logistic_step = None
        if hasattr(self.model, "named_steps"):
            logistic_step = self.model.named_steps.get("logisticregression")
        elif hasattr(self.model, "steps"):
            for _, step in self.model.steps:
                if hasattr(step, "coef_"):
                    logistic_step = step
                    break

        if logistic_step is not None and hasattr(logistic_step, "coef_"):
            coefficients = logistic_step.coef_[0]
            feature_importance = list(zip(self.features, coefficients, feature_values))

            print("\n=== ACTUAL CONTRIBUTION TO PREDICTION (Top 10) ===")
            contributions = [
                (feature, coef * value, value)
                for feature, coef, value in feature_importance
            ]
            contributions_sorted = sorted(
                contributions, key=lambda item: abs(item[1]), reverse=True
            )

            for feature, contribution, value in contributions_sorted[:10]:
                print(
                    f"  {feature:40s} | Contrib: {contribution:+.4f} | Value: {value:6.2f}"
                )
        else:
            print(
                "\n=== ACTUAL CONTRIBUTION TO PREDICTION ===\n"
                "Coefficient-based contributions unavailable for this model."
            )


__all__ = ["SingleBattlePredictor"]
