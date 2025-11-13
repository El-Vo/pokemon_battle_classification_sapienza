from typing import Any

from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

from prepare_data.import_source import ImportSource
from analysis.create_simple_features import BattleFeatureExtractor
from analysis.logistic_regression import RunLogisticRegression
from analysis.stacking_model import RunStackingModel
from analysis.gradient_boosting import RunGradientBoosting
from analysis.test_model import TestModel
from analysis.feature_correlation import FeatureCorrelationAnalyzer


def create_training_features(
    selected_features: list[str],
) -> tuple[DataFrame, list]:
    # Check if training data is available under data/train.jsonl
    train_data_importer = ImportSource()
    train_data_importer.load_jsonl("./data/train.jsonl")
    # train_data_importer.display_first_battle()

    train_data_extractor = BattleFeatureExtractor(
        train_data_importer.data, list(selected_features)
    )
    train_df = train_data_extractor.process()

    # Exclude battle_id (because it has no informative value) and player_won (because this is the variable we want to predict) from training features
    train_feature_names = [
        col for col in train_df.columns if col not in ["battle_id", "player_won"]
    ]

    return train_df, train_feature_names


def train_model(
    train_dataframe: DataFrame,
    train_feature_names: list,
    log_results: bool = False,
    algorithm: str = "logistic_regression",
) -> Any:
    trainers = {
        "logistic_regression": RunLogisticRegression,
        "gradient_boosting": RunGradientBoosting,
    }

    trainer_cls = trainers.get(algorithm)
    if trainer_cls is None:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    trainer = trainer_cls(train_dataframe, train_feature_names)
    model = trainer.train_model(log_results)

    performance_report = trainer.evaluate_training_performance()
    print("\nTraining performance summary:\n")
    performance_report.print(top_k=10)

    return model


def train_stacking_model(
    train_dataframe: DataFrame, train_feature_names: list, log_results: bool = False
) -> StackingClassifier:
    trainer = RunStackingModel(train_dataframe, train_feature_names)
    model = trainer.train_stacking_model(log_results)
    performance_report = trainer.evaluate_training_performance()
    print("\nTraining performance summary:\n")
    performance_report.print(top_k=10)
    return model


def test_trained_model(model: Any, features: list) -> None:
    test_data_importer = ImportSource()
    test_data_importer.load_jsonl("./data/test.jsonl")
    test_data_extractor = BattleFeatureExtractor(
        test_data_importer.data, list(features)
    )
    test_df = test_data_extractor.process()
    test = TestModel(model, test_df, features)
    test.test()


def predict_single_battle(
    model: Any,
    features: list,
    battles_df: DataFrame,
    random_state: int | None = None,
):
    if battles_df.empty:
        print("No battle data provided – unable to run prediction.")
        return

    sample_df = (
        battles_df
        if len(battles_df) == 1
        else battles_df.sample(n=1, random_state=random_state)
    )

    battle_row = sample_df.iloc[0]
    battle_features = sample_df[features]

    # Prediction
    test_predictions = model.predict(battle_features)
    test_probabilities = model.predict_proba(battle_features)

    battle_identifier = battle_row.get("battle_id", sample_df.index[0])

    print("\n=== PREDICTION ANALYSIS ===")
    print(f"Battle {battle_identifier} was chosen at random.")
    print(
        f"\nPrediction: {'Player 1 wins' if test_predictions[0] == 1 else 'Player 1 loses'}"
    )
    print(f"Probability of loss: {test_probabilities[0][0]:.2%}")
    print(f"Probability of win: {test_probabilities[0][1]:.2%}")

    # Feature values for this prediction
    print("\n=== FEATURE VALUES ===")
    feature_values = battle_row[features]
    for feature, value in feature_values.items():
        print(f"{feature}: {value}")

    # Feature importance (model coefficients)
    logistic_step = None
    if hasattr(model, "named_steps"):
        logistic_step = model.named_steps.get("logisticregression")
    elif hasattr(model, "steps"):
        for _, step in model.steps:
            if hasattr(step, "coef_"):
                logistic_step = step
                break

    if logistic_step is not None and hasattr(logistic_step, "coef_"):
        coefficients = logistic_step.coef_[0]
        feature_importance = list(zip(features, coefficients, feature_values))

        # Actual contribution of each feature to the prediction
        print("\n=== ACTUAL CONTRIBUTION TO PREDICTION (Top 10) ===")
        contributions = [
            (feature, coef * value, value)
            for feature, coef, value in feature_importance
        ]
        contributions_sorted = sorted(
            contributions, key=lambda x: abs(x[1]), reverse=True
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


if __name__ == "__main__":
    all_features = [
        "p1_mean_hp",
        "p1_mean_spe",
        "p1_mean_atk",
        "p1_mean_def",
        "p1_dead_pokemons",
        "p1_hp_loss",
        "p1_total_damage",
        "p1_avg_status",
        "p1_type_compatibility",
        "p1_avg_team_winrate",
        "p1_positive_boosts",
        "p1_negative_boosts",
        "p2_lead_hp",
        "p2_lead_spe",
        "p2_lead_atk",
        "p2_lead_def",
        "p2_dead_pokemons",
        "p2_hp_loss",
        "p2_total_damage",
        "p2_avg_status",
        "p2_type_compatibility",
        "p2_avg_team_winrate",
        "p2_positive_boosts",
        "p2_negative_boosts",
        "successful_explosion",
        "successful_explosion_p2",
        "attacks_2x_p1",
        "attacks_0_5x_p1",
        "attacks_0x_p1",
        "attacks_2x_p2",
        "attacks_0_5x_p2",
        "attacks_0x_p2",
    ]

    configured_features = [
        "p1_hp_loss",
        "p1_avg_status",
        "p1_type_compatibility",
        "p2_hp_loss",
        "p2_avg_status",
        "p2_type_compatibility",
        "attacks_2x_p1",
        "attacks_0_5x_p1",
        "attacks_0x_p1",
        "attacks_2x_p2",
        "attacks_0_5x_p2",
        "attacks_0x_p2",
        "p1_avg_team_winrate",
    ]

    [train_df, feature_names] = create_training_features(configured_features)

    feature_correlation = FeatureCorrelationAnalyzer(train_df, feature_names)
    feature_correlation.compute_correlation()
    print(feature_correlation.top_correlated_pairs(0))

    algorithm = "logistic_regression"
    model = train_model(train_df, feature_names, True, algorithm)

    # Run this block to analyse a live match you are watching in the browser
    """ battle_file_name = (
        "./visualization/battle_htmls/json/Gen1OU-2020-04-19-eightylewis-nsh526625.json"
    )
    live_battle_importer = ImportSource()
    live_battle_importer.load_json(battle_file_name)
    live_battle_df = BattleFeatureExtractor(
        live_battle_importer.data, list(feature_names)
    ).process()
    predict_single_battle(model, feature_names, live_battle_df) """

    # Run this block to analyse a random battle from the test set instead
    test_data_importer = ImportSource()
    test_data_importer.load_jsonl("./data/test.jsonl")
    test_df = BattleFeatureExtractor(
        test_data_importer.data, list(feature_names)
    ).process()
    predict_single_battle(model, feature_names, test_df)

    # Run this part if you want to let the model predict the outcomes for the
    # test dataset, create a submission csv file and save it to the 'results' directory
    test_trained_model(model, feature_names)
