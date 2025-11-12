from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from prepare_data.import_source import ImportSource
from analysis.create_simple_features import BattleFeatureExtractor
from analysis.logistic_regression import RunLogisticRegression
from analysis.stacking_model import RunStackingModel
from analysis.test_model import TestModel
from analysis.feature_correlation import FeatureCorrelationAnalyzer
from sklearn.pipeline import Pipeline


def create_training_features() -> tuple[DataFrame, list]:
    # Check if training data is available under data/train.jsonl
    train_data_importer = ImportSource()
    train_data_importer.load_jsonl("./data/train.jsonl")
    # train_data_importer.display_first_battle()

    train_data_extractor = BattleFeatureExtractor(train_data_importer.data)
    train_df = train_data_extractor.process()

    # Exclude battle_id (because it has no informative value) and player_won (because this is the variable we want to predict) from training features
    train_feature_names = [
        col for col in train_df.columns if col not in ["battle_id", "player_won"]
    ]

    return train_df, train_feature_names


def train_logistic_regression_model(
    train_dataframe: DataFrame, train_feature_names: list, log_results: bool = False
) -> Pipeline:
    trainer = RunLogisticRegression(train_dataframe, train_feature_names)
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


def test_logistic_regression_model(model: Pipeline, features: list) -> None:
    test_data_importer = ImportSource()
    test_data_importer.load_jsonl("./data/test.jsonl")
    test_data_extractor = BattleFeatureExtractor(test_data_importer.data)
    test_df = test_data_extractor.process()
    test = TestModel(model, test_df, features)
    test.test()


def predict_single_battle(
    model: Pipeline,
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
    [train_df, feature_names] = create_training_features()

    feature_correlation = FeatureCorrelationAnalyzer(train_df, feature_names)
    feature_correlation.compute_correlation()
    print(feature_correlation.top_correlated_pairs(0))

    # model = train_logistic_regression_model(train_df, feature_names)
    model = train_stacking_model(train_df, feature_names)

    # Run this block to analyse a live match you are watching in the browser
    """ battle_file_name = (
        "./visualization/battle_htmls/json/Gen1OU-2020-04-19-eightylewis-nsh526625.json"
    )
    live_battle_importer = ImportSource()
    live_battle_importer.load_json(battle_file_name)
    live_battle_df = BattleFeatureExtractor(live_battle_importer.data).process()
    predict_single_battle(model, feature_names, live_battle_df) """

    # Run this block to analyse a random battle from the test set instead
    test_data_importer = ImportSource()
    test_data_importer.load_jsonl("./data/test.jsonl")
    test_df = BattleFeatureExtractor(test_data_importer.data).process()
    predict_single_battle(model, feature_names, test_df)

    # Run this part if you want to let the model predict the outcomes for the
    # test dataset, create a submission csv file and save it to the 'results' directory
    """ test_logistic_regression_model(model, feature_names) """
