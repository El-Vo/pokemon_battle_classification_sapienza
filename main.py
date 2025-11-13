from typing import Any

from pandas import DataFrame

from prepare_data.import_source import ImportSource
from analysis.extract_battle_features import BattleFeatureExtractor
from analysis.logistic_regression import RunLogisticRegression
from analysis.stacking_model import RunStackingModel
from analysis.gradient_boosting import RunGradientBoosting
from analysis.test_model import TestModel
from analysis.feature_correlation import FeatureCorrelationAnalyzer
from tools.single_battle_predictor import SingleBattlePredictor


def create_training_features(
    selected_features: list[str], train_data_filepath: str
) -> tuple[DataFrame, list]:
    train_data_importer = ImportSource()
    train_data_importer.load_jsonl(train_data_filepath)
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
        "stacking": RunStackingModel,
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


def test_trained_model(model: Any, features: list, test_data_filepath: str) -> None:
    test_data_importer = ImportSource()
    test_data_importer.load_jsonl(test_data_filepath)
    test_data_extractor = BattleFeatureExtractor(
        test_data_importer.data, list(features)
    )
    test_df = test_data_extractor.process()
    test = TestModel(model, test_df, features)
    test.test("results")


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
        "p1_avg_team_winrate",
        "p2_lead_hp",
        "p2_hp_loss",
        "p2_avg_status",
        "p2_type_compatibility",
        "p2_avg_team_winrate",
        "attacks_0x_p1",
        "attacks_2x_p2",
        "attacks_0x_p2",
    ]

    [train_df, feature_names] = create_training_features(
        configured_features, "./data/train.jsonl"
    )

    feature_correlation = FeatureCorrelationAnalyzer(train_df, feature_names)
    feature_correlation.compute_correlation()
    print(feature_correlation.top_correlated_pairs(0))

    # algorithm = "gradient_boosting"
    algorithm = "logistic_regression"
    # algorithm = "stacking"
    model = train_model(train_df, feature_names, True, algorithm)

    # Run this block to analyse a random battle from the test set
    """ test_data_importer = ImportSource()
    test_data_importer.load_jsonl("./data/test.jsonl")
    test_df = BattleFeatureExtractor(
        test_data_importer.data, list(feature_names)
    ).process()
    predictor = SingleBattlePredictor(model, feature_names)
    predictor.predict(test_df) """

    # Run this part if you want to let the model predict the outcomes for the
    # test dataset, create a submission csv file and save it to the 'results' directory
    test_trained_model(model, feature_names, "./data/test.jsonl")
