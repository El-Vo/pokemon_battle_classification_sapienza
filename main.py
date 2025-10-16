from prepare_data.import_source_jsonl import ImportSourceJsonl
from analysis.create_simple_features import BattleFeatureExtractor
from analysis.logistic_regression import RunLogisticRegression
from analysis.test_model import TestModel

if __name__ == '__main__':
    # Check if training data is available under data/train.jsonl
    importer = ImportSourceJsonl()
    importer.load_train()
    #importer.display_first_battle()

    train_data_extractor = BattleFeatureExtractor(importer.train_data)
    print("\nTraining features preview:")
    print(train_data_extractor.data_df.head().to_string(index=False))

    # Exclude battle_id (because it has no informative value) and player_won (because this is the variable we want to predict)
    features = [col for col in train_data_extractor.data_df.columns if col not in ['battle_id', 'player_won']]

    trainer = RunLogisticRegression(train_data_extractor.data_df, features)
    model = trainer.train_model()

    # Run this part if you want to let the model predict the outcomes for the 
    # test dataset, create a submission csv file and save it to the 'results' directory
    importer.load_test()
    test_data_extractor = BattleFeatureExtractor(importer.test_data)
    test = TestModel(model, test_data_extractor.data_df, features)
    test.test()