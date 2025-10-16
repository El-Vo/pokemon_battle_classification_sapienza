import json
import pandas as pd
import os


class ImportSourceJsonl:
    """Helper class to load `train.jsonl`/`test.jsonl` files.

    Provides simple methods to load the training file and display the
    structure of the first battle for inspection.
    """

    def __init__(self, data_path: str = './data'):
        self.data_path = data_path
        self.train_file_path = os.path.join(self.data_path, 'train.jsonl')
        self.test_file_path = os.path.join(self.data_path, 'test.jsonl')
        self.train_data = []
        self.test_data = []

    def load_train(self) -> None:
        """Load `train.jsonl` line-by-line into `self.train_data`.
        """
        print(f"Loading data from '{self.train_file_path}'...")
        with open(self.train_file_path, 'r') as f:
            for line in f:
                # json.loads() parses one line (one JSON object) into a Python dictionary
                self.train_data.append(json.loads(line))

        print(f"Successfully loaded {len(self.train_data)} battles.")

    def load_test(self) -> None:
        """Load `test.jsonl` line-by-line into `self.test_data`.
        """
        print(f"Loading data from '{self.test_file_path}'...")
        with open(self.test_file_path, 'r') as f:
            for line in f:
                # json.loads() parses one line (one JSON object) into a Python dictionary
                self.test_data.append(json.loads(line))

        print(f"Successfully loaded {len(self.test_data)} battles.")

    def display_first_battle(self, truncate: int = 1, truncate_threshold: int = 1) -> None:
        """Print the structure of the first loaded battle.

        Args:
            truncate: number of turns to show from `battle_timeline`.
            truncate_threshold: length above which a truncation notice is shown.
        """
        print("\n--- Structure of the first train battle: ---")
        if self.train_data:
            first_battle = self.train_data[0]

            # To keep the output clean, we can create a copy and truncate the timeline
            battle_for_display = first_battle.copy()
            battle_for_display['battle_timeline'] = battle_for_display.get('battle_timeline', [])[:truncate]

            # Use json.dumps for pretty-printing the dictionary
            print(json.dumps(battle_for_display, indent=4))
            if len(first_battle.get('battle_timeline', [])) > truncate_threshold:
                print("    ...")
                print("    (battle_timeline has been truncated for display)")
        else:
            print("No train data available to display. Did you call load_train()?")