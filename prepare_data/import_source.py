import json
import pandas as pd
import os


class ImportSource:
    """Helper class to load `train.jsonl`/`test.jsonl` files and regular json files.

    Provides simple methods to load the training file and display the
    structure of the first battle for inspection.
    """

    def __init__(self):
        self.data = []

    def load_jsonl(self, file_path: str) -> None:
        """Load `.jsonl` file line-by-line into `self.train_data`.
        """
        print(f"Loading data from '{file_path}'...")
        with open(file_path, 'r') as f:
            for line in f:
                # json.loads() parses one line (one JSON object) into a Python dictionary
                self.data.append(json.loads(line))

        print(f"Successfully loaded {len(self.data)} battles.")

    def load_json(self, file_path: str) -> None:
        """Load a single `.json` file into `self.data`.
        
        If the JSON file contains a single battle object, it will be wrapped in a list.
        If it contains a list of battles, it will be extended to `self.data`.
        """
        print(f"Loading data from '{file_path}'...")
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        
        # Check if the loaded data is a list or a single object
        if isinstance(json_data, list):
            self.data.extend(json_data)
            print(f"Successfully loaded {len(json_data)} battles.")
        else:
            self.data.append(json_data)
            print(f"Successfully loaded 1 battle.")

    def display_first_battle(self, truncate: int = 1, truncate_threshold: int = 1) -> None:
        """Print the structure of the first loaded battle.

        Args:
            truncate: number of turns to show from `battle_timeline`.
            truncate_threshold: length above which a truncation notice is shown.
        """
        print("\n--- Structure of the first train battle: ---")
        if self.data:
            first_battle = self.data[0]

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