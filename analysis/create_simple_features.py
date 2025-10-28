from tqdm.auto import tqdm
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Iterable
from tools.calculate_attack_effectiveness import AttackEffectivenessCalculator

NEGATIVE_STATUS = ["par", "brn", "frz", "slp", "psn", "tox"]


class BattleFeatureExtractor:
    """Encapsulate feature extraction from battleJSONL data."""

    def __init__(self, data: List[Dict]):
        self.data = data
        self.data_df: Optional[pd.DataFrame] = None
        self._team_rosters: Optional[Dict[int, Dict[str, List[str]]]] = None
        self._pokemon_types: Optional[Dict[str, List[str]]] = None
        self._attack_calc = AttackEffectivenessCalculator()
        self._active_battle_id: Optional[int] = None

    def process(self) -> pd.DataFrame:
        """Compute the feature DataFrame once and cache the result."""
        if self.data_df is None:
            self.data_df = self.create_simple_features(self.data)
        return self.data_df

    def create_simple_features(self, data: List[Dict]) -> pd.DataFrame:
        """
        A feature extraction function.
        Returns a pandas DataFrame with missing values filled as 0.
        """
        feature_list: List[Dict] = []
        for battle in tqdm(data, desc="Extracting features"):
            features: Dict = {}

            self._active_battle_id = battle.get("battle_id")

            # Get the last documented hp for each players pokemons
            battle_info_p1, battle_info_p2 = self.get_battle_info(battle.get("battle_timeline", []))

            # --- Player 1 Team Features ---
            p1_team = battle.get("p1_team_details", [])
            if p1_team:

                # features['p1_mean_hp'] = np.mean([p.get('base_hp', 0) for p in p1_team])
                # features['p1_mean_spe'] = np.mean([p.get('base_spe', 0) for p in p1_team])
                # features['p1_mean_atk'] = np.mean([p.get('base_atk', 0) for p in p1_team])
                # features['p1_mean_def'] = np.mean([p.get('base_def', 0) for p in p1_team])

                features["p1_dead_pokemons"] = self.count_dead_pokemons(
                    battle_info_p1["pokemon_hp"]
                )
                features["p1_hp_loss"] = self.calculate_hp_loss(
                    battle_info_p1["pokemon_hp"]
                )
                features["p1_avg_status"] = (
                    battle_info_p1["status_count"] / 30
                )  # Divide by the total number of rounds
                features["p1_type_compatibility"] = self.calculate_type_compatibility(
                    battle_info_p1["pokemon_hp"], battle_info_p2["pokemon_hp"]
                )
                # features['p1_positive_boosts'] = battle_info_p1['positive_boosts']
                # features['p1_negative_boosts'] = battle_info_p1['negative_boosts']

            # --- Player 2 Lead Features ---
            p2_lead = battle.get("p2_lead_details")
            if p2_lead:
                # Player 2's lead Pokémon's stats
                # features['p2_lead_hp'] = p2_lead.get('base_hp', 0)
                # features['p2_lead_spe'] = p2_lead.get('base_spe', 0)
                # features['p2_lead_atk'] = p2_lead.get('base_atk', 0)
                # features['p2_lead_def'] = p2_lead.get('base_def', 0)

                features["p2_dead_pokemons"] = self.count_dead_pokemons(
                    battle_info_p2["pokemon_hp"]
                )
                features["p2_hp_loss"] = self.calculate_hp_loss(
                    battle_info_p2["pokemon_hp"]
                )
                features["p2_avg_status"] = (
                    battle_info_p2["status_count"] / 30
                )  # Divide by the total number of rounds
                features["p2_type_compatibility"] = self.calculate_type_compatibility(
                    battle_info_p2["pokemon_hp"], battle_info_p1["pokemon_hp"]
                )
                # features['p2_positive_boosts'] = battle_info_p2['positive_boosts']
                # features['p2_negative_boosts'] = battle_info_p2['negative_boosts']
               
            # We also need the ID and the target variable (if it exists)
            features["battle_id"] = battle.get("battle_id")
            if "player_won" in battle:
                features["player_won"] = int(battle["player_won"])

            feature_list.append(features)

        return pd.DataFrame(feature_list).fillna(0)

    def get_battle_info(self, battle_timeline: List[Dict]) -> tuple:
        """Collect battle information from the last 30 rounds. The last documented HP for each
        Pokémon in the battle is collected as well as the number of rounds with a negative status for each player.

        Returns:
            A tuple with two dictionaries with each pokemons hp and the total number of rounds with a status.
        """

        battle_info_p1 = {'pokemon_hp': {}, 'status_count': 0, 'positive_boosts' : 0, 'negative_boosts' : 0}
        battle_info_p2 = {'pokemon_hp': {}, 'status_count': 0, 'positive_boosts' : 0, 'negative_boosts' : 0}
   
        # Add additional health and status information from battle turns
        for turn in battle_timeline:
            for player_key in ["p1_pokemon_state", "p2_pokemon_state"]:
                pokemon_state = turn.get(player_key, {})
                pokemon_name = pokemon_state.get('name')
                current_hp = pokemon_state.get('hp_pct')
                boost_dict = pokemon_state.get('boosts') # Boosts can be values between +6 and -6
                

                # Get the name of the dictonary to store the information based on which player it is
                dict_name = (
                    battle_info_p1
                    if player_key == "p1_pokemon_state"
                    else battle_info_p2
                )
                dict_name['pokemon_hp'][pokemon_name] = current_hp

                for value in boost_dict.values():
                    if value > 0:
                        dict_name['positive_boosts']+= value
                    elif value < 0:
                        # The boost is just a negative value to show that it is a bad boost
                        dict_name['negative_boosts']+= -1*value 

                if pokemon_state.get("status") in NEGATIVE_STATUS:
                    dict_name["status_count"] += 1

        return battle_info_p1, battle_info_p2

    def count_dead_pokemons(self, hp_dict: Dict[str, float]) -> int:
        """
        Counts the total number of dead pokemons
        during the first 30 rounds
        """
        return sum(1 for hp in hp_dict.values() if hp == 0.0)

    def calculate_hp_loss(self, hp_dict: Dict[str, float]) -> float:
        """
        Calculate the total hp loss for a player during the first 30 rounds
        """
        total_hp_loss = 0
        for end_hp in hp_dict.values():
            start_hp = 1.0  # Assuming all Pokémon start with full HP (1.0)
            hp_loss = start_hp - end_hp
            total_hp_loss += hp_loss

        return total_hp_loss

    def calculate_type_compatibility(
        self, hp_dict_attacker: Dict[str, float], hp_dict_defender: Dict[str, float]
    ):
        battle_id = self._active_battle_id
        attacker_names = {
            name
            for name, hp in hp_dict_attacker.items()
            if name and hp is not None and hp > 0
        }
        defender_names = {
            name
            for name, hp in hp_dict_defender.items()
            if name and hp is not None and hp > 0
        }

        if not attacker_names or not defender_names:
            return 0.0

        type_lookup = self._load_pokemon_types()

        total_multiplier = 0.0
        combinations = 0

        for attacker in attacker_names:
            attacker_types = type_lookup.get(attacker.lower())
            if not attacker_types:
                continue
            for defender in defender_names:
                defender_types = type_lookup.get(defender.lower())
                if not defender_types:
                    continue
                multiplier = self._attack_calc.calculate(attacker_types, defender_types)
                total_multiplier += multiplier
                combinations += 1

        if combinations == 0:
            return 0.0

        return total_multiplier / len(attacker_names)

    def _load_team_rosters(self) -> Dict[int, Dict[str, List[str]]]:
        if self._team_rosters is None:
            roster_path = self._data_path("team_rosters_train.json")
            if roster_path.exists():
                with roster_path.open("r", encoding="utf-8") as f:
                    entries = json.load(f)
                self._team_rosters = {
                    int(entry["battle_id"]): {
                        "p1_team": entry.get("p1_team", []),
                        "p2_team": entry.get("p2_team", []),
                    }
                    for entry in entries
                }
            else:
                self._team_rosters = {}
        return self._team_rosters

    def _load_pokemon_types(self) -> Dict[str, List[str]]:
        if self._pokemon_types is None:
            roster_path = self._data_path("pokemon_roster.json")
            type_map: Dict[str, List[str]] = {}
            if roster_path.exists():
                with roster_path.open("r", encoding="utf-8") as f:
                    pokedex = json.load(f)
                for entry in pokedex:
                    name = entry.get("name")
                    types = entry.get("p1_attributes", {}).get("types", [])
                    if not name:
                        continue
                    normalized = [
                        t.strip().upper() for t in types if t and t.lower() != "notype"
                    ]
                    if not normalized:
                        normalized = ["NORMAL"]
                    type_map[name.lower()] = normalized
            self._pokemon_types = type_map
        return self._pokemon_types

    @staticmethod
    def _data_path(filename: str) -> Path:
        return Path(__file__).resolve().parents[1] / "data" / filename
