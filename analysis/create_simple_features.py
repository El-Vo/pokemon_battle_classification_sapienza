from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from tools.calculate_attack_effectiveness import AttackEffectivenessCalculator

NEGATIVE_STATUS = ["par", "brn", "frz", "slp", "psn", "tox"]

class BattleFeatureExtractor:
    """Encapsulate feature extraction from battleJSONL data.
    """

    def __init__(self, data: List[Dict]):
        self.data = data
        self.data_df: Optional[pd.DataFrame] = None

    def process(self) -> pd.DataFrame:
        """Compute the feature DataFrame once and cache the result."""
        if self.data_df is None:
            self.data_df = self.create_simple_features(self.data)
        return self.data_df

    def create_simple_features(self, data: List[Dict]) -> pd.DataFrame:
        """
        A very basic feature extraction function.
        It only uses the aggregated base stats of the player's team and opponent's lead.
        Returns a pandas DataFrame with missing values filled as 0.
        """
        feature_list: List[Dict] = []
        for battle in tqdm(data, desc="Extracting features"):
            features: Dict = {}

            # Get the last documented hp for each players pokemons
            battle_info_p1, battle_info_p2 = self.get_battle_info(battle.get('battle_timeline', []))

            # --- Player 1 Team Features ---
            p1_team = battle.get('p1_team_details', [])
            if p1_team:
                features['p1_mean_hp'] = np.mean([p.get('base_hp', 0) for p in p1_team])
                features['p1_mean_spe'] = np.mean([p.get('base_spe', 0) for p in p1_team])
                features['p1_mean_atk'] = np.mean([p.get('base_atk', 0) for p in p1_team])
                features['p1_mean_def'] = np.mean([p.get('base_def', 0) for p in p1_team])
                features['p1_dead_pokemons'] = self.count_dead_pokemons(battle_info_p1['pokemon_hp'])
                features['p1_hp_loss'] = self.calculate_hp_loss(battle_info_p1['pokemon_hp'])
                features['p1_avg_status'] = battle_info_p1['status_count'] / 30 # Divide by the total number of rounds


            # --- Player 2 Lead Features ---
            p2_lead = battle.get('p2_lead_details')
            if p2_lead:
                # Player 2's lead Pokémon's stats
                features['p2_lead_hp'] = p2_lead.get('base_hp', 0)
                features['p2_lead_spe'] = p2_lead.get('base_spe', 0)
                features['p2_lead_atk'] = p2_lead.get('base_atk', 0)
                features['p2_lead_def'] = p2_lead.get('base_def', 0)
                features['p2_dead_pokemons'] = self.count_dead_pokemons(battle_info_p2['pokemon_hp'])
                features['p2_hp_loss'] = self.calculate_hp_loss(battle_info_p2['pokemon_hp'])
                features['p2_avg_status'] = battle_info_p2['status_count'] / 30 # Divide by the total number of rounds

            # We also need the ID and the target variable (if it exists)
            features['battle_id'] = battle.get('battle_id')
            if 'player_won' in battle:
                features['player_won'] = int(battle['player_won'])

            feature_list.append(features)

        return pd.DataFrame(feature_list).fillna(0)
    
    def get_battle_info(self, battle_timeline: List[Dict]) -> Dict:
        """Collect battle information from the last 30 rounds. The last documented HP for each 
        Pokémon in the battle is collected as well as the number of rounds with a negative status for each player.

        Returns:
            A dictionary with each pokemons hp and the total number of rounds with a status.
        """

        battle_info_p1 = {'pokemon_hp': {}, 'status_count': 0}
        battle_info_p2 = {'pokemon_hp': {}, 'status_count': 0}

        for turn in battle_timeline:
            for player_key in ['p1_pokemon_state', 'p2_pokemon_state']:
                pokemon_state = turn.get(player_key, {})
                pokemon_name = pokemon_state.get('name')
                current_hp = pokemon_state.get('hp_pct')

                # Get the name of the dictonary to store the information based on which player it is
                dict_name = battle_info_p1 if player_key == 'p1_pokemon_state' else battle_info_p2
                dict_name['pokemon_hp'][pokemon_name] = current_hp

                if pokemon_state.get("status") in NEGATIVE_STATUS:
                    dict_name['status_count'] += 1

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

