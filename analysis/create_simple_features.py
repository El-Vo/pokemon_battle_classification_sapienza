from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from typing import List, Dict, Optional


class BattleFeatureExtractor:
    """Encapsulate feature extraction from battleJSONL data.
    """

    def __init__(self, data: List[Dict]):
        self.data = data
        self.data_df = self.create_simple_features(self.data)

    def create_simple_features(self, data: List[Dict]) -> pd.DataFrame:
        """
        A very basic feature extraction function.
        It only uses the aggregated base stats of the player's team and opponent's lead.
        Returns a pandas DataFrame with missing values filled as 0.
        """
        feature_list: List[Dict] = []
        for battle in tqdm(data, desc="Extracting features"):
            features: Dict = {}

            # --- Player 1 Team Features ---
            p1_team = battle.get('p1_team_details', [])
            if p1_team:
                features['p1_mean_hp'] = np.mean([p.get('base_hp', 0) for p in p1_team])
                features['p1_mean_spe'] = np.mean([p.get('base_spe', 0) for p in p1_team])
                features['p1_mean_atk'] = np.mean([p.get('base_atk', 0) for p in p1_team])
                features['p1_mean_def'] = np.mean([p.get('base_def', 0) for p in p1_team])

            # --- Player 2 Lead Features ---
            p2_lead = battle.get('p2_lead_details')
            if p2_lead:
                # Player 2's lead Pokémon's stats
                features['p2_lead_hp'] = p2_lead.get('base_hp', 0)
                features['p2_lead_spe'] = p2_lead.get('base_spe', 0)
                features['p2_lead_atk'] = p2_lead.get('base_atk', 0)
                features['p2_lead_def'] = p2_lead.get('base_def', 0)

            # We also need the ID and the target variable (if it exists)
            features['battle_id'] = battle.get('battle_id')
            if 'player_won' in battle:
                features['player_won'] = int(battle['player_won'])

            feature_list.append(features)

        return pd.DataFrame(feature_list).fillna(0)