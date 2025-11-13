from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from tools.calculate_attack_effectiveness import AttackEffectivenessCalculator
from tools.list_all_pokemon import PokemonRosterSummary

NEGATIVE_STATUS = ["par", "brn", "frz", "slp", "psn", "tox"]


@dataclass
class BattleContext:
    battle: Dict
    battle_id: Optional[int]
    timeline: List[Dict]
    p1_team: List[Dict]
    p2_team: List[Dict]
    p1_lead: Optional[Dict]
    p2_lead: Optional[Dict]
    battle_info_p1: Dict
    battle_info_p2: Dict
    type_lookup: Dict[str, List[str]]
    caches: Dict[str, Any] = field(default_factory=dict)


class BattleFeatureExtractor:
    """Encapsulate feature extraction from battleJSONL data."""

    FEATURE_BUILDERS = {
        "p1_mean_hp": "_feature_p1_mean_hp",
        "p1_mean_spe": "_feature_p1_mean_spe",
        "p1_mean_atk": "_feature_p1_mean_atk",
        "p1_mean_def": "_feature_p1_mean_def",
        "p1_dead_pokemons": "_feature_p1_dead_pokemons",
        "p1_hp_loss": "_feature_p1_hp_loss",
        "p1_total_damage": "_feature_p1_total_damage",
        "p1_avg_status": "_feature_p1_avg_status",
        "p1_type_compatibility": "_feature_p1_type_compatibility",
        "p1_avg_team_winrate": "_feature_p1_avg_team_winrate",
        "p1_positive_boosts": "_feature_p1_positive_boosts",
        "p1_negative_boosts": "_feature_p1_negative_boosts",
        "p2_lead_hp": "_feature_p2_lead_hp",
        "p2_lead_spe": "_feature_p2_lead_spe",
        "p2_lead_atk": "_feature_p2_lead_atk",
        "p2_lead_def": "_feature_p2_lead_def",
        "p2_dead_pokemons": "_feature_p2_dead_pokemons",
        "p2_hp_loss": "_feature_p2_hp_loss",
        "p2_total_damage": "_feature_p2_total_damage",
        "p2_avg_status": "_feature_p2_avg_status",
        "p2_type_compatibility": "_feature_p2_type_compatibility",
        "p2_avg_team_winrate": "_feature_p2_avg_team_winrate",
        "p2_positive_boosts": "_feature_p2_positive_boosts",
        "p2_negative_boosts": "_feature_p2_negative_boosts",
        "successful_explosion": "_feature_successful_explosion",
        "successful_explosion_p2": "_feature_successful_explosion_p2",
        "high_damage_moves_p1": "_feature_high_damage_moves_p1",
        "high_damage_moves_p2": "_feature_high_damage_moves_p2",
        "attacks_2x_p1": "_feature_attacks_2x_p1",
        "attacks_0_5x_p1": "_feature_attacks_0_5x_p1",
        "attacks_0x_p1": "_feature_attacks_0x_p1",
        "attacks_2x_p2": "_feature_attacks_2x_p2",
        "attacks_0_5x_p2": "_feature_attacks_0_5x_p2",
        "attacks_0x_p2": "_feature_attacks_0x_p2",
    }

    HIGH_DAMAGE_THRESHOLD = 0.66

    def __init__(self, data: List[Dict], enabled_features: List[str]):
        self.data = data
        self.enabled_features = self._validate_features(enabled_features)
        self.data_df: Optional[pd.DataFrame] = None
        self._team_rosters: Optional[Dict[int, Dict[str, List[str]]]] = None
        self._pokemon_types: Optional[Dict[str, List[str]]] = None
        self._attack_calc = AttackEffectivenessCalculator()
        self._active_battle_id: Optional[int] = None
        self._pokemon_win_rates: Optional[Dict[str, float]] = None

    @classmethod
    def available_features(cls) -> List[str]:
        return list(cls.FEATURE_BUILDERS.keys())

    def process(self) -> pd.DataFrame:
        """Compute the feature DataFrame once and cache the result."""
        if self.data_df is None:
            self.data_df = self.create_simple_features(self.data)
        return self.data_df

    def create_simple_features(self, data: List[Dict]) -> pd.DataFrame:
        """Return a pandas DataFrame with the selected features for each battle."""

        feature_rows: List[Dict[str, Any]] = []
        for battle in tqdm(data, desc="Extracting features"):
            context = self._build_context(battle)
            row: Dict[str, Any] = {}

            for feature_name in self.enabled_features:
                builder_name = self.FEATURE_BUILDERS[feature_name]
                # Retrieve function names for individual attributes and run their computation
                builder = getattr(self, builder_name)
                try:
                    row[feature_name] = builder(context)
                except Exception as exc:  # pragma: no cover - defensive guard
                    raise RuntimeError(
                        f"Failed to compute feature '{feature_name}' for battle "
                        f"{context.battle_id}"
                    ) from exc

            row["battle_id"] = context.battle_id
            if "player_won" in battle:
                row["player_won"] = int(battle["player_won"])

            feature_rows.append(row)

        return pd.DataFrame(feature_rows).fillna(0)

    def _validate_features(self, enabled_features: List[str]) -> List[str]:
        if not enabled_features:
            raise ValueError("At least one feature must be provided.")

        features = list(enabled_features)
        seen = set()
        validated: List[str] = []
        for name in features:
            if name not in self.FEATURE_BUILDERS:
                available = ", ".join(sorted(self.FEATURE_BUILDERS))
                raise ValueError(
                    f"Unknown feature '{name}'. Available features: {available}"
                )
            if name not in seen:
                validated.append(name)
                seen.add(name)

        return validated

    def _build_context(self, battle: Dict) -> BattleContext:
        battle_id = battle.get("battle_id")
        timeline = battle.get("battle_timeline") or []
        p1_team = battle.get("p1_team_details") or []
        p2_team = battle.get("p2_team_details") or []
        p1_lead = battle.get("p1_lead_details")
        p2_lead = battle.get("p2_lead_details")
        battle_info_p1, battle_info_p2 = self.get_battle_info(timeline)
        type_lookup = self._load_pokemon_types()

        self._active_battle_id = battle_id

        return BattleContext(
            battle=battle,
            battle_id=battle_id,
            timeline=timeline,
            p1_team=p1_team,
            p2_team=p2_team,
            p1_lead=p1_lead,
            p2_lead=p2_lead,
            battle_info_p1=battle_info_p1,
            battle_info_p2=battle_info_p2,
            type_lookup=type_lookup,
        )

    def _compute_timeline_metrics(
        self, context: BattleContext
    ) -> Dict[str, Dict[str, int]]:
        cached = context.caches.get("timeline_metrics")
        if cached is not None:
            return cached

        attack_counts = {
            "attacks_2x_p1": 0,
            "attacks_0_5x_p1": 0,
            "attacks_0x_p1": 0,
            "attacks_2x_p2": 0,
            "attacks_0_5x_p2": 0,
            "attacks_0x_p2": 0,
        }
        explosion_flags = {"p1": 0, "p2": 0}

        prev_p2_hp = 1.0
        prev_p1_hp = 1.0

        for turn in context.timeline:
            p1_move = turn.get("p1_move_details") or {}
            p2_move = turn.get("p2_move_details") or {}
            p2_state = turn.get("p2_pokemon_state") or {}
            p1_state = turn.get("p1_pokemon_state") or {}

            if (p1_move.get("name") or "").lower() == "explosion" and p2_state.get(
                "hp_pct"
            ) == 0.0:
                explosion_flags["p1"] = 1
            if (p2_move.get("name") or "").lower() == "explosion" and p1_state.get(
                "hp_pct"
            ) == 0.0:
                explosion_flags["p2"] = 1

            curr_p2_hp = p2_state.get("hp_pct")
            if curr_p2_hp is None:
                curr_p2_hp = prev_p2_hp
            curr_p1_hp = p1_state.get("hp_pct")
            if curr_p1_hp is None:
                curr_p1_hp = prev_p1_hp

            self._update_attack_counts(
                attack_counts, p1_move, p2_state, context.type_lookup, "p1"
            )
            self._update_attack_counts(
                attack_counts, p2_move, p1_state, context.type_lookup, "p2"
            )

            prev_p2_hp = curr_p2_hp
            prev_p1_hp = curr_p1_hp

        metrics = {
            "attack_counts": attack_counts,
            "explosions": explosion_flags,
        }
        context.caches["timeline_metrics"] = metrics
        return metrics

    def _update_attack_counts(
        self,
        attack_counts: Dict[str, int],
        move: Dict,
        defender_state: Dict,
        type_lookup: Dict[str, List[str]],
        prefix: str,
    ) -> None:
        if not move:
            return

        move_type = move.get("type")
        defender_name = defender_state.get("name")
        if not move_type or not defender_name:
            return

        defender_types = type_lookup.get(defender_name.lower())
        if not defender_types:
            return

        try:
            multiplier = self._attack_calc.calculate(move_type, defender_types)
        except Exception:
            return

        mapping = {
            2.0: f"attacks_2x_{prefix}",
            0.5: f"attacks_0_5x_{prefix}",
            0.0: f"attacks_0x_{prefix}",
        }
        for value, key in mapping.items():
            if abs(multiplier - value) < 1e-8:
                attack_counts[key] += 1
                break

    @staticmethod
    def _mean_team_stat(team: List[Dict], stat_key: str) -> float:
        values: List[float] = []
        for pokemon in team:
            if not pokemon:
                continue
            value = pokemon.get(stat_key)
            if value is None:
                continue
            values.append(float(value))
        if not values:
            return 0.0
        return float(np.mean(values))

    @staticmethod
    def _lead_pokemon_stat(lead: Optional[Dict], stat_key: str) -> float:
        if not lead:
            return 0.0
        value = lead.get(stat_key)
        if value is None:
            return 0.0
        return float(value)

    def _feature_p1_mean_hp(self, context: BattleContext) -> float:
        return self._mean_team_stat(context.p1_team, "base_hp")

    def _feature_p1_mean_spe(self, context: BattleContext) -> float:
        return self._mean_team_stat(context.p1_team, "base_spe")

    def _feature_p1_mean_atk(self, context: BattleContext) -> float:
        return self._mean_team_stat(context.p1_team, "base_atk")

    def _feature_p1_mean_def(self, context: BattleContext) -> float:
        return self._mean_team_stat(context.p1_team, "base_def")

    def _feature_p1_dead_pokemons(self, context: BattleContext) -> int:
        return self.count_dead_pokemons(context.battle_info_p1["pokemon_hp"])

    def _feature_p1_hp_loss(self, context: BattleContext) -> float:
        return self.calculate_hp_loss(context.battle_info_p1["pokemon_hp"])

    def _feature_p1_total_damage(self, context: BattleContext) -> float:
        return self.calculate_total_damage(context.battle_info_p1["total_lost_health"])

    def _feature_p1_avg_status(self, context: BattleContext) -> float:
        """Returns the percentage of rounds with a bad status"""
        return context.battle_info_p1["status_count"] / 30.0

    def _feature_p1_type_compatibility(self, context: BattleContext) -> float:
        return self.calculate_type_compatibility(
            context.battle_info_p1["pokemon_hp"], context.battle_info_p2["pokemon_hp"]
        )

    def _feature_p1_avg_team_winrate(self, context: BattleContext) -> float:
        return self.calculate_team_winrate(context.battle_info_p1["pokemon_hp"])

    def _feature_p1_positive_boosts(self, context: BattleContext) -> float:
        return float(context.battle_info_p1["positive_boosts"])

    def _feature_p1_negative_boosts(self, context: BattleContext) -> float:
        return float(context.battle_info_p1["negative_boosts"])

    def _feature_p2_lead_hp(self, context: BattleContext) -> float:
        return self._lead_pokemon_stat(context.p2_lead, "base_hp")

    def _feature_p2_lead_spe(self, context: BattleContext) -> float:
        return self._lead_pokemon_stat(context.p2_lead, "base_spe")

    def _feature_p2_lead_atk(self, context: BattleContext) -> float:
        return self._lead_pokemon_stat(context.p2_lead, "base_atk")

    def _feature_p2_lead_def(self, context: BattleContext) -> float:
        return self._lead_pokemon_stat(context.p2_lead, "base_def")

    def _feature_p2_dead_pokemons(self, context: BattleContext) -> int:
        return self.count_dead_pokemons(context.battle_info_p2["pokemon_hp"])

    def _feature_p2_hp_loss(self, context: BattleContext) -> float:
        return self.calculate_hp_loss(context.battle_info_p2["pokemon_hp"])

    def _feature_p2_total_damage(self, context: BattleContext) -> float:
        return self.calculate_total_damage(context.battle_info_p2["total_lost_health"])

    def _feature_p2_avg_status(self, context: BattleContext) -> float:
        """Returns the percentage of rounds with a bad status"""
        return context.battle_info_p2["status_count"] / 30.0

    def _feature_p2_type_compatibility(self, context: BattleContext) -> float:
        return self.calculate_type_compatibility(
            context.battle_info_p2["pokemon_hp"], context.battle_info_p1["pokemon_hp"]
        )

    def _feature_p2_avg_team_winrate(self, context: BattleContext) -> float:
        return self.calculate_team_winrate(context.battle_info_p2["pokemon_hp"])

    def _feature_p2_positive_boosts(self, context: BattleContext) -> float:
        return float(context.battle_info_p2["positive_boosts"])

    def _feature_p2_negative_boosts(self, context: BattleContext) -> float:
        return float(context.battle_info_p2["negative_boosts"])

    def _feature_successful_explosion(self, context: BattleContext) -> int:
        """Counts if player had at least one successful explosion (1) or none (0)
        in a boolean value"""
        metrics = self._compute_timeline_metrics(context)
        return metrics["explosions"]["p1"]

    def _feature_successful_explosion_p2(self, context: BattleContext) -> int:
        """Counts if player had at least one successful explosion (1) or none (0)
        in a boolean value"""
        metrics = self._compute_timeline_metrics(context)
        return metrics["explosions"]["p2"]

    def _feature_attacks_2x_p1(self, context: BattleContext) -> int:
        metrics = self._compute_timeline_metrics(context)
        return metrics["attack_counts"]["attacks_2x_p1"]

    def _feature_attacks_0_5x_p1(self, context: BattleContext) -> int:
        metrics = self._compute_timeline_metrics(context)
        return metrics["attack_counts"]["attacks_0_5x_p1"]

    def _feature_attacks_0x_p1(self, context: BattleContext) -> int:
        metrics = self._compute_timeline_metrics(context)
        return metrics["attack_counts"]["attacks_0x_p1"]

    def _feature_attacks_2x_p2(self, context: BattleContext) -> int:
        metrics = self._compute_timeline_metrics(context)
        return metrics["attack_counts"]["attacks_2x_p2"]

    def _feature_attacks_0_5x_p2(self, context: BattleContext) -> int:
        metrics = self._compute_timeline_metrics(context)
        return metrics["attack_counts"]["attacks_0_5x_p2"]

    def _feature_attacks_0x_p2(self, context: BattleContext) -> int:
        metrics = self._compute_timeline_metrics(context)
        return metrics["attack_counts"]["attacks_0x_p2"]

    def get_battle_info(self, battle_timeline: List[Dict]) -> tuple:
        """Collect battle information from the last 30 rounds. The last documented HP for each
        Pokémon in the battle is collected as well as the number of rounds with a negative status for each player.

        Returns:
            A tuple with two dictionaries with each pokemons hp and the total number of rounds with a status.
        """

        battle_info_p1 = {
            "pokemon_hp": {},
            "status_count": 0,
            "positive_boosts": 0,
            "negative_boosts": 0,
            "total_lost_health": {},
        }
        battle_info_p2 = {
            "pokemon_hp": {},
            "status_count": 0,
            "positive_boosts": 0,
            "negative_boosts": 0,
            "total_lost_health": {},
        }

        # Add additional health and status information from battle turns
        for turn in battle_timeline:
            for player_key in ["p1_pokemon_state", "p2_pokemon_state"]:
                pokemon_state = turn.get(player_key) or {}
                pokemon_name = pokemon_state.get("name")
                if not pokemon_name:
                    continue

                current_hp = pokemon_state.get("hp_pct")
                boost_dict = pokemon_state.get("boosts") or {}

                dict_name = (
                    battle_info_p1
                    if player_key == "p1_pokemon_state"
                    else battle_info_p2
                )

                former_hp = dict_name["pokemon_hp"].get(pokemon_name, 1.0)

                if current_hp is not None:
                    dict_name["pokemon_hp"][pokemon_name] = current_hp
                    lost = abs(former_hp - current_hp)
                    dict_name["total_lost_health"][pokemon_name] = (
                        dict_name["total_lost_health"].get(pokemon_name, 0.0) + lost
                    )

                for value in boost_dict.values():
                    if value > 0:
                        dict_name["positive_boosts"] += value
                    elif value < 0:
                        dict_name["negative_boosts"] += -1 * value

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
            if end_hp is None:
                continue
            start_hp = 1.0  # Assuming all Pokémon start with full HP (1.0)
            hp_loss = start_hp - end_hp
            total_hp_loss += hp_loss

        return total_hp_loss

    def calculate_total_damage(self, hp_dict: Dict[str, float]) -> float:
        """
        Calculate the total damage for a player during the first 30 rounds.
        This is different from calculate_hp_loss in that healed hp do not affect
        this metric.
        """
        total_damage_pct = 0
        for end_damage in hp_dict.values():
            if end_damage is None:
                continue
            total_damage_pct += end_damage

        return total_damage_pct

    def calculate_type_compatibility(
        self, hp_dict_attacker: Dict[str, float], hp_dict_defender: Dict[str, float]
    ):
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

    def calculate_team_winrate(self, pokemon_hp: Dict[str, float]) -> float:
        """Return the mean win rate for all Pokémon referenced in ``pokemon_hp``."""

        win_rates = self._load_pokemon_win_rates()
        values: List[float] = []

        for name in pokemon_hp.keys():
            if not name:
                continue
            normalized = name.strip().lower()
            if not normalized:
                continue
            rate = win_rates.get(normalized)
            if rate is not None:
                values.append(rate)

        if not values:
            return 0.0

        return float(np.mean(values))

    def _load_pokemon_types(self) -> Dict[str, List[str]]:
        if self._pokemon_types is None:
            summary = PokemonRosterSummary(self.data).build_summary()
            type_map: Dict[str, List[str]] = {}

            for name, entry in summary.items():
                attributes = entry.get("p1_attributes") or {}
                types = attributes.get("types") or []
                normalized_types = [
                    t.strip().upper()
                    for t in types
                    if t and t.strip().lower() != "notype"
                ]
                if not normalized_types:
                    normalized_types = ["NORMAL"]
                type_map[name] = normalized_types

            self._pokemon_types = type_map
        return self._pokemon_types

    def _load_pokemon_win_rates(self) -> Dict[str, float]:
        if self._pokemon_win_rates is None:
            win_counts = self._compute_win_counts_from_data()
            appearance_counter: Counter[str] = Counter()

            for battle in self.data:
                for pokemon in battle.get("p1_team_details", []) or []:
                    name = pokemon.get("name")
                    if not name:
                        continue
                    normalized = name.strip().lower()
                    if normalized:
                        appearance_counter[normalized] += 1

            win_rates: Dict[str, float] = {}
            for name, total in appearance_counter.items():
                wins = win_counts.get(name, 0)
                win_rates[name] = wins / total

            self._pokemon_win_rates = win_rates

        return self._pokemon_win_rates

    def _compute_win_counts_from_data(self) -> Dict[str, int]:
        counter: Counter[str] = Counter()
        for battle in self.data:
            player_won = battle.get("player_won")
            if player_won is None:
                continue
            elif bool(player_won) is False:
                continue

            for pokemon in battle.get("p1_team_details", []) or []:
                name = pokemon.get("name")
                if not name:
                    continue
                normalized = name.strip().lower()
                if normalized:
                    counter[normalized] += 1

        return dict(counter)
