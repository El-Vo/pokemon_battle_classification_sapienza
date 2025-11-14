from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set


class PokemonRosterSummary:
    """Aggregate how often each Pokémon appears in player teams.

    The summary keeps track of how many battles a Pokémon appears in for both
    players and returns the aggregation as a dictionary keyed by normalized
    Pokémon names.
    """

    def __init__(self, battles: Iterable[Dict[str, Any]]):
        self.battles: List[Dict[str, Any]] = list(battles)
        self._summary: Optional[Dict[str, Dict[str, Any]]] = None

    def build_summary(self) -> Dict[str, Dict[str, Any]]:
        """Build (and cache) the aggregated Pokémon summary keyed by name."""
        if self._summary is None:
            summary_map: Dict[str, Dict[str, Any]] = {}

            for battle in self.battles:
                p1_seen_in_battle: Set[str] = set()
                for pokemon in battle.get("p1_team_details") or []:
                    if not isinstance(pokemon, dict):
                        continue

                    normalized_name = self._normalize_name(pokemon.get("name"))
                    if normalized_name is None:
                        continue

                    entry = summary_map.setdefault(
                        normalized_name, self._empty_entry(normalized_name)
                    )

                    if normalized_name not in p1_seen_in_battle:
                        entry["count_p1"] += 1
                        p1_seen_in_battle.add(normalized_name)

                    if entry.get("p1_attributes") is None:
                        entry["p1_attributes"] = self._extract_pokemon_attributes(
                            pokemon
                        )

                for name in self._collect_p2_names(battle):
                    entry = summary_map.setdefault(name, self._empty_entry(name))
                    entry["count_p2"] += 1

            self._summary = summary_map

        return self._summary

    @staticmethod
    def _normalize_name(name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        normalized = name.strip().lower()
        return normalized or None

    def _collect_p2_names(self, battle: Dict[str, Any]) -> Set[str]:
        names: Set[str] = set()

        lead = battle.get("p2_lead_details")
        if isinstance(lead, dict):
            normalized = self._normalize_name(lead.get("name"))
            if normalized is not None:
                names.add(normalized)

        for turn in battle.get("battle_timeline") or []:
            if not isinstance(turn, dict):
                continue
            state = turn.get("p2_pokemon_state")
            if isinstance(state, dict):
                normalized = self._normalize_name(state.get("name"))
                if normalized is not None:
                    names.add(normalized)

        return names

    @staticmethod
    def _empty_entry(name: str) -> Dict[str, Any]:
        return {
            "name": name,
            "count_p1": 0,
            "count_p2": 0,
            "p1_attributes": None,
        }

    @staticmethod
    def _extract_pokemon_attributes(pokemon: Dict[str, Any]) -> Dict[str, Any]:
        """Return a shallow copy of all attributes except the name."""
        return {key: value for key, value in pokemon.items() if key != "name"}
