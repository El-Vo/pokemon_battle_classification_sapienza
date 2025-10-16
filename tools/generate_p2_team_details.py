"""Extract battle-level p2 roster information.

This tool reads battle dictionaries (as loaded from `train.jsonl` / `test.jsonl`)
and produces a JSON file documenting, per battle, the battle id, the player-1 team
members, and all known player-2 Pokémon names observed throughout the match.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional


class P2TeamRosterExtractor:
	"""Collect p2 roster names for each battle."""

	def __init__(self, battles: Iterable[Dict[str, Any]]):
		self.battles: List[Dict[str, Any]] = list(battles)
		self._roster: Optional[List[Dict[str, Any]]] = None

	def build_roster(self) -> List[Dict[str, Any]]:
		"""Return a cached list of battle roster summaries."""
		if self._roster is None:
			entries: List[Dict[str, Any]] = []
			for idx, battle in enumerate(self.battles):
				battle_id = battle.get('battle_id')
				if battle_id is None:
					battle_id = idx

				p1_team = self._collect_p1_names(battle)
				p2_team = self._collect_p2_names(battle)

				entries.append(
					{
						'battle_id': battle_id,
						'p1_team': p1_team,
						'p2_team': p2_team,
					}
				)

			self._roster = entries

		return self._roster

	def to_json(self, indent: int = 2) -> str:
		"""Return the roster summaries encoded as JSON."""
		return json.dumps(self.build_roster(), indent=indent, ensure_ascii=False)

	def save(self, filepath: str, indent: int = 2) -> None:
		"""Persist roster summaries to ``filepath`` (creating directories if needed)."""
		payload = self.to_json(indent=indent)
		directory = os.path.dirname(filepath)
		if directory:
			os.makedirs(directory, exist_ok=True)
		with open(filepath, 'w', encoding='utf-8') as f:
			f.write(payload)

	@staticmethod
	def _collect_p1_names(battle: Dict[str, Any]) -> List[str]:
		names: List[str] = []
		for member in (battle.get('p1_team_details') or []):
			if isinstance(member, dict):
				name = member.get('name')
				if name:
					names.append(str(name))
		return _deduplicate(names)

	@staticmethod
	def _collect_p2_names(battle: Dict[str, Any]) -> List[str]:
		names: List[str] = []

		# Lead Pokémon details (if present)
		P2TeamRosterExtractor._maybe_append_name(battle.get('p2_lead_details'), names)

		# Declared team list
		for member in (battle.get('p2_team_details') or []):
			P2TeamRosterExtractor._maybe_append_name(member, names)

		# Timeline states reveal the active Pokémon at each turn
		for turn in (battle.get('battle_timeline') or []):
			if isinstance(turn, dict):
				state = turn.get('p2_pokemon_state')
				P2TeamRosterExtractor._maybe_append_name(state, names)

		return _deduplicate(names)

	@staticmethod
	def _maybe_append_name(candidate: Any, names: List[str]) -> None:
		if isinstance(candidate, dict):
			name = candidate.get('name')
			if name:
				names.append(str(name))
		elif isinstance(candidate, str):
			names.append(candidate)


def _deduplicate(items: Iterable[str]) -> List[str]:
	"""Return items without duplicates while preserving order."""
	seen = set()
	result: List[str] = []
	for item in items:
		if item not in seen:
			seen.add(item)
			result.append(item)
	return result


if __name__ == '__main__':
	# Ensure we can import from the project root when running from the tools directory
	import sys
	from pathlib import Path

	root = Path(__file__).resolve().parents[1]
	if str(root) not in sys.path:
		sys.path.insert(0, str(root))

	from prepare_data.import_source_jsonl import ImportSourceJsonl

	importer = ImportSourceJsonl()
	importer.load_train()

	extractor = P2TeamRosterExtractor(importer.train_data)
	output_path = 'data/p2_team_rosters_train.json'
	extractor.save(output_path)

	print(f"Saved {len(extractor.build_roster())} battle roster entries to {output_path}.")
