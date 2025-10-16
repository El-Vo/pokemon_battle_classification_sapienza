from __future__ import annotations
import json
import os
from typing import Any, Dict, Iterable, List, Optional


class PokemonRosterSummary:
	"""Aggregate how often each Pokémon appears in player teams.

	The summary keeps track of how many battles a Pokémon appears in for
	player 1 (p1) and player 2 (p2). When the Pokémon is part of a
	player-1 team, its full attribute payload (level, types, base stats, …)
	is stored so the resulting JSON mirrors the structure found in the
	training data.
	"""

	def __init__(self, battles: Iterable[Dict[str, Any]]):
		# Store a list so we can iterate multiple times without consuming generators
		self.battles: List[Dict[str, Any]] = list(battles)
		self._summary: Optional[List[Dict[str, Any]]] = None

	def build_summary(self) -> List[Dict[str, Any]]:
		"""Build (and cache) the aggregated Pokémon summary."""
		if self._summary is None:
			summary_map: Dict[str, Dict[str, Any]] = {}

			for battle in self.battles:
				# --- Player 1 team members ---
				p1_seen_in_battle = set()
				# this construction with .get('p1_team_details', []) or [] is a defensive way to protect us from missing/None/'falsy' values
				for pokemon in battle.get('p1_team_details', []) or []:
					name = pokemon.get('name')
					if not name:
						continue

					if name not in summary_map:
						summary_map[name] = self._empty_entry(name)
					entry = summary_map[name]
					if name not in p1_seen_in_battle:
						entry['count_p1'] += 1
						p1_seen_in_battle.add(name)

					# Persist the attribute payload the first time we encounter it
					if 'p1_attributes' not in entry:
						entry['p1_attributes'] = self._extract_pokemon_attributes(pokemon)

				# --- Player 2 roster ---
				p2_names = set()

				lead = battle.get('p2_lead_details')
				if isinstance(lead, dict):
					name = lead.get('name')
					if name:
						p2_names.add(name)

				for opponent in battle.get('p2_team_details', []) or []:
					name = opponent.get('name') if isinstance(opponent, dict) else None
					if name:
						p2_names.add(name)

				for turn in battle.get('battle_timeline', []) or []:
					state = turn.get('p2_pokemon_state')
					if isinstance(state, dict):
						name = state.get('name')
						if name:
							p2_names.add(name)

				for name in p2_names:
					if name not in summary_map:
						summary_map[name] = self._empty_entry(name)
					entry = summary_map[name]
					entry['count_p2'] += 1

			self._summary = sorted(summary_map.values(), key=lambda item: item['name'])

		return self._summary

	def to_json(self, indent: int = 2) -> str:
		"""Return the summary encoded as a JSON string."""
		summary = self.build_summary()
		return json.dumps(summary, indent=indent, ensure_ascii=False)

	def save(self, filepath: str, indent: int = 2) -> None:
		"""Persist the summary JSON to ``filepath`` (creating directories if required)."""
		summary_json = self.to_json(indent=indent)
		directory = os.path.dirname(filepath)
		if directory:
			os.makedirs(directory, exist_ok=True)
		with open(filepath, 'w', encoding='utf-8') as f:
			f.write(summary_json)

	@staticmethod
	def _empty_entry(name: str) -> Dict[str, Any]:
		return {
			'name': name,
			'count_p1': 0,
			'count_p2': 0,
		}

	@staticmethod
	def _extract_pokemon_attributes(pokemon: Dict[str, Any]) -> Dict[str, Any]:
		"""Return a shallow copy of all attributes except the name."""
		return {key: value for key, value in pokemon.items() if key != 'name'}

if __name__ == '__main__':
	# If the script is executed directly from the 'analysis' folder, ensure the
	# project root is on sys.path so 'prepare_data' can be imported.
	import sys
	from pathlib import Path

	root = Path(__file__).resolve().parents[1]
	if str(root) not in sys.path:
		sys.path.insert(0, str(root))

	# Local import after ensuring the project root is on sys.path
	from prepare_data.import_source_jsonl import ImportSourceJsonl

	importer = ImportSourceJsonl()
	importer.load_train()
	importer.load_test()

	# Combine train and test battles so the roster includes Pokémon from both splits
	all_battles = importer.train_data + importer.test_data

	summary = PokemonRosterSummary(all_battles)
	output_path = 'data/pokemon_roster.json'
	summary.save(output_path)

	print(f"Loaded {len(importer.train_data)} train battles and {len(importer.test_data)} test battles.")
	print(f"Exported {len(summary.build_summary())} Pokémon to {output_path}")
