"""Utility for calculating Pokémon attack effectiveness multipliers.

This module provides a single public class, :class:`AttackEffectivenessCalculator`,
which encapsulates the type effectiveness chart for Generation 1 Pokémon games.
It can be used to compute the combined damage multiplier for attacks from one
or multiple attacking types against one or multiple defending types.
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence


TypeName = str


def _build_gen1_type_chart() -> Mapping[TypeName, Mapping[TypeName, float]]:

	return {
		"NORMAL": {
			"ROCK": 0.5,
			"GHOST": 0.0,
		},
		"FIRE": {
			"FIRE": 0.5,
			"WATER": 0.5,
			"GRASS": 2.0,
			"ICE": 2.0,
			"BUG": 2.0,
			"ROCK": 0.5,
			"DRAGON": 0.5,
		},
		"WATER": {
			"FIRE": 2.0,
			"WATER": 0.5,
			"GRASS": 0.5,
			"GROUND": 2.0,
			"ROCK": 2.0,
			"DRAGON": 0.5,
		},
		"ELECTRIC": {
			"WATER": 2.0,
			"ELECTRIC": 0.5,
			"GRASS": 0.5,
			"GROUND": 0.0,
			"FLYING": 2.0,
			"DRAGON": 0.5,
		},
		"GRASS": {
			"FIRE": 0.5,
			"WATER": 2.0,
			"GRASS": 0.5,
			"POISON": 0.5,
			"GROUND": 2.0,
			"FLYING": 0.5,
			"BUG": 0.5,
			"ROCK": 2.0,
			"DRAGON": 0.5,
		},
		"ICE": {
			"WATER": 0.5,
			"GRASS": 2.0,
			"ICE": 0.5,
			"GROUND": 2.0,
			"FLYING": 2.0,
			"DRAGON": 2.0,
		},
		"FIGHTING": {
			"NORMAL": 2.0,
			"ICE": 2.0,
			"POISON": 0.5,
			"FLYING": 0.5,
			"PSYCHIC": 0.5,
			"BUG": 0.5,
			"ROCK": 2.0,
			"GHOST": 0.0,
		},
		"POISON": {
			"GRASS": 2.0,
			"POISON": 0.5,
			"GROUND": 0.5,
			"BUG": 2.0,
			"ROCK": 0.5,
			"GHOST": 0.5,
		},
		"GROUND": {
			"FIRE": 2.0,
			"ELECTRIC": 2.0,
			"GRASS": 0.5,
			"POISON": 2.0,
			"FLYING": 0.0,
			"BUG": 0.5,
			"ROCK": 2.0,
		},
		"FLYING": {
			"ELECTRIC": 0.5,
			"GRASS": 2.0,
			"FIGHTING": 2.0,
			"BUG": 2.0,
			"ROCK": 0.5,
		},
		"PSYCHIC": {
			"FIGHTING": 2.0,
			"POISON": 2.0,
			"PSYCHIC": 0.5,
		},
		"BUG": {
			"FIRE": 0.5,
			"GRASS": 2.0,
			"FIGHTING": 0.5,
			"POISON": 2.0,
			"FLYING": 0.5,
			"PSYCHIC": 2.0,
			"GHOST": 0.5,
		},
		"ROCK": {
			"FIRE": 2.0,
			"ICE": 2.0,
			"FIGHTING": 0.5,
			"GROUND": 0.5,
			"FLYING": 2.0,
			"BUG": 2.0,
		},
		"GHOST": {
			"NORMAL": 0.0,
			"PSYCHIC": 0.0,
			"GHOST": 2.0,
		},
		"DRAGON": {
			"DRAGON": 2.0,
		},
	}


_GEN1_TYPE_CHART = _build_gen1_type_chart()
_KNOWN_TYPES = frozenset(
	set(_GEN1_TYPE_CHART.keys())
	| {t for chart in _GEN1_TYPE_CHART.values() for t in chart.keys()}
)


class AttackEffectivenessCalculator:
	"""Calculate damage multipliers for Pokémon attacks.

	The calculator defaults to the Generation 1 type chart. Call :meth:`calculate`
	with the attacking Pokémon types and defending Pokémon types to obtain the
	resulting multipliers.
	"""

	def __init__(
		self,
		type_chart: Mapping[TypeName, Mapping[TypeName, float]] | None = None,
	):
		self.type_chart = type_chart or _GEN1_TYPE_CHART

	def calculate(
		self,
		attacker_types: Sequence[TypeName] | TypeName,
		defender_types: Sequence[TypeName] | TypeName,
	) -> float:
		"""Return the combined effectiveness multiplier.

		Args:
			attacker_types: Single type string or sequence of type strings for the
				attacking Pokémon.
			defender_types: Single type string or sequence of type strings for the
				defending Pokémon.

		Returns:
			The resulting multiplier as a floating-point number.

		Raises:
			ValueError: If any provided type name is unknown to the chart or the
				collections are empty.
		"""

		attackers = self._normalize_types(attacker_types, role="attacker")
		defenders = self._normalize_types(defender_types, role="defender")

		multiplier = 1.0
		for atk_type in attackers:
			chart_row = self.type_chart.get(atk_type)
			if chart_row is None:
				raise ValueError(f"Unknown attacker type: {atk_type}")

			for def_type in defenders:
				value = chart_row.get(def_type, 1.0)
				multiplier *= value

		return multiplier

	@staticmethod
	def _normalize_types(
		value: Sequence[TypeName] | TypeName,
		*,
		role: str,
	) -> List[TypeName]:
		"""Convert user input into a normalized list of uppercase type names."""

		if isinstance(value, str):
			type_candidates = [value]
		elif isinstance(value, Iterable):
			type_candidates = list(value)
		else:
			raise TypeError(
				f"{role}_types must be a string or an iterable of strings, got {type(value)!r}"
			)

		normalized: List[TypeName] = []
		for type_name in type_candidates:
			if not isinstance(type_name, str):
				raise TypeError(
					f"Type names must be strings, got {type(type_name)!r} in {role}_types"
				)
			normalized_name = type_name.strip().upper()
			if not normalized_name:
				raise ValueError(f"Empty type name in {role}_types")
			if normalized_name not in _KNOWN_TYPES:
				raise ValueError(
					f"Unknown type '{normalized_name}' provided for {role}_types"
				)
			normalized.append(normalized_name)

		if not normalized:
			raise ValueError(f"No {role} types provided")

		return normalized

