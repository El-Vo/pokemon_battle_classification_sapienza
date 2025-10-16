# Pokémon Gen 1: Attack Effectiveness (documentation)

This document explains how attack effectiveness is calculated in Generation 1 Pokémon games,
how multiple types are combined, and how to apply the rule in code. It also documents the
behaviour implemented by `tools/calculate_attack_effectiveness.py` in this repository.

## High-level rule

- Each attack has exactly one type (the move's type).
- Each Pokémon has one or two types (defender types).
- For multi-typed defenders, the total effect of a single attack against the defender is
  the product of the effectiveness multipliers against each defender type.

In formula form:

multiplier = \prod_{i=1..n} effectiveness(move_type, defender_type_i)

Where `effectiveness(move_type, defender_type)` is typically one of the values: `0`, `0.5`,
`1`, `2` (or other decimals in fractional combinations). If any factor is `0` (an immunity),
then the full product becomes `0`.

## STAB (Same-Type Attack Bonus)

- If the move's type matches at least one of the attacker's types, the move receives a
  Same-Type Attack Bonus (STAB).
- In Gen 1, the canonical STAB multiplier is `1.5`.
- STAB is applied after computing the type-effectiveness product.

Full computation including STAB:

final_damage_multiplier = multiplier * (1.5 if move_type in attacker_types else 1.0)

## Examples

- Electric move vs Water/Flying defender: Electric→Water = 2, Electric→Flying = 2 -> 2 * 2 = 4.0
- Electric move vs Ground defender: Electric→Ground = 0 -> total = 0 (immunity)
- Fire move vs Grass/Poison defender: Fire→Grass = 2, Fire→Poison = 1 -> 2 * 1 = 2.0
- Rock move vs Fire/Flying defender: Rock→Fire = 2, Rock→Flying = 2 -> 4.0

If the attacker is Electric/Flying and uses an Electric move against Water/Flying
(defender), the base multiplier is 4.0 and STAB applies (move type Electric is one of the
attacker's types), so final multiplier = 4.0 * 1.5 = 6.0.

## Implementation notes (this repository)

- The file `tools/calculate_attack_effectiveness.py` implements an
  `AttackEffectivenessCalculator` class that:
  - contains a Gen 1 type chart (mapping attacker-type -> defender-type -> multiplier),
  - normalizes input type names (trims whitespace and normalizes case),
  - validates type names and raises clear errors for unknown or empty names,
  - computes the product of per-type effectiveness values.

- The class expects move types and defender types as either a single string or an
  iterable (list/tuple) of strings. Example usage:

```py
from tools.calculate_attack_effectiveness import AttackEffectivenessCalculator

calc = AttackEffectivenessCalculator()
# single-type move vs single-type defender
calc.calculate("Electric", "Water")  # -> 2.0
# single-type move vs dual-type defender
calc.calculate("Electric", ["Water", "Flying"])  # -> 4.0
```

## Edge cases and validation

- If any per-type multiplier is `0` the final result becomes `0`.
- Input type names are normalized to uppercase and whitespace trimmed. Unknown names
  raise `ValueError`.
- The implementation multiplies default value `1.0` when no explicit entry exists in the
  type chart for a particular attacker->defender pair.

## Gen 1 quirks and historical notes

- The documented product rule is the canonical and widely used approach for Gen 1 type
  math. There are a few engine quirks that historically affected damage calculation
  (e.g. rounding behavior, special cases) but the multiplicative type effectiveness
  rule is the correct conceptual model.