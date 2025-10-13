# Pokémon Generation 1 OverUsed (OU) Battle Dataset: Background for Data Scientists

## Introduction
This document provides a high-level overview of the **Pokémon Generation 1 OverUsed (OU) Battle Dataset**, designed for data scientists without prior knowledge of Pokémon. The dataset contains records of competitive Pokémon battles from the first generation of games (Red, Blue, Green, and Yellow).

## Pokémon Basics
### What is Pokémon?
Pokémon is a franchise created by Nintendo, centered around fictional creatures called Pokémon that players train to battle each other.

### Competitive Battling
Competitive Pokémon battling involves constructing teams of Pokémon to compete under standardized rules. The **OverUsed (OU)** tier represents the most commonly used Pokémon in competitive play.

## Dataset Context
### What is the Dataset?
The **Generation 1 OU Battle Dataset** is a collection of battle records from competitive matches in the first generation of Pokémon games. It includes details such as Pokémon teams, battle outcomes, and moves used.

### Why Generation 1?
- **Simplicity:** Fewer Pokémon and mechanics make it easier to analyze.
- **Nostalgia and Popularity:** Many players and researchers are familiar with the original games.
- **Data Availability:** A large number of recorded battles exist from this era.

## Key Rules and Mechanics
### Battle Format
- Single Battles: One Pokémon vs. one Pokémon at a time.
- Team Size: Up to 6 Pokémon per team.
- Objective: Defeat all of the opponent's Pokémon to win.

### Pokémon Stats
- **HP:** Determines how much damage a Pokémon can take.
- **Attack:** Influences the damage dealt by physical moves.
- **Defense:** Reduces damage taken from physical moves.
- **Speed:** Determines the order of moves in battle.
- **Special:** A combined stat for special attack and defense in Generation 1.

### Moves and Damage Calculation
- **Types:** There are 15 types in Generation 1 (e.g., Fire, Water, Psychic).
- **Type Effectiveness:** Some types are more effective against others.
- **Damage Formula:**
  ```
  Damage = (((((2 * Level / 5) + 2) * Attack * Power / Defense) / 50) + 2) * Type Effectiveness * Random (0.85-1.0)
  ```

### Status Conditions
- **Sleep:** Pokémon cannot move for 1-7 turns.
- **Poison:** Pokémon loses HP every turn.
- **Paralysis:** Speed is halved, and there is a chance the Pokémon cannot move.
- **Burn:** Attack is halved.
- **Freeze:** Pokémon cannot move until thawed.

### Effects
- Modify mechanics beyond basic stats/status.
- Field / team effects: affect a whole side (examples: Reflect, Light Screen, Safeguard); usually last a fixed number of turns.
- Volatile / individual effects: apply to a single Pokémon (examples: Confusion, Substitute, Leech Seed); often cleared on switch or after a few turns.
- Multiple effects can coexist; some persist, others are time-limited.
- "No effect" indicates no special effects are active.

### Items
- **Consumable Items:** Players can use items like Potion or X Attack once per battle.
- **Held Items:** Not present in Generation 1.

## Competitive Rules and Bans
### OverUsed (OU) Tier
- The OU tier includes strong but not overpowered Pokémon.
- **Banned Pokémon:** Legendary Pokémon like Mewtwo, Mew, Articuno, Zapdos, and Moltres are often banned.

### Banned Moves
- **One-Hit Knock Out (OHKO) Moves:** Moves like Guillotine and Horn Drill are often banned.

## Conclusion
The **Pokémon Generation 1 OU Battle Dataset** offers a rich source of data for exploring competitive gaming strategies, predictive modeling, and game theory.

## Further Reading
- [Smogon University: Generation 1 OU](https://www.smogon.com/dex/sm/formats/ou/)
- [Pokémon Showdown](https://pokemonshowdown.com/)
- [Kaggle Pokémon Datasets](https://www.kaggle.com/datasets?search=pokémon)

## Example data (from data/sample_battle.json)

The following section shows a small, hand-picked excerpt from a single battle record in the dataset (`data/sample_battle.json`). This example demonstrates how metadata, team details, and a short excerpt of the battle timeline are structured. It helps connect the concepts described above to real field data.

### Metadata
- battle_id: 0
- player_won: true

These metadata indicate which side (the dataset considers the "player") won and provide a unique ID for the battle.

### Player 1 — Team (excerpt)
Player 1 has six Pokémon in this example (all at level 100):

| Name | Types | base_hp | base_atk | base_spa | base_spe |
|---|---|---:|---:|---:|---:|
| starmie | Psychic, Water | 60 | 75 | 100 | 115 |
| exeggutor | Grass, Psychic | 95 | 95 | 125 | 55 |
| chansey | Normal | 250 | 5 | 105 | 50 |
| snorlax | Normal | 160 | 110 | 65 | 30 |
| tauros | Normal | 75 | 100 | 70 | 110 |
| alakazam | Psychic | 55 | 50 | 135 | 120 |

This team composition is typical for Generation-1 OU: strong special attackers (Starmie, Alakazam, Exeggutor), defensive or stall-oriented Pokémon (Chansey, Snorlax), and a fast physical sweeper (Tauros).

### Opponent (lead)
The opponent's lead (`p2_lead_details`) is also a Starmie (level 100). The field often contains mirror or similar leads that create early tempo interactions.

### Selected timeline excerpts (early turns)
The battle record contains a step-by-step turn log. Here is a heavily shortened and readable rendition of the first turns:

- Turn 1:
  - P1 (Starmie) uses Ice Beam
  - P2 (Exeggutor) is partially frozen / has hp_pct ≈ 0.69

- Turn 2:
  - P1 switches to Exeggutor (full HP)
  - P2 switches to Starmie

- Turn 3:
  - P1 (Exeggutor) uses Sleep Powder (status move, accuracy 0.75)
  - P2 (Starmie) uses Blizzard (powerful special attack)

- Turn 5–7 (example of status and recovery interactions):
  - Chansey becomes paralyzed (par) and uses Thunderbolt
  - Starmie uses Recover multiple times to regain HP

- Turn 23:
  - Exeggutor uses Explosion and then falls to hp_pct = 0.0 (fnt)

The JSON stores for each turn, besides the move, the HP percentage (`hp_pct`), status (e.g., `par`, `frz`, `slp`), as well as `effects` and `boosts`. These fine-grained states enable replaying sequences and feature engineering for temporal models.

### How to use this example
- Feature engineering: Use `hp_pct` trajectories, status history, and move sequences as temporal features (RNNs, Transformers, or classical time-series features).
- Labeling: `player_won` is the target label for supervised learning tasks.
- Analysis: Investigate which opening moves, lead combinations, or status effects correlate with higher win probability.

This example is only a snippet; the full dataset contains many thousands of such battle logs with comparable structure.