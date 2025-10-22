# Pokemon Battle Data Visualizations

This document describes how to use the visualization tools to analyze Pokemon battle data.

## Overview

The `visualize_battle_data.py` script creates comprehensive visualizations from Pokemon battle data to help understand:

1. **Pokemon Win Rates**: Which Pokemon appear most frequently in winning teams
2. **Active Rounds**: Which Pokemon stay active for the most turns
3. **Team Constellations**: Which team compositions are most successful
4. **Attack Usage**: Which attacks are used most frequently in battles

## Installation

First, ensure all required dependencies are installed:

```bash
pip install -r requirements.txt
```

The visualization tool requires:
- pandas
- numpy
- matplotlib
- seaborn

## Usage

### Basic Usage

To create visualizations from a battle data file:

```bash
python3 analysis/visualize_battle_data.py <input_file>
```

For example, with the sample training data:

```bash
python3 analysis/visualize_battle_data.py data/sample_battle_train.json
```

### Specifying Output Directory

By default, visualizations are saved to the `visualizations/` directory. You can specify a different location:

```bash
python3 analysis/visualize_battle_data.py data/sample_battle_train.json --output-dir my_visualizations
```

### Supported Data Formats

The script supports both single JSON files and JSONL format:

- **Single battle JSON**: `sample_battle_train.json`
- **Multiple battles JSON array**: `[battle1, battle2, ...]`
- **JSONL format**: One battle per line in `train.jsonl`

## Data Structure Requirements

The script expects battle data with the following structure:

```json
{
  "battle_id": 0,
  "player_won": true,
  "p1_team_details": [
    {
      "name": "starmie",
      "level": 100,
      "types": ["psychic", "water"],
      ...
    }
  ],
  "p2_lead_details": {...},
  "battle_timeline": [
    {
      "turn": 1,
      "p1_pokemon_state": {"name": "starmie", ...},
      "p1_move_details": {"name": "icebeam", ...},
      "p2_pokemon_state": {"name": "exeggutor", ...},
      "p2_move_details": {...}
    }
  ]
}
```

## Generated Visualizations

The script generates four PNG files:

### 1. pokemon_win_rates.png
Shows the Pokemon with the highest win rates and total battle appearances. Includes:
- Bar chart of win rates (percentage)
- Stacked bar chart showing wins vs losses

### 2. pokemon_active_rounds.png
Displays which Pokemon are active for the most turns across all battles. This helps identify:
- Pokemon that stay on the field longer
- Most commonly used Pokemon in battles

### 3. team_success.png
Analyzes which team compositions (combinations of 6 Pokemon) are most successful:
- Win rates for different team configurations
- Total battles for each team composition

### 4. attack_usage.png
Shows the most frequently used attacks across all battles, helping identify:
- Popular move choices
- Meta attack strategies

## Output Example

When running the script, you'll see output like:

```
Loading data from data/sample_battle_train.json...
Loaded 1 battle(s)
Analyzing 1 battles...
Analysis complete!

Creating visualizations...
✓ Pokemon win rates visualization saved
✓ Pokemon active rounds visualization saved
✓ Team success visualization saved
✓ Attack usage visualization saved

============================================================
SUMMARY STATISTICS
============================================================

Total battles analyzed: 1
Unique Pokemon: 6
Unique team compositions: 1
Unique attacks: 11

Top 5 Pokemon by Win Rate:
  exeggutor: 100.0% (1W-0L in 1 battles)
  chansey: 100.0% (1W-0L in 1 battles)
  ...

Top 5 Most Used Attacks:
  psychic: 10 times
  sleeppowder: 8 times
  ...
============================================================
```

## Using with Larger Datasets

The script is designed to scale to much larger datasets. When you have a `train.jsonl` file with thousands of battles:

```bash
python3 analysis/visualize_battle_data.py data/train.jsonl --output-dir visualizations_full
```

The visualizations will automatically show the top N entries (configurable in the code) to keep the charts readable.

## Customization

You can modify the number of top entries shown in visualizations by editing the default parameters in `visualize_battle_data.py`:

- `visualize_pokemon_win_rates(top_n=20)`: Top 20 Pokemon
- `visualize_active_rounds(top_n=20)`: Top 20 most active
- `visualize_team_success(top_n=15)`: Top 15 team compositions
- `visualize_attack_usage(top_n=25)`: Top 25 attacks

## Troubleshooting

### Missing Data Warning

If you see warnings like "No data available for ...", ensure your input file:
1. Is properly formatted JSON
2. Contains the required fields (battle_timeline, p1_team_details, etc.)
3. Has at least one complete battle

### Import Errors

If you get import errors, make sure all dependencies are installed:

```bash
pip install pandas numpy matplotlib seaborn
```

## Technical Details

The visualization script consists of three main classes:

- **BattleDataLoader**: Handles loading data from various formats
- **BattleDataAnalyzer**: Processes battle data and extracts statistics
- **BattleDataVisualizer**: Creates and saves visualization plots

All visualizations are saved as high-resolution PNG files (300 DPI) suitable for reports and presentations.
