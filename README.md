# FDS: Pokemon Battles prediction 2025

This repository is a space to track our progress on the [Foundations/Fundamentals of Data Science kaggle competition](https://www.kaggle.com/competitions/fds-pokemon-battles-prediction-2025/overview).

A general primer is written (AI) in the document [intro_to_pokemon.md](./intro_to_pokemon.md).

The battles were fought based on the "Generation 1 OverUsed (OU)" competitive tier. All of the extra rules when fighting in this style can be found [here](https://www.smogon.com/dex/rb/formats/ou/).

## Data Visualization

To visualize battle data and analyze Pokemon statistics, use the visualization script:

```bash
python3 analysis/visualize_battle_data.py data/sample_battle_train.json
```

This creates visualizations showing:
- Pokemon win rates and battle appearances
- Most active Pokemon (by number of rounds)
- Successful team constellations
- Most commonly used attacks

See [docs/visualizations.md](./docs/visualizations.md) for detailed documentation.