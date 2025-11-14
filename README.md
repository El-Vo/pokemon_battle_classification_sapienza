# FDS: Pokemon Battles Prediction 2025

> Machine-learning workspace for the FDS Pokemon Battles Prediction 2025 Kaggle challenge.

## Overview
- Build and compare battle outcome classifiers (logistic regression, gradient boosting, stacking).
- Extract handcrafted battle features from the raw competition dataset.
- Generate submission files and performance logs for leaderboard tracking.

## Prerequisites
- Python 3.10 or newer.
- Project dependencies from `requirements.txt` (install with `pip install -r requirements.txt`).

## Quick Start
1. Download `train.jsonl` and `test.jsonl` from the [competition page](https://www.kaggle.com/competitions/fds-pokemon-battles-prediction-2025/overview) and place them in `data/`.
2. (Optional) Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`.
3. Install dependencies: `pip install -r requirements.txt`.
4. Train and evaluate a model: `python main.py`.
	- Toggle the model by editing the `algorithm` variable in `main.py` (options: `logistic_regression`, `gradient_boosting`, `stacking`).
	- The script writes submission-ready predictions to `results/` and prints training metrics.

## Repository Tour
| Path | Purpose |
| --- | --- |
| `main.py` | End-to-end entry point: feature extraction, training, evaluation, submission export. |
| `analysis/` | Model-specific trainers, feature engineering utilities, correlation analysis, and testing helpers. |
| `prepare_data/` | Data ingestion helpers for the JSONL competition files. |
| `results/` | Generated submission files and accuracy logs. |
| `docs/` | Background reading on game mechanics and previous experiment notes. |
| `visualization/` | Scripts and exports for exploratory plots and showdown battle dumps. |

## Additional Resources
- Read the AI-generated primer in `docs/intro_to_pokemon.md` for game mechanics context.
- Consult `docs/attack_effectiveness.md` when tuning feature engineering around type matchups.
- Use `tools/single_battle_predictor.py` to inspect model behaviour on individual battles.
- These tools have been used by us to gain an inital understanding of the game. Everything in 
the `docs/` and `visualization/` should be viewed as illustrative and has not been used in the prediction model training process.
