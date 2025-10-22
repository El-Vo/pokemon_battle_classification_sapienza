# Examples

This directory contains example scripts demonstrating how to use various tools in the repository.

## Visualization Example

The `visualize_example.py` script demonstrates how to use the Pokemon battle data visualization tools programmatically.

### Running the Example

```bash
cd examples
python3 visualize_example.py
```

This will:
1. Load sample battle data from `data/sample_battle_train.json`
2. Analyze the battle data to extract statistics
3. Generate visualizations in the `example_visualizations/` directory
4. Print summary statistics to the console

### What You'll See

The script outputs:
- Number of battles loaded
- Analysis progress
- Key statistics (unique Pokemon, attacks, top performers)
- Visualization generation status

### Output Files

Four PNG visualizations are created:
- `pokemon_win_rates.png` - Win rates and battle appearances
- `pokemon_active_rounds.png` - Most active Pokemon by round count
- `team_success.png` - Successful team compositions
- `attack_usage.png` - Most commonly used attacks

### Using with Your Own Data

To analyze your own battle data, modify the `data_file` path in the script:

```python
data_file = Path('path/to/your/data.json')
```

The script supports both single JSON files and JSONL format.

## Additional Examples

More examples can be added here as the project grows. Some ideas:
- Feature extraction examples
- Model training examples
- Custom analysis pipelines
