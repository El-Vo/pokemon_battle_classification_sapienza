#!/usr/bin/env python3
"""
Example usage of the Pokemon battle data visualization script.

This script demonstrates how to use the visualization tools
with sample data.
"""

from pathlib import Path
import sys

# Add analysis directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'analysis'))

from visualize_battle_data import (
    BattleDataLoader,
    BattleDataAnalyzer,
    BattleDataVisualizer
)


def main():
    """Run a simple visualization example."""
    
    # Define paths
    data_file = Path(__file__).parent.parent / 'data' / 'sample_battle_train.json'
    output_dir = Path(__file__).parent.parent / 'example_visualizations'
    
    print("="*60)
    print("Pokemon Battle Data Visualization Example")
    print("="*60)
    
    # Load data
    print(f"\n1. Loading data from: {data_file.name}")
    battles = BattleDataLoader.load_data(data_file)
    print(f"   ✓ Loaded {len(battles)} battle(s)")
    
    # Analyze data
    print("\n2. Analyzing battle data...")
    analyzer = BattleDataAnalyzer(battles)
    analyzer.analyze()
    print("   ✓ Analysis complete")
    
    # Get some statistics
    win_rates_df = analyzer.get_pokemon_win_rates()
    active_rounds_df = analyzer.get_active_rounds_data()
    attacks_df = analyzer.get_attack_usage_data()
    
    print("\n3. Sample Statistics:")
    print(f"   - Unique Pokemon: {len(analyzer.pokemon_wins)}")
    print(f"   - Unique Attacks: {len(analyzer.attack_usage)}")
    
    if not win_rates_df.empty:
        top_pokemon = win_rates_df.iloc[0]
        print(f"   - Top Pokemon: {top_pokemon['pokemon']} "
              f"({top_pokemon['win_rate']:.1%} win rate)")
    
    if not active_rounds_df.empty:
        most_active = active_rounds_df.iloc[0]
        print(f"   - Most Active: {most_active['pokemon']} "
              f"({most_active['active_rounds']} rounds)")
    
    if not attacks_df.empty:
        top_attack = attacks_df.iloc[0]
        print(f"   - Top Attack: {top_attack['attack']} "
              f"({top_attack['usage_count']} uses)")
    
    # Create visualizations
    print(f"\n4. Creating visualizations in: {output_dir}")
    visualizer = BattleDataVisualizer(analyzer, output_dir)
    visualizer.create_all_visualizations()
    
    print("\n" + "="*60)
    print("Example complete! Check the output directory for charts.")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
