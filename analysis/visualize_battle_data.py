"""
Pokemon Battle Data Visualization

This script creates meaningful visualizations from Pokemon battle data to analyze:
1. Pokemon who tend to win most battles
2. Pokemon with the most active rounds
3. Most successful team constellations
4. Most commonly used attacks

The script can handle both single JSON files and JSONL format.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class BattleDataLoader:
    """Load battle data from various formats."""
    
    @staticmethod
    def load_json(filepath: Path) -> List[Dict]:
        """Load a single JSON file containing one battle."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Single battle - wrap in list
        if isinstance(data, dict):
            return [data]
        return data
    
    @staticmethod
    def load_jsonl(filepath: Path) -> List[Dict]:
        """Load a JSONL file containing multiple battles."""
        battles = []
        with open(filepath, 'r') as f:
            for line in f:
                battles.append(json.loads(line))
        return battles
    
    @staticmethod
    def load_data(filepath: Path) -> List[Dict]:
        """Load battle data from file, auto-detecting format."""
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filepath.suffix == '.jsonl':
            return BattleDataLoader.load_jsonl(filepath)
        else:
            return BattleDataLoader.load_json(filepath)


class BattleDataAnalyzer:
    """Analyze Pokemon battle data and extract statistics."""
    
    def __init__(self, battles: List[Dict]):
        self.battles = battles
        self.pokemon_wins = defaultdict(lambda: {'wins': 0, 'losses': 0, 'appearances': 0})
        self.pokemon_active_rounds = defaultdict(int)
        self.team_wins = defaultdict(lambda: {'wins': 0, 'losses': 0})
        self.attack_usage = Counter()
        
    def analyze(self):
        """Run all analyses on the battle data."""
        print(f"Analyzing {len(self.battles)} battles...")
        
        for battle in self.battles:
            self._analyze_pokemon_wins(battle)
            self._analyze_active_rounds(battle)
            self._analyze_team_constellations(battle)
            self._analyze_attacks(battle)
        
        print("Analysis complete!")
        
    def _analyze_pokemon_wins(self, battle: Dict):
        """Track which Pokemon appear in winning/losing teams."""
        p1_won = battle.get('player_won', False)
        
        # P1 team
        p1_team = battle.get('p1_team_details', [])
        for pokemon in p1_team:
            name = pokemon.get('name', 'unknown')
            self.pokemon_wins[name]['appearances'] += 1
            if p1_won:
                self.pokemon_wins[name]['wins'] += 1
            else:
                self.pokemon_wins[name]['losses'] += 1
        
        # P2 lead (if available)
        p2_lead = battle.get('p2_lead_details')
        if p2_lead:
            name = p2_lead.get('name', 'unknown')
            self.pokemon_wins[name]['appearances'] += 1
            if not p1_won:
                self.pokemon_wins[name]['wins'] += 1
            else:
                self.pokemon_wins[name]['losses'] += 1
    
    def _analyze_active_rounds(self, battle: Dict):
        """Count how many rounds each Pokemon is active."""
        timeline = battle.get('battle_timeline', [])
        
        for turn in timeline:
            # P1 Pokemon
            p1_state = turn.get('p1_pokemon_state', {})
            if p1_state and p1_state.get('name'):
                self.pokemon_active_rounds[p1_state['name']] += 1
            
            # P2 Pokemon
            p2_state = turn.get('p2_pokemon_state', {})
            if p2_state and p2_state.get('name'):
                self.pokemon_active_rounds[p2_state['name']] += 1
    
    def _analyze_team_constellations(self, battle: Dict):
        """Track which team compositions win or lose."""
        p1_won = battle.get('player_won', False)
        p1_team = battle.get('p1_team_details', [])
        
        # Create sorted tuple of Pokemon names for the team
        if p1_team:
            team_names = tuple(sorted([p.get('name', 'unknown') for p in p1_team]))
            if p1_won:
                self.team_wins[team_names]['wins'] += 1
            else:
                self.team_wins[team_names]['losses'] += 1
    
    def _analyze_attacks(self, battle: Dict):
        """Count which attacks are used most frequently."""
        timeline = battle.get('battle_timeline', [])
        
        for turn in timeline:
            # P1 moves
            p1_move = turn.get('p1_move_details')
            if p1_move and p1_move.get('name'):
                self.attack_usage[p1_move['name']] += 1
            
            # P2 moves
            p2_move = turn.get('p2_move_details')
            if p2_move and p2_move.get('name'):
                self.attack_usage[p2_move['name']] += 1
    
    def get_pokemon_win_rates(self, min_appearances: int = 1) -> pd.DataFrame:
        """Get Pokemon win rates as a DataFrame."""
        data = []
        for name, stats in self.pokemon_wins.items():
            if stats['appearances'] >= min_appearances:
                win_rate = stats['wins'] / stats['appearances'] if stats['appearances'] > 0 else 0
                data.append({
                    'pokemon': name,
                    'wins': stats['wins'],
                    'losses': stats['losses'],
                    'appearances': stats['appearances'],
                    'win_rate': win_rate
                })
        
        df = pd.DataFrame(data)
        return df.sort_values('win_rate', ascending=False)
    
    def get_active_rounds_data(self) -> pd.DataFrame:
        """Get Pokemon active rounds as a DataFrame."""
        data = [{'pokemon': name, 'active_rounds': count} 
                for name, count in self.pokemon_active_rounds.items()]
        df = pd.DataFrame(data)
        return df.sort_values('active_rounds', ascending=False)
    
    def get_team_success_data(self, min_battles: int = 1) -> pd.DataFrame:
        """Get team constellation success rates."""
        data = []
        for team, stats in self.team_wins.items():
            total = stats['wins'] + stats['losses']
            if total >= min_battles:
                win_rate = stats['wins'] / total if total > 0 else 0
                # Create readable team name
                team_str = ', '.join(team[:3]) + ('...' if len(team) > 3 else '')
                data.append({
                    'team': team_str,
                    'full_team': team,
                    'wins': stats['wins'],
                    'losses': stats['losses'],
                    'total_battles': total,
                    'win_rate': win_rate
                })
        
        df = pd.DataFrame(data)
        return df.sort_values('win_rate', ascending=False)
    
    def get_attack_usage_data(self) -> pd.DataFrame:
        """Get attack usage statistics as a DataFrame."""
        data = [{'attack': name, 'usage_count': count} 
                for name, count in self.attack_usage.items()]
        df = pd.DataFrame(data)
        return df.sort_values('usage_count', ascending=False)


class BattleDataVisualizer:
    """Create visualizations from battle data analysis."""
    
    def __init__(self, analyzer: BattleDataAnalyzer, output_dir: Path):
        self.analyzer = analyzer
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_all_visualizations(self):
        """Generate all visualizations."""
        print("\nCreating visualizations...")
        
        self.visualize_pokemon_win_rates()
        self.visualize_active_rounds()
        self.visualize_team_success()
        self.visualize_attack_usage()
        
        print(f"All visualizations saved to {self.output_dir}/")
    
    def visualize_pokemon_win_rates(self, top_n: int = 20):
        """Visualize Pokemon with highest win rates."""
        df = self.analyzer.get_pokemon_win_rates(min_appearances=1)
        
        if df.empty:
            print("No data available for Pokemon win rates")
            return
        
        # Take top N by appearances (most relevant)
        df_top = df.nlargest(top_n, 'appearances')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Win rate
        ax1.barh(df_top['pokemon'], df_top['win_rate'], color='skyblue')
        ax1.set_xlabel('Win Rate')
        ax1.set_ylabel('Pokemon')
        ax1.set_title(f'Top {top_n} Pokemon by Win Rate\n(sorted by appearances)')
        ax1.set_xlim(0, 1)
        ax1.grid(axis='x', alpha=0.3)
        
        # Add percentage labels
        for i, (idx, row) in enumerate(df_top.iterrows()):
            ax1.text(row['win_rate'] + 0.01, i, f"{row['win_rate']:.1%}", 
                    va='center', fontsize=8)
        
        # Plot 2: Total appearances with wins/losses breakdown
        x = np.arange(len(df_top))
        ax2.barh(x, df_top['wins'], label='Wins', color='green', alpha=0.7)
        ax2.barh(x, df_top['losses'], left=df_top['wins'], label='Losses', 
                color='red', alpha=0.7)
        ax2.set_yticks(x)
        ax2.set_yticklabels(df_top['pokemon'])
        ax2.set_xlabel('Number of Battles')
        ax2.set_title(f'Top {top_n} Pokemon by Total Appearances')
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pokemon_win_rates.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Pokemon win rates visualization saved")
    
    def visualize_active_rounds(self, top_n: int = 20):
        """Visualize Pokemon with most active rounds."""
        df = self.analyzer.get_active_rounds_data()
        
        if df.empty:
            print("No data available for active rounds")
            return
        
        df_top = df.head(top_n)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(df_top['pokemon'], df_top['active_rounds'], 
                       color=sns.color_palette("viridis", len(df_top)))
        plt.xlabel('Total Active Rounds')
        plt.ylabel('Pokemon')
        plt.title(f'Top {top_n} Pokemon by Active Rounds')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (idx, row) in enumerate(df_top.iterrows()):
            plt.text(row['active_rounds'] + max(df_top['active_rounds']) * 0.01, i, 
                    f"{row['active_rounds']}", va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pokemon_active_rounds.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Pokemon active rounds visualization saved")
    
    def visualize_team_success(self, top_n: int = 15):
        """Visualize most successful team constellations."""
        df = self.analyzer.get_team_success_data(min_battles=1)
        
        if df.empty:
            print("No data available for team success")
            return
        
        # Take top teams by total battles (most data)
        df_top = df.nlargest(top_n, 'total_battles')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Win rates
        ax1.barh(range(len(df_top)), df_top['win_rate'], color='coral')
        ax1.set_yticks(range(len(df_top)))
        ax1.set_yticklabels(df_top['team'], fontsize=8)
        ax1.set_xlabel('Win Rate')
        ax1.set_title(f'Top {top_n} Team Constellations by Win Rate\n(sorted by total battles)')
        ax1.set_xlim(0, 1)
        ax1.grid(axis='x', alpha=0.3)
        
        # Add percentage labels
        for i, (idx, row) in enumerate(df_top.iterrows()):
            ax1.text(row['win_rate'] + 0.01, i, f"{row['win_rate']:.1%}", 
                    va='center', fontsize=8)
        
        # Plot 2: Total battles
        ax2.barh(range(len(df_top)), df_top['total_battles'], color='steelblue')
        ax2.set_yticks(range(len(df_top)))
        ax2.set_yticklabels(df_top['team'], fontsize=8)
        ax2.set_xlabel('Total Battles')
        ax2.set_title(f'Top {top_n} Team Constellations by Total Battles')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'team_success.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Team success visualization saved")
    
    def visualize_attack_usage(self, top_n: int = 25):
        """Visualize most commonly used attacks."""
        df = self.analyzer.get_attack_usage_data()
        
        if df.empty:
            print("No data available for attack usage")
            return
        
        df_top = df.head(top_n)
        
        plt.figure(figsize=(12, 10))
        bars = plt.barh(df_top['attack'], df_top['usage_count'], 
                       color=sns.color_palette("rocket", len(df_top)))
        plt.xlabel('Usage Count')
        plt.ylabel('Attack')
        plt.title(f'Top {top_n} Most Used Attacks')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (idx, row) in enumerate(df_top.iterrows()):
            plt.text(row['usage_count'] + max(df_top['usage_count']) * 0.01, i, 
                    f"{row['usage_count']}", va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'attack_usage.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Attack usage visualization saved")


def main():
    """Main function to run the visualization pipeline."""
    parser = argparse.ArgumentParser(
        description='Create visualizations from Pokemon battle data'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input data file (.json or .jsonl)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations',
        help='Directory to save visualizations (default: visualizations)'
    )
    
    args = parser.parse_args()
    
    # Load data
    input_path = Path(args.input_file)
    print(f"Loading data from {input_path}...")
    
    try:
        battles = BattleDataLoader.load_data(input_path)
        print(f"Loaded {len(battles)} battle(s)")
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    # Analyze data
    analyzer = BattleDataAnalyzer(battles)
    analyzer.analyze()
    
    # Create visualizations
    output_path = Path(args.output_dir)
    visualizer = BattleDataVisualizer(analyzer, output_path)
    visualizer.create_all_visualizations()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nTotal battles analyzed: {len(battles)}")
    print(f"Unique Pokemon: {len(analyzer.pokemon_wins)}")
    print(f"Unique team compositions: {len(analyzer.team_wins)}")
    print(f"Unique attacks: {len(analyzer.attack_usage)}")
    
    # Top 5 by win rate
    win_rates_df = analyzer.get_pokemon_win_rates(min_appearances=1)
    if not win_rates_df.empty:
        print("\nTop 5 Pokemon by Win Rate:")
        for idx, row in win_rates_df.head(5).iterrows():
            print(f"  {row['pokemon']}: {row['win_rate']:.1%} "
                  f"({row['wins']}W-{row['losses']}L in {row['appearances']} battles)")
    
    # Top 5 by active rounds
    active_df = analyzer.get_active_rounds_data()
    if not active_df.empty:
        print("\nTop 5 Pokemon by Active Rounds:")
        for idx, row in active_df.head(5).iterrows():
            print(f"  {row['pokemon']}: {row['active_rounds']} rounds")
    
    # Top 5 attacks
    attacks_df = analyzer.get_attack_usage_data()
    if not attacks_df.empty:
        print("\nTop 5 Most Used Attacks:")
        for idx, row in attacks_df.head(5).iterrows():
            print(f"  {row['attack']}: {row['usage_count']} times")
    
    print("\n" + "="*60)
    print(f"All visualizations saved to: {output_path.absolute()}")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
