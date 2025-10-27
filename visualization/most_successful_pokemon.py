from pathlib import Path
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from prepare_data.import_source import ImportSource


class MostSuccessfulPokemonVisualizer:
	"""Create a bar chart for Pokémon and the number of battles they won."""

	def __init__(self, data_path: str = "./data") -> None:
		self.data_source = ImportSource()

	def _ensure_train_data_loaded(self) -> None:
		if not self.data_source.data:
			self.data_source.load_jsonl('./data/train.jsonl')

	def compute_win_counts(self) -> Dict[str, int]:
		"""Return a mapping of Pokémon name to the number of battles won."""

		self._ensure_train_data_loaded()

		win_counter: Counter[str] = Counter()
		for battle in self.data_source.data:
			if not battle.get("player_won"):
				continue

			for pokemon in battle.get("p1_team_details", []):
				name = pokemon.get("name")
				if not name:
					continue
				win_counter[name] += 1

		return dict(win_counter)

	def _prepare_ranked_counts(
		self, win_counts: Dict[str, int], top_n: Optional[int]
	) -> Tuple[List[str], List[int]]:
		# Sort by win count descending, then alphabetically for deterministic layouts.
		ranked: List[Tuple[str, int]] = sorted(
			win_counts.items(), key=lambda item: (-item[1], item[0])
		)

		if top_n is not None:
			ranked = ranked[:top_n]

		if not ranked:
			raise ValueError("No winning Pokémon found in the data set.")

		names, counts = zip(*ranked)
		return list(names), list(counts)

	def plot(
		self,
		top_n: Optional[int] = None,
		save_path: Optional[str] = None,
	) -> Axes:
		"""Plot the win counts as a bar chart and optionally save the figure."""

		win_counts = self.compute_win_counts()
		names, counts = self._prepare_ranked_counts(win_counts, top_n)

		# Scale figure width with the number of Pokémon to keep labels readable.
		width = max(6.0, len(names) * 0.5)
		sns.set_theme(style="whitegrid")  # Subtle grid makes bar comparisons easier to see.
		fig, ax = plt.subplots(figsize=(width, 6.0))
		sns.barplot(x=names, y=counts, ax=ax, palette="viridis")
		ax.set_ylabel("Battles Won")
		ax.set_xlabel("Pokémon")
		ax.set_title("Pokémon Ranked by Wins (train.jsonl)")
		ax.tick_params(axis="x", rotation=45)
		fig.tight_layout()

		if save_path:
			fig.savefig(save_path, dpi=300)

		return ax


__all__ = ["MostSuccessfulPokemonVisualizer", "main"]

DEFAULT_DATA_PATH = PROJECT_ROOT / "data"
DEFAULT_TOP_N: Optional[int] = None
DEFAULT_SAVE_PATH: Optional[Path] = None
SHOW_PLOT = True


def main() -> None:
	data_path = str(DEFAULT_DATA_PATH)
	top_n = DEFAULT_TOP_N
	save_path = str(DEFAULT_SAVE_PATH) if DEFAULT_SAVE_PATH else None
	show_plot = SHOW_PLOT or save_path is None

	visualizer = MostSuccessfulPokemonVisualizer(data_path=data_path)
	visualizer.plot(top_n=top_n, save_path=save_path)

	if show_plot:
		plt.show()


if __name__ == "__main__":
	main()
