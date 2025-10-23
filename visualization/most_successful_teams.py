from pathlib import Path
import sys
from collections import Counter
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.patches import Circle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from prepare_data.import_source_jsonl import ImportSourceJsonl


TeamKey = Tuple[str, ...]


class MostSuccessfulTeamsRadarVisualizer:
	"""Create a spider network that connects Pokémon winning together."""

	def __init__(self, data_path: str = "./data") -> None:
		self.data_source = ImportSourceJsonl(data_path)

	def _ensure_train_data_loaded(self) -> None:
		if not self.data_source.train_data:
			self.data_source.load_train()

	def compute_team_win_counts(self) -> Counter[TeamKey]:
		"""Return win counts per team combination."""

		self._ensure_train_data_loaded()
		team_win_counter: Counter[TeamKey] = Counter()

		for battle in self.data_source.train_data:
			if not battle.get("player_won"):
				continue

			team_details = [
				pokemon
				for pokemon in battle.get("p1_team_details", [])
				if pokemon.get("name")
			]
			if not team_details:
				continue

			team_key: TeamKey = tuple(sorted(pokemon["name"] for pokemon in team_details))
			team_win_counter[team_key] += 1

		return team_win_counter

	def _select_top_teams(
		self,
		team_counts: Counter[TeamKey],
		top_n: Optional[int],
	) -> List[Tuple[TeamKey, int]]:
		# Sort teams by win count descending, then alphabetically for determinism.
		ranked = sorted(team_counts.items(), key=lambda item: (-item[1], item[0]))

		if top_n is not None:
			ranked = ranked[:top_n]

		if not ranked:
			raise ValueError("No winning team combinations found in the data set.")

		return ranked

	def _build_network_data(
		self,
		top_teams: Iterable[Tuple[TeamKey, int]],
	) -> Tuple[List[str], Dict[str, int], Dict[Tuple[str, str], int]]:
		"""Return nodes with weights and pair co-occurrence counts."""

		node_weight: Counter[str] = Counter()
		edge_weight: Counter[Tuple[str, str]] = Counter()

		for team_key, win_count in top_teams:
			for name in team_key:
				node_weight[name] += win_count
			for p1, p2 in combinations(team_key, 2):
				pair = (p1, p2) if p1 <= p2 else (p2, p1)
				edge_weight[pair] += win_count

		nodes = sorted(
			node_weight.keys(),
			key=lambda name: (-node_weight[name], name),
		)

		return nodes, dict(node_weight), dict(edge_weight)

	def plot(
		self,
		top_n: Optional[int] = 4,
		save_path: Optional[str] = None,
	) -> Axes:
		"""Plot a spider network of Pokémon that won together and optionally save it."""

		team_counts = self.compute_team_win_counts()
		top_teams = self._select_top_teams(team_counts, top_n)
		nodes, node_weight, edge_weight = self._build_network_data(top_teams)

		if not nodes:
			raise ValueError("No Pokémon nodes available after filtering winning teams.")

		angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
		radius = 1.0
		positions = {
			name: (radius * np.cos(angle), radius * np.sin(angle))
			for name, angle in zip(nodes, angles)
		}

		sns.set_theme(style="white")
		fig, ax = plt.subplots(figsize=(8.0, 8.0))
		ax.set_aspect("equal")
		ax.axis("off")
		ax.set_title("Winning Team Combinations (Spider Network)")

		# Draw a light circular frame for context.
		circle = Circle((0.0, 0.0), radius, color="lightgray", fill=False, linewidth=1.0, linestyle="--")
		ax.add_patch(circle)

		if edge_weight:
			edge_values = np.array(list(edge_weight.values()), dtype=float)
			min_weight = edge_values.min()
			max_weight = edge_values.max()
			weight_range = max(max_weight - min_weight, 1.0)
			cmap = plt.get_cmap("viridis")
			for (p1, p2), weight in sorted(edge_weight.items(), key=lambda item: (-item[1], item[0])):
				x1, y1 = positions[p1]
				x2, y2 = positions[p2]
				normalized = (weight - min_weight) / weight_range
				line_width = 1.2 + normalized * 4.0
				color = cmap(normalized)
				ax.plot([x1, x2], [y1, y2], color=color, linewidth=line_width, alpha=0.75)

		max_node_weight = max(node_weight.values()) if node_weight else 1
		for name in nodes:
			x, y = positions[name]
			size = 100 + (node_weight.get(name, 0) / max_node_weight) * 200
			ax.scatter([x], [y], s=size, color="darkorange", edgecolors="black", linewidth=0.8, zorder=3)
			text_dx = 0.06 * np.sign(x) if x != 0 else 0
			text_dy = 0.06 * np.sign(y) if y != 0 else 0
			ax.text(
				x * 1.1 + text_dx,
				y * 1.1 + text_dy,
				name,
				ha="center",
				va="center",
				fontsize=11,
				fontweight="bold",
			)

		if save_path:
			fig.savefig(save_path, dpi=300, bbox_inches="tight")

		return ax


__all__ = ["MostSuccessfulTeamsRadarVisualizer", "main"]

DEFAULT_DATA_PATH = PROJECT_ROOT / "data"
DEFAULT_TOP_N: Optional[int] = 4
DEFAULT_SAVE_PATH: Optional[Path] = None
SHOW_PLOT = True


def main() -> None:
	data_path = str(DEFAULT_DATA_PATH)
	top_n = DEFAULT_TOP_N
	save_path = str(DEFAULT_SAVE_PATH) if DEFAULT_SAVE_PATH else None
	show_plot = SHOW_PLOT or save_path is None

	visualizer = MostSuccessfulTeamsRadarVisualizer(data_path=data_path)
	visualizer.plot(top_n=top_n, save_path=save_path)

	if show_plot:
		plt.show()


if __name__ == "__main__":
	main()

