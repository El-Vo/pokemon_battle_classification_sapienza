"""
Convert Pokemon Showdown HTML battle replays to JSON format.

This module parses HTML replay files and converts them into the structured
JSON format used for battle training data.
"""

from __future__ import annotations
import json
import os
from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup


class BattleHTMLConverter:
    """Convert Pokemon Showdown HTML battle replays to structured JSON format."""

    def __init__(self, pokemon_roster_path: str = "data/pokemon_roster.json"):
        """Initialize the converter with Pokemon roster data.

        Args:
            pokemon_roster_path: Path to the pokemon_roster.json file containing
                                 Pokemon base stats and type information.
        """
        self.pokemon_roster_path = pokemon_roster_path
        self.pokemon_data = self._load_pokemon_roster()
        self.move_data = self._initialize_move_data()

    def _load_pokemon_roster(self) -> Dict[str, Dict[str, Any]]:
        """Load Pokemon roster data from JSON file."""
        with open(self.pokemon_roster_path, "r", encoding="utf-8") as f:
            roster = json.load(f)

        # Convert to dict with pokemon name as key
        return {
            pokemon["name"]: pokemon["p1_attributes"]
            for pokemon in roster
            if "p1_attributes" in pokemon
        }

    def _initialize_move_data(self) -> Dict[str, Dict[str, Any]]:
        """Initialize move data with known moves from the training data."""
        # Based on moves seen in train.jsonl
        return {
            "icebeam": {
                "type": "ICE",
                "category": "SPECIAL",
                "base_power": 95,
                "accuracy": 1.0,
                "priority": 0,
            },
            "sleeppowder": {
                "type": "GRASS",
                "category": "STATUS",
                "base_power": 0,
                "accuracy": 0.75,
                "priority": 0,
            },
            "blizzard": {
                "type": "ICE",
                "category": "SPECIAL",
                "base_power": 120,
                "accuracy": 0.9,
                "priority": 0,
            },
            "thunderbolt": {
                "type": "ELECTRIC",
                "category": "SPECIAL",
                "base_power": 95,
                "accuracy": 1.0,
                "priority": 0,
            },
            "thunderwave": {
                "type": "ELECTRIC",
                "category": "STATUS",
                "base_power": 0,
                "accuracy": 1.0,
                "priority": 0,
            },
            "recover": {
                "type": "NORMAL",
                "category": "STATUS",
                "base_power": 0,
                "accuracy": 1.0,
                "priority": 0,
            },
            "bodyslam": {
                "type": "NORMAL",
                "category": "PHYSICAL",
                "base_power": 85,
                "accuracy": 1.0,
                "priority": 0,
            },
            "reflect": {
                "type": "PSYCHIC",
                "category": "STATUS",
                "base_power": 0,
                "accuracy": 1.0,
                "priority": 0,
            },
            "rest": {
                "type": "PSYCHIC",
                "category": "STATUS",
                "base_power": 0,
                "accuracy": 1.0,
                "priority": 0,
            },
            "psychic": {
                "type": "PSYCHIC",
                "category": "SPECIAL",
                "base_power": 90,
                "accuracy": 1.0,
                "priority": 0,
            },
            "explosion": {
                "type": "NORMAL",
                "category": "PHYSICAL",
                "base_power": 170,
                "accuracy": 1.0,
                "priority": 0,
            },
            "earthquake": {
                "type": "GROUND",
                "category": "PHYSICAL",
                "base_power": 100,
                "accuracy": 1.0,
                "priority": 0,
            },
            "hyperbeam": {
                "type": "NORMAL",
                "category": "PHYSICAL",
                "base_power": 150,
                "accuracy": 0.9,
                "priority": 0,
            },
            "lovelykiss": {
                "type": "NORMAL",
                "category": "STATUS",
                "base_power": 0,
                "accuracy": 0.75,
                "priority": 0,
            },
            "seismictoss": {
                "type": "FIGHTING",
                "category": "PHYSICAL",
                "base_power": 1,
                "accuracy": 1.0,
                "priority": 0,
            },
            "doubleedge": {
                "type": "NORMAL",
                "category": "PHYSICAL",
                "base_power": 100,
                "accuracy": 1.0,
                "priority": 0,
            },
            "counter": {
                "type": "FIGHTING",
                "category": "PHYSICAL",
                "base_power": 1,
                "accuracy": 1.0,
                "priority": -1,
            },
            "selfdestruct": {
                "type": "NORMAL",
                "category": "PHYSICAL",
                "base_power": 130,
                "accuracy": 1.0,
                "priority": 0,
            },
            "megadrain": {
                "type": "GRASS",
                "category": "SPECIAL",
                "base_power": 40,
                "accuracy": 1.0,
                "priority": 0,
            },
            "substitute": {
                "type": "NORMAL",
                "category": "STATUS",
                "base_power": 0,
                "accuracy": 1.0,
                "priority": 0,
            },
            "amnesia": {
                "type": "PSYCHIC",
                "category": "STATUS",
                "base_power": 0,
                "accuracy": 1.0,
                "priority": 0,
            },
            "hypnosis": {
                "type": "PSYCHIC",
                "category": "STATUS",
                "base_power": 0,
                "accuracy": 0.6,
                "priority": 0,
            },
            "softboiled": {
                "type": "NORMAL",
                "category": "STATUS",
                "base_power": 0,
                "accuracy": 1.0,
                "priority": 0,
            },
            "fireblast": {
                "type": "FIRE",
                "category": "SPECIAL",
                "base_power": 120,
                "accuracy": 0.85,
                "priority": 0,
            },
            "agility": {
                "type": "PSYCHIC",
                "category": "STATUS",
                "base_power": 0,
                "accuracy": 1.0,
                "priority": 0,
            },
            "drillpeck": {
                "type": "FLYING",
                "category": "PHYSICAL",
                "base_power": 80,
                "accuracy": 1.0,
                "priority": 0,
            },
            "sing": {
                "type": "NORMAL",
                "category": "STATUS",
                "base_power": 0,
                "accuracy": 0.55,
                "priority": 0,
            },
            "headbutt": {
                "type": "NORMAL",
                "category": "PHYSICAL",
                "base_power": 70,
                "accuracy": 1.0,
                "priority": 0,
            },
            "stunspore": {
                "type": "GRASS",
                "category": "STATUS",
                "base_power": 0,
                "accuracy": 0.75,
                "priority": 0,
            },
            "razorleaf": {
                "type": "GRASS",
                "category": "SPECIAL",
                "base_power": 55,
                "accuracy": 0.95,
                "priority": 0,
            },
        }

    def convert_html_to_json(
        self, html_path: str, player_perspective: str = "p1", battle_id: int = 0
    ) -> Dict[str, Any]:
        """Convert an HTML battle replay to JSON format.

        Args:
            html_path: Path to the HTML replay file.
            player_perspective: Which player's perspective ('p1' or 'p2').
            battle_id: Battle ID to assign to this battle.

        Returns:
            Dictionary containing the battle data in the training format.
        """
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, "html.parser")

        # Extract battle log data
        battle_log_script = soup.find(
            "script", {"type": "text/plain", "class": "battle-log-data"}
        )
        if not battle_log_script:
            raise ValueError("Could not find battle log data in HTML")

        log_lines = battle_log_script.string.strip().split("\n")

        # Parse the battle log
        battle_data = self._parse_battle_log(log_lines, player_perspective, battle_id)

        return battle_data

    def _parse_battle_log(
        self, log_lines: List[str], player_perspective: str, battle_id: int
    ) -> Dict[str, Any]:
        """Parse the battle log lines into structured data."""

        # Initialize result structure
        result = {
            "p1_team_details": [],
            "p2_lead_details": {},
            "battle_timeline": [],
            "battle_id": battle_id,
        }

        # Track state
        p1_team = set()
        p2_team = set()
        current_turn = 0
        turn_data: Dict[str, Optional[str]] = {}
        p1_active = None
        p2_active = None
        p1_hp = 1.0
        p2_hp = 1.0
        p1_status = "nostatus"
        p2_status = "nostatus"
        p1_effects = ["noeffect"]
        p2_effects = ["noeffect"]
        p1_boosts = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
        p2_boosts = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
        # Track pending switches (Pokemon that will be active next turn)
        p1_pending_switch = None
        p2_pending_switch = None

        for line in log_lines:
            if not line.strip() or line.startswith("|j|") or line.startswith("|t:|"):
                continue

            parts = line.split("|")

            if len(parts) < 2:
                continue

            command = parts[1]

            # Switch command - track active Pokemon
            if command == "switch":
                player = parts[2].split(":")[0]  # e.g., 'p1a' -> 'p1'
                pokemon_name = parts[2].split(":")[1].strip().lower()
                hp_info = parts[3]

                if player.startswith("p1"):
                    # If p1 is fainted, store as pending switch for next turn
                    if p1_status == "fnt":
                        p1_pending_switch = {
                            "name": pokemon_name,
                            "hp": self._parse_hp(hp_info),
                        }
                    else:
                        # Otherwise, switch immediately
                        p1_active = pokemon_name
                        p1_hp = self._parse_hp(hp_info)
                        p1_status = "nostatus"
                        p1_effects = ["noeffect"]
                        p1_boosts = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
                    p1_team.add(pokemon_name)

                elif player.startswith("p2"):
                    # If p2 is fainted, store as pending switch for next turn
                    if p2_status == "fnt":
                        p2_pending_switch = {
                            "name": pokemon_name,
                            "hp": self._parse_hp(hp_info),
                        }
                    else:
                        # Otherwise, switch immediately
                        p2_active = pokemon_name
                        p2_hp = self._parse_hp(hp_info)
                        p2_status = "nostatus"
                        p2_effects = ["noeffect"]
                        p2_boosts = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
                    p2_team.add(pokemon_name)

                    # First p2 switch is the lead
                    if not result["p2_lead_details"]:
                        result["p2_lead_details"] = self._get_pokemon_details(
                            pokemon_name
                        )

            # Turn marker
            elif command == "turn":
                # Save previous turn if it exists
                if current_turn > 0 and p1_active and p2_active:
                    result["battle_timeline"].append(
                        self._create_turn_entry(
                            current_turn,
                            p1_active,
                            p1_hp,
                            p1_status,
                            p1_effects,
                            p1_boosts,
                            turn_data.get("p1_move"),
                            p2_active,
                            p2_hp,
                            p2_status,
                            p2_effects,
                            p2_boosts,
                            turn_data.get("p2_move"),
                        )
                    )

                current_turn = int(parts[2])

                # Stop processing after turn 30
                if current_turn > 30:
                    break

                # Apply pending switches from previous turn
                if p1_pending_switch:
                    p1_active = p1_pending_switch["name"]
                    p1_hp = p1_pending_switch["hp"]
                    p1_status = "nostatus"
                    p1_effects = ["noeffect"]
                    p1_boosts = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
                    p1_pending_switch = None

                if p2_pending_switch:
                    p2_active = p2_pending_switch["name"]
                    p2_hp = p2_pending_switch["hp"]
                    p2_status = "nostatus"
                    p2_effects = ["noeffect"]
                    p2_boosts = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
                    p2_pending_switch = None

                turn_data = {"p1_move": None, "p2_move": None}

            # Move command
            elif command == "move":
                player = parts[2].split(":")[0]
                move_name = parts[3].lower()

                if player.startswith("p1"):
                    turn_data["p1_move"] = move_name
                elif player.startswith("p2"):
                    turn_data["p2_move"] = move_name

            # Damage
            elif command == "-damage":
                player = parts[2].split(":")[0]
                hp_info = parts[3]

                if player.startswith("p1"):
                    p1_hp = self._parse_hp(hp_info)
                elif player.startswith("p2"):
                    p2_hp = self._parse_hp(hp_info)

            # Status
            elif command == "-status":
                player = parts[2].split(":")[0]
                status = parts[3].lower()

                if player.startswith("p1"):
                    p1_status = status
                elif player.startswith("p2"):
                    p2_status = status

            # Cure status
            elif command == "-curestatus":
                player = parts[2].split(":")[0]

                if player.startswith("p1"):
                    p1_status = "nostatus"
                elif player.startswith("p2"):
                    p2_status = "nostatus"

            # Faint
            elif command == "faint":
                player = parts[2].split(":")[0]

                if player.startswith("p1"):
                    p1_hp = 0.0
                    p1_status = "fnt"
                elif player.startswith("p2"):
                    p2_hp = 0.0
                    p2_status = "fnt"

            # Boost/Unboost
            elif command in ["-boost", "-unboost"]:
                player = parts[2].split(":")[0]
                stat = parts[3].lower()
                amount = int(parts[4])

                if command == "-unboost":
                    amount = -amount

                if player.startswith("p1") and stat in p1_boosts:
                    p1_boosts[stat] += amount
                elif player.startswith("p2") and stat in p2_boosts:
                    p2_boosts[stat] += amount

            # Start effect (like Reflect, Substitute)
            elif command == "-start":
                player = parts[2].split(":")[0]
                effect = parts[3].lower()

                if player.startswith("p1"):
                    if "noeffect" in p1_effects:
                        p1_effects = [effect]
                    else:
                        p1_effects.append(effect)
                elif player.startswith("p2"):
                    if "noeffect" in p2_effects:
                        p2_effects = [effect]
                    else:
                        p2_effects.append(effect)

            # End effect
            elif command == "-end":
                player = parts[2].split(":")[0]
                effect = parts[3].lower()

                if player.startswith("p1") and effect in p1_effects:
                    p1_effects.remove(effect)
                    if not p1_effects:
                        p1_effects = ["noeffect"]
                elif player.startswith("p2") and effect in p2_effects:
                    p2_effects.remove(effect)
                    if not p2_effects:
                        p2_effects = ["noeffect"]

            # Heal
            elif command == "-heal":
                player = parts[2].split(":")[0]
                hp_info = parts[3]

                if player.startswith("p1"):
                    p1_hp = self._parse_hp(hp_info)
                elif player.startswith("p2"):
                    p2_hp = self._parse_hp(hp_info)

        # Add final turn
        if current_turn > 0 and p1_active and p2_active:
            result["battle_timeline"].append(
                self._create_turn_entry(
                    current_turn,
                    p1_active,
                    p1_hp,
                    p1_status,
                    p1_effects,
                    p1_boosts,
                    turn_data.get("p1_move"),
                    p2_active,
                    p2_hp,
                    p2_status,
                    p2_effects,
                    p2_boosts,
                    turn_data.get("p2_move"),
                )
            )

        # Build p1 team details
        for pokemon in p1_team:
            details = self._get_pokemon_details(pokemon)
            if details:
                result["p1_team_details"].append(details)

        return result

    def _parse_hp(self, hp_string: str) -> float:
        """Parse HP string to percentage (0.0 to 1.0)."""
        # Format: "75/100" or "0 fnt"
        if "fnt" in hp_string.lower():
            return 0.0

        # Remove status conditions if present
        hp_part = hp_string.split()[0]

        if "/" in hp_part:
            parts = hp_part.split("/")
            # Clean up any escape characters or whitespace
            current = parts[0].strip().replace("\\", "")
            maximum = parts[1].strip().replace("\\", "")
            try:
                return float(current) / float(maximum)
            except (ValueError, ZeroDivisionError):
                return 1.0

        return 1.0

    def _get_pokemon_details(self, pokemon_name: str) -> Dict[str, Any]:
        """Get Pokemon details from roster."""
        pokemon_name = pokemon_name.lower().strip()

        if pokemon_name in self.pokemon_data:
            details = {"name": pokemon_name}
            details.update(self.pokemon_data[pokemon_name])
            return details

        # Return minimal structure if not found
        return {
            "name": pokemon_name,
            "level": 100,
            "types": ["notype", "notype"],
            "base_hp": 0,
            "base_atk": 0,
            "base_def": 0,
            "base_spa": 0,
            "base_spd": 0,
            "base_spe": 0,
        }

    def _get_move_details(self, move_name: str) -> Optional[Dict[str, Any]]:
        """Get move details."""
        move_name = move_name.lower().strip().replace(" ", "")

        if move_name in self.move_data:
            details = {"name": move_name}
            details.update(self.move_data[move_name])
            return details

        return None

    def _create_turn_entry(
        self,
        turn: int,
        p1_pokemon: str,
        p1_hp: float,
        p1_status: str,
        p1_effects: List[str],
        p1_boosts: Dict[str, int],
        p1_move: Optional[str],
        p2_pokemon: str,
        p2_hp: float,
        p2_status: str,
        p2_effects: List[str],
        p2_boosts: Dict[str, int],
        p2_move: Optional[str],
    ) -> Dict[str, Any]:
        """Create a turn entry for the battle timeline."""
        return {
            "turn": turn,
            "p1_pokemon_state": {
                "name": p1_pokemon,
                "hp_pct": p1_hp,
                "status": p1_status,
                "effects": p1_effects.copy(),
                "boosts": p1_boosts.copy(),
            },
            "p1_move_details": self._get_move_details(p1_move) if p1_move else None,
            "p2_pokemon_state": {
                "name": p2_pokemon,
                "hp_pct": p2_hp,
                "status": p2_status,
                "effects": p2_effects.copy(),
                "boosts": p2_boosts.copy(),
            },
            "p2_move_details": self._get_move_details(p2_move) if p2_move else None,
        }

    def save_to_json(self, battle_data: Dict[str, Any], output_path: str) -> None:
        """Save battle data to JSON file."""
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(battle_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # Convert all HTML files in data/battle_htmls/
    converter = BattleHTMLConverter()

    html_dir = "data/battle_htmls"

    if not os.path.exists(html_dir):
        print(f"Directory not found: {html_dir}")
        exit(1)

    # Find all HTML files in the directory
    html_files = [f for f in os.listdir(html_dir) if f.endswith(".html")]

    if not html_files:
        print(f"No HTML files found in {html_dir}")
        exit(1)

    print(f"Found {len(html_files)} HTML file(s) to convert\n")

    for idx, html_filename in enumerate(html_files):
        html_path = os.path.join(html_dir, html_filename)

        # Create output filename: same name as HTML but with .json extension
        # and in the data/battle_htmls/json/ directory
        json_filename = os.path.splitext(html_filename)[0] + ".json"
        output_file = os.path.join(html_dir, "json", json_filename)

        try:
            print(f"[{idx + 1}/{len(html_files)}] Converting {html_filename}...")
            battle_data = converter.convert_html_to_json(html_path, battle_id=idx)
            converter.save_to_json(battle_data, output_file)
            print(f"  ✓ Saved to {output_file}")
            print(f"    - P1 team size: {len(battle_data['p1_team_details'])}")
            print(f"    - Total turns: {len(battle_data['battle_timeline'])}")
        except Exception as e:
            print(f"  ✗ Error converting {html_filename}: {e}")

        print()

    print(f"Conversion complete! Processed {len(html_files)} file(s).")
