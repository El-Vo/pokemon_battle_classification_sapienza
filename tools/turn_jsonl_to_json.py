from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Union
from datetime import datetime
import argparse
import sys

class JsonlToJsonConverter:
    """
    Class that converts a .jsonl file into a valid JSON file.
    The original file remains unchanged; a new file is created.
    """

    def __init__(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
    ) -> None:
        self.input_path = Path(input_path)
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        if self.input_path.suffix.lower() not in {".jsonl", ".ndl", ".txt"}:
            # .txt and .ndl optionally allowed, but warning is omitted to keep it concise
            pass

        if output_path is None:
            self.output_path = self._default_output_path()
        else:
            self.output_path = Path(output_path)

        if self.output_path.exists() and not overwrite:
            # By default, do not overwrite: append a timestamp suffix
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = self.output_path.with_name(
                f"{self.output_path.stem}_converted_{ts}{self.output_path.suffix or '.json'}"
            )

    def _default_output_path(self) -> Path:
        # Replace .jsonl with .json, otherwise append .json
        if self.input_path.suffix.lower() == ".jsonl":
            return self.input_path.with_suffix(".json")
        return self.input_path.with_suffix(self.input_path.suffix or ".json")

    def convert(self) -> Path:
        """
        Converts the jsonl file into a JSON file (as an array).
        Returns the path to the generated file.
        If a JSON line is invalid, a ValueError with the line number is raised.
        """
        first = True
        out_parent = self.output_path.parent
        out_parent.mkdir(parents=True, exist_ok=True)

        with self.input_path.open("r", encoding="utf-8") as fin, \
             self.output_path.open("w", encoding="utf-8") as fout:
            fout.write("[\n")
            for i, raw_line in enumerate(fin, start=1):
                line = raw_line.strip()
                if not line:
                    continue  # skip empty lines
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in line {i}: {e.msg}") from e

                dumped = json.dumps(obj, ensure_ascii=False)
                if not first:
                    fout.write(",\n")
                fout.write(dumped)
                first = False
            fout.write("\n]\n")

        return self.output_path


# Short example (for illustration only, not automatically executed):
def main(argv: Optional[list[str]] = None) -> int:
    """Small CLI: converts a jsonl file into a JSON file.

    Return values: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(description="Converts a .jsonl file into a .json file (array).")
    parser.add_argument("input", help="Path to the input .jsonl file")
    parser.add_argument("-o", "--output", help="Path to the output file (optional)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")
    args = parser.parse_args(argv)

    try:
        converter = JsonlToJsonConverter(args.input, output_path=args.output, overwrite=args.overwrite)
        output_path = converter.convert()
        print(f"Created: {output_path}")
        return 0
    except Exception as e:
        # Short, user-friendly error message on stderr
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())