from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Union
from datetime import datetime
import argparse
import sys

class JsonlToJsonConverter:
    """
    Klasse, die eine .jsonl-Datei in eine valide JSON-Datei umwandelt.
    Die Originaldatei bleibt unverändert; es wird eine neue Datei erstellt.
    """

    def __init__(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
    ) -> None:
        self.input_path = Path(input_path)
        if not self.input_path.exists():
            raise FileNotFoundError(f"Eingabedatei nicht gefunden: {self.input_path}")
        if self.input_path.suffix.lower() not in {".jsonl", ".ndl", ".txt"}:
            # .txt and .ndl optionally allowed, but warn is omitted to keep kurz
            pass

        if output_path is None:
            self.output_path = self._default_output_path()
        else:
            self.output_path = Path(output_path)

        if self.output_path.exists() and not overwrite:
            # Standardmäßig nicht überschreiben: Suffix mit Zeitstempel anhängen
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = self.output_path.with_name(
                f"{self.output_path.stem}_converted_{ts}{self.output_path.suffix or '.json'}"
            )

    def _default_output_path(self) -> Path:
        # Ersetze .jsonl durch .json, sonst hänge .json an
        if self.input_path.suffix.lower() == ".jsonl":
            return self.input_path.with_suffix(".json")
        return self.input_path.with_suffix(self.input_path.suffix or ".json")

    def convert(self) -> Path:
        """
        Konvertiert die jsonl-Datei in eine JSON-Datei (als Array).
        Liefert den Pfad zur erzeugten Datei zurück.
        Bei fehlerhafter JSON-Zeile wird eine ValueError mit Zeilennummer geworfen.
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
                    continue  # leere Zeilen überspringen
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Ungültiges JSON in Zeile {i}: {e.msg}") from e

                dumped = json.dumps(obj, ensure_ascii=False)
                if not first:
                    fout.write(",\n")
                fout.write(dumped)
                first = False
            fout.write("\n]\n")

        return self.output_path


# Kurzbeispiel (nur zur Illustration, nicht automatisch ausgeführt):
def main(argv: Optional[list[str]] = None) -> int:
    """Kleine CLI: konvertiert eine jsonl-Datei in eine JSON-Datei.

    Rückgabewerte: 0 bei Erfolg, 1 bei Fehler.
    """
    parser = argparse.ArgumentParser(description="Konvertiert eine .jsonl-Datei in eine .json-Datei (Array).")
    parser.add_argument("input", help="Pfad zur Eingabe-.jsonl-Datei")
    parser.add_argument("-o", "--output", help="Pfad zur Ausgabedatei (optional)")
    parser.add_argument("--overwrite", action="store_true", help="Bestehende Ausgabedatei überschreiben")
    args = parser.parse_args(argv)

    try:
        converter = JsonlToJsonConverter(args.input, output_path=args.output, overwrite=args.overwrite)
        output_path = converter.convert()
        print(f"Erstellt: {output_path}")
        return 0
    except Exception as e:
        # Kurze, nutzerfreundliche Fehlermeldung auf stderr
        print(f"Fehler: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())