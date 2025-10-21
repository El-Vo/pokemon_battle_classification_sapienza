from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Union


DEFAULT_HEADER = ["accuracy", "git_commit", "timestamp", "features"]


def _resolve_default_path() -> Path:
    """Return the default path to the accuracy results csv file."""
    return (Path(__file__).resolve().parent.parent / "results" / "accuracy_results.csv").resolve()


def _format_accuracy(value: float) -> str:
    """Format accuracy value to a fixed precision string."""
    return f"{float(value):.6f}"


def _stringify_features(features: Union[str, Sequence[str]]) -> str:
    if isinstance(features, str):
        return features
    return " | ".join(str(feature) for feature in features)


def _current_git_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if commit:
            return commit
    except (OSError, subprocess.CalledProcessError):
        pass
    return "unknown"


@dataclass
class AccuracyResultEntry:
    accuracy: float
    features: Union[str, Sequence[str]]
    git_commit: Optional[str] = None
    timestamp: Optional[datetime] = None

    def as_dict(self) -> dict:
        timestamp = (
            self.timestamp.astimezone(timezone.utc).isoformat()
            if self.timestamp
            else datetime.now(timezone.utc).isoformat()
        )
        return {
            "accuracy": _format_accuracy(self.accuracy),
            "git_commit": self.git_commit or _current_git_commit(),
            "timestamp": timestamp,
            "features": _stringify_features(self.features),
        }


class AccuracyResultsLogger:
    """Append training metrics to the accuracy_results.csv file."""

    def __init__(self, csv_path: Optional[Union[str, Path]] = None):
        self.csv_path = Path(csv_path) if csv_path else _resolve_default_path()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: AccuracyResultEntry) -> Path:
        self._validate_accuracy(entry.accuracy)
        file_exists_with_content = self.csv_path.exists() and self.csv_path.stat().st_size > 0

        with self.csv_path.open("a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=DEFAULT_HEADER)
            if not file_exists_with_content:
                writer.writeheader()
            writer.writerow(entry.as_dict())
        return self.csv_path

    def append_result(
        self,
        accuracy: float,
        features: Union[str, Sequence[str]],
        *,
        git_commit: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> Path:
        entry = AccuracyResultEntry(
            accuracy=accuracy,
            features=features,
            git_commit=git_commit,
            timestamp=timestamp,
        )
        return self.append(entry)

    @staticmethod
    def _validate_accuracy(value: float) -> None:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("Accuracy must be between 0 and 1 inclusive.")