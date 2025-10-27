from __future__ import annotations
from typing import Iterable, Literal, Optional
import numpy as np
import pandas as pd


class FeatureCorrelationAnalyzer:
    """Compute feature correlations based on the extracted battle features."""

    def __init__(
        self,
        feature_data: pd.DataFrame,
        feature_names: Optional[Iterable[str]] = None,
    ) -> None:
        self.feature_data = feature_data
        self.feature_names = feature_names
        self._correlation: Optional[pd.DataFrame] = None

    def compute_correlation(
        self,
        method: Literal["pearson", "kendall", "spearman"] = "pearson",
        include_target: bool = False,
    ) -> pd.DataFrame:
        """Compute and cache the correlation matrix for the selected features."""
        self._correlation = self.feature_data[pd.Series(self.feature_names)].corr(
            method
        )
        return self._correlation

    def top_correlated_pairs(
        self,
        threshold: float = 0.7,
        absolute: bool = True,
    ) -> pd.DataFrame:
        """Return feature pairs with correlation above the given threshold."""

        if self._correlation is None:
            raise RuntimeError(
                "Call compute_correlation() before requesting top pairs."
            )

        corr_matrix = self._correlation.abs() if absolute else self._correlation.copy()
        # Use an upper-triangular mask to avoid duplicate pairs and self-correlations.
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        filtered = corr_matrix.where(mask)
        pairs = filtered.stack().reset_index()
        pairs.columns = ["feature_1", "feature_2", "correlation"]

        if threshold is not None:
            pairs = pairs[pairs["correlation"] >= threshold]

        return pairs.sort_values("correlation", ascending=False).reset_index(drop=True)
