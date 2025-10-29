from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


@dataclass
class ModelPerformanceReport:
    """Container with convenience helpers for inspecting model performance."""

    accuracy: float
    confusion_matrix: pd.DataFrame
    classification_report: str
    feature_weights: pd.DataFrame
    coefficients: pd.DataFrame

    def top_features(self, n: int = 20) -> pd.DataFrame:
        return self.feature_weights.head(n)

    def to_text(self, *, top_k: int = 10) -> str:
        parts = [f"Accuracy: {self.accuracy:.4f}"]

        if not self.coefficients.empty:
            parts.extend(
                [
                    "Coefficients:",
                    _format_coefficients_table(self.coefficients),
                    _significance_legend(),
                ]
            )

        parts.extend(
            [
                "Confusion Matrix:",
                self.confusion_matrix.to_string(),
                "Classification Report:",
                self.classification_report,
            ]
        )
        return "\n\n".join(parts)

    def print(self, *, top_k: int = 10) -> None:
        print(self.to_text(top_k=top_k))


def summarize_model_performance(
    model: ClassifierMixin,
    X: pd.DataFrame,
    y: Sequence,
    feature_names: Optional[Sequence[str]] = None,
    *,
    label_names: Optional[Iterable] = None,
) -> ModelPerformanceReport:
    """Compute standard performance diagnostics for a fitted classifier."""

    if feature_names is None:
        feature_names = list(X.columns)

    predictions = model.predict(X)
    accuracy = float(accuracy_score(y, predictions))

    labels = list(label_names) if label_names is not None else sorted(set(y))

    cm = confusion_matrix(y, predictions, labels=labels)
    cm_index = [f"true_{label}" for label in labels]
    cm_columns = [f"pred_{label}" for label in labels]
    cm_df = pd.DataFrame(cm, index=cm_index, columns=cm_columns)

    class_report = classification_report(y, predictions, labels=labels)
    feature_weights = _extract_feature_weights(model, feature_names)
    coefficients = _summarize_coefficients(model, X, y, feature_names)

    return ModelPerformanceReport(
        accuracy=accuracy,
        confusion_matrix=cm_df,
        classification_report=class_report,
        feature_weights=feature_weights,
        coefficients=coefficients,
    )


def _extract_feature_weights(
    model: ClassifierMixin, feature_names: Sequence[str]
) -> pd.DataFrame:
    estimator, transformer = _unwrap_linear_components(model)

    if estimator is None:
        return pd.DataFrame({"feature": feature_names})

    coef = getattr(estimator, "coef_", None)
    if coef is None:
        return pd.DataFrame({"feature": feature_names})

    coef_array = np.asarray(coef)
    if coef_array.ndim == 1:
        coef_array = coef_array.reshape(1, -1)

    resolved_feature_names = _resolve_feature_names(
        transformer, feature_names, coef_array.shape[1]
    )

    abs_importance = np.mean(np.abs(coef_array), axis=0)
    data = {
        "feature": resolved_feature_names,
        "mean_coefficient": coef_array.mean(axis=0),
        "abs_importance": abs_importance,
    }

    for class_idx, class_coef in enumerate(coef_array):
        data[f"coef_class_{class_idx}"] = class_coef

    weights = pd.DataFrame(data)
    return weights.sort_values("abs_importance", ascending=False).reset_index(drop=True)


def _summarize_coefficients(
    model: ClassifierMixin,
    X: pd.DataFrame,
    y: Sequence,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    estimator, transformer = _unwrap_linear_components(model)
    if estimator is None:
        return pd.DataFrame()

    if not hasattr(estimator, "coef_") or not hasattr(estimator, "intercept_"):
        return pd.DataFrame()

    coef_attr = getattr(estimator, "coef_", None)
    if coef_attr is None:
        return pd.DataFrame()
    coef_array = np.asarray(coef_attr)
    if coef_array.ndim != 2 or coef_array.shape[0] != 1:
        # Currently only support binary logistic regression output
        return pd.DataFrame()

    intercept_attr = getattr(estimator, "intercept_", None)
    if intercept_attr is None:
        return pd.DataFrame()
    intercept = np.asarray(intercept_attr).ravel()
    if intercept.size != 1:
        return pd.DataFrame()

    X_matrix = _transform_design_matrix(X, transformer)
    if X_matrix is None:
        return pd.DataFrame()

    if X_matrix.shape[1] != coef_array.shape[1]:
        return pd.DataFrame()

    resolved_feature_names = _resolve_feature_names(
        transformer, feature_names, coef_array.shape[1]
    )

    n_samples = X_matrix.shape[0]
    if n_samples == 0:
        return pd.DataFrame()

    # Build design matrix with intercept first
    design_matrix = np.hstack([np.ones((n_samples, 1)), X_matrix])
    linear_response = design_matrix @ np.concatenate([intercept, coef_array.ravel()])
    probabilities = 1.0 / (1.0 + np.exp(-linear_response))
    probabilities = np.clip(probabilities, 1e-9, 1 - 1e-9)
    weights = probabilities * (1 - probabilities)

    weighted_design = design_matrix * weights[:, None]
    xtwx = design_matrix.T @ weighted_design

    try:
        covariance = np.linalg.inv(xtwx)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(xtwx)

    standard_errors = np.sqrt(np.diag(covariance))
    coefficients = np.concatenate([intercept, coef_array.ravel()])
    z_values = coefficients / np.where(standard_errors == 0, np.nan, standard_errors)

    p_values = 2 * (1 - _normal_cdf(np.abs(z_values)))
    terms = ["(Intercept)"] + list(resolved_feature_names)
    significance = [_significance_code(p) for p in p_values]

    summary_df = pd.DataFrame(
        {
            "term": terms,
            "estimate": coefficients,
            "std_error": standard_errors,
            "z_value": z_values,
            "p_value": p_values,
            "significance": significance,
        }
    )
    return summary_df


def _unwrap_linear_components(
    model: ClassifierMixin,
) -> tuple[Optional[ClassifierMixin], Optional[Pipeline]]:
    if hasattr(model, "best_estimator_"):
        best_estimator = getattr(model, "best_estimator_")
        if best_estimator is None:
            return None, None
        return _unwrap_linear_components(best_estimator)

    if isinstance(model, Pipeline):
        if not model.steps:
            return None, None
        final_estimator = model.steps[-1][1]
        feature_pipeline = model[:-1] if len(model.steps) > 1 else None
        return final_estimator, feature_pipeline

    if hasattr(model, "coef_"):
        return model, None

    return None, None


def _transform_design_matrix(
    X: pd.DataFrame, transformer: Optional[Pipeline]
) -> Optional[np.ndarray]:
    if transformer is None:
        return np.asarray(X, dtype=float)

    try:
        transformed = transformer.transform(X)
    except Exception:
        return None

    return np.asarray(transformed, dtype=float)


def _resolve_feature_names(
    transformer: Optional[Pipeline],
    feature_names: Sequence[str],
    expected_length: int,
) -> list[str]:
    candidates: list[str] | None = None

    if transformer is not None and hasattr(transformer, "get_feature_names_out"):
        try:
            candidates = list(transformer.get_feature_names_out(feature_names))
        except TypeError:
            try:
                candidates = list(transformer.get_feature_names_out())
            except Exception:
                candidates = None
        except Exception:
            candidates = None

    if candidates is None and isinstance(transformer, Pipeline) and transformer.steps:
        last_step = transformer.steps[-1][1]
        if hasattr(last_step, "get_feature_names_out"):
            try:
                candidates = list(last_step.get_feature_names_out(feature_names))
            except TypeError:
                try:
                    candidates = list(last_step.get_feature_names_out())
                except Exception:
                    candidates = None
            except Exception:
                candidates = None

    if candidates is not None and len(candidates) == expected_length:
        return candidates

    if len(feature_names) == expected_length:
        return list(feature_names)

    return [f"feature_{idx}" for idx in range(expected_length)]


def _normal_cdf(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    erf_vec = np.vectorize(erf)
    return 0.5 * (1 + erf_vec(value / sqrt(2)))


def _significance_code(p_value: float) -> str:
    if np.isnan(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    if p_value < 0.1:
        return "."
    return ""


def _format_coefficients_table(coefficients: pd.DataFrame) -> str:
    if coefficients.empty:
        return "(coefficients unavailable)"

    df = coefficients.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    table = pd.DataFrame(
        {
            "Estimate": df["estimate"].map(lambda v: _format_float(v, 4)).values,
            "Std. Error": df["std_error"].map(lambda v: _format_float(v, 4)).values,
            "z value": df["z_value"].map(lambda v: _format_float(v, 3)).values,
            "Pr(>|z|)": df["p_value"].map(_format_p_value).values,
            "": df["significance"].values,
        }
    )
    table.index = pd.Index(df["term"].values)

    return table.to_string()


def _format_float(value: float, digits: int) -> str:
    if value is None or np.isnan(value):
        return "NaN"
    return f"{value:.{digits}f}"


def _format_p_value(value: float) -> str:
    if value is None or np.isnan(value):
        return "NaN"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.5f}"


def _significance_legend() -> str:
    return "---\nSignif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
