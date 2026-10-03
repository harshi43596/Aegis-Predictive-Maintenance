"""Model training and prediction helpers for predictive maintenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

SENSOR_FEATURES = ["temperature", "vibration", "pressure", "current"]
FEATURE_LABELS = {
    "temperature": "Temperature (°C)",
    "vibration": "Vibration (mm/s)",
    "pressure": "Pressure (kPa)",
    "current": "Current (A)",
}
ALIASES = {
    "temperature": {"temperature", "temperature_c", "temp", "temp_c"},
    "vibration": {"vibration", "vibration_mm_s", "vibration_mms", "vib"},
    "pressure": {"pressure", "pressure_kpa", "pressure_bar"},
    "current": {"current", "current_a", "amps", "amperage"},
}


@dataclass
class ModelArtifact:
    estimator: RandomForestClassifier
    metrics: dict[str, float]
    source: str
    target_name: str
    rows_used: int
    feature_importance: dict[str, float]


def normalize_sensor_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename common sensor header variants to the app's canonical names."""
    renamed = frame.copy()
    lookup = {str(column).strip().lower().replace(" ", "_"): column for column in renamed.columns}
    for canonical, aliases in ALIASES.items():
        alias_name = next((name for name in aliases if name in lookup), None)
        match = lookup[alias_name] if alias_name is not None else None
        if match is not None and canonical not in renamed.columns:
            renamed = renamed.rename(columns={match: canonical})
            if canonical == "pressure" and alias_name == "pressure_bar":
                renamed[canonical] = pd.to_numeric(renamed[canonical], errors="coerce") * 100
    return renamed


def generate_labeled_data(rows: int = 2400, seed: int = 42) -> pd.DataFrame:
    """Create reproducible, explicitly synthetic telemetry for the demo model."""
    rng = np.random.default_rng(seed)
    wear = rng.beta(1.7, 2.0, rows)
    temperature = 61 + 18 * wear + rng.normal(0, 5.0, rows)
    vibration = 1.25 + 5.0 * wear + rng.normal(0, 0.72, rows)
    pressure = 116 - 25 * wear + rng.normal(0, 5.0, rows)
    current = 11.5 + 11.0 * wear + rng.normal(0, 2.3, rows)

    # Rare independent excursions keep the synthetic training set from relying
    # on one perfectly correlated wear pattern.
    temperature += rng.binomial(1, 0.045, rows) * rng.uniform(5, 14, rows)
    vibration += rng.binomial(1, 0.055, rows) * rng.uniform(1.0, 3.2, rows)
    pressure -= rng.binomial(1, 0.04, rows) * rng.uniform(7, 18, rows)
    current += rng.binomial(1, 0.04, rows) * rng.uniform(3, 8, rows)

    temp_score = np.clip((temperature - 70) / 18, 0, 1.5)
    vibration_score = np.clip((vibration - 2.0) / 4.7, 0, 1.5)
    pressure_score = np.clip((110 - pressure) / 26, 0, 1.5)
    current_score = np.clip((current - 14) / 14, 0, 1.5)
    stress = 0.27 * temp_score + 0.36 * vibration_score + 0.19 * pressure_score + 0.18 * current_score
    probability = 1 / (1 + np.exp(-9 * (stress - 0.43)))
    failure = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "temperature": temperature.round(2),
            "vibration": vibration.round(2),
            "pressure": pressure.round(2),
            "current": current.round(2),
            "failure_within_24h": failure,
        }
    )


def _binary_target(values: pd.Series) -> pd.Series:
    normalized = values.dropna()
    if normalized.nunique() != 2:
        raise ValueError("The selected target must contain exactly two classes, such as 0/1 or healthy/failure.")

    numeric = pd.to_numeric(normalized, errors="coerce")
    if numeric.notna().all():
        classes = sorted(numeric.unique())
        positive = 1 if set(classes) == {0, 1} else classes[-1]
        return pd.to_numeric(values, errors="coerce").eq(positive).astype("int8")

    text = normalized.astype(str).str.strip().str.lower()
    positives = {"1", "yes", "true", "failure", "failed", "fault", "faulty", "positive", "alarm", "anomaly"}
    negatives = {"0", "no", "false", "healthy", "normal", "ok", "negative", "no_failure", "none"}
    present_positive = set(text.unique()) & positives
    present_negative = set(text.unique()) & negatives
    if present_positive and present_negative:
        return values.astype(str).str.strip().str.lower().isin(present_positive).astype("int8")
    raise ValueError("Could not identify the failure class. Use 0/1, yes/no, healthy/failure, or fault/normal labels.")


def fit_failure_model(frame: pd.DataFrame, target_column: str, source: str = "Uploaded labeled CSV") -> ModelArtifact:
    """Validate, hold out data for evaluation, then fit a random forest."""
    data = normalize_sensor_columns(frame)
    missing = [feature for feature in SENSOR_FEATURES if feature not in data.columns]
    if missing:
        raise ValueError("Missing sensor columns: " + ", ".join(missing))
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    working = data[SENSOR_FEATURES + [target_column]].copy()
    for feature in SENSOR_FEATURES:
        working[feature] = pd.to_numeric(working[feature], errors="coerce")
    working[target_column] = working[target_column].replace(r"^\s*$", np.nan, regex=True)
    working = working.replace([np.inf, -np.inf], np.nan).dropna()
    if len(working) < 40:
        raise ValueError("At least 40 complete labeled rows are needed to train a useful demonstration model.")

    labels = _binary_target(working[target_column])
    class_counts = labels.value_counts()
    if class_counts.min() < 5:
        raise ValueError("The minority class needs at least 5 examples for a meaningful holdout evaluation.")

    x = working[SENSOR_FEATURES].astype(float)
    x_train, x_valid, y_train, y_valid = train_test_split(x, labels, test_size=0.2, random_state=42, stratify=labels)
    options: dict[str, Any] = {
        "n_estimators": 180,
        "max_depth": 10,
        "min_samples_leaf": 3,
        "class_weight": "balanced_subsample",
        "random_state": 42,
        "n_jobs": -1,
    }
    evaluator = RandomForestClassifier(**options)
    evaluator.fit(x_train, y_train)
    predicted = evaluator.predict(x_valid)
    metrics = {
        "accuracy": float(accuracy_score(y_valid, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_valid, predicted)),
        "precision": float(precision_score(y_valid, predicted, zero_division=0)),
        "recall": float(recall_score(y_valid, predicted, zero_division=0)),
        "f1": float(f1_score(y_valid, predicted, zero_division=0)),
        "validation_rows": float(len(y_valid)),
    }

    estimator = RandomForestClassifier(**options)
    estimator.fit(x, labels)
    importance = dict(zip(SENSOR_FEATURES, estimator.feature_importances_.astype(float)))
    return ModelArtifact(estimator, metrics, source, target_column, len(working), importance)


def predict_failure_probability(model: ModelArtifact, frame: pd.DataFrame) -> np.ndarray:
    data = normalize_sensor_columns(frame)
    missing = [feature for feature in SENSOR_FEATURES if feature not in data.columns]
    if missing:
        raise ValueError("Missing sensor columns: " + ", ".join(missing))
    features = data[SENSOR_FEATURES].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any() or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Sensor readings must be valid finite numbers before scoring.")
    positive_index = list(model.estimator.classes_).index(1)
    return model.estimator.predict_proba(features.astype(float))[:, positive_index]
