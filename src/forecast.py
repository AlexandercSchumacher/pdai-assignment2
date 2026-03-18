from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.feature_engineering import MODEL_FEATURE_COLUMNS, TAG_FEATURES, add_energy_and_model_features



def load_model_bundle(path: str | Path = "models/energy_model.pkl") -> dict[str, Any] | None:
    model_path = Path(path)
    if not model_path.exists():
        return None

    with model_path.open("rb") as f:
        return pickle.load(f)



def _risk_from_p10(p10: float) -> str:
    if p10 < 55:
        return "High"
    if p10 < 70:
        return "Medium"
    return "Low"


def _scenario_prediction_shift(scenario: dict[str, Any], day_index: int) -> float:
    if not scenario:
        return 0.0

    decay = [1.0, 0.65, 0.4][min(day_index - 1, 2)]
    bedtime_shift = float(scenario.get("bedtime_shift_hours", 0.0))
    sleep_delta = float(scenario.get("sleep_delta_min", 0.0))
    training_delta = float(scenario.get("training_load_delta", 0.0))
    caffeine_cutoff = float(scenario.get("caffeine_cutoff_hour", 15.0))
    alcohol = bool(scenario.get("alcohol", False))
    late_meal = bool(scenario.get("late_meal", False))

    shift = (
        sleep_delta * 0.03
        - max(bedtime_shift, 0.0) * 1.4
        + max(-bedtime_shift, 0.0) * 1.0
        - max(training_delta, 0.0) * 0.26
        + max(-training_delta, 0.0) * 0.16
        - max(caffeine_cutoff - 15.0, 0.0) * 0.8
        + max(15.0 - caffeine_cutoff, 0.0) * 0.45
        - (6.0 if alcohol else 0.0)
        - (3.5 if late_meal else 0.0)
    )
    return float(shift * decay)



def _serialize_tags_from_state(state: pd.Series) -> str:
    tags = []
    if int(state.get("tag_alcohol", 0)) == 1:
        tags.append("alcohol")
    if int(state.get("tag_late_meal", 0)) == 1:
        tags.append("late_meal")
    if int(state.get("tag_late_caffeine", 0)) == 1:
        tags.append("late_caffeine")
    if int(state.get("tag_hard_training", 0)) == 1:
        tags.append("hard_training")
    return json.dumps(tags)



def _bounded_state(state: pd.Series) -> pd.Series:
    bounded = state.copy()
    bounded["sleep_score"] = np.clip(float(bounded.get("sleep_score", 70.0)), 0, 100)
    bounded["readiness_score"] = np.clip(float(bounded.get("readiness_score", 70.0)), 0, 100)
    bounded["activity_score"] = np.clip(float(bounded.get("activity_score", 70.0)), 0, 100)
    bounded["total_sleep_duration_min"] = np.clip(float(bounded.get("total_sleep_duration_min", 420.0)), 120, 720)
    bounded["bedtime_start_hour"] = np.clip(float(bounded.get("bedtime_start_hour", 23.0)), 18, 30)
    bounded["resting_heart_rate"] = np.clip(float(bounded.get("resting_heart_rate", 55.0)), 35, 120)
    bounded["hrv_ms"] = np.clip(float(bounded.get("hrv_ms", 40.0)), 5, 250)
    bounded["stress_high_min"] = np.clip(float(bounded.get("stress_high_min", 75.0)), 0, 360)
    bounded["recovery_min"] = np.clip(float(bounded.get("recovery_min", 95.0)), 0, 600)
    bounded["steps"] = np.clip(float(bounded.get("steps", 8000.0)), 0, 50000)
    return bounded



def _apply_scenario_effects(state: pd.Series, scenario: dict[str, Any], day_index: int) -> pd.Series:
    if not scenario:
        return state

    decay = [1.0, 0.65, 0.4][min(day_index - 1, 2)]
    adjusted = state.copy()

    bedtime_shift = float(scenario.get("bedtime_shift_hours", 0.0)) * decay
    sleep_delta = float(scenario.get("sleep_delta_min", 0.0)) * decay
    training_delta = float(scenario.get("training_load_delta", 0.0)) * decay
    caffeine_cutoff = float(scenario.get("caffeine_cutoff_hour", 15.0))
    alcohol = bool(scenario.get("alcohol", False))
    late_meal = bool(scenario.get("late_meal", False))

    late_caffeine_penalty = max(caffeine_cutoff - 15.0, 0.0)
    early_cutoff_bonus = max(15.0 - caffeine_cutoff, 0.0)
    hard_load = max(training_delta, 0.0)
    recovery_load = max(-training_delta, 0.0)

    adjusted["bedtime_start_hour"] += bedtime_shift
    adjusted["total_sleep_duration_min"] += sleep_delta
    adjusted["activity_score"] += hard_load * 0.2 - recovery_load * 0.05
    adjusted["steps"] += hard_load * 55.0 - recovery_load * 25.0

    adjusted["stress_high_min"] += (
        hard_load * 2.6
        - recovery_load * 1.4
        + late_caffeine_penalty * 2.2
        + (15.0 if alcohol else 0.0) * decay
        + (10.0 if late_meal else 0.0) * decay
        - early_cutoff_bonus * 0.8
    )
    adjusted["recovery_min"] += (
        max(sleep_delta, 0.0) * 0.34
        - hard_load * 0.9
        + recovery_load * 0.8
        - (12.0 if alcohol else 0.0) * decay
        - (8.0 if late_meal else 0.0) * decay
        + early_cutoff_bonus * 1.2
    )

    adjusted["sleep_score"] += (
        sleep_delta * 0.06
        - hard_load * 0.3
        + recovery_load * 0.2
        - max(late_caffeine_penalty, 0.0) * 0.9
        - (6.0 if alcohol else 0.0) * decay
        - (4.0 if late_meal else 0.0) * decay
        - max(bedtime_shift, 0.0) * 1.7
        + max(-bedtime_shift, 0.0) * 1.3
    )
    adjusted["readiness_score"] += (
        sleep_delta * 0.05
        - hard_load * 0.75
        + recovery_load * 0.35
        - late_caffeine_penalty * 0.8
        - (8.0 if alcohol else 0.0) * decay
        - (5.0 if late_meal else 0.0) * decay
        + early_cutoff_bonus * 0.5
    )

    adjusted["resting_heart_rate"] += (
        hard_load * 0.15
        - recovery_load * 0.1
        + late_caffeine_penalty * 0.15
        + (1.8 if alcohol else 0.0) * decay
        - early_cutoff_bonus * 0.06
    )
    adjusted["hrv_ms"] += (
        max(sleep_delta, 0.0) * 0.05
        - hard_load * 0.24
        + recovery_load * 0.14
        - late_caffeine_penalty * 0.35
        - (3.5 if alcohol else 0.0) * decay
        - (2.2 if late_meal else 0.0) * decay
        + early_cutoff_bonus * 0.25
    )

    direct_energy_delta = (
        sleep_delta * 0.045
        - hard_load * 0.55
        + recovery_load * 0.25
        - max(bedtime_shift, 0.0) * 2.4
        + max(-bedtime_shift, 0.0) * 1.6
        - late_caffeine_penalty * 1.3
        + early_cutoff_bonus * 0.5
        - (7.0 if alcohol else 0.0) * decay
        - (4.0 if late_meal else 0.0) * decay
    )
    adjusted["readiness_score"] += direct_energy_delta * 0.9
    adjusted["sleep_score"] += direct_energy_delta * 0.7
    adjusted["stress_high_min"] += np.clip(-direct_energy_delta * 1.8, -25, 35)
    adjusted["recovery_min"] += np.clip(direct_energy_delta * 1.5, -30, 30)

    adjusted["sleep_score"] = np.clip(
        adjusted["sleep_score"],
        float(state["sleep_score"]) - 18.0,
        float(state["sleep_score"]) + 18.0,
    )
    adjusted["readiness_score"] = np.clip(
        adjusted["readiness_score"],
        float(state["readiness_score"]) - 20.0,
        float(state["readiness_score"]) + 20.0,
    )
    adjusted["stress_high_min"] = np.clip(
        adjusted["stress_high_min"],
        float(state["stress_high_min"]) - 70.0,
        float(state["stress_high_min"]) + 90.0,
    )
    adjusted["recovery_min"] = np.clip(
        adjusted["recovery_min"],
        float(state["recovery_min"]) - 70.0,
        float(state["recovery_min"]) + 80.0,
    )
    adjusted["resting_heart_rate"] = np.clip(
        adjusted["resting_heart_rate"],
        float(state["resting_heart_rate"]) - 6.0,
        float(state["resting_heart_rate"]) + 8.0,
    )
    adjusted["hrv_ms"] = np.clip(
        adjusted["hrv_ms"],
        float(state["hrv_ms"]) - 14.0,
        float(state["hrv_ms"]) + 14.0,
    )
    adjusted["total_sleep_duration_min"] = np.clip(
        adjusted["total_sleep_duration_min"],
        float(state["total_sleep_duration_min"]) - 120.0,
        float(state["total_sleep_duration_min"]) + 120.0,
    )
    adjusted["bedtime_start_hour"] = np.clip(
        adjusted["bedtime_start_hour"],
        float(state["bedtime_start_hour"]) - 2.5,
        float(state["bedtime_start_hour"]) + 2.5,
    )
    adjusted["steps"] = np.clip(
        adjusted["steps"],
        float(state["steps"]) - 3500.0,
        float(state["steps"]) + 4500.0,
    )

    adjusted["tag_alcohol"] = int(alcohol)
    adjusted["tag_late_meal"] = int(late_meal)
    adjusted["tag_late_caffeine"] = int(caffeine_cutoff >= 17)
    adjusted["tag_hard_training"] = int(training_delta > 12)

    adjusted = _bounded_state(adjusted)
    adjusted["tags_json"] = _serialize_tags_from_state(adjusted)

    enriched = add_energy_and_model_features(pd.DataFrame([adjusted])).iloc[0]
    return enriched



def _roll_state_forward(state: pd.Series, predicted_energy: float) -> pd.Series:
    next_state = state.copy()
    current_date = pd.to_datetime(next_state.get("date"), errors="coerce")
    if pd.isna(current_date):
        current_date = pd.Timestamp.today().normalize()
    next_state["date"] = current_date + pd.Timedelta(days=1)

    next_state["energy_score"] = np.clip(predicted_energy, 0, 100)
    next_state["readiness_score"] = np.clip(0.66 * state["readiness_score"] + 0.34 * predicted_energy, 0, 100)
    next_state["sleep_score"] = np.clip(0.72 * state["sleep_score"] + 0.28 * (predicted_energy + 4.0), 0, 100)
    next_state["activity_score"] = np.clip(0.82 * state["activity_score"] + 0.18 * predicted_energy, 0, 100)
    next_state["stress_high_min"] = np.clip(0.72 * state["stress_high_min"] + 0.28 * (100.0 - predicted_energy), 0, 360)
    next_state["recovery_min"] = np.clip(0.75 * state["recovery_min"] + 0.25 * (predicted_energy * 1.2), 0, 600)
    next_state["resting_heart_rate"] = np.clip(
        0.8 * state["resting_heart_rate"] + 0.2 * (65.0 - predicted_energy * 0.12),
        35,
        120,
    )
    next_state["hrv_ms"] = np.clip(0.78 * state["hrv_ms"] + 0.22 * (predicted_energy * 0.6), 5, 250)
    next_state["total_sleep_duration_min"] = np.clip(
        0.85 * state["total_sleep_duration_min"] + 0.15 * (360.0 + predicted_energy * 1.5),
        120,
        720,
    )
    next_state["bedtime_start_hour"] = np.clip(
        0.9 * state["bedtime_start_hour"] + 0.1 * (23.2 - (predicted_energy - 70.0) * 0.01),
        18,
        30,
    )
    next_state["steps"] = np.clip(0.78 * state["steps"] + 0.22 * (3000.0 + predicted_energy * 80.0), 0, 50000)

    for tag_col in TAG_FEATURES:
        next_state[tag_col] = int(next_state.get(tag_col, 0))

    next_state["tags_json"] = _serialize_tags_from_state(next_state)

    enriched = add_energy_and_model_features(pd.DataFrame([next_state])).iloc[0]
    return enriched



def simulate_forecast(
    model_bundle: dict[str, Any],
    base_state: pd.Series,
    scenario: dict[str, Any] | None = None,
    horizon_days: int = 3,
    n_samples: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    model = model_bundle["model"]
    feature_columns = model_bundle.get("feature_columns", MODEL_FEATURE_COLUMNS)
    residuals = np.array(model_bundle.get("residuals", []), dtype=float)
    rng = np.random.default_rng(random_state)

    if residuals.size < 20:
        residuals = rng.normal(0, 4.5, size=300)

    state = add_energy_and_model_features(pd.DataFrame([base_state])).iloc[0]
    state = _bounded_state(state)

    current_date = pd.to_datetime(state.get("date"), errors="coerce")
    if pd.isna(current_date):
        current_date = pd.Timestamp.today().normalize()
    state["date"] = current_date

    rows: list[dict[str, Any]] = []

    for day in range(1, horizon_days + 1):
        effective_state = _apply_scenario_effects(state, scenario or {}, day)

        x_df = pd.DataFrame([effective_state[feature_columns]], columns=feature_columns)
        scenario_shift = _scenario_prediction_shift(scenario or {}, day)
        point_forecast = float(np.clip(model.predict(x_df)[0] + scenario_shift, 0, 100))

        sampled_residuals = rng.choice(residuals, size=n_samples, replace=True)
        scaled_samples = np.clip(point_forecast + sampled_residuals * (1.0 + 0.18 * (day - 1)), 0, 100)

        p10, p50, p90 = np.quantile(scaled_samples, [0.1, 0.5, 0.9])

        rows.append(
            {
                "day": day,
                "date": pd.to_datetime(state["date"]) + pd.Timedelta(days=1),
                "p10": float(p10),
                "median": float(p50),
                "p90": float(p90),
                "risk": _risk_from_p10(float(p10)),
            }
        )

        state = _roll_state_forward(effective_state, float(p50))

    return pd.DataFrame(rows)



def compare_forecasts(baseline: pd.DataFrame, scenario: pd.DataFrame) -> pd.DataFrame:
    merged = baseline.merge(
        scenario,
        on=["day", "date"],
        suffixes=("_baseline", "_scenario"),
        how="inner",
    )
    merged["delta_median"] = merged["median_scenario"] - merged["median_baseline"]
    merged["delta_p10"] = merged["p10_scenario"] - merged["p10_baseline"]
    merged["delta_p90"] = merged["p90_scenario"] - merged["p90_baseline"]
    return merged
