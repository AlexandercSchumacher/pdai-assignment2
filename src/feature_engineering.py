from __future__ import annotations

import json
from typing import Iterable

import numpy as np
import pandas as pd

BASE_NUMERIC_FEATURES = [
    "sleep_score",
    "readiness_score",
    "activity_score",
    "total_sleep_duration_min",
    "bedtime_start_hour",
    "resting_heart_rate",
    "hrv_ms",
    "stress_high_min",
    "recovery_min",
    "steps",
]

TAG_FEATURES = [
    "tag_alcohol",
    "tag_late_meal",
    "tag_late_caffeine",
    "tag_hard_training",
]

MODEL_FEATURE_COLUMNS = [
    "sleep_score",
    "readiness_score",
    "activity_score",
    "total_sleep_duration_min",
    "bedtime_start_hour",
    "bedtime_sin",
    "bedtime_cos",
    "resting_heart_rate",
    "hrv_ms",
    "stress_high_min",
    "recovery_min",
    "steps",
    "stress_component",
    "energy_score",
    "tag_alcohol",
    "tag_late_meal",
    "tag_late_caffeine",
    "tag_hard_training",
]



def _safe_json_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip().lower() for x in value if str(x).strip()]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            loaded = json.loads(text)
            if isinstance(loaded, list):
                return [str(x).strip().lower() for x in loaded if str(x).strip()]
            if isinstance(loaded, dict):
                return [str(k).strip().lower() for k in loaded.keys() if str(k).strip()]
        except json.JSONDecodeError:
            pass

        if "," in text:
            return [part.strip().lower() for part in text.split(",") if part.strip()]

        return [text.lower()]

    return []



def clamp_frame(df: pd.DataFrame) -> pd.DataFrame:
    bounded = df.copy()
    bounded["sleep_score"] = bounded["sleep_score"].clip(0, 100)
    bounded["readiness_score"] = bounded["readiness_score"].clip(0, 100)
    bounded["activity_score"] = bounded["activity_score"].clip(0, 100)
    bounded["total_sleep_duration_min"] = bounded["total_sleep_duration_min"].clip(120, 720)
    bounded["bedtime_start_hour"] = bounded["bedtime_start_hour"].clip(18, 30)
    bounded["resting_heart_rate"] = bounded["resting_heart_rate"].clip(35, 120)
    bounded["hrv_ms"] = bounded["hrv_ms"].clip(5, 250)
    bounded["stress_high_min"] = bounded["stress_high_min"].clip(0, 360)
    bounded["recovery_min"] = bounded["recovery_min"].clip(0, 600)
    bounded["steps"] = bounded["steps"].clip(0, 50000)
    return bounded



def compute_stress_component(stress_high_min: pd.Series, recovery_min: pd.Series) -> pd.Series:
    denominator = np.maximum(stress_high_min + recovery_min, 1.0)
    stress_component = 100.0 * stress_high_min / denominator
    return stress_component.clip(0, 100)



def compute_energy_score(df: pd.DataFrame) -> pd.Series:
    stress_component = compute_stress_component(df["stress_high_min"], df["recovery_min"])
    stress_relief = (100.0 - stress_component).clip(0, 100)
    energy_score = 0.5 * df["readiness_score"] + 0.3 * df["sleep_score"] + 0.2 * stress_relief
    return energy_score.clip(0, 100)



def add_tag_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    tags = enriched["tags_json"].apply(_safe_json_list)

    def has_any(tag_list: Iterable[str], candidates: set[str]) -> int:
        return int(any(tag in candidates for tag in tag_list))

    enriched["tag_alcohol"] = tags.apply(lambda x: has_any(x, {"alcohol", "drinks", "wine", "beer"}))
    enriched["tag_late_meal"] = tags.apply(
        lambda x: has_any(x, {"late_meal", "late meal", "late_eating", "heavy_dinner"})
    )
    enriched["tag_late_caffeine"] = tags.apply(
        lambda x: has_any(x, {"late_caffeine", "late caffeine", "coffee_late", "caffeine_late"})
    )
    enriched["tag_hard_training"] = tags.apply(
        lambda x: has_any(x, {"hard_training", "hard workout", "intense_training", "heavy_training"})
    )

    return enriched



def add_energy_and_model_features(df: pd.DataFrame) -> pd.DataFrame:
    prepared = clamp_frame(df.copy())
    prepared = add_tag_features(prepared)

    prepared["stress_component"] = compute_stress_component(
        prepared["stress_high_min"], prepared["recovery_min"]
    )
    prepared["energy_score"] = compute_energy_score(prepared)

    bedtime_rad = (prepared["bedtime_start_hour"] % 24) / 24.0 * 2.0 * np.pi
    prepared["bedtime_sin"] = np.sin(bedtime_rad)
    prepared["bedtime_cos"] = np.cos(bedtime_rad)

    return prepared



def prepare_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    prepared = add_energy_and_model_features(df)
    prepared = prepared.sort_values("date").reset_index(drop=True)
    prepared["target_energy_t_plus_1"] = prepared["energy_score"].shift(-1)
    prepared = prepared.dropna(subset=["target_energy_t_plus_1"]).reset_index(drop=True)
    return prepared, MODEL_FEATURE_COLUMNS



def get_latest_state(df: pd.DataFrame, reference_date: pd.Timestamp | None = None) -> pd.Series:
    prepared = add_energy_and_model_features(df)
    prepared = prepared.sort_values("date")

    if reference_date is None:
        return prepared.iloc[-1].copy()

    eligible = prepared[prepared["date"] <= reference_date]
    if eligible.empty:
        return prepared.iloc[0].copy()
    return eligible.iloc[-1].copy()
