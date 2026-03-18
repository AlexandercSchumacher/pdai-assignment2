from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "date",
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
    "tags_json",
]

DEFAULT_VALUES = {
    "sleep_score": 72.0,
    "readiness_score": 70.0,
    "activity_score": 68.0,
    "total_sleep_duration_min": 420.0,
    "bedtime_start_hour": 23.0,
    "resting_heart_rate": 55.0,
    "hrv_ms": 42.0,
    "stress_high_min": 70.0,
    "recovery_min": 95.0,
    "steps": 8000.0,
    "tags_json": "[]",
}

COLUMN_ALIASES = {
    "date": ["date", "day", "summary_date", "calendar_date"],
    "sleep_score": ["sleep_score", "score_sleep", "sleep", "sleepscore"],
    "readiness_score": ["readiness_score", "score_readiness", "readiness", "readinessscore"],
    "activity_score": ["activity_score", "score_activity", "activity", "activityscore"],
    "total_sleep_duration_min": [
        "total_sleep_duration_min",
        "total_sleep_duration",
        "sleep_duration_min",
        "sleep_duration",
        "total_sleep_time",
    ],
    "bedtime_start_hour": [
        "bedtime_start_hour",
        "bedtime_hour",
        "bedtime_start",
        "sleep_start_hour",
    ],
    "resting_heart_rate": ["resting_heart_rate", "resting_hr", "rhr", "lowest_heart_rate"],
    "hrv_ms": ["hrv_ms", "hrv", "avg_hrv", "rmssd"],
    "stress_high_min": ["stress_high_min", "high_stress_min", "stress_minutes", "stress_high"],
    "recovery_min": ["recovery_min", "restorative_minutes", "recovery_minutes", "restoration_min"],
    "steps": ["steps", "step_count", "daily_steps"],
    "tags_json": ["tags_json", "tags", "labels", "events"],
}

NUMERIC_COLUMNS = [
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



def _normalize_name(value: str) -> str:
    return "".join(ch for ch in value.strip().lower() if ch.isalnum() or ch == "_")



def _coerce_tags_json(value: Any) -> str:
    if isinstance(value, list):
        return json.dumps([str(x).strip() for x in value if str(x).strip()])

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "[]"
        try:
            loaded = json.loads(text)
            if isinstance(loaded, list):
                return json.dumps([str(x).strip() for x in loaded if str(x).strip()])
            if isinstance(loaded, dict):
                return json.dumps([str(k).strip() for k in loaded.keys() if str(k).strip()])
        except json.JSONDecodeError:
            pass

        if "," in text:
            items = [part.strip() for part in text.split(",") if part.strip()]
            return json.dumps(items)

        return json.dumps([text])

    return "[]"


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "ja"}
    return False



def create_demo_dataset(path: Path, n_days: int = 120, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    end_date = pd.Timestamp.today().normalize()
    dates = pd.date_range(end=end_date, periods=n_days, freq="D")

    day_idx = np.arange(n_days)
    weekly = np.sin(2 * np.pi * day_idx / 7)

    sleep_duration = np.clip(425 + 22 * weekly + rng.normal(0, 18, n_days), 320, 560)
    bedtime_hour = np.clip(23.3 + 0.7 * np.cos(2 * np.pi * day_idx / 7) + rng.normal(0, 0.35, n_days), 20, 26)
    stress_high = np.clip(78 - (sleep_duration - 420) * 0.11 + rng.normal(0, 12, n_days), 15, 220)
    recovery = np.clip(92 + (sleep_duration - 420) * 0.22 + rng.normal(0, 14, n_days), 30, 240)
    readiness = np.clip(72 + (sleep_duration - 420) * 0.07 - (stress_high - 75) * 0.14 + rng.normal(0, 6, n_days), 35, 99)
    sleep_score = np.clip(74 + (sleep_duration - 420) * 0.06 - (bedtime_hour - 23.5) * 2.3 + rng.normal(0, 5, n_days), 35, 99)
    activity = np.clip(70 + rng.normal(0, 10, n_days), 30, 98)
    rhr = np.clip(56 + (75 - readiness) * 0.13 + rng.normal(0, 2.0, n_days), 42, 78)
    hrv = np.clip(44 + (readiness - 72) * 0.55 + rng.normal(0, 7, n_days), 12, 120)
    steps = np.clip(8300 + (activity - 70) * 130 + rng.normal(0, 1300, n_days), 1000, 25000)

    tag_options = [
        [],
        ["late_meal"],
        ["alcohol"],
        ["hard_training"],
        ["late_caffeine"],
        ["late_meal", "alcohol"],
    ]
    tags = [json.dumps(tag_options[idx]) for idx in rng.integers(0, len(tag_options), size=n_days)]

    df = pd.DataFrame(
        {
            "date": dates,
            "sleep_score": sleep_score,
            "readiness_score": readiness,
            "activity_score": activity,
            "total_sleep_duration_min": sleep_duration,
            "bedtime_start_hour": bedtime_hour,
            "resting_heart_rate": rhr,
            "hrv_ms": hrv,
            "stress_high_min": stress_high,
            "recovery_min": recovery,
            "steps": steps,
            "tags_json": tags,
            "is_demo": True,
            "is_synthetic": False,
            "data_source": "demo_personal",
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df



def load_personal_data(path: str | Path = "data/oura_personal.csv") -> tuple[pd.DataFrame, dict[str, Any]]:
    csv_path = Path(path)
    assumptions: list[str] = []
    created_demo = False

    if not csv_path.exists():
        create_demo_dataset(csv_path)
        assumptions.append(
            "No personal CSV found; a demo dataset was created automatically at data/oura_personal.csv."
        )
        created_demo = True

    raw_df = pd.read_csv(csv_path)
    original_columns = list(raw_df.columns)
    normalized_map = {_normalize_name(col): col for col in original_columns}

    standardized = pd.DataFrame()
    column_mapping: dict[str, str] = {}

    for canonical in REQUIRED_COLUMNS:
        selected_source = None
        alias_candidates = [canonical] + COLUMN_ALIASES.get(canonical, [])
        for alias in alias_candidates:
            norm_alias = _normalize_name(alias)
            if norm_alias in normalized_map:
                selected_source = normalized_map[norm_alias]
                break

        if selected_source is not None:
            standardized[canonical] = raw_df[selected_source]
            column_mapping[canonical] = selected_source
        else:
            default_value = DEFAULT_VALUES.get(canonical, np.nan)
            standardized[canonical] = default_value
            assumptions.append(
                f"Column '{canonical}' was missing in the CSV and was filled with default value '{default_value}'."
            )

    if "date" in standardized.columns:
        standardized["date"] = pd.to_datetime(standardized["date"], errors="coerce")
    missing_dates = standardized["date"].isna().sum()
    if missing_dates:
        assumptions.append(f"{missing_dates} date entries were invalid and removed.")
    standardized = standardized.dropna(subset=["date"]).copy()

    for column in NUMERIC_COLUMNS:
        standardized[column] = pd.to_numeric(standardized[column], errors="coerce")
        if standardized[column].isna().all():
            standardized[column] = DEFAULT_VALUES[column]
            assumptions.append(
                f"Column '{column}' contained no numeric values and was replaced with '{DEFAULT_VALUES[column]}'."
            )
        else:
            fallback_value = float(np.nanmedian(standardized[column]))
            standardized[column] = standardized[column].fillna(fallback_value)

    standardized["tags_json"] = standardized["tags_json"].apply(_coerce_tags_json)

    standardized["sleep_score"] = standardized["sleep_score"].clip(0, 100)
    standardized["readiness_score"] = standardized["readiness_score"].clip(0, 100)
    standardized["activity_score"] = standardized["activity_score"].clip(0, 100)
    standardized["total_sleep_duration_min"] = standardized["total_sleep_duration_min"].clip(120, 720)
    standardized["bedtime_start_hour"] = standardized["bedtime_start_hour"].clip(18, 30)
    standardized["resting_heart_rate"] = standardized["resting_heart_rate"].clip(35, 120)
    standardized["hrv_ms"] = standardized["hrv_ms"].clip(5, 250)
    standardized["stress_high_min"] = standardized["stress_high_min"].clip(0, 360)
    standardized["recovery_min"] = standardized["recovery_min"].clip(0, 600)
    standardized["steps"] = standardized["steps"].clip(0, 50000)

    standardized = standardized.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    is_demo_column = None
    for candidate in ["is_demo", "demo", "is_sample"]:
        if candidate in raw_df.columns:
            is_demo_column = candidate
            break
    if is_demo_column is not None:
        standardized["is_demo"] = raw_df.loc[standardized.index, is_demo_column].fillna(False).map(_to_bool)
    else:
        standardized["is_demo"] = created_demo

    standardized["is_synthetic"] = False
    standardized["data_source"] = np.where(standardized["is_demo"], "demo_personal", "personal")

    info = {
        "path": str(csv_path),
        "created_demo": created_demo,
        "column_mapping": column_mapping,
        "assumptions": assumptions,
        "row_count": int(len(standardized)),
    }
    return standardized.reset_index(drop=True), info



def load_synthetic_data(path: str | Path = "data/synthetic.csv") -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ["is_synthetic", "is_demo", "data_source"])

    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "tags_json" in df.columns:
        df["tags_json"] = df["tags_json"].apply(_coerce_tags_json)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = DEFAULT_VALUES.get(col, np.nan)

    df["is_synthetic"] = True
    if "is_demo" in df.columns:
        df["is_demo"] = df["is_demo"].map(_to_bool)
    else:
        df["is_demo"] = False
    df["data_source"] = df.get("data_source", "synthetic")

    return df.sort_values("date").reset_index(drop=True)
