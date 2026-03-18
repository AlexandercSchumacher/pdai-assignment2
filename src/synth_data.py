from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineering import clamp_frame

NUMERIC_COLS = [
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



def _sample_tags(real_df: pd.DataFrame, size: int, rng: np.random.Generator) -> list[str]:
    if "tags_json" not in real_df.columns or real_df["tags_json"].dropna().empty:
        return [json.dumps([]) for _ in range(size)]

    pool = real_df["tags_json"].fillna("[]").astype(str).tolist()
    idx = rng.integers(0, len(pool), size=size)
    return [pool[i] for i in idx]



def generate_synthetic_data(
    real_df: pd.DataFrame,
    n_days: int = 730,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    n_days = max(int(n_days), 730)

    if real_df.empty:
        raise ValueError("real_df darf nicht leer sein, um synthetische Daten zu erzeugen.")

    base = real_df.sample(n=n_days, replace=True, random_state=random_state).reset_index(drop=True)
    numeric = base[NUMERIC_COLS].to_numpy(dtype=float)

    cov_source = real_df[NUMERIC_COLS].to_numpy(dtype=float)
    cov = np.cov(cov_source.T)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    ridge = np.eye(cov.shape[0]) * max(1e-3, float(np.mean(np.diag(cov))) * 1e-3)
    stable_cov = cov + ridge

    noise = rng.multivariate_normal(
        mean=np.zeros(len(NUMERIC_COLS)),
        cov=stable_cov,
        size=n_days,
        check_valid="ignore",
    )

    noisy_numeric = numeric + noise * 0.18

    synth = pd.DataFrame(noisy_numeric, columns=NUMERIC_COLS)
    synth = clamp_frame(synth)

    max_real_date = pd.to_datetime(real_df["date"], errors="coerce").max()
    if pd.isna(max_real_date):
        max_real_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    synth_dates = pd.date_range(start=max_real_date + pd.Timedelta(days=1), periods=n_days, freq="D")

    synth["date"] = synth_dates
    synth["tags_json"] = _sample_tags(real_df, n_days, rng)
    synth["is_synthetic"] = True
    synth["is_demo"] = False
    synth["data_source"] = "synthetic"

    return synth[
        [
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
            "is_demo",
            "is_synthetic",
            "data_source",
        ]
    ]



def create_or_load_synthetic(
    real_df: pd.DataFrame,
    path: str | Path = "data/synthetic.csv",
    n_days: int = 730,
    force_regenerate: bool = True,
) -> pd.DataFrame:
    csv_path = Path(path)

    if csv_path.exists() and not force_regenerate:
        existing = pd.read_csv(csv_path)
        if len(existing) >= 730:
            existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
            existing["is_synthetic"] = True
            existing["data_source"] = "synthetic"
            return existing

    synthetic = generate_synthetic_data(real_df=real_df, n_days=n_days)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(csv_path, index=False)
    return synthetic
