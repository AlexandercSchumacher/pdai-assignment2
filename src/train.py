from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_load import load_personal_data
from src.feature_engineering import prepare_training_frame
from src.synth_data import create_or_load_synthetic



def train_model(
    personal_csv: str,
    synthetic_csv: str,
    model_out: str,
    metadata_out: str,
    n_synth_days: int,
    random_state: int,
) -> dict:
    personal_df, load_info = load_personal_data(personal_csv)
    synthetic_df = create_or_load_synthetic(
        personal_df,
        path=synthetic_csv,
        n_days=n_synth_days,
        force_regenerate=True,
    )

    full_df = pd.concat([personal_df, synthetic_df], ignore_index=True)
    full_df["date"] = pd.to_datetime(full_df["date"], errors="coerce")
    full_df = full_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    training_df, feature_columns = prepare_training_frame(full_df)
    if len(training_df) < 60:
        raise ValueError(
            "Too few training rows after feature engineering. At least 60 rows are required."
        )

    X = training_df[feature_columns]
    y = training_df["target_energy_t_plus_1"]

    split_idx = max(int(len(training_df) * 0.8), len(training_df) - 60)
    split_idx = min(split_idx, len(training_df) - 20)

    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(
        n_estimators=450,
        max_depth=10,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred_val = model.predict(X_val)
    mae = float(mean_absolute_error(y_val, pred_val))
    rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
    r2 = float(r2_score(y_val, pred_val))

    if len(y_val) >= 20:
        residuals = (y_val - pred_val).to_numpy(dtype=float)
    else:
        residuals = (y - model.predict(X)).to_numpy(dtype=float)

    try:
        perm = permutation_importance(
            model,
            X_val,
            y_val,
            n_repeats=12,
            random_state=random_state,
            n_jobs=-1,
        )
        importances = perm.importances_mean
    except Exception:
        importances = model.feature_importances_

    importance_map = {
        col: float(imp)
        for col, imp in sorted(
            zip(feature_columns, importances),
            key=lambda pair: pair[1],
            reverse=True,
        )
    }

    model_bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "residuals": residuals.tolist(),
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
        "feature_importance": importance_map,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    model_out_path = Path(model_out)
    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    with model_out_path.open("wb") as f:
        pickle.dump(model_bundle, f)

    metadata = {
        "trained_at_utc": model_bundle["trained_at"],
        "rows_personal": int(len(personal_df)),
        "rows_synthetic": int(len(synthetic_df)),
        "rows_total": int(len(full_df)),
        "rows_training": int(len(training_df)),
        "feature_columns": feature_columns,
        "metrics": model_bundle["metrics"],
        "energy_formula": "energy_score = 0.5*readiness_score + 0.3*sleep_score + 0.2*clamp(100 - stress_component)",
        "stress_formula": "stress_component = clamp(100 * stress_high_min / (stress_high_min + recovery_min), 0, 100)",
        "load_assumptions": load_info["assumptions"],
        "column_mapping": load_info["column_mapping"],
    }

    metadata_out_path = Path(metadata_out)
    metadata_out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_out_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "model_path": str(model_out_path),
        "metadata_path": str(metadata_out_path),
        "synthetic_path": synthetic_csv,
        "metrics": metadata["metrics"],
        "rows_training": metadata["rows_training"],
        "rows_synthetic": metadata["rows_synthetic"],
    }



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training script for the Personal Energy Forecast model")
    parser.add_argument("--personal-csv", default="data/oura_personal.csv")
    parser.add_argument("--synthetic-csv", default="data/synthetic.csv")
    parser.add_argument("--model-out", default="models/energy_model.pkl")
    parser.add_argument("--metadata-out", default="models/metadata.json")
    parser.add_argument("--n-synth-days", type=int, default=730)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    result = train_model(
        personal_csv=args.personal_csv,
        synthetic_csv=args.synthetic_csv,
        model_out=args.model_out,
        metadata_out=args.metadata_out,
        n_synth_days=args.n_synth_days,
        random_state=args.random_state,
    )

    print("Training completed")
    print(f"Model: {result['model_path']}")
    print(f"Metadata: {result['metadata_path']}")
    print(f"Synthetic data: {result['synthetic_path']}")
    print(f"Rows training: {result['rows_training']}")
    print(f"Rows synthetic: {result['rows_synthetic']}")
    print(
        "Metrics - "
        f"MAE: {result['metrics']['mae']:.2f}, "
        f"RMSE: {result['metrics']['rmse']:.2f}, "
        f"R2: {result['metrics']['r2']:.3f}"
    )


if __name__ == "__main__":
    main()
