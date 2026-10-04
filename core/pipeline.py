from __future__ import annotations

import csv
import io
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

SCHEMA = [
    "date", "year", "month", "quarter", "season", "monsoon", "is_holiday_peak",
    "temp_mean_c", "temp_min_c", "temp_max_c", "rainfall_mm", "rainy_days",
    "humidity_pct", "typhoon_count", "typhoon_max_wind_kt", "storm_signal_days",
    "pm25_ugm3", "wave_height_m", "arrivals",
]
CANDIDATES = [c for c in SCHEMA if c not in {"date", "year", "month", "season", "monsoon", "arrivals"}]
CATEGORIES = {"season": ["Dry", "Transition", "Wet"], "monsoon": ["Amihan", "Habagat", "Transition"]}
LABELS = {
    "quarter": "Quarter", "is_holiday_peak": "Holiday peak", "temp_mean_c": "Mean temperature (°C)",
    "temp_min_c": "Minimum temperature (°C)", "temp_max_c": "Maximum temperature (°C)",
    "rainfall_mm": "Rainfall (mm)", "rainy_days": "Rainy days", "humidity_pct": "Humidity (%)",
    "typhoon_count": "Typhoon count", "typhoon_max_wind_kt": "Maximum wind (kt)",
    "storm_signal_days": "Storm warning days", "pm25_ugm3": "PM2.5 (µg/m³)",
    "wave_height_m": "Wave height (m)", "season_Transition": "Transition season",
    "season_Wet": "Wet season", "monsoon_Habagat": "Habagat", "monsoon_Transition": "Transition monsoon",
}


class PipelineError(ValueError):
    """An actionable data or prerequisite error, safe to show in either UI."""


def safe(value: Any) -> Any:
    """Convert reports to strict JSON: unavailable numbers become null, never NaN."""
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [safe(v) for v in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def records(df: pd.DataFrame) -> list[dict]:
    return safe(df.to_dict(orient="records"))


@dataclass(frozen=True)
class Config:
    train_ratio: float = 0.80
    validation_ratio: float = 0.15  # tail of the development/training portion
    lookback: int = 12
    seed: int = 42


def boundaries(n: int, config: Config) -> tuple[int, int]:
    if not 0.60 <= config.train_ratio <= 0.95:
        raise PipelineError("Choose a training share between 60% and 95%.")
    if not 0.10 <= config.validation_ratio <= 0.30:
        raise PipelineError("Choose a validation share between 10% and 30% of the training portion.")
    if not 6 <= config.lookback <= 24:
        raise PipelineError("Choose a lookback between 6 and 24 months.")
    development_end = int(n * config.train_ratio)
    fitting_end = int(development_end * (1 - config.validation_ratio))
    if min(fitting_end, development_end - fitting_end, n - development_end) <= config.lookback + 1:
        raise PipelineError("This split leaves too little history. Lower the training share or lookback, or use more data.")
    return fitting_end, development_end


def load_csv(content: bytes) -> tuple[pd.DataFrame, dict]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise PipelineError("Export the dataset as a UTF-8 CSV, then try again.") from exc
    lines = text.splitlines()
    header_row = None
    for i, line in enumerate(lines[:20]):
        fields = [c.strip() for c in next(csv.reader([line]))]
        if "date" in fields and "arrivals" in fields:
            header_row = i
            break
    if header_row is None:
        raise PipelineError("No date/arrivals header found. Use the full 19-column laboratory CSV.")
    try:
        df = pd.read_csv(io.StringIO("\n".join(lines[header_row:])))
    except Exception as exc:
        raise PipelineError(f"The CSV could not be read: {exc}") from exc
    df.columns = df.columns.str.strip()
    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        raise PipelineError("Missing required columns: " + ", ".join(missing))
    df = df[SCHEMA].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df.empty or df["date"].isna().any():
        raise PipelineError("Every record needs a valid date in YYYY-MM-01 format.")
    if (df["date"].dt.day != 1).any():
        raise PipelineError("Dates must be the first day of each month.")
    sentinel_counts = {}
    for col in CANDIDATES + ["year", "month", "arrivals"]:
        original = df[col]
        numeric = pd.to_numeric(original, errors="coerce")
        if (original.notna() & numeric.isna()).any():
            raise PipelineError(f"{col} has a non-numeric reading. Correct the source CSV first.")
        if np.isinf(numeric).any():
            raise PipelineError(f"{col} contains an infinite reading. Correct the source CSV first.")
        sentinel_counts[col] = int((numeric == -999).sum())
        df[col] = numeric.replace(-999, np.nan)
    if (df["arrivals"].dropna() < 0).any():
        raise PipelineError("Arrival counts must be non-negative. Check the source records.")
    for col, allowed in CATEGORIES.items():
        df[col] = df[col].astype("string").str.strip()
        if (~df[col].isin(allowed)).any():
            raise PipelineError(f"{col} must use: {', '.join(allowed)}. Check missing or misspelled categories.")
    for col, expected in [("year", df.date.dt.year), ("month", df.date.dt.month), ("quarter", df.date.dt.quarter)]:
        if not np.array_equal(df[col].to_numpy(), expected.to_numpy()):
            raise PipelineError(f"{col} does not match the date column. Correct the source CSV first.")
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    months = pd.date_range(df.date.min(), df.date.max(), freq="MS")
    report = {
        "rows": len(df), "columns": len(df.columns), "unique_months": df.date.nunique(),
        "start": df.date.min(), "end": df.date.max(), "header_rows_skipped": header_row,
        "missing_months": months.difference(df.date).strftime("%Y-%m-%d").tolist(),
        "duplicate_records": int(df.date.duplicated().sum()),
        "missing": df.isna().sum().to_dict(), "sentinels": {k: v for k, v in sentinel_counts.items() if v},
        "dtypes": [{"column": c, "type": str(df[c].dtype), "missing": int(df[c].isna().sum())} for c in df],
    }
    return df, safe(report)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    removed = raw[raw.date.duplicated(keep="first")]
    df = raw.drop_duplicates("date", keep="first").reset_index(drop=True).copy()
    before = df.isna().sum()
    fill_log = []
    # Causal, bounded filling: never borrow from a later month; never fill the target.
    for col in CANDIDATES:
        filled = df[col].ffill(limit=2)
        for idx in df.index[df[col].isna() & filled.notna()]:
            fill_log.append({"date": df.at[idx, "date"], "feature": col, "value": filled.at[idx], "method": "Past reading (forward-fill, limit 2)"})
        df[col] = filled
    q1, q3 = df.arrivals.quantile([.25, .75])
    lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    flagged = df[(df.arrivals < lower) | (df.arrivals > upper)][["date", "arrivals"]]
    for col, levels in CATEGORIES.items():
        # Fixed categories from the lab schema; one reference category is omitted.
        for level in levels[1:]:
            df[f"{col}_{level}"] = (df[col] == level).astype(float)
    report = {
        "rows": len(df), "duplicates_removed": len(removed), "duplicate_log": records(removed),
        "missing_before": before.to_dict(), "missing_after": df.isna().sum().to_dict(),
        "fill_log": fill_log, "missing_targets": records(df.loc[df.arrivals.isna(), ["date", "arrivals"]]),
        "outliers": records(flagged), "iqr": {"q1": q1, "q3": q3, "lower": lower, "upper": upper},
        "outlier_policy": "IQR flags are diagnostic only. Every flagged value is retained pending source verification.",
        "encoding": "Dry season and Amihan are reference categories; no arbitrary numeric ranks are assigned to text.",
    }
    return df, safe(report)


def vif_values(df: pd.DataFrame) -> list[float]:
    if df.shape[1] == 1:
        return [1.0]
    # Including an intercept makes VIF measure redundancy, rather than distance from zero.
    matrix = np.column_stack([np.ones(len(df)), df.to_numpy(dtype=float)])
    with np.errstate(divide="ignore", invalid="ignore"):
        return [float(variance_inflation_factor(matrix, i + 1)) for i in range(df.shape[1])]


def select_features(df: pd.DataFrame, config: Config) -> tuple[list[str], dict]:
    fitting_end, development_end = boundaries(len(df), config)
    fitting = df.iloc[:fitting_end]
    candidates = CANDIDATES + [f"{c}_{level}" for c, levels in CATEGORIES.items() for level in levels[1:]]
    results = []
    kept = []
    for col in candidates:
        pair = fitting[[col, "arrivals"]].dropna()
        rho, p = (np.nan, np.nan)
        if len(pair) >= 3 and pair[col].nunique() > 1:
            rho, p = spearmanr(pair[col], pair.arrivals)
        passed = bool(np.isfinite(rho) and abs(rho) > .10 and p < .05)
        results.append({"feature": col, "rho": rho, "p_value": p, "n": len(pair), "passed": passed})
        if passed:
            kept.append(col)
    if not kept:
        raise PipelineError("No predictor passes |rho| > 0.10 and p < 0.05 on training data. Report this result; do not quietly weaken the rule.")
    vif_log = []
    iteration = 1
    while True:
        X = fitting[kept].dropna()
        if len(X) <= len(kept) + 1:
            raise PipelineError("Not enough complete training readings for VIF. Review gaps or use more history.")
        values = vif_values(X)
        maximum = max(values)
        if maximum < 5:
            break
        # Stable tie rule: discard the weaker absolute target correlation among tied VIFs.
        ties = [i for i, v in enumerate(values) if v == maximum or (np.isfinite(v) and np.isfinite(maximum) and np.isclose(v, maximum))]
        rho_map = {r["feature"]: abs(r["rho"]) for r in results if r["passed"]}
        drop_i = min(ties, key=lambda i: rho_map[kept[i]])
        vif_log.append({"iteration": iteration, "dropped": kept[drop_i], "vif": maximum,
                        "infinite": not np.isfinite(maximum), "before": dict(zip(kept, values))})
        kept.pop(drop_i)
        iteration += 1
    report = {"results": results, "vif_log": vif_log, "final_vif": dict(zip(kept, values)), "selected": kept,
              "fitting_start": fitting.date.iloc[0], "fitting_end": fitting.date.iloc[-1], "fitting_rows": fitting_end,
              "development_end": development_end, "config": asdict(config),
              "selection_scope": "Fitting rows only. Validation and test data are excluded from selection."}
    return kept, safe(report)


def sequences(X: np.ndarray, y: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, indices = [], [], []
    for target in range(lookback, len(X)):
        window = X[target-lookback:target]
        if np.isfinite(y[target]).all() and np.isfinite(window).all():
            xs.append(window)
            ys.append(y[target])
            indices.append(target)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32).reshape(-1, 1), np.asarray(indices, dtype=int)


def prepare(df: pd.DataFrame, features: list[str], config: Config) -> tuple[dict, dict]:
    fitting_end, development_end = boundaries(len(df), config)
    fitting = df.iloc[:fitting_end]
    scaler_x = MinMaxScaler().fit(fitting[features])
    scaler_y = MinMaxScaler().fit(fitting[["arrivals"]].dropna())
    parts = {"train": (0, fitting_end), "validation": (fitting_end, development_end), "test": (development_end, len(df))}
    bundle = {"scaler_x": scaler_x, "scaler_y": scaler_y, "features": features, "config": config, "df": df}
    report = {"config": asdict(config), "partitions": [], "scaler_ranges": [], "shape_order": "(windows, lookback months, features)"}
    for name, (start, end) in parts.items():
        part = df.iloc[start:end]
        X = scaler_x.transform(part[features])
        y = scaler_y.transform(part[["arrivals"]])
        xs, ys, target_indices = sequences(X, y, config.lookback)
        if len(xs) < 2:
            raise PipelineError(f"The {name} partition has fewer than two usable windows. Review missing values or choose a different split.")
        bundle[f"X_{name}"] = xs
        bundle[f"y_{name}"] = ys
        bundle[f"indices_{name}"] = target_indices + start
        report["partitions"].append({"name": name, "start": part.date.iloc[0], "end": part.date.iloc[-1],
            "rows": len(part), "windows": len(xs), "shape": list(xs.shape),
            "warmup_rows": config.lookback, "unusable_targets": len(part) - config.lookback - len(xs),
            "first_target": df.date.iloc[start+target_indices[0]], "last_target": df.date.iloc[start+target_indices[-1]]})
    for i, col in enumerate(features):
        report["scaler_ranges"].append({"feature": col, "min": scaler_x.data_min_[i], "max": scaler_x.data_max_[i]})
    report["target_range"] = {"min": scaler_y.data_min_[0], "max": scaler_y.data_max_[0]}
    report["policy"] = "Split before fitting either scaler. Train, validation and test are windowed separately; inputs precede their targets. Missing targets are excluded, not invented."
    return bundle, safe(report)
