import numpy as np
import pandas as pd

from core.pipeline import PipelineError, safe
from core.training import predict


def forecast_window(bundle):
    L = bundle["config"].lookback
    df = bundle["df"].iloc[-L:]
    return {"date": (df.date.iloc[-1] + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d"),
            "history_dates": df.date.dt.strftime("%Y-%m-%d").tolist(),
            "features": bundle["features"], "values": safe(df[bundle["features"]].to_numpy()),
            "default_values": safe(df[bundle["features"]].to_numpy())}


def forecast(model, bundle, readings):
    features = bundle["features"]
    try:
        array = np.asarray(readings, dtype=float)
    except (ValueError, TypeError) as exc:
        raise PipelineError("Enter a numeric value in every reading cell.") from exc
    expected = (bundle["config"].lookback, len(features))
    if array.shape != expected or not np.isfinite(array).all():
        raise PipelineError(f"Supply {expected[0]} complete months with {expected[1]} selected features in the saved order.")
    for i, feature in enumerate(features):
        values = array[:, i]
        if feature == "quarter" and ((values < 1) | (values > 4) | (values != np.round(values))).any():
            raise PipelineError("Quarter must be a whole number from 1 to 4.")
        if feature == "is_holiday_peak" or feature.startswith(("season_", "monsoon_")):
            if not np.isin(values, [0, 1]).all():
                raise PipelineError(f"{feature} must be 0 or 1.")
        if feature == "humidity_pct" and ((values < 0) | (values > 100)).any():
            raise PipelineError("Humidity must be between 0% and 100%.")
        if feature in {"rainy_days", "storm_signal_days", "typhoon_count"}:
            if ((values < 0) | (values != np.round(values))).any():
                raise PipelineError(f"{feature} must be a non-negative whole number.")
            if feature != "typhoon_count" and (values > 31).any():
                raise PipelineError(f"{feature} cannot exceed 31 days.")
        if feature in {"rainfall_mm", "wave_height_m", "pm25_ugm3", "typhoon_max_wind_kt"} and (values < 0).any():
            raise PipelineError(f"{feature} cannot be negative.")
    for prefix in ["season_", "monsoon_"]:
        cols = [i for i, f in enumerate(features) if f.startswith(prefix)]
        if cols and (array[:, cols].sum(axis=1) > 1).any():
            raise PipelineError("A month cannot belong to two season or monsoon categories at once.")
    scaler = bundle["scaler_x"]
    outside = (array < scaler.data_min_) | (array > scaler.data_max_)
    scaled = scaler.transform(pd.DataFrame(array, columns=features)).reshape(1, *expected)
    value = float(bundle["scaler_y"].inverse_transform(predict(model, scaled))[0, 0])
    return safe({"arrivals": value, "date": forecast_window(bundle)["date"],
        "outside_training_range": [{"feature": f, "cells": int(outside[:, i].sum())} for i, f in enumerate(features) if outside[:, i].any()],
        "negative": value < 0, "readings": array.tolist(), "lookback": expected[0],
        "policy": "One month after the latest dataset date. Edits are what-if inputs; no confidence interval or causal claim is implied."})
