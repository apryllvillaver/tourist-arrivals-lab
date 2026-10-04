import numpy as np

from core.pipeline import PipelineError, safe
from core.training import predict


def score(actual, predicted):
    actual, predicted = np.asarray(actual).ravel(), np.asarray(predicted).ravel()
    errors = actual - predicted
    nonzero = actual != 0
    total = np.sum((actual - actual.mean()) ** 2)
    return safe({"MAE": np.mean(np.abs(errors)), "RMSE": np.sqrt(np.mean(errors ** 2)),
        "MAPE": np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100 if nonzero.any() else None,
        "R2": 1 - np.sum(errors ** 2) / total if total > 0 else None,
        "MAPE_n": int(nonzero.sum()), "n": len(actual)})


def evaluate(model, bundle):
    indices = bundle["indices_test"]
    arrivals = bundle["df"].arrivals.to_numpy(dtype=float)
    actual = arrivals[indices]
    predicted = bundle["scaler_y"].inverse_transform(predict(model, bundle["X_test"])).ravel()
    naive, seasonal = arrivals[indices-1], arrivals[indices-12]
    # Use the same target months for all models, including short lookbacks.
    valid = np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(naive) & np.isfinite(seasonal)
    if valid.sum() < 2:
        raise PipelineError("Fewer than two test months have complete predictions and both baseline readings.")
    metrics = [{"model": label, **score(actual[valid], values[valid])}
               for label, values in [("LSTM", predicted), ("Naive", naive), ("Seasonal naive", seasonal)]]
    comparison = [{"date": bundle["df"].date.iloc[i], "actual": float(a), "lstm": float(p), "naive": float(n),
        "seasonal": float(s), "error": float(a-p)} for i, a, p, n, s in zip(indices[valid], actual[valid], predicted[valid], naive[valid], seasonal[valid])]
    best = min(metrics, key=lambda m: m["MAE"])
    return safe({"metrics": metrics, "predictions": comparison, "lowest_mae": best["model"],
        "excluded_months": int((~valid).sum()), "negative_predictions": int((predicted[valid] < 0).sum()),
        "policy": "One-step forecasts on identical held-out months. Prior observed arrivals are available to the baselines. MAPE excludes zero actuals; undefined R² is left unavailable."})
