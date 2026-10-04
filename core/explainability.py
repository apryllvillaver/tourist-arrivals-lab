import numpy as np

from core.pipeline import PipelineError, safe
from core.training import predict


def explain(model, bundle, samples=10, nsamples=256, progress=None):
    import shap
    if not 2 <= samples <= 20 or not 128 <= nsamples <= 1024:
        raise PipelineError("Explain 2–20 test windows with 128–1024 SHAP samples.")
    rng = np.random.default_rng(bundle["config"].seed)
    training = bundle["X_train"]
    count = min(50, len(training))
    bg = training[rng.choice(len(training), count, replace=False)]
    lookback, features = training.shape[1:]
    background = shap.kmeans(bg.reshape(count, -1), min(10, count))
    tested = bundle["X_test"][:min(samples, len(bundle["X_test"]))]

    def original_scale(flat):
        seq = np.asarray(flat, dtype=np.float32).reshape(-1, lookback, features)
        return bundle["scaler_y"].inverse_transform(predict(model, seq)).ravel()

    explainer = shap.KernelExplainer(original_scale, background)
    # Enough masks for an unregularized regression; avoid the version-dependent top-10 default.
    masks = max(nsamples, 2 * lookback * features + 64)
    np.random.seed(bundle["config"].seed)
    all_values = []
    for i, window in enumerate(tested):
        values = np.asarray(explainer.shap_values(window.reshape(1, -1), nsamples=masks, l1_reg=0.0, silent=True))
        all_values.append(values.reshape(lookback, features))
        if progress:
            progress((i+1)/len(tested), f"Explained window {i+1}/{len(tested)}")
    values = np.asarray(all_values)
    base = float(np.asarray(explainer.expected_value).ravel()[0])
    outputs = original_scale(tested.reshape(len(tested), -1))
    residuals = outputs - (base + values.sum(axis=(1, 2)))
    if np.max(np.abs(residuals)) > max(.1, np.max(np.abs(outputs)) * 1e-5):
        raise PipelineError("SHAP contributions did not reconstruct the predictions. Increase the sampling budget.")
    # Importance measures the average total absolute contribution over every lag of a feature.
    importance = np.abs(values).sum(axis=1).mean(axis=0)
    top = int(np.argmax(importance))
    raw = bundle["scaler_x"].inverse_transform(tested.reshape(-1, features)).reshape(tested.shape)
    local = []
    for i, window_values in enumerate(values):
        contributions = window_values.sum(axis=0)
        local.append({"date": bundle["df"].date.iloc[bundle["indices_test"][i]], "base": base,
            "prediction": outputs[i], "residual": residuals[i],
            "contributions": sorted([{"feature": f, "value": v} for f, v in zip(bundle["features"], contributions)], key=lambda r: abs(r["value"]), reverse=True)})
    return safe({"method": "KernelSHAP", "samples": len(tested), "background_windows": count,
        "background_centroids": min(10, count), "masks": masks, "unit": "arrivals", "lookback": lookback,
        "importance": sorted([{"feature": f, "value": v} for f, v in zip(bundle["features"], importance)], key=lambda r: r["value"], reverse=True),
        "local": local, "top_feature": bundle["features"][top],
        "dependence": [{"value": raw[i, -1, top], "shap": values[i, -1, top], "date": local[i]["date"]} for i in range(len(tested))],
        "policy": "Background uses fitting windows only. The waterfall sums each feature's lags; dependence uses the most recent lag only. Values explain this model's behaviour, not causation."})
