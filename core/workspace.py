from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

import joblib

from core.pipeline import Config, LABELS, PipelineError, clean, load_csv, prepare, records, safe, select_features
from core.forecasting import forecast, forecast_window

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["dataset", "clean", "features", "prepare", "train", "evaluate", "explain", "forecast"]


class Workspace:
    def __init__(self):
        self.raw = self.cleaned = self.bundle = self.model = None
        self.features = []
        self.config = Config()
        self.reports = {}
        self.filename = "tourist_arrivals.csv"

    def invalidate(self, after):
        for stage in STAGES[STAGES.index(after)+1:]:
            self.reports.pop(stage, None)
        if STAGES.index(after) < 1:
            self.cleaned = None
        if STAGES.index(after) < 2:
            self.features = []
        if STAGES.index(after) < 3:
            self.bundle = None
        if STAGES.index(after) < 4:
            self.model = None

    def need(self, stage):
        if stage not in self.reports:
            raise PipelineError(f"Complete the {stage.title()} step first.")

    def load(self, content=None, filename="tourist_arrivals.csv"):
        df, report = load_csv(content if content is not None else (ROOT / "data/tourist_arrivals.csv").read_bytes())
        self.raw, self.filename = df, filename
        self.invalidate("dataset")
        self.reports["dataset"] = report

    def run_clean(self):
        self.need("dataset")
        if self.reports["dataset"]["missing_months"]:
            raise PipelineError("There are missing months in the timeline. Resolve them in the source CSV before windowing.")
        df, report = clean(self.raw)
        self.cleaned = df
        self.invalidate("clean")
        self.reports["clean"] = report

    def run_features(self, config: Config):
        self.need("clean")
        features, report = select_features(self.cleaned, config)
        self.features, self.config = features, config
        self.invalidate("features")
        self.reports["features"] = report

    def run_prepare(self):
        self.need("features")
        bundle, report = prepare(self.cleaned, self.features, self.config)
        self.bundle = bundle
        self.invalidate("prepare")
        self.reports["prepare"] = report

    def run_train(self, epochs=100, search="full", progress=None):
        from core.training import train
        self.need("prepare")
        model, report = train(self.bundle, epochs, search, progress)
        self.model = model
        self.invalidate("train")
        self.reports["train"] = report

    def run_evaluate(self):
        from core.evaluation import evaluate
        self.need("train")
        self.reports["evaluate"] = evaluate(self.model, self.bundle)

    def run_explain(self, samples=10, nsamples=256, progress=None):
        from core.explainability import explain
        self.need("train")
        self.reports["explain"] = explain(self.model, self.bundle, samples, nsamples, progress)

    def run_forecast(self, readings):
        self.need("train")
        self.reports["forecast"] = forecast(self.model, self.bundle, readings)

    def snapshot(self):
        df = self.cleaned if self.cleaned is not None else self.raw
        return safe({"filename": self.filename, "reports": self.reports, "completed": list(self.reports),
            "features": self.features, "labels": LABELS,
            "history": records(df[["date", "arrivals", "rainfall_mm", "temp_mean_c"]]) if df is not None else [],
            "preview": records(self.raw.head(12)) if self.raw is not None else [],
            "window": forecast_window(self.bundle) if self.bundle is not None else None})

    def export(self):
        """Download an auditable run without placing temporary files in the source tree."""
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("reports.json", json.dumps(safe(self.reports), indent=2, allow_nan=False))
            archive.writestr("README.txt", "This export contains this workspace's computed results.\nThe saved Keras model and preprocessing.joblib belong together.\nSHAP explains model behaviour, not cause and effect.\n")
            if self.cleaned is not None:
                archive.writestr("cleaned_data.csv", self.cleaned.to_csv(index=False))
            if "features" in self.reports:
                import pandas as pd
                archive.writestr("spearman.csv", pd.DataFrame(self.reports["features"]["results"]).to_csv(index=False))
            if "evaluate" in self.reports:
                import pandas as pd
                archive.writestr("metrics.csv", pd.DataFrame(self.reports["evaluate"]["metrics"]).to_csv(index=False))
                archive.writestr("test_predictions.csv", pd.DataFrame(self.reports["evaluate"]["predictions"]).to_csv(index=False))
            if self.model is not None:
                with tempfile.TemporaryDirectory() as temp:
                    model_path = Path(temp) / "model.keras"
                    self.model.save(model_path)
                    archive.write(model_path, "model.keras")
                    prep_path = Path(temp) / "preprocessing.joblib"
                    joblib.dump({"scaler_x": self.bundle["scaler_x"], "scaler_y": self.bundle["scaler_y"], "features": self.features, "config": self.config}, prep_path)
                    archive.write(prep_path, "preprocessing.joblib")
        return stream.getvalue()
