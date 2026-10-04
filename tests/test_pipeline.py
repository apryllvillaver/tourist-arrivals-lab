"""Meaningful checks for time leakage, missing targets, and forecast alignment."""
import unittest

import numpy as np
import pandas as pd

from core.evaluation import score
from core.forecasting import forecast_window
from core.pipeline import Config, PipelineError, boundaries, clean, load_csv, prepare, select_features
from core.workspace import ROOT, Workspace


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.dataset = load_csv((ROOT / "data/tourist_arrivals.csv").read_bytes())
        cls.df, cls.clean_report = clean(cls.raw)
        cls.config = Config()
        cls.features, _ = select_features(cls.df, cls.config)
        cls.bundle, cls.prep_report = prepare(cls.df, cls.features, cls.config)

    def test_export_header_and_timeline(self):
        self.assertEqual(self.dataset["header_rows_skipped"], 2)
        self.assertEqual(self.raw.shape, (314, 19))
        self.assertEqual(self.dataset["missing_months"], [])
        self.assertEqual(self.dataset["unique_months"], 312)

    def test_duplicates_and_unknown_targets_are_preserved_correctly(self):
        originals = self.raw.loc[self.raw.date == pd.Timestamp("2006-06-01"), "arrivals"]
        actual = self.df.loc[self.df.date == pd.Timestamp("2006-06-01"), "arrivals"].iloc[0]
        self.assertEqual(actual, originals.iloc[0])
        self.assertEqual(self.clean_report["duplicates_removed"], 2)
        self.assertEqual(self.df.arrivals.isna().sum(), 3)
        self.assertEqual(self.df.loc[self.df.date == pd.Timestamp("2001-02-01"), "arrivals"].iloc[0], 34240153)

    def test_predictor_fill_uses_past_not_future(self):
        broken = self.raw.copy()
        idx = broken.index[broken.date == pd.Timestamp("2017-07-01")][0]
        broken.loc[idx+1, "temp_min_c"] = 999
        cleaned, _ = clean(broken)
        value = cleaned.loc[cleaned.date == pd.Timestamp("2017-07-01"), "temp_min_c"].iloc[0]
        previous = cleaned.loc[cleaned.date == pd.Timestamp("2017-06-01"), "temp_min_c"].iloc[0]
        self.assertEqual(value, previous)

    def test_future_changes_cannot_change_selection_scalers_or_training_windows(self):
        fit_end, _ = boundaries(len(self.df), self.config)
        changed = self.df.copy()
        columns = self.features + ["arrivals"]
        changed.loc[fit_end:, columns] = changed.loc[fit_end:, columns] * 1000 + 123456
        features, _ = select_features(changed, self.config)
        bundle, _ = prepare(changed, features, self.config)
        self.assertEqual(features, self.features)
        np.testing.assert_array_equal(bundle["scaler_x"].data_min_, self.bundle["scaler_x"].data_min_)
        np.testing.assert_array_equal(bundle["scaler_x"].data_max_, self.bundle["scaler_x"].data_max_)
        np.testing.assert_array_equal(bundle["scaler_y"].data_max_, self.bundle["scaler_y"].data_max_)
        np.testing.assert_array_equal(bundle["X_train"], self.bundle["X_train"])
        np.testing.assert_array_equal(bundle["y_train"], self.bundle["y_train"])

    def test_windows_stay_in_their_own_partitions(self):
        fit_end, development_end = boundaries(len(self.df), self.config)
        for name, start, end in [("train", 0, fit_end), ("validation", fit_end, development_end), ("test", development_end, len(self.df))]:
            indices = self.bundle[f"indices_{name}"]
            self.assertTrue((indices - self.config.lookback >= start).all())
            self.assertTrue((indices < end).all())
            self.assertTrue(np.isfinite(self.bundle[f"y_{name}"]).all())
        self.assertEqual(self.bundle["X_train"].shape, (196, 12, len(self.features)))
        self.assertEqual(len(self.bundle["X_test"]), 51)

    def test_forecast_uses_latest_actual_window(self):
        window = forecast_window(self.bundle)
        self.assertEqual(window["date"], "2026-01-01")
        self.assertEqual(window["history_dates"][-1], "2025-12-01")
        np.testing.assert_array_equal(window["values"], self.df.iloc[-12:][self.features].to_numpy())

    def test_metric_edge_cases(self):
        result = score(np.array([0, 10]), np.array([1, 12]))
        self.assertEqual(result["MAE"], 1.5)
        self.assertAlmostEqual(result["RMSE"], np.sqrt(2.5))
        self.assertEqual(result["MAPE"], 20.0)
        self.assertEqual(result["MAPE_n"], 1)
        self.assertIsNone(score([5, 5], [4, 6])["R2"])

    def test_upstream_change_clears_dependent_state(self):
        w=Workspace();w.load();w.run_clean();w.run_features(self.config);w.run_prepare()
        w.model=object();w.reports["train"]={"old":True}
        w.run_clean()
        self.assertIsNone(w.bundle)
        self.assertIsNone(w.model)
        self.assertEqual(set(w.reports), {"dataset", "clean"})
        with self.assertRaises(PipelineError):
            w.run_prepare()


if __name__ == "__main__":
    unittest.main()
