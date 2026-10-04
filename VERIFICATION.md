# Verification record

The shared pipeline was exercised on the supplied CSV. This verification ran on Linux using Python 3.12, TensorFlow 2.16.2 and Keras 3.3.3. The supplied Mac instructions use Python 3.11 for the pinned installation; the application has not been run on your physical MacBook.

## Completed checks

- Eight pipeline tests passed, including an adversarial check that changing validation/test values cannot change selected features, scalers or training windows.
- Streamlit Home and all eight pages rendered through the registered multipage app without exceptions.
- Next.js TypeScript checks and the production build passed.
- A real browser exercised cleaning, selection, preparation, full eight-candidate training, evaluation, ten-window SHAP and forecasting through the FastAPI endpoints.
- Changing a forecast input hid the stale prediction. Recalculating preserved the edited scenario readings.
- Every result page stayed inside the viewport at 390, 768 and 1280 pixels. The final browser layout checks reported no page errors.
- SHAP's base value plus contributions reconstructed each explained model prediction within numerical tolerance.
- Model/scaler/report export was exercised on a separate short setup run.
- API checks passed for report export, workspace isolation and invalid-upload recovery without losing the previous pipeline state.

## Full-search reference run

Settings: development share 80%, validation tail 15%, lookback 12, random seed 42, maximum 100 epochs, early-stopping patience 8. All IQR-flagged values were retained. The best candidate used 64 units, dropout 0.1 and batch size 16.

| Forecast | MAE | RMSE | MAPE | R² |
| --- | ---: | ---: | ---: | ---: |
| LSTM | 296,838.4 | 406,393.3 | 220.58% | -3.658 |
| Naive | 64,242.2 | 83,920.9 | 16.73% | 0.801 |
| Seasonal naive | 134,876.6 | 176,339.6 | 39.20% | 0.123 |

All three were scored on the same 51 months, October 2021–December 2025. The naive baseline was best on all four displayed metrics. These values are verification results, not hard-coded application outputs. Your run may differ slightly across machines.

In ten explained test windows, the Habagat indicator had the largest average total absolute SHAP contribution across its lags. This is reliance within this model and sample, rather than proof of a causal monsoon effect. The large retained February 2001 arrival value affects the scale of the training target; source verification is necessary before correcting it.

The forecast date was January 2026, based on the latest January–December 2025 readings. It is determined by the dataset's end, rather than the present calendar date.
