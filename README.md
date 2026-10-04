# Philippine Tourist Arrivals Forecasting Lab

This application follows the forecasting pipeline in Module 2 and Laboratory Exercise 1. It loads the supplied monthly dataset, documents cleaning decisions, selects predictors, trains an LSTM, evaluates it against two baselines, explains its predictions with SHAP and forecasts the next dataset month.

The graded Streamlit application and the optional FastAPI + Next.js interface call the same Python functions. Each interface has its own workspace. Running one does not silently change the other.

## Start on a Mac

Use Python 3.11 and Node.js 22. The project pins compatible Python packages and includes an npm lockfile.

From the project folder:

```bash
bash scripts/setup-mac.sh
bash scripts/start-streamlit.sh
```

Open [http://localhost:8501](http://localhost:8501). Confirm Home loads, then follow the eight sidebar pages in order.

For the optional frontend, keep two separate terminals open:

```bash
# Terminal 1, from the project root
bash scripts/start-backend.sh
```

```bash
# Terminal 2, from the project root
bash scripts/start-frontend.sh
```

Open [http://localhost:3000](http://localhost:3000). API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

Read **MACBOOK_GUIDE.md** for first-time installation, the complete lab walkthrough, stopping and restarting, GitHub submission and the three-minute demonstration plan.

## Dataset checks

The included course CSV has two title rows before its actual header. The loader detects the header rather than assuming the first row is a column list. The attached dataset README describes a four-row preview, but this uploaded CSV contains the full multi-year series.

The supplied file has:

| Check | Finding |
| --- | --- |
| Source records | 314 |
| Source columns | 19 |
| Unique months | 312, January 2000 to December 2025 |
| Duplicate records removed | Two additional entries for June 2006 |
| Missing arrivals | December 2002, August 2005, July 2010 |
| Missing predictors | Minimum temperature in July 2017; maximum temperature in November 2018 and July 2023 |
| Monthly timeline gaps | None |

Cleaning keeps the first original entry for each date with a stable sort. Short predictor gaps use a past-only forward-fill with a two-month limit. Missing targets remain unknown and are omitted as target windows. No target is imputed for test scoring.

The IQR check flags three arrival values: February 2001 (34,240,153), January 2020 (1,041,025) and February 2020 (1,090,211). All are retained. The unusually large 2001 reading needs verification against its source before any correction is justified. IQR is a descriptive flag, calculated on the cleaned series; it is not used to change training data.

## Pipeline and lab requirements

| Lab stage | Implementation |
| --- | --- |
| Set up and confirm the app | `Home.py`, requirements, Mac scripts |
| Dataset summary | Header detection, shape, dtypes, monthly gaps, raw preview |
| Clean | Stable deduplication, missing-value log, causal predictor filling, IQR review |
| Feature selection | Spearman: absolute rho > 0.10 and p < 0.05; iterative VIF < 5 |
| Leakage-safe preparation | Chronological fitting/validation/test periods; fitting-only scalers; separate `(N, L, F)` windows |
| Train and tune | Keras LSTM, dropout, dense output; full eight-candidate search; chronological validation and early stopping |
| Evaluate | MAE, RMSE, MAPE and R² in original units; naive and seasonal-naive comparisons on identical months |
| Explain | KernelSHAP, a real floating-bar waterfall, global importance and raw-unit dependence plot |
| Forecast | Editable latest lookback window; one-step prediction for the month after the last dataset record |
| Optional challenge | FastAPI endpoints and a responsive Next.js interface |

The lab lists 13 numeric candidates even though its prose uses a different count. The application uses all 13 listed numeric fields and four encoded category indicators, for 17 candidates. Dry and Amihan are reference categories. `date`, `year`, `month`, raw text fields and the arrival target are excluded from predictor screening.

## Corrections to the starter code

- Feature selection uses only fitting rows, rather than examining the test-period target before splitting.
- The validation tail is reserved before fitting either scaler. Validation also stays outside Spearman and VIF selection.
- VIF includes an intercept. A single survivor has VIF 1; empty selection produces an explicit finding rather than an arbitrary fallback.
- Training explicitly disables shuffling. Early stopping and model selection use validation loss only.
- Missing targets are excluded. MAPE excludes zero actuals and reports its sample count; undefined R² remains unavailable.
- Both baselines use the same test months. A six-month lookback still supports seasonal naive by using the observed arrival one year earlier in the original timeline.
- SHAP predictions and contributions are expressed in original arrival counts. The local waterfall reconstructs the actual model output; a signed bar chart alone is not labelled a waterfall.
- The forecast uses the final dataset window, rather than the last training sequence. For this file, the forecast month is **January 2026**, with January–December 2025 inputs.
- Changing an upstream stage invalidates dependent models and results. Editing forecast inputs hides the old result until recalculated.

With the default settings, the development/test split is 80/20. A validation tail of 15% is taken from development. The actual chronological periods are:

| Partition | Period | Rows | Usable windows, 12-month lookback |
| --- | --- | ---: | ---: |
| Fitting | January 2000–July 2017 | 211 | 196 |
| Validation | August 2017–September 2020 | 38 | 26 |
| Test | October 2020–December 2025 | 63 | 51 |

Every partition spends its first 12 rows building context. No window crosses a partition boundary, matching the lab's separate-window requirement. Scored test targets run from October 2021 to December 2025.

## Interface decisions

The Next.js interface uses a stable sidebar, one primary action per stage, clear prerequisites and progress messages. Charts occupy the main reading area; settings sit alongside them. The layout uses whitespace, rules and typography rather than a grid of decorative cards.

Body text, axes and table cells use readable system fonts. Navy and teal provide contrast on a warm light background; colour is accompanied by labels. Forms have visible labels, errors identify the next action, and inputs preserve their values after a forecast. Keyboard users can reach controls, skip navigation and inspect time-series points with arrow keys. Charts have corresponding value tables. The layout adapts to phone widths and respects reduced-motion preferences.

These are implemented HCI practices, not a claim that a usability study has been conducted.

## Verify and export

```bash
.venv/bin/python -m unittest discover -s tests -v
npm --prefix challenge/frontend run typecheck
npm --prefix challenge/frontend run build
```

Eight meaningful pipeline checks cover header handling, stable deduplication, causal filling, training-only selection/scalers, partition boundaries, next-month alignment, metric edge cases and stale-state invalidation.

**Download run** exports computed reports, cleaned data, Spearman results, test predictions and metrics. If a model has been trained, its Keras file and preprocessing bundle are included together. In Streamlit, click **Prepare downloadable run**, then download it. Prepare the download again after changing the run.

Browser refresh preserves a Next.js workspace while the backend process runs. Restarting the backend clears it. Streamlit state belongs to its browser session. Exports are snapshots; they do not automatically restore a running session.

## Interpretation limits

A baseline beating the LSTM is a valid result. Report the winner, the test period and errors rather than claiming that training guarantees a better forecast. The large retained arrival outlier can dominate MSE and target scaling; performance should be read alongside that data-quality issue.

Global SHAP importance is computed over the explained test sample, not every month or every possible forecast. It sums absolute contributions across each feature's lags and averages over explained windows. Local contributions sum signed lag effects. The dependence plot isolates the most recent lag in raw predictor units. Its pattern describes model behaviour, not a controlled causal relationship.

KernelSHAP approximates attribution and depends on its background/sample budget. Correlated inputs can produce combinations that do not occur naturally. Forecasts are point estimates, without calibrated prediction intervals. These are one-step forecasts using past environmental readings, not long-range predictions based on unknown future weather.

## Technical references

- [TensorFlow installation](https://www.tensorflow.org/install/pip)
- [Keras LSTM](https://keras.io/api/layers/recurrent_layers/lstm/)
- [Next.js installation](https://nextjs.org/docs/app/getting-started/installation)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [Streamlit documentation](https://docs.streamlit.io/)
- [SHAP KernelExplainer](https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html)

The primary course references are the supplied Module 2 lecture and Laboratory Exercise 1 PDF. The included CSV is the supplied course file; its tourism and climate provenance has not been independently verified.
