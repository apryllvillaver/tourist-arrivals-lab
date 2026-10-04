# MacBook setup and laboratory walkthrough

The project includes all code, the supplied full CSV, the graded Streamlit app, and the optional FastAPI + Next.js interface. You do not need to create every file manually or paste a long series of Python files into VS Code.

## 1. Download and open the project

1. Download `tourist-arrivals-lab.zip` from this conversation.
2. Double-click the ZIP in Finder. It creates a folder named `tourist-arrivals-lab`.
3. Move that folder into **Documents**.
4. Open **VS Code → File → Open Folder → Documents → tourist-arrivals-lab**.
5. Open **Terminal → New Terminal** inside VS Code. Run:

```bash
cd "$HOME/Documents/tourist-arrivals-lab"
pwd
```

The printed path should end with `/Documents/tourist-arrivals-lab`. If you chose a different folder, adjust that first `cd` command to its actual location. Use this new project folder rather than copying over an older project.

## 2. Check Python and Node

In the same VS Code terminal:

```bash
python3.11 --version
node --version
```

Use Python 3.11 for this project and Node.js 22 or newer. Next.js requires at least Node 20.9; Node 22 is the recommended setup here. The Python versions and libraries are pinned to keep TensorFlow compatible on Apple Silicon and Intel Macs.

If either command is missing, check Homebrew:

```bash
brew --version
```

If Homebrew works, install the missing software:

```bash
brew install python@3.11 node@22
export PATH="$(brew --prefix python@3.11)/bin:$(brew --prefix node@22)/bin:$PATH"
python3.11 --version
node --version
```

The `export` affects this terminal only. If a new terminal cannot find Node 22, run that export line again. If Homebrew itself is missing, follow the official installation command at [brew.sh](https://brew.sh/), then run the shell setup instructions printed at the end before continuing. VS Code is the editor; Python runs the model, and Node runs the optional frontend.

## 3. Install once

From the project root:

```bash
bash scripts/setup-mac.sh
```

The script creates `.venv`, installs the Python requirements and installs the exact frontend dependencies from the lockfile. The first install can take several minutes because TensorFlow is large. Wait until it prints **Setup finished**.

You do not need to activate `.venv` for the provided start scripts; they call its Python directly. You also do not need a database, Docker, an API key, a GPU plugin or a paid service.

## 4. Confirm the graded Streamlit app first

Run:

```bash
bash scripts/start-streamlit.sh
```

Open [http://localhost:8501](http://localhost:8501). The Home page should show the lab title and all eight stages. This is the setup confirmation required before proceeding through the lab.

Keep the terminal running. Move through the sidebar in order:

| Page | Action | Check before continuing |
| --- | --- | --- |
| Dataset | The included file loads automatically | 314 source records, 19 columns, 312 unique months and no timeline gaps |
| Clean | Run cleaning | Two duplicate records removed, three predictor readings filled, three missing targets left unknown; inspect three IQR flags |
| Features | Use 80% development, 15% validation tail and 12-month lookback; Run Spearman + VIF | Screening table, removal log and final VIF table appear |
| Prepare | Prepare sequences | Train `(196, 12, 8)`, validation `(26, 12, 8)`, test `(51, 12, 8)` with the supplied CSV and defaults |
| Train | Full search, maximum 100 epochs; Train & tune | Eight candidate results, selected settings and training/validation loss history appear |
| Evaluate | Score on the test set | LSTM, naive and seasonal-naive MAE/RMSE/MAPE/R² on identical held-out months |
| Explain | Ten test windows, SHAP budget 256; Compute SHAP values | Global importance, one-forecast waterfall and dependence plot appear |
| Forecast | Inspect the latest 12-month inputs; Forecast next month | A prediction for January 2026 appears; editing an input requires recalculation |

Training and SHAP take longer than the data checks. Leave the tab and server open while their progress updates run. Values may differ slightly across machines. Report your own computed metrics.

With the provided defaults, the selected features are quarter, holiday peak, rainy days, humidity, maximum typhoon wind, storm-warning days, PM2.5 and the Habagat indicator. Other splits can select a different set. Do not copy the lecture's illustrative numbers into your own results.

The February 2001 arrival value is **34,240,153** and is flagged. Keep it unless you obtain source evidence for a correction. Its size can strongly affect the model, especially target scaling and MSE. A poor LSTM score is a result to explain, not permission to edit the dataset until it wins.

## 5. Open the polished optional frontend

The baseline remains available. For the optional challenge, open two additional VS Code terminals using the **+** button in the terminal panel.

**Backend terminal:**

```bash
cd "$HOME/Documents/tourist-arrivals-lab"
bash scripts/start-backend.sh
```

Wait for **Application startup complete**. API docs are at [http://localhost:8000/docs](http://localhost:8000/docs).

**Frontend terminal:**

```bash
cd "$HOME/Documents/tourist-arrivals-lab"
bash scripts/start-frontend.sh
```

Wait for **Ready**, then open [http://localhost:3000](http://localhost:3000).

The optional UI starts a separate workspace, so follow its sidebar from Dataset again. It uses the same algorithm and defaults as Streamlit. It offers charts with value inspection, progress updates, labelled forms, a clearer layout and responsive sizing.

| Running process | Browser address | Terminal to keep open |
| --- | --- | --- |
| Streamlit baseline | `localhost:8501` | Streamlit terminal |
| FastAPI backend | `localhost:8000/docs` | Backend terminal |
| Next.js frontend | `localhost:3000` | Frontend terminal |

You can stop Streamlit after confirming and completing the baseline if you only want the optional UI running. FastAPI and Next.js must both remain running for the optional UI.

## 6. Save evidence and write the interpretation

In Next.js, click **Download run** after finishing the stages. In Streamlit, click **Prepare downloadable run**, then **Download saved run**. Prepare a new download whenever the results change.

The ZIP contains the computed reports, cleaned data, Spearman table and, after evaluation, the metrics and test predictions. After training it also includes the model and the matching scalers/feature order. Save the ZIP and useful screenshots before restarting the server.

Write one short interpretation using your actual run. Cover these connected points:

1. What cleaning changed, what remained uncertain and why outliers were retained.
2. Which predictors survived Spearman and VIF, with the fitting period stated.
3. Which model had the lowest test MAE, and whether the LSTM beat both baselines.
4. Where the forecasts missed the observed pattern, rather than only stating an overall score.
5. Which SHAP predictors mattered most in the explained sample and how the waterfall moved one prediction from the base value.
6. A cautious connection to seasonal/climate patterns. Describe the model's reliance on predictors, without claiming a proven cause.

Use your results to complete a paragraph like this, then remove the brackets:

> The app removed two duplicate records, filled three missing predictor readings from past months, and left three missing arrival targets unknown. I retained the IQR-flagged values because I had no source evidence for a correction. Using the stated chronological split, the LSTM recorded a test MAE of [value], compared with [value] for naive and [value] for seasonal naive. [Model] had the lowest average error. The model's largest misses occurred in [months/pattern], which suggests [careful interpretation]. Across [number] explained test windows, SHAP ranked [predictor] highest. For [forecast month], it [increased/decreased] the prediction relative to the background reference. This describes what the model relied on, rather than proving a causal effect on tourism. The large retained arrival outlier and [other observed limit] should be considered before using the forecast for a decision.

The forecast month is January 2026 because the dataset ends in December 2025. Do not present it as a forecast for the current month if you run the app later.

## 7. Record the demonstration, at most three minutes

Finish training and SHAP before recording. Demonstrate the saved results rather than spending the video waiting for calculations. A useful sequence is:

| Time | Show and explain |
| --- | --- |
| 0:00–0:15 | Open the Streamlit Home and briefly confirm the graded baseline runs; introduce the optional interface |
| 0:15–0:40 | Dataset and Clean: full monthly timeline, duplicates, missing values and retained outlier flags |
| 0:40–1:05 | Features and Prepare: training-only Spearman/VIF, separate partitions and sequence shapes |
| 1:05–1:25 | Train: selected settings, validation-based candidate comparison and learning curves |
| 1:25–1:55 | Evaluate: compare LSTM with both baselines; explain your actual winner and error |
| 1:55–2:30 | Explain: global importance, one waterfall and dependence plot; clarify that SHAP is not causation |
| 2:30–2:55 | Forecast: make one prediction and demonstrate an edited reading; mention the dataset-based forecast date |
| 2:55–3:00 | Show that the results can be downloaded |

Keep the same dataset and settings throughout the demonstration. Avoid rereading every table entry aloud.

## 8. Upload the repository and submit

Add your name/course details to the README as appropriate. Keep the code, requirements, frontend lockfile, dataset and documentation in the repository. The included `.gitignore` excludes `.venv`, `node_modules`, `.next` and generated caches.

Run from the project root:

```bash
git init
git add .
git status
git commit -m "Build tourist arrivals forecasting lab and optional frontend"
git branch -M main
```

On GitHub, create an empty repository. Do not add a separate README during repository creation because this project already has one. Copy your new repository's HTTPS URL, then run:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Replace the example URL with your actual repository URL. If a remote is already set, inspect it with `git remote -v` before adding another. Follow GitHub's sign-in prompt if required. Check the repository page to confirm the files uploaded.

Upload the demo video to a service your instructor can open. Submit **the GitHub repository link and the demo video link** in the lab's **Submit your deliverables** form. The video must be **three minutes maximum**. These are the explicit submission requirements in the supplied lab.

## 9. Stop safely and restart later

Save your exports first. In each terminal running Streamlit, FastAPI or Next.js, press **Control + C**. Wait for the normal shell prompt before closing VS Code. Closing a browser tab alone does not stop its server.

Next time, open the project and run the same start scripts. You do not need to run setup again unless dependencies change. New sessions need the pipeline calculations rerun; downloaded results are saved evidence rather than automatically restored sessions.

## If something fails

| What you see | What to do |
| --- | --- |
| `python3.11: command not found` | Install Python 3.11, then check its command before running setup |
| `node: command not found` or unsupported version | Install Node 22 and run the Homebrew PATH line in that terminal |
| `.venv/bin/python` missing | Check `pwd`, then run `bash scripts/setup-mac.sh` from this project root |
| File or module not found | The terminal must be in the project root when starting the app |
| Backend could not be reached | Start the backend terminal and wait for startup; both backend and frontend must run |
| Address already in use | Stop the older server using Control+C in its terminal, then restart this one |
| Later page requests an earlier step | Run the pipeline in order in the current interface/session |
| Few/no usable windows | Lower the training share or lookback; review source gaps; rerun Features then Prepare |
| Poor LSTM score or negative R² | Inspect the data and actual misses; report the result rather than changing the test score |
| No predictor passes the filter | Report that finding; do not silently reduce the required thresholds |

Optional verification commands:

```bash
.venv/bin/python -m unittest discover -s tests -v
npm --prefix challenge/frontend run typecheck
npm --prefix challenge/frontend run build
```

Primary technical references: [TensorFlow](https://www.tensorflow.org/install/pip), [Next.js](https://nextjs.org/docs/app/getting-started/installation), [SHAP](https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html). Course procedure: the supplied Module 2 and Laboratory Exercise 1 PDFs.
