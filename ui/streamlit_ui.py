from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from core.pipeline import Config, LABELS, PipelineError
from core.workspace import Workspace

PAGE_FILES = ["pages/1_Dataset.py", "pages/2_Clean.py", "pages/3_Features.py", "pages/4_Prepare.py",
              "pages/5_Train.py", "pages/6_Evaluate.py", "pages/7_Explain.py", "pages/8_Forecast.py"]
STAGES = ["dataset", "clean", "features", "prepare", "train", "evaluate", "explain", "forecast"]
TITLES = ["Dataset summary", "Clean the monthly record", "Select useful predictors", "Prepare without leakage",
          "Train and tune the LSTM", "Evaluate against baselines", "Explain with SHAP", "Forecast the next dataset month"]


def table(rows):
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def attempt(action):
    try:
        action()
        st.success("Calculation complete. Review the result below.")
    except (PipelineError, ValueError, RuntimeError) as exc:
        st.error(str(exc))


def progress():
    bar = st.progress(0, text="Starting calculation…")
    return lambda fraction, message: bar.progress(min(float(fraction), 1.0), text=message)


def waterfall(local):
    running = local["base"]
    rows = [{"term": "Base", "start": 0, "end": running, "kind": "Total", "order": 0, "value": running}]
    for i, item in enumerate(local["contributions"], 1):
        previous = running
        running += item["value"]
        rows.append({"term": LABELS.get(item["feature"], item["feature"]), "start": min(previous, running),
                     "end": max(previous, running), "value": item["value"], "kind": "Adds" if item["value"] >= 0 else "Subtracts", "order": i})
    rows.append({"term": "Prediction", "start": 0, "end": local["prediction"], "kind": "Total", "order": len(rows), "value": local["prediction"]})
    chart = alt.Chart(pd.DataFrame(rows)).mark_bar().encode(
        x=alt.X("term:N", sort=alt.SortField("order"), title=None, axis=alt.Axis(labelAngle=-35)),
        y=alt.Y("start:Q", title="Arrivals"), y2="end:Q",
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Total", "Adds", "Subtracts"], range=["#193746", "#006C67", "#B55026"])),
        tooltip=["term:N", alt.Tooltip("value:Q", format=",.2f")]).properties(height=320)
    st.altair_chart(chart, use_container_width=True)
    st.caption("Each floating bar starts where the previous term ended. Feature contributions sum over all lookback months.")


def render(stage):
    st.set_page_config(page_title="Arrivals | Forecasting Lab", page_icon="🧭", layout="wide")
    st.markdown("""<style>
    .block-container{padding-top:2rem;max-width:1250px;padding-bottom:3rem}
    h1{font-weight:650;letter-spacing:-1px;font-size:2.25rem}
    h2{font-size:1.3rem;letter-spacing:-.25px}
    [data-testid="stMetric"]{border-top:1px solid #D8DFD9;padding-top:12px}
    [data-testid="stMetricValue"]{font-size:1.8rem}
    .stButton>button{border-radius:5px;min-height:42px}
    a{color:#006C67}
    </style>""", unsafe_allow_html=True)
    if "workspace" not in st.session_state:
        st.session_state.workspace = Workspace()
    w = st.session_state.workspace
    with st.sidebar:
        st.markdown("**Arrivals / Philippines**")
        st.caption("Forecasting and explainable AI")
        st.caption(f"{len(w.reports)} of 8 stages complete. Run the pages in order.")
        if w.reports:
            if st.button("Prepare downloadable run"):
                st.session_state.export_bytes = w.export()
            if "export_bytes" in st.session_state:
                st.download_button("Download saved run", st.session_state.export_bytes, "forecast-run.zip", "application/zip")
                st.caption("Click Prepare downloadable run again after making changes.")
    if stage == "home":
        st.caption("MODULE 2 / LABORATORY EXERCISE 1")
        st.title("Philippine tourist arrivals")
        st.write("Build one connected forecast, from the source record to a prediction you can inspect.")
        st.divider()
        left, right = st.columns([2, 1], gap="large")
        with left:
            st.subheader("Follow the monthly data through the pipeline")
            st.write("Start with the CSV. Review duplicates and missing readings, select useful predictors, prepare ordered windows, train an LSTM, compare simple baselines, and explain the result with SHAP.")
            st.page_link(PAGE_FILES[0], label="Begin with Dataset", icon=":material/arrow_forward:")
            table([{"Step": f"{i+1:02}", "Stage": title, "Status": "Complete" if key in w.reports else "To do"} for i,(key,title) in enumerate(zip(STAGES,TITLES))])
        with right:
            st.subheader("Your lab workspace")
            st.write("The supplied full CSV is included. Its two export title rows are handled automatically.")
            st.caption("Each stage is kept in this browser session. Restarting Streamlit starts a new run. Download results before closing your work.")
            st.caption("The optional FastAPI + Next.js interface uses the same Python pipeline, with a separate workspace.")
        return
    index = STAGES.index(stage)
    st.caption(f"{index+1:02} / FORECASTING PIPELINE")
    st.title(TITLES[index])
    required = [None, "dataset", "clean", "features", "prepare", "train", "train", "train"][index]
    if required and required not in w.reports:
        st.info(f"Complete the {required.title()} page first in this session.")
        st.page_link(PAGE_FILES[STAGES.index(required)], label=f"Go to {required.title()}")
        st.stop()

    if stage == "dataset":
        if w.raw is None:
            attempt(w.load)
        uploaded = st.file_uploader("Replace the supplied CSV", type="csv")
        if uploaded is not None and st.button("Load this CSV", type="primary"):
            attempt(lambda: w.load(uploaded.getvalue(), uploaded.name))
        if "dataset" not in w.reports:
            st.stop()
        r = w.reports[stage]
        cols = st.columns(4)
        for col, title, value in zip(cols, ["Source records", "Source columns", "Unique months", "Duplicate records"], [r["rows"],r["columns"],r["unique_months"],r["duplicate_records"]]):
            col.metric(title, value)
        st.caption(f"{r['start']} to {r['end']}. Skipped {r['header_rows_skipped']} export title rows.")
        if r["missing_months"]:
            st.error(f"Missing months: {', '.join(r['missing_months'])}. Resolve the source timeline before proceeding.")
        else:
            st.success("The monthly sequence has no gaps.")
        recent = st.toggle("Show the latest five years", value=True)
        plot = w.raw.tail(60) if recent else w.raw
        st.line_chart(plot.set_index("date")[["arrivals"]], color="#006C67")
        table(r["dtypes"])
        with st.expander("Inspect source records"):
            st.dataframe(w.raw, hide_index=True, use_container_width=True)

    if stage == "clean":
        st.write("Keep the first entry for a repeated date. Fill predictor gaps from past readings only, with a two-month limit. Keep unknown arrival targets unknown.")
        if st.button("Run cleaning", type="primary"):
            attempt(w.run_clean)
        if stage in w.reports:
            r=w.reports[stage]
            a,b,c=st.columns(3);a.metric("Unique months",r["rows"]);b.metric("Duplicates removed",r["duplicates_removed"]);c.metric("Predictor readings filled",len(r["fill_log"]))
            st.subheader("Missing predictor readings")
            table(r["fill_log"])
            st.caption(f"{r['missing_after']['arrivals']} missing arrival targets will be excluded when constructing usable target windows.")
            st.subheader("Outlier review")
            st.warning(r["outlier_policy"])
            st.caption(f"Full-dataset IQR diagnostic only. Bounds: {r['iqr']['lower']:,.0f} to {r['iqr']['upper']:,.0f}. No model step uses the flag to alter data.")
            table(r["outliers"])
            st.caption(r["encoding"])
            with st.expander("Inspect duplicate removal and remaining gaps"):
                table(r["duplicate_log"])
                table([{"column": k,"missing": v} for k,v in r["missing_after"].items()])

    if stage == "features":
        st.write("Keep |ρ| > 0.10 with p < 0.05, then remove the largest VIF until every retained VIF is below 5. Fitting rows alone make these decisions.")
        a,b,c=st.columns(3)
        ratio=a.slider("Development / training share",.60,.95,w.config.train_ratio,.05)
        val=b.selectbox("Validation tail of development", [.10,.15,.20,.25,.30], index=[.10,.15,.20,.25,.30].index(w.config.validation_ratio),format_func=lambda v:f"{v:.0%}")
        L=c.selectbox("Lookback months",[6,12,18,24], index=[6,12,18,24].index(w.config.lookback))
        if st.button("Run Spearman + VIF",type="primary"):
            attempt(lambda:w.run_features(Config(ratio,val,L)))
        if stage in w.reports:
            r=w.reports[stage]
            if Config(ratio,val,L)!=w.config:
                st.warning("The controls differ from the saved selection. Run selection again to apply them.")
            st.caption(f"Fitting data: {r['fitting_start']} to {r['fitting_end']}, {r['fitting_rows']} rows.")
            st.subheader("Spearman screening");table(r["results"])
            st.subheader("VIF removal log")
            table([{"iteration": row["iteration"],"dropped":row["dropped"],"VIF":"infinite" if row["infinite"] else row["vif"]} for row in r["vif_log"]])
            st.subheader("Selected features and final VIF");table([{"feature":k,"VIF":v} for k,v in r["final_vif"].items()])
            st.caption("VIF includes an intercept. Dry and Amihan are reference categories. No target, year or raw text category is fed into the predictor selection.")

    if stage == "prepare":
        st.write("Split before scaling. Fit feature and target scalers only on the fitting period, then window train, validation and test separately.")
        st.caption(f"Saved settings: {w.config.train_ratio:.0%} development / {1-w.config.train_ratio:.0%} test; validation tail {w.config.validation_ratio:.0%}; lookback {w.config.lookback} months. Change them on Features.")
        if st.button("Prepare sequences",type="primary"):
            attempt(w.run_prepare)
        if stage in w.reports:
            r=w.reports[stage]
            table([{**p,"shape":" × ".join(map(str,p["shape"]))} for p in r["partitions"]])
            st.subheader("Feature scaler ranges");table(r["scaler_ranges"])
            st.caption(r["policy"])
            st.caption("A transformed later reading can exceed 1. Do not refit a scaler to hide that extrapolation.")

    if stage == "train":
        a,b=st.columns(2)
        search=a.selectbox("Candidate search",["full","quick"],format_func=lambda v:"Full search: 8 candidates" if v=="full" else "Setup check: 2 candidates")
        epochs=b.number_input("Maximum epochs per candidate",5,150,100)
        if st.button("Train & tune",type="primary"):
            with st.spinner("Fitting on training windows and choosing by validation error…"):
                attempt(lambda:w.run_train(int(epochs),search,progress()))
        if stage in w.reports:
            r=w.reports[stage]
            st.write("Selected hyperparameters:",r["best_params"])
            st.caption(f"Validation MSE: {r['validation_mse']:.6f}; seed {r['seed']}. No test-set values choose the winner.")
            st.line_chart(pd.DataFrame(r["history"]).set_index("epoch")[["training","validation"]],color=["#006C67","#B55026"])
            table(r["trials"])
            st.caption(r["policy"])

    if stage == "evaluate":
        st.write("Score the selected LSTM, last-month naive and same-month-last-year seasonal naive on identical held-out target months.")
        if st.button("Score on the test set",type="primary"):
            attempt(w.run_evaluate)
        if stage in w.reports:
            r=w.reports[stage];table(r["metrics"])
            st.line_chart(pd.DataFrame(r["predictions"]).set_index("date")[["actual","lstm","naive","seasonal"]])
            if r["lowest_mae"]!="LSTM":
                st.warning(f"{r['lowest_mae']} has the lowest MAE in this run. Report that outcome honestly; the lab does not require the LSTM to win.")
            if r["negative_predictions"]:
                st.warning(f"{r['negative_predictions']} negative LSTM outputs were retained in scoring. They are unusable arrival forecasts.")
            st.caption(r["policy"])
            with st.expander("Inspect each test month"):
                table(r["predictions"])

    if stage == "explain":
        a,b=st.columns(2)
        n=a.number_input("Test windows to explain",2,20,10)
        masks=b.selectbox("SHAP sampling budget",[128,256,512,1024],index=1)
        if st.button("Compute SHAP values",type="primary"):
            with st.spinner("Explaining the trained model in original arrival units…"):
                attempt(lambda:w.run_explain(int(n),masks,progress()))
        if stage in w.reports:
            r=w.reports[stage]
            st.subheader("Global feature importance")
            st.caption(f"Average total absolute SHAP contribution over {r['lookback']} lags, across {r['samples']} explained test windows.")
            st.bar_chart(pd.DataFrame(r["importance"]).set_index("feature"),horizontal=True,color="#006C67")
            chosen=st.selectbox("Forecast to explain",range(len(r["local"])),format_func=lambda i:r["local"][i]["date"])
            local=r["local"][chosen]
            st.subheader("Waterfall for one forecast")
            waterfall(local)
            st.caption(f"Base {local['base']:,.2f} + signed contributions = {local['prediction']:,.2f} arrivals. Reconstruction residual {local['residual']:.6f}.")
            table(local["contributions"])
            st.subheader(f"Dependence: {LABELS.get(r['top_feature'],r['top_feature'])}")
            st.scatter_chart(pd.DataFrame(r["dependence"]),x="value",y="shap",color="#006C67")
            st.caption("Horizontal axis uses the raw most-recent predictor reading. Vertical axis is that lag's SHAP contribution in arrival units.")
            st.caption(r["policy"])

    if stage == "forecast":
        window=w.snapshot()["window"]
        st.write(f"Forecast month: **{window['date']}**, the month after the latest dataset record. This date is not tied to the current calendar month.")
        st.caption("Edit the latest input history in original units. The month column is fixed, and selected feature order is preserved.")
        initial=pd.DataFrame(window["values"],columns=w.features)
        initial.insert(0,"month",window["history_dates"])
        edited=st.data_editor(initial,num_rows="fixed",disabled=["month"],hide_index=True,use_container_width=True,key="forecast_editor")
        if st.button("Forecast next month",type="primary"):
            attempt(lambda:w.run_forecast(edited[w.features].to_numpy().tolist()))
        if stage in w.reports:
            r=w.reports[stage]
            if edited[w.features].to_numpy().tolist()!=r["readings"]:
                st.info("Inputs have changed. Run the forecast again to update the result.")
            else:
                st.metric(f"Predicted arrivals for {r['date']}",f"{r['arrivals']:,.0f}")
                if r["negative"]:
                    st.warning("This negative output is not a usable arrival forecast. Report the model limitation.")
                if r["outside_training_range"]:
                    st.warning("Some inputs exceed fitted training ranges: "+", ".join(v["feature"] for v in r["outside_training_range"])+". Interpret this extrapolation cautiously.")
                st.caption(r["policy"])
    if stage in w.reports and index<7:
        st.divider()
        st.page_link(PAGE_FILES[index+1],label=f"Continue to {STAGES[index+1].title()}",icon=":material/arrow_forward:")
