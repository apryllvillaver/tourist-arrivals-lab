"""Run from the project root: python -m uvicorn challenge.backend.main:app --port 8000"""
from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from core.pipeline import Config, PipelineError
from core.workspace import Workspace

app = FastAPI(title="Tourist Arrivals Forecasting Lab", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-Workspace-ID"])
workspaces: dict[str, Workspace] = {}
jobs: dict[str, dict] = {}
busy: set[str] = set()
guard = threading.RLock()
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="forecast-lab")


@app.exception_handler(PipelineError)
async def pipeline_error(_, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=422, content={"detail": str(exc)})


def workspace(x_workspace_id: str = Header(...)):
    if x_workspace_id not in workspaces:
        raise HTTPException(404, "This workspace has expired. Reload the page to start a new session.")
    return x_workspace_id, workspaces[x_workspace_id]


def available(pair):
    if pair[0] in busy:
        raise HTTPException(409, "A calculation is running. Wait for it to finish before changing the pipeline.")


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/api/workspaces")
def create_workspace():
    with guard:
        if len(workspaces) >= 32:
            raise HTTPException(503, "This local server has 32 open workspaces. Restart it to release old sessions.")
        identifier = str(uuid.uuid4())
        w = Workspace()
        w.load()
        workspaces[identifier] = w
    return {"workspace_id": identifier, "snapshot": w.snapshot()}


@app.get("/api/state")
def state(pair=Depends(workspace)):
    return pair[1].snapshot()


@app.post("/api/dataset")
async def upload_dataset(file: UploadFile = File(...), pair=Depends(workspace)):
    content = await file.read(10 * 1024 * 1024 + 1)
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "Use a CSV smaller than 10 MB.")
    with guard:
        available(pair)
        pair[1].load(content, file.filename or "dataset.csv")
    return pair[1].snapshot()


class FeatureConfig(BaseModel):
    train_ratio: float = Field(.80, ge=.60, le=.95)
    validation_ratio: float = Field(.15, ge=.10, le=.30)
    lookback: int = Field(12, ge=6, le=24)
    seed: int = 42


class TrainConfig(BaseModel):
    epochs: int = Field(100, ge=5, le=150)
    search: str = "full"


class ExplainConfig(BaseModel):
    samples: int = Field(10, ge=2, le=20)
    nsamples: int = Field(256, ge=128, le=1024)


class ForecastConfig(BaseModel):
    readings: list[list[float]]


@app.post("/api/clean")
def clean_data(pair=Depends(workspace)):
    with guard:
        available(pair)
        pair[1].run_clean()
    return pair[1].snapshot()


@app.post("/api/features")
def features(config: FeatureConfig, pair=Depends(workspace)):
    with guard:
        available(pair)
        pair[1].run_features(Config(**config.model_dump()))
    return pair[1].snapshot()


@app.post("/api/prepare")
def preparation(pair=Depends(workspace)):
    with guard:
        available(pair)
        pair[1].run_prepare()
    return pair[1].snapshot()


def start_job(pair, kind, action):
    with guard:
        available(pair)
        pair[1].need("prepare" if kind == "train" else "train")
        job_id = str(uuid.uuid4())
        jobs[job_id] = {"id": job_id, "workspace_id": pair[0], "kind": kind, "status": "queued", "progress": 0, "message": "Waiting for the calculation worker", "error": None}
        busy.add(pair[0])

    def update(value, message):
        jobs[job_id].update(progress=value, message=message)

    def run():
        jobs[job_id].update(status="running", message="Starting calculation")
        try:
            action(update)
            jobs[job_id].update(status="completed", progress=1, message="Calculation complete")
        except Exception as exc:
            jobs[job_id].update(status="failed", error=str(exc), message="Calculation stopped")
        finally:
            with guard:
                busy.discard(pair[0])
    executor.submit(run)
    return {"job_id": job_id}


@app.post("/api/train", status_code=202)
def training(config: TrainConfig, pair=Depends(workspace)):
    if config.search not in {"quick", "full"}:
        raise PipelineError("Choose quick or full search.")
    return start_job(pair, "train", lambda update: pair[1].run_train(config.epochs, config.search, update))


@app.post("/api/explain", status_code=202)
def explanations(config: ExplainConfig, pair=Depends(workspace)):
    return start_job(pair, "explain", lambda update: pair[1].run_explain(config.samples, config.nsamples, update))


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, pair=Depends(workspace)):
    job = jobs.get(job_id)
    if not job or job["workspace_id"] != pair[0]:
        raise HTTPException(404, "Calculation not found in this workspace.")
    return {k: v for k, v in job.items() if k != "workspace_id"}


@app.post("/api/evaluate")
def evaluation(pair=Depends(workspace)):
    with guard:
        available(pair)
        pair[1].run_evaluate()
    return pair[1].snapshot()


@app.post("/api/forecast")
def forecasting(config: ForecastConfig, pair=Depends(workspace)):
    with guard:
        available(pair)
        pair[1].run_forecast(config.readings)
    return pair[1].snapshot()


@app.get("/api/export")
def export(pair=Depends(workspace)):
    with guard:
        available(pair)
        content = pair[1].export()
    return Response(content, media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="forecast-run.zip"'})
