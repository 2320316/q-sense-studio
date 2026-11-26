import csv
import io
import shutil
import subprocess
import sys
import tempfile
import threading
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from filelock import FileLock, Timeout
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .experiment import Experiment, preview
from .library import (
    LAYERS,
    SIDECARS,
    TABLES,
    commit,
    data,
    inspect_files,
    normalize,
    thin,
)
from .sources import SOURCES, collect
from .storage import ROOT, catalog, folder, now, read, save
from .worker import update

STATIC = Path(__file__).parent / "static"


def supervise(stop):
    process = None
    while not stop.wait(0.6):
        if process and process.poll() is None:
            continue
        # A worker may outlive a restarted web service.
        try:
            with FileLock(str(ROOT / "training.lock"), timeout=0):
                records = catalog("runs", "state.json")
                for state in records:
                    if state["status"] in ("running", "pausing", "cancelling"):
                        root = folder("runs", state["id"])
                        update(
                            root,
                            status="cancelled"
                            if state["status"] == "cancelling"
                            else "paused",
                            note="Recovered at the last saved epoch",
                        )
        except Timeout:
            continue
        queued = sorted(
            (s for s in records if s["status"] == "queued"), key=lambda s: s["created"]
        )
        if queued:
            key = queued[0]["id"]
            with (folder("runs", key) / "worker.log").open("a") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "studio.worker", key],
                    cwd=Path(__file__).resolve().parents[1],
                    stdout=log,
                    stderr=log,
                )


@asynccontextmanager
async def lifespan(app):
    ROOT.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    thread = threading.Thread(target=supervise, args=(stop,), daemon=True)
    thread.start()
    yield
    stop.set()
    thread.join(2)


app = FastAPI(title="Q-Sense Studio", lifespan=lifespan)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"],
)


@app.middleware("http")
async def local_mutations(req: Request, call_next):
    if req.method not in ("GET", "HEAD", "OPTIONS"):
        origin = req.headers.get("origin")
        if req.headers.get("x-qsense") != "studio" or (
            origin and urlsplit(origin).netloc != req.headers.get("host")
        ):
            return JSONResponse(
                {"detail": "Local application request required"}, status_code=403
            )
    return await call_next(req)


@app.exception_handler(ValueError)
@app.exception_handler(KeyError)
def invalid(req, exc):
    return JSONResponse({"detail": str(exc)}, status_code=422)


@app.get("/api/library")
def library():
    return [
        {
            k: v
            for k, v in item.items()
            if k not in ("series", "paths", "grid", "preview", "mapping")
        }
        for item in catalog("datasets", "info.json")
    ]


@app.get("/api/library/{key}")
def dataset(key: str):
    return data(key)[1]


@app.post("/api/stage")
def upload(files: list[UploadFile] = File(...), source: str = Form("Local import")):
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for f in files:
            name = (f.filename or "").replace("\\", "/")
            parts = Path(name).parts
            if not parts or any(p in ("..", "/") or ":" in p for p in parts):
                raise ValueError("Invalid upload name")
            path = root.joinpath(*parts)
            if (
                path.name.startswith(".")
                or path.suffix.lower() not in TABLES | LAYERS | SIDECARS
            ):
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as out:
                shutil.copyfileobj(f.file, out)
        records = []
        for parent in sorted({p.parent for p in root.rglob("*") if p.is_file()}):
            records += inspect_files(parent, source)
        if not records:
            raise ValueError("No supported datasets found")
        return records


@app.delete("/api/stage/{key}")
def discard_preview(key: str):
    shutil.rmtree(folder("staged", key))
    return {"discarded": key}


@app.post("/api/stage/{key}/commit")
def accept(key: str):
    return commit(key)


@app.post("/api/library/{key}/map")
def map_columns(key: str, cfg: dict):
    return normalize(key, cfg)


@app.get("/api/sources")
def sources():
    return SOURCES


@app.post("/api/sources/{key}/collect")
def acquire(key: str):
    try:
        return collect(key)
    except Exception as exc:
        raise ValueError(
            f"Collection failed: {exc}. Retry or import the provider file."
        ) from exc


@lru_cache
def devices():
    import torch
    from .qt_model import Forecast, QuantumWeights, generated_parameters

    result = {}
    for dev, available in [
        ("cpu", True),
        ("mps", torch.backends.mps.is_available()),
        ("cuda", torch.cuda.is_available()),
    ]:
        if not available:
            result[dev] = "Unavailable"
            continue
        try:
            target = Forecast(2, 4).to(dev)
            generator = QuantumWeights(sum(p.numel() for p in target.parameters())).to(
                dev
            )
            loss = torch.func.functional_call(
                target,
                generated_parameters(target, generator),
                (torch.ones(2, 3, 2, device=dev),),
            ).sum()
            loss.backward()
            if not torch.isfinite(generator.angles.grad).all():
                raise ValueError("Non-finite gradient")
            result[dev] = "Ready"
        except Exception as exc:
            result[dev] = str(exc)[:220]
    return result


@app.get("/api/devices")
def hardware():
    return devices()


@app.post("/api/preview")
def preflight(c: Experiment):
    if devices().get(c.device) != "Ready":
        raise ValueError(
            "Selected training device did not pass its compatibility check"
        )
    return preview(c)


@app.post("/api/runs")
def enqueue(c: Experiment):
    plan = preflight(c)
    key = uuid4().hex
    root = ROOT / "runs" / key
    root.mkdir(parents=True)
    save(root / "config.json", c.model_dump(mode="json"))
    plan["source_sha256"] = data(c.dataset)[1]["sha256"]
    if c.warm_dataset:
        plan["warm_sha256"] = data(c.warm_dataset)[1]["sha256"]
    save(root / "plan.json", plan)
    state = dict(id=key, name=c.name, status="queued", created=now(), jobs=plan["jobs"])
    save(root / "state.json", state)
    return state


@app.get("/api/runs")
def runs():
    return catalog("runs", "state.json")


@app.get("/api/runs/{key}")
def inspect_run(key: str):
    root = folder("runs", key)

    def optional(name, default):
        path = root / name
        return read(path) if path.exists() else default

    return dict(
        state=read(root / "state.json"),
        config=read(root / "config.json"),
        plan=read(root / "plan.json"),
        progress=optional("progress.json", {}),
        results=[read(p) for p in sorted(root.glob("job-*/result.json"))],
        statistics=optional("statistics.json", {}),
    )


@app.post("/api/runs/{key}/{action}")
def control(key: str, action: str):
    root = folder("runs", key)
    transitions = {
        "pause": {"running": "pausing", "queued": "paused"},
        "resume": {"paused": "queued", "failed": "queued"},
        "cancel": {
            "running": "cancelling",
            "pausing": "cancelling",
            "queued": "cancelled",
            "paused": "cancelled",
        },
    }
    with FileLock(str(root / "state.lock")):
        st = read(root / "state.json")
        new = transitions.get(action, {}).get(st["status"])
        if not new:
            raise ValueError("This action is unavailable for the current state")
        st.update(status=new, error=None, ended=None)
        save(root / "state.json", st)
        return st


def job_folder(key, job):
    path = folder("runs", key) / f"job-{job:03}"
    if not (path / "result.json").exists():
        raise ValueError("Job has no finished result")
    return path


@app.get("/api/runs/{key}/series/{job}")
def points(key: str, job: int):
    with (job_folder(key, job) / "forecast.csv").open() as f:
        rows = list(csv.DictReader(f))
    return thin(rows, 1800)


@app.get("/api/runs/{key}/download/{artifact}")
def download(key: str, artifact: str, job: int = 0):
    root = folder("runs", key)
    if artifact in ("config.json", "statistics.json", "plan.json"):
        file = root / artifact
    elif artifact in ("forecast.csv", "inference.pt", "result.json", "trained.pt"):
        file = job_folder(key, job) / artifact
    elif artifact == "figure.svg":
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        from datetime import datetime

        jobroot = job_folder(key, job)
        meta = read(jobroot / "result.json")
        cfg = read(root / "config.json")
        with (jobroot / "forecast.csv").open() as f:
            rows = list(csv.DictReader(f))
        fig = Figure(figsize=(10, 4), layout="constrained")
        ax = fig.subplots()
        dates = [datetime.fromisoformat(r["timestamp"]) for r in rows]
        for field, color in [
            ("observed", "#16364b"),
            (meta["arm"], "#078569"),
            ("persistence", "#a26437"),
        ]:
            ax.plot(
                dates,
                [float(r[field]) for r in rows],
                label=field,
                color=color,
                linewidth=1.1,
            )
        ax.set_ylabel("Rainfall (mm)" if cfg["mode"] == "rainfall" else "Stage (cm)")
        ax.set_title(cfg["name"] + " · held-out forecasts")
        ax.legend(frameon=False)
        out = io.BytesIO()
        fig.savefig(out, format="svg")
        return Response(
            out.getvalue(),
            media_type="image/svg+xml",
            headers={"Content-Disposition": "attachment; filename=forecast.svg"},
        )
    else:
        raise ValueError("Unknown artifact")
    if not file.exists():
        raise ValueError("Artifact is not ready")
    return FileResponse(file, filename=file.name)


app.mount("/", StaticFiles(directory=STATIC, html=True), name="app")
