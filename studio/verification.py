"""Replay an exported model on its original test fold and stress it on repeated blocks."""

import csv
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, Field

from .experiment import Experiment, prepare
from .library import data
from .qt_model import Forecast
from .storage import digest, now, read, save


class TrialSettings(BaseModel):
    rounds: int = Field(default=20, ge=1, le=100)
    max_mae: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


@torch.inference_mode()
def verify(job: Path, settings: TrialSettings):
    root = job.parent
    result = read(job / "result.json")
    config = Experiment(**read(root / "config.json"))
    plan = read(root / "plan.json")
    if digest(data(config.dataset)[0]) != plan["source_sha256"]:
        raise ValueError("Dataset checksum differs from the training manifest")
    p = prepare(config, result["fold"])
    raw, truth, ids = p["raw"][-1]
    bundle = torch.load(job / "inference.pt", map_location="cpu", weights_only=True)
    if bundle["features"] != p["names"] or bundle["lookback"] != config.lookback:
        raise ValueError("Exported model input schema differs from the test fold")
    model = Forecast(len(p["names"]), bundle["hidden"]).eval()
    model.load_state_dict(bundle["state_dict"])
    scale = bundle["scale"]
    x = torch.tensor(
        (raw - scale["feature_mean"]) / scale["feature_std"], dtype=torch.float32
    )

    def predict(values):
        preds = torch.cat([model(b) for b in values.split(128)]).numpy().astype(float)
        return preds * scale["target_std"] + scale["target_mean"]

    clean = predict(x)
    with (job / "forecast.csv").open() as f:
        rows = list(csv.DictReader(f))
    if [r["timestamp"] for r in rows] != [p["dates"][i].isoformat() for i in ids]:
        raise ValueError("Archived forecast timestamps differ from the test fold")
    replay = bool(
        np.allclose(
            clean, [float(r[result["arm"]]) for r in rows], atol=2e-4, rtol=2e-4
        )
    )
    labels_match = bool(
        np.allclose(truth, [float(r["observed"]) for r in rows], atol=1e-8, rtol=1e-8)
    )
    baseline = p["origin"][-1]
    block = min(32, len(x))
    rng = np.random.default_rng(settings.seed)
    rounds = []
    for _ in range(settings.rounds):
        start = int(rng.integers(len(x) - block + 1))
        section = slice(start, start + block)
        batch = x[section]
        # Simulate late delivery by carrying the entire previous feature vector forward.
        stale = batch.clone()
        stale[:, -1] = stale[:, -2]
        stressed = predict(stale)
        repeated = predict(batch)
        stable = bool(np.allclose(repeated, clean[section], atol=2e-4, rtol=2e-4))
        finite = bool(np.isfinite(stressed).all())
        rounds.append(
            dict(
                start=rows[start]["timestamp"],
                end=rows[start + block - 1]["timestamp"],
                repeatable=stable,
                finite=finite,
                clean_mae=float(np.abs(clean[section] - truth[section]).mean())
                if np.isfinite(clean[section]).all()
                else None,
                stale_mae=float(np.abs(stressed - truth[section]).mean())
                if finite
                else None,
            )
        )
    valid = np.isfinite(clean).all()
    comparable = np.isfinite(baseline)

    def mae(values, actual):
        return float(np.abs(values - actual).mean())

    report = dict(
        created=now(),
        model_sha256=digest(job / "inference.pt"),
        source_sha256=plan["source_sha256"],
        job=result["job"],
        fold=result["fold"],
        arm=result["arm"],
        seed=settings.seed,
        count=len(truth),
        unit=bundle["unit"],
        exported_matches=replay,
        labels_match=labels_match,
        passed=bool(
            valid
            and replay
            and labels_match
            and all(r["repeatable"] and r["finite"] for r in rounds)
        ),
        mae=mae(clean, truth) if valid else None,
        paired_count=int(comparable.sum()),
        paired_mae=mae(clean[comparable], truth[comparable])
        if valid and comparable.any()
        else None,
        baseline_mae=mae(baseline[comparable], truth[comparable])
        if comparable.any()
        else None,
        max_mae=settings.max_mae,
        within_limit=sum(
            r["stale_mae"] is not None and r["stale_mae"] <= settings.max_mae
            for r in rounds
        )
        if settings.max_mae is not None
        else None,
        block=block,
        rounds=rounds,
    )
    save(job / "verification.json", report)
    return report
