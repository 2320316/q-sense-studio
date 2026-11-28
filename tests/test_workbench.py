"""Research protocol and local browser API acceptance tests; all observations are fixtures."""

import csv
import io
import json
from unittest.mock import patch
import math
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

sandbox = tempfile.TemporaryDirectory()
os.environ["QSENSE_HOME"] = sandbox.name
from fastapi.testclient import TestClient
from studio.engine import Halt, train_arm
from studio.experiment import Experiment, partition, prepare, preview
from studio.library import data, commit, inspect_files, normalize
from studio.server import app
from studio.quantum_trace import frames_since
from studio.statistics import holm, indicators, report
from studio.storage import ROOT, read, save
from studio.worker import execute


def make_records(path, mode, n, start):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "timestamp",
                "site_id",
                "qa_flag",
                "stage_cm",
                "rain_mm",
                "temp_c",
                "rh_pct",
                "pressure_hpa",
            ]
        )
        for i in range(n):
            ts = start + timedelta(minutes=(10 if mode == "local" else 60) * i)
            for j, site in enumerate(
                ["C1", "C2", "C3", "C4"] if mode == "local" else ["basin"]
            ):
                w.writerow(
                    [
                        ts.isoformat(),
                        site,
                        "STALE" if i == 501 else "VALID",
                        15 + j + 3 * math.sin(i / 16),
                        i % 3,
                        29,
                        70,
                        1000,
                    ]
                )
    return commit(inspect_files(path.parent)[0]["id"])["id"]


class Workbench(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        parent = ROOT / "fixtures"
        for sub in ("local", "hourly"):
            (parent / sub).mkdir(parents=True)
        cls.local = make_records(
            parent / "local" / "s.csv", "local", 2500, datetime(2025, 1, 1)
        )
        cls.hour = make_records(
            parent / "hourly" / "h.csv", "hourly", 350, datetime(2023, 1, 1)
        )
        cls.config = Experiment(
            name="Fixture",
            mode="local",
            dataset=cls.local,
            site="C2",
            hidden=4,
            lookback=6,
            epochs=1,
            batch=128,
            arms=["qt", "classical", "compact", "softmax"],
            folds=2,
            seeds=[42],
            warm_dataset=cls.hour,
            warm_site="basin",
            horizon=1,
        )
        cls.client = TestClient(app, headers={"x-qsense": "studio"})

    def test_splits_and_transfer_layout(self):
        p = prepare(self.config)
        w = prepare(self.config, warm=True)
        self.assertEqual(p["names"], w["names"])
        self.assertIn("C3_lag1_stage_cm", p["names"])
        x, y, ids = p["raw"][0]
        i = ids[0] - p["horizon"]
        self.assertAlmostEqual(x[0, -1, -2], 17 + 3 * math.sin((i - 1) / 16))
        self.assertFalse(any(501 <= j - p["horizon"] <= 644 for j in ids))
        for (_, _, a), (_, _, b) in zip(p["raw"], p["raw"][1:]):
            self.assertLess(a.max() + 143 + self.config.lookback, b.min())
        self.assertTrue(np.all(w["raw"][0][0][:, :, 2:] == 0))
        bad = self.config.model_copy(
            update={"val_start": "2030-01-01", "test_start": "2031-01-01"}
        )
        with self.assertRaises(ValueError):
            prepare(bad)

    def test_scratch_transfer_arms_and_exports(self):
        c = self.config
        plan = preview(c)
        self.assertEqual(plan["jobs"], 16)
        root = ROOT / "runs" / "integration"
        root.mkdir(parents=True)
        save(root / "config.json", c.model_dump(mode="json"))
        plan["source_sha256"] = data(c.dataset)[1]["sha256"]
        save(root / "plan.json", plan)
        save(
            root / "state.json",
            dict(
                id=root.name,
                status="queued",
                created="2026-01-01",
                name=c.name,
                jobs=16,
            ),
        )
        execute(root.name)
        state = read(root / "state.json")
        self.assertEqual(state["status"], "completed", state)
        self.assertEqual(len(list(root.glob("job-*/inference.pt"))), 16)
        stats = read(root / "statistics.json")
        self.assertEqual(stats["warning_tier"], "Regression only")
        self.assertTrue(stats["comparisons"])
        for file in (
            "forecast.csv",
            "figure.svg",
            "inference.pt",
            "statistics.json",
            "config.json",
            "plan.json",
        ):
            self.assertEqual(
                self.client.get("/api/runs/integration/download/" + file).status_code,
                200,
            )

        job = root / "job-000"
        trace = read(job / "diagnostics.json")
        self.assertEqual(len(trace["gates"]), c.lookback)
        self.assertEqual(len(trace["quantum"]["angles"]), c.depth)
        url = "/api/runs/integration/quantum/0"
        frames = self.client.get(url).json()
        self.assertEqual([f["epoch"] for f in frames], [1])
        self.assertEqual(frames[0]["angles"], trace["quantum"]["angles"])
        self.assertEqual(self.client.get(url + "?after=1").json(), [])
        self.assertEqual(
            self.client.get("/api/runs/integration/quantum/-1").status_code, 422
        )
        from studio.qt_model import QuantumWeights

        generator = QuantumWeights(plan["weights"], c.depth, c.width)
        generator.load_state_dict(torch.load(job / "trained.pt", weights_only=True))
        probs = generator.probabilities().detach().numpy()
        for q, reported in enumerate(frames[0]["one_probabilities"]):
            self.assertAlmostEqual(
                reported,
                float(probs[(np.arange(len(probs)) >> q) % 2 == 1].sum()),
                places=5,
            )
        exported = torch.load(job / "inference.pt", weights_only=True)
        from studio.qt_model import Forecast

        model = Forecast(len(exported["features"]), exported["hidden"])
        model.load_state_dict(exported["state_dict"])
        first = prepare(c)["raw"][0][0][:1]
        scale = exported["scale"]
        x = torch.tensor(
            (first - scale["feature_mean"]) / scale["feature_std"], dtype=torch.float32
        )
        self.assertAlmostEqual(trace["output"], model(x).item(), places=5)
        endpoint = "/api/runs/integration/jobs/0/verify"
        args = {"rounds": 4, "seed": 7, "max_mae": 0}
        check = self.client.post(endpoint, json=args).json()
        self.assertTrue(check["passed"], check)
        self.assertEqual(len(check["rounds"]), 4)
        self.assertEqual(check["within_limit"], 0)
        self.assertEqual(
            check["rounds"], self.client.post(endpoint, json=args).json()["rounds"]
        )
        self.assertEqual(
            self.client.post(endpoint, json={"rounds": 0}).status_code, 422
        )
        self.assertEqual(
            self.client.get("/api/runs/integration/download/verification.json").json()[
                "seed"
            ],
            7,
        )
        path = job / "forecast.csv"
        with path.open() as f:
            archived = list(csv.DictReader(f))
        archived[0][check["arm"]] = "100000"
        with path.open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=archived[0].keys())
            writer.writeheader()
            writer.writerows(archived)
        self.assertFalse(self.client.post(endpoint, json=args).json()["passed"])
        child = self.client.post("/api/runs/integration/retrain", json={}).json()
        self.assertEqual(child["parent"], "integration")
        self.assertEqual(
            read(ROOT / "runs" / child["id"] / "config.json"),
            read(root / "config.json"),
        )

    def test_gridded_rain_collectors(self):
        query = dict(
            latitude=14.08, longitude=121.15, start="2024-01-01", end="2024-01-01"
        )
        cases = [
            (
                "power",
                {
                    "properties": {
                        "parameter": {
                            "PRECTOTCORR": {"2024010100": 1.5, "2024010101": -999}
                        }
                    },
                    "parameters": {"PRECTOTCORR": {"units": "mm/hour"}},
                    "header": {"fill_value": -999, "time_standard": "UTC"},
                },
            ),
            (
                "era5",
                {
                    "hourly_units": {"precipitation": "mm"},
                    "utc_offset_seconds": 0,
                    "hourly": {
                        "time": ["2024-01-01T01:00", "2024-01-01T02:00"],
                        "precipitation": [1.5, None],
                    },
                },
            ),
        ]
        for provider, payload in cases:
            with patch(
                "studio.weather.fetch", return_value=io.StringIO(json.dumps(payload))
            ):
                route = f"/api/sources/{provider}/hourly"
                response = self.client.post(route, json=query)
                self.assertEqual(response.status_code, 200, response.text)
                info = response.json()[0]
                self.assertEqual(info["provenance"], "gridded estimate")
                self.assertEqual(
                    info["preview"][0]["timestamp"], "2024-01-01T01:00:00+00:00"
                )
                self.assertEqual(info["preview"][1]["rain_mm"], "")
                self.assertEqual(
                    self.client.post(route, json={**query, "latitude": 91}).status_code,
                    422,
                )

    def test_local_presentation_figures(self):
        self.assertIsNone(self.client.get("/api/presentation/reference").json())
        self.assertEqual(
            self.client.get("/api/presentation/reference/image/0").status_code, 404
        )
        dest = ROOT / "panel-reference"
        dest.mkdir()
        save(
            dest / "reference.json", {"title": "Fixture", "figures": [{"title": "Run"}]}
        )
        (dest / "figure-0.png").write_bytes(b"test image")
        self.assertEqual(
            self.client.get("/api/presentation/reference").json()["title"], "Fixture"
        )
        self.assertEqual(
            self.client.get("/api/presentation/reference/image/0").content,
            b"test image",
        )
        self.assertEqual(
            self.client.get("/api/presentation/reference/image/-1").status_code, 404
        )

    def test_checkpoint_resume(self):
        c = self.config.model_copy(update={"epochs": 3})
        p = prepare(c)
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:

            def stop(history, best):
                raise Halt("requested")

            with self.assertRaises(Halt):
                train_arm(c, p, "qt", 42, Path(a), stop, lambda: None)
            resumed, _ = train_arm(
                c, p, "qt", 42, Path(a), lambda *args: None, lambda: None
            )
            clean, _ = train_arm(
                c, p, "qt", 42, Path(b), lambda *args: None, lambda: None
            )
            recorded = frames_since(Path(a), 0)
            self.assertEqual([f["epoch"] for f in recorded], [1, 2, 3])
            self.assertEqual(recorded, frames_since(Path(b), 0))
        np.testing.assert_array_equal(clean, resumed)

    def test_import_preview_mapping_join_and_control(self):
        file = ("files", ("stage.csv", b"t,height\n2024-01-01T00:00,1.5\n", "text/csv"))
        staged = self.client.post("/api/stage", files=[file]).json()[0]
        self.assertFalse((ROOT / "datasets" / staged["id"]).exists())
        self.client.post("/api/stage/" + staged["id"] + "/commit")
        rain = self.client.post(
            "/api/stage",
            files=[("files", ("rain.csv", b"t,r\n2024-01-01T00:00,0.8\n", "text/csv"))],
        ).json()[0]
        self.client.post("/api/stage/" + rain["id"] + "/commit")
        mapped = normalize(
            staged["id"],
            dict(
                mapping={"timestamp": "t", "stage_cm": "height"},
                site="basin",
                name="Aligned",
                rain_dataset=rain["id"],
                rain_time_column="t",
                rain_column="r",
                factors={"stage_cm": 100},
            ),
        )
        self.assertEqual(mapped["series"][0][1:3], [150, 0.8])
        cfg = self.config.model_copy(
            update={"arms": ["classical"], "folds": 1, "warm_dataset": ""}
        )
        st = self.client.post("/api/runs", json=cfg.model_dump(mode="json")).json()
        for action, status in [
            ("pause", "paused"),
            ("resume", "queued"),
            ("cancel", "cancelled"),
        ]:
            self.assertEqual(
                self.client.post(f"/api/runs/{st['id']}/{action}").json()["status"],
                status,
            )
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/app.js").status_code, 200)

    def test_event_statistics(self):
        from studio.experiment import Event

        dates = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(100)]
        c = self.config.model_copy(
            update={"events": [Event(name="block", start=dates[68], end=dates[75])]}
        )
        self.assertEqual(partition(c, dates)[0][2], 68)
        flags = indicators(np.array([True, True, False]), np.array([False, True, True]))
        self.assertEqual(flags["mwr"], 0.5)
        self.assertEqual(flags["pofd"], 1.0)
        self.assertEqual(holm({"a": 0.01, "b": 0.04}), {"a": 0.02, "b": 0.04})
        events = [
            Event(name=str(i), start=dates[k], end=dates[k + 1])
            for i, k in enumerate([10, 20, 70])
        ]
        c = c.model_copy(update={"threshold": 10, "events": events})
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for fold, indices in enumerate([[10, 20], [70]]):
                job = root / f"job-{fold:03}"
                job.mkdir()
                save(
                    job / "result.json",
                    dict(
                        fold=fold,
                        seed=1,
                        arm="qt",
                        init="scratch",
                        metrics={"qt": dict(mae=1, rmse=1, nse=None)},
                    ),
                )
                with (job / "forecast.csv").open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["timestamp", "observed", "origin_6h_ago", "qt"])
                    for i in indices:
                        w.writerow([dates[i].isoformat(), 11, 5, 9])
            result = report(root, c)
            self.assertEqual(result["warning_tier"], "Flood exceedance")
            self.assertEqual(result["warnings"][0]["mwr"], 1)
