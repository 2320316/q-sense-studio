"""Build causal per-site feature sequences. NaNs and timestamp gaps mask windows."""

import csv
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def dekad(date):
    if date.day not in (1, 11, 21):
        raise ValueError(f"Unexpected dekad date {date}")
    return 36 * date.year + 3 * (date.month - 1) + (date.day - 1) // 10


def rainfall(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        rows = sorted(
            (
                r
                for r in csv.DictReader(f)
                if r.get("PCODE", "PH04010") == "PH04010" and r["version"] == "final"
            ),
            key=lambda r: r["date"],
        )
    dates = [datetime.fromisoformat(r["date"]) for r in rows]
    ticks = np.array([dekad(d) for d in dates])
    rain = np.array([float(r["rfh"]) for r in rows])
    if not np.isfinite(rain).all() or (rain < 0).any():
        raise ValueError("Invalid final rainfall")
    angle = (ticks % 36) * (2 * math.pi / 36)
    return (
        dates,
        ticks,
        np.column_stack((rain, np.sin(angle), np.cos(angle))),
        ["rain_mm", "season_sin", "season_cos"],
    )


def stage(path, mode, site):
    # Local files use a common Manila civil time and an interval-ending timestamp.
    minutes = 60 if mode == "hourly" else 10
    base = ["stage_cm", "rain_mm"]
    if mode == "local":
        base += ["temp_c", "rh_pct", "pressure_hpa"]
    series = {}
    with Path(path).open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"timestamp", "site_id", "qa_flag", *base}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns {sorted(missing)}")
        for r in reader:
            t = datetime.fromisoformat(r["timestamp"])
            if t.tzinfo or t.second or t.microsecond or t.minute % minutes:
                raise ValueError(f"Expected aligned Manila local time, received {t}")
            key = (r["site_id"], t)
            if key in series:
                raise ValueError(f"Duplicate site/time {key}")
            ok = r["qa_flag"] in ("VALID", "INTERPOLATED")
            values = [float(r[k]) if ok and r[k].strip() else np.nan for k in base]
            series[key] = (values, r["qa_flag"] == "VALID")
    dates = sorted(t for station, t in series if station == site)
    if not dates:
        raise ValueError(f"No rows for site {site}")
    step = timedelta(minutes=minutes)
    ticks = np.array([(t - dates[0]) // step for t in dates])
    values = np.array([series[site, t][0] for t in dates])
    # Only measured stage values are eligible as targets.
    targets = np.array(
        [series[site, t][0][0] if series[site, t][1] else np.nan for t in dates]
    )
    features = list(values.T)
    names = list(base)
    if mode == "local":
        rain = {t: v for t, v in zip(dates, values[:, 1])}
        for hours in (1, 6, 24):
            features.append(
                [
                    sum(rain.get(t - i * step, np.nan) for i in range(hours * 6))
                    for t in dates
                ]
            )
            names.append(f"rain_{hours}h_mm")
        nanrow = ([np.nan] * len(base), False)

        def lag(station, column):
            return np.array(
                [series.get((station, t - step), nanrow)[0][column] for t in dates]
            )

        features.append((values[:, 0] - lag(site, 0)) * 6)
        names.append("rise_cm_per_hour")
        links = {"C2": [("C3", 0), ("C3", 1)], "C4": [("C2", 0), ("C1", 0)]}
        for upstream, col in links.get(site, []):
            features.append(lag(upstream, col))
            names.append(f"{upstream}_lag1_{base[col]}")
    return dates, ticks, np.column_stack(features), names, targets


def windows(ticks, features, targets, lookback, horizon, start, end):
    xs, ys, ids = [], [], []
    for label in range(start + lookback + horizon - 1, end):
        origin = label - horizon
        first = origin - lookback + 1
        x = features[first : origin + 1]
        if ticks[label] - ticks[first] != label - first:
            continue
        if np.isfinite(x).all() and np.isfinite(targets[label]):
            xs.append(x)
            ys.append(targets[label])
            ids.append(label)
    if not xs:
        raise ValueError(
            "No complete windows in a split. Check dates, coverage, QA and upstream features."
        )
    return np.array(xs), np.array(ys), np.array(ids)
