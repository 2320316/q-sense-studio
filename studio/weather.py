"""Collect hourly gridded precipitation as interval-ending rain for stage joins."""

import csv
import json
import math
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

from pydantic import BaseModel, Field, model_validator

from .library import inspect_files
from .sources import fetch

EXTRA_SOURCES = [
    dict(
        id="era5",
        name="ERA5 precipitation",
        provider="Open-Meteo · ECMWF",
        url="https://open-meteo.com/en/docs/historical-weather-api",
        kind="Hourly gridded reanalysis",
        area="Selected coordinates",
        group="additional",
        action="Collect",
    ),
    dict(
        id="power",
        name="POWER precipitation",
        provider="NASA POWER",
        url="https://power.larc.nasa.gov/docs/services/api/temporal/hourly/",
        kind="Hourly corrected precipitation",
        area="Selected coordinates",
        group="additional",
        action="Collect",
    ),
]


class WeatherQuery(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    start: date
    end: date

    @model_validator(mode="after")
    def dates(self):
        if (
            not 0 <= (self.end - self.start).days <= 365
            or self.end > datetime.now(timezone.utc).date()
        ):
            raise ValueError("Choose 1–366 historical UTC dates in chronological order")
        return self


def collect_weather(key, q):
    point = dict(latitude=q.latitude, longitude=q.longitude)
    if key == "era5":
        url = "https://archive-api.open-meteo.com/v1/archive?" + urlencode(
            dict(
                **point,
                start_date=q.start,
                end_date=q.end,
                hourly="precipitation",
                models="era5",
                timezone="GMT",
            )
        )
    elif key == "power":
        url = "https://power.larc.nasa.gov/api/temporal/hourly/point?" + urlencode(
            dict(
                **point,
                start=q.start.strftime("%Y%m%d"),
                end=q.end.strftime("%Y%m%d"),
                parameters="PRECTOTCORR",
                community="AG",
                format="JSON",
                **{"time-standard": "UTC"},
            )
        )
    else:
        raise ValueError("Unknown weather provider")
    with fetch(url) as response:
        payload = json.load(response)
    if key == "era5":
        if (
            payload["hourly_units"]["precipitation"] != "mm"
            or payload["utc_offset_seconds"] != 0
        ):
            raise ValueError("Unexpected ERA5 units or time zone")
        hourly = payload["hourly"]
        rows = [
            (datetime.fromisoformat(t), v)
            for t, v in zip(hourly["time"], hourly["precipitation"], strict=True)
        ]
    else:
        if (
            payload["parameters"]["PRECTOTCORR"]["units"] != "mm/hour"
            or payload["header"]["time_standard"] != "UTC"
        ):
            raise ValueError("Unexpected POWER units or time zone")
        fill = payload["header"]["fill_value"]
        # POWER labels the start of each 1-hour average. Stage joins use its end.
        rows = [
            (
                datetime.strptime(t, "%Y%m%d%H") + timedelta(hours=1),
                None if v == fill else v,
            )
            for t, v in sorted(
                payload["properties"]["parameter"]["PRECTOTCORR"].items()
            )
        ]
    if not any(v is not None and math.isfinite(v) and v >= 0 for _, v in rows):
        raise ValueError(
            "The provider returned no usable precipitation for these dates"
        )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{key}-{q.latitude}-{q.longitude}-{q.start}-{q.end}.csv"
        with path.open("w", newline="", encoding="utf8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "rain_mm", "requested_lat", "requested_lon"])
            for t, v in rows:
                writer.writerow(
                    [
                        t.replace(tzinfo=timezone.utc).isoformat(),
                        v if v is not None and math.isfinite(v) and v >= 0 else "",
                        q.latitude,
                        q.longitude,
                    ]
                )
        return inspect_files(path.parent, url, provenance="gridded estimate")
