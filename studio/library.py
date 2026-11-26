"""Local input staging, coordinate-aware map previews, and observation preparation."""

import csv
import json
import math
import shutil
from collections import Counter
from datetime import datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
from .storage import ROOT, digest, folder, now, read, save

TABLES = {".csv", ".xlsx"}
LAYERS = {".shp", ".geojson", ".tif", ".tiff"}
SIDECARS = {".shx", ".dbf", ".prj", ".cpg"}
CANON = [
    "timestamp",
    "site_id",
    "qa_flag",
    "stage_cm",
    "rain_mm",
    "temp_c",
    "rh_pct",
    "pressure_hpa",
    "lat",
    "lon",
]


def thin(seq, limit):
    if len(seq) <= limit:
        return seq
    return [seq[i] for i in np.linspace(0, len(seq) - 1, limit, dtype=int)]


def summarize_csv(path, name):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if not cols or len(cols) != len(set(cols)):
            raise ValueError(f"Missing or duplicate column names in {name}")
        preview, series, stations, locations = [], [], set(), {}
        rain = {"date", "rfh", "version"}.issubset(cols)
        kind = (
            "rainfall"
            if rain
            else (
                "stage"
                if {"timestamp", "stage_cm", "site_id", "qa_flag", "rain_mm"}.issubset(
                    cols
                )
                else "table"
            )
        )
        if "adm4_pcode" in cols:
            kind = "hazard table"
        dates, repeats, last, intervals, missing = set(), 0, {}, Counter(), 0
        rain_values = []
        n = 0
        for n, row in enumerate(reader, 1):
            if len(preview) < 5:
                preview.append({k: str(row.get(k) or "")[:160] for k in cols})
            if rain and row.get("PCODE", "PH04010") != "PH04010":
                continue
            site = row.get("site_id") or ("PH04010" if rain else "")
            if site:
                stations.add(site)
            t = row.get("timestamp") or row.get("date")
            if t and (rain or kind == "stage"):
                repeats += (site, t) in dates
                dates.add((site, t))
                d = datetime.fromisoformat(t)
                if rain:
                    rain_values.append((t, row["rfh"], row["version"]))
                if site in last:
                    seconds = int((d - last[site]).total_seconds())
                    if seconds > 0:
                        intervals[seconds] += 1
                last[site] = d

                def number(key):
                    try:
                        v = float(row[key])
                        return v if math.isfinite(v) else None
                    except (TypeError, ValueError, KeyError):
                        return None

                value = number("rfh" if rain else "stage_cm")
                missing += value is None
                series.append([t, value, number("rain_mm"), site])
                if number("lon") is not None and number("lat") is not None:
                    locations[site] = [number("lon"), number("lat")]
        cadence = intervals.most_common(1)[0][0] if intervals else 0
        gaps = sum(
            n
            for delta, n in intervals.items()
            if delta > (11 * 86400 if rain else cadence)
        )
        detail = dict(
            kind=kind,
            columns=cols,
            rows=n,
            preview=preview,
            sites=sorted(stations),
            series=thin(sorted(series), 1200),
            locations=locations,
            duplicates=repeats,
            missing=missing,
            gaps=int(gaps),
            interval_minutes=cadence // 60,
        )
        if rain:
            detail["note"] = (
                f"{len(rain_values):,} Batangas dekads · next-period rainfall (mm)"
            )
        elif kind == "hazard table":
            detail["note"] = "Static susceptibility percentages · geographic context"
        return detail


def vector_preview(geometries, convert):
    paths = []

    def add(coordinates):
        if len(paths) < 800 and coordinates:
            pts = (
                coordinates
                if isinstance(coordinates[0], (list, tuple))
                else [coordinates]
            )
            paths.append([list(convert(*p[:2])) for p in thin(pts, 100)])

    for geom in geometries:
        coords = geom.get("coordinates", [])
        typ = geom.get("type")
        if typ in ("Point", "LineString", "MultiPoint"):
            add(coords)
        elif typ in ("Polygon", "MultiLineString"):
            for ring in coords:
                add(ring)
        elif typ == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    add(ring)
    return paths


def layer(path):
    identity = lambda x, y: (x, y)
    extra = {}
    if path.suffix.lower() == ".shp":
        import shapefile
        from pyproj import Transformer

        if not all(path.with_suffix(ext).exists() for ext in (".dbf", ".shx", ".prj")):
            raise ValueError(f"{path.name} needs matching .dbf, .shx and .prj files")
        tf = Transformer.from_crs(
            path.with_suffix(".prj").read_text(), 4326, always_xy=True
        )
        with shapefile.Reader(str(path)) as reader:
            box = tf.transform_bounds(*reader.bbox)
            ids = thin(list(range(len(reader))), 250)
            paths = []
            for i in ids:
                shp = reader.shape(i)
                edges = (
                    list(shp.parts) + [len(shp.points)]
                    if hasattr(shp, "parts")
                    else [0, len(shp.points)]
                )
                for k in thin(list(range(len(edges) - 1)), 4):
                    start, end = edges[k : k + 2]
                    if end > start:
                        take = np.linspace(
                            start, end - 1, min(100, end - start), dtype=int
                        )
                        paths.append(
                            [list(tf.transform(*shp.points[j][:2])) for j in take]
                        )
            count = len(reader)
    elif path.suffix.lower() == ".geojson":
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        feats = obj.get("features", [obj])
        geoms = [f.get("geometry", f) for f in feats if f.get("geometry", f)]
        paths = vector_preview(thin(geoms, 250), identity)
        pts = np.array([p for ring in paths for p in ring])
        if not pts.size:
            raise ValueError("No supported GeoJSON geometry")
        box = [*pts.min(axis=0), *pts.max(axis=0)]
        count = len(geoms)
    else:
        import rasterio
        from rasterio.warp import transform_bounds

        with rasterio.open(path) as raster:
            box = transform_bounds(raster.crs, 4326, *raster.bounds)
            pixels = raster.read(1, out_shape=(48, 64), masked=True).astype(float)
            extra["grid"] = [
                [None if not np.isfinite(v) else float(v) for v in row]
                for row in pixels.filled(np.nan)
            ]
            count = raster.width * raster.height
        paths = []
    west, south, east, north = box
    overlaps = west <= 121.47 and east >= 120.57 and south <= 14.23 and north >= 13.59
    return dict(
        kind="map",
        bounds=list(box),
        paths=paths,
        rows=count,
        note="Intersects Batangas bounds" if overlaps else "Outside Batangas bounds",
        **extra,
    )


def inspect_files(raw, source="Local import"):
    records = []
    for file in sorted(raw.iterdir()):
        ext = file.suffix.lower()
        if ext not in TABLES | LAYERS:
            continue
        work = ROOT / "staged" / uuid4().hex
        work.mkdir(parents=True)
        try:
            trusted = work / ("input" + ext)
            shutil.copyfile(file, trusted)
            members = {file.name: digest(trusted)}
            if ext == ".shp":
                for peer in raw.iterdir():
                    if (
                        peer.stem.lower() == file.stem.lower()
                        and peer.suffix.lower() in SIDECARS
                    ):
                        shutil.copyfile(peer, trusted.with_suffix(peer.suffix.lower()))
                        members[peer.name] = digest(peer)
            if ext == ".xlsx":
                from openpyxl import load_workbook

                book = load_workbook(trusted, read_only=True, data_only=True)
                try:
                    with (work / "table.csv").open(
                        "w", encoding="utf8", newline=""
                    ) as f:
                        csv.writer(f).writerows(book.active.values)
                finally:
                    book.close()
                trusted = work / "table.csv"
            detail = (
                summarize_csv(trusted, file.name) if ext in TABLES else layer(trusted)
            )
            detail.update(
                id=work.name,
                name=file.name,
                source=source,
                created=now(),
                file=trusted.name,
                hashes=members,
                sha256=digest(trusted),
                provenance="observed",
            )
            save(work / "info.json", detail)
            records.append(detail)
        except Exception:
            shutil.rmtree(work)
            for earlier in records:
                shutil.rmtree(ROOT / "staged" / earlier["id"], ignore_errors=True)
            raise
    if not records:
        raise ValueError("Choose CSV, XLSX, a complete shapefile, GeoJSON, or GeoTIFF")
    # Detect the short export only after comparing its actual values against the full resource.
    rain = [r for r in records if r["kind"] == "rainfall"]

    def values(r):
        with (ROOT / "staged" / r["id"] / r["file"]).open(encoding="utf-8-sig") as f:
            return {
                (v["date"], v["rfh"], v["version"])
                for v in csv.DictReader(f)
                if v.get("PCODE", "PH04010") == "PH04010"
            }

    sets = {r["id"]: values(r) for r in rain} if len(rain) > 1 else {}
    for r in rain:
        for other in rain:
            if r != other and sets[r["id"]] < sets[other["id"]]:
                r["note"] += " · repeated subset of " + other["name"]
                save(ROOT / "staged" / r["id"] / "info.json", r)
    return records


def commit(key):
    src = folder("staged", key)
    info = read(src / "info.json")
    from filelock import FileLock

    dest = ROOT / "datasets" / key
    dest.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(dest.parent / "import.lock")):
        for saved in dest.parent.glob("*/info.json"):
            previous = read(saved)
            if (
                info["name"],
                info["source"],
                info["sha256"],
                info.get("hashes"),
                info.get("provenance"),
            ) == (
                previous["name"],
                previous["source"],
                previous["sha256"],
                previous.get("hashes"),
                previous.get("provenance"),
            ):
                shutil.rmtree(src)
                return previous
        src.rename(dest)
    return info


def data(key):
    root = folder("datasets", key)
    info = read(root / "info.json")
    return root / info["file"], info


def normalize(key, cfg):
    """Convert explicit field mappings and join interval rain at exact timestamps."""
    path, info = data(key)
    if path.suffix != ".csv":
        raise ValueError("Column mapping requires a table")
    mapping = cfg["mapping"]
    zone = ZoneInfo(cfg.get("timezone", "Asia/Manila"))
    manila = ZoneInfo("Asia/Manila")

    def stamp(value):
        parsed = datetime.fromisoformat(value)
        return (
            (parsed if parsed.tzinfo else parsed.replace(tzinfo=zone))
            .astimezone(manila)
            .replace(tzinfo=None)
            .isoformat()
        )

    companion = {}
    sources = [key]
    if cfg.get("rain_dataset"):
        p, _ = data(cfg["rain_dataset"])
        sources.append(cfg["rain_dataset"])
        with p.open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                t = stamp(r[cfg["rain_time_column"]])
                if t in companion:
                    raise ValueError(
                        "Rainfall join has multiple observations at one timestamp; select a single station series"
                    )
                companion[t] = r[cfg["rain_column"]]
    new = ROOT / "staged" / uuid4().hex
    new.mkdir(parents=True)
    try:
        out = new / "table.csv"
        with (
            path.open(encoding="utf-8-sig") as f,
            out.open("w", newline="", encoding="utf8") as g,
        ):
            writer = csv.DictWriter(g, fieldnames=CANON)
            writer.writeheader()
            for r in csv.DictReader(f):
                row = {c: r.get(mapping.get(c, ""), "") for c in CANON}
                row["timestamp"] = stamp(row["timestamp"])
                row["site_id"] = row["site_id"] or cfg["site"]
                row["qa_flag"] = row["qa_flag"] or cfg.get("qa_default", "VALID")
                if companion:
                    row["rain_mm"] = companion.get(row["timestamp"], "")
                for col, factor in cfg.get("factors", {}).items():
                    if row.get(col):
                        row[col] = float(row[col]) * float(factor)
                writer.writerow(row)
        result = summarize_csv(out, info["name"])
        result.update(
            id=new.name,
            name=cfg["name"],
            source=info["source"],
            created=now(),
            file=out.name,
            sha256=digest(out),
            provenance=cfg.get("provenance", "observed"),
            parents=sources,
            mapping=cfg,
        )
        save(new / "info.json", result)
        return result
    except Exception:
        shutil.rmtree(new)
        raise
