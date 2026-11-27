import React, { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { Field } from "./common";

const studio = "https://noah.up.edu.ph/noah-studio";
const weather = "https://noah.up.edu.ph/weather-updates/rainfall-contour";
const bounds = [
  [3.85, 115.35],
  [21.55, 128.25],
];

export function WeatherMap() {
  const host = useRef(null),
    map = useRef(null);
  const [enabled, setEnabled] = useState(true),
    [hours, setHours] = useState(1);
  const [version, setVersion] = useState(0),
    [status, setStatus] = useState("");
  useEffect(() => {
    const m = L.map(host.current, { scrollWheelZoom: false }).setView(
      [13.5, 122],
      5,
    );
    map.current = m;
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18,
      attribution:
        '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(m);
    L.circleMarker([14.08, 121.15], {
      radius: 4,
      color: "#1c2625",
      fillColor: "white",
      fillOpacity: 1,
    })
      .bindTooltip("Tanauan City")
      .addTo(m);
    return () => {
      m.remove();
      map.current = null;
    };
  }, []);
  useEffect(() => {
    if (!enabled) {
      setStatus("Weather layer off");
      return;
    }
    const m = map.current;
    let layer,
      pending,
      timer,
      disposed = false;
    const load = () => {
      const url = `https://webgis-static.up.edu.ph/api/contours/${hours}hr_latest_rainfall_contour.png?t=${Date.now()}`;
      setStatus("Fetching rainfall layer…");
      pending = L.imageOverlay(url, bounds, {
        opacity: 0,
        interactive: false,
        attribution: "UP NOAH rainfall contours",
      });
      const finished = (message) => {
        setStatus(message);
        timer = setTimeout(load, 5 * 60 * 1000);
      };
      pending
        .once("load", () => {
          if (disposed) return;
          layer?.remove();
          layer = pending.setOpacity(0.72);
          finished(
            "Retrieved " +
              new Date().toLocaleTimeString() +
              " · see image for product time",
          );
        })
        .once("error", () => {
          if (disposed) return;
          pending.remove();
          finished(
            layer
              ? "Update failed · showing previous image"
              : "Weather layer unavailable",
          );
        })
        .addTo(m);
    };
    load();
    return () => {
      disposed = true;
      clearTimeout(timer);
      pending?.off().remove();
      layer?.remove();
    };
  }, [enabled, hours, version]);
  return (
    <section className="panel weatherPanel">
      <div className="panelHead">
        <div>
          <h2>NOAH weather</h2>
          <small>Philippines · latest rainfall contours</small>
        </div>
        <label className="toggle">
          <input
            type="checkbox"
            checked={enabled}
            onChange={(e) => setEnabled(e.target.checked)}
          />{" "}
          Weather updates
        </label>
      </div>
      <div
        className="weatherMap"
        ref={host}
        aria-label="Interactive Philippine rainfall map"
      />
      <div className="weatherTools">
        <Field label="Accumulation">
          <select value={hours} onChange={(e) => setHours(+e.target.value)}>
            {[1, 3, 6, 12, 24].map((n) => (
              <option key={n} value={n}>
                {n} hour{n > 1 ? "s" : ""}
              </option>
            ))}
          </select>
        </Field>
        <div>
          <small aria-live="polite">{status}</small>
          <small>Checks every 5 minutes while enabled</small>
        </div>
        <button
          className="secondary"
          disabled={!enabled}
          onClick={() => setVersion((v) => v + 1)}
        >
          Refresh layer
        </button>
        <a
          href={`https://webgis-static.up.edu.ph/api/contours/${hours}hr_latest_rainfall_contour.png`}
          target="_blank"
          rel="noreferrer"
        >
          Full-size rainfall image ↗
        </a>
        <a href={weather} target="_blank" rel="noreferrer">
          NOAH Weather Updates ↗
        </a>
        <a href={studio} target="_blank" rel="noreferrer">
          NOAH Studio ↗
        </a>
      </div>
      {enabled && (
        <details>
          <summary>Rainfall legend</summary>
          <img
            className="rainLegend"
            alt={hours + "-hour rainfall contour legend from UP NOAH"}
            src={`https://noah.up.edu.ph/assets/legends/rainfall-contour/rainfall-contour-legend-${hours}hr.png`}
          />
        </details>
      )}
    </section>
  );
}

export function WeatherCollection({ sources, onCollect }) {
  const [provider, setProvider] = useState("era5");
  const [query, setQuery] = useState({
    latitude: "14.08",
    longitude: "121.15",
    start: "2024-01-01",
    end: "2024-01-31",
  });
  const source = sources.find((s) => s.id === provider);
  if (!source) return null;
  return (
    <section className="panel">
      <h2>Additional sources</h2>
      <p>
        Collect hourly gridded rainfall for a point. Join the result to hourly
        gauge readings by interval-ending timestamp in Datasets.
      </p>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          onCollect(source, {
            ...query,
            latitude: +query.latitude,
            longitude: +query.longitude,
          });
        }}
      >
        <div className="formGrid">
          <Field label="Weather archive">
            <select
              value={provider}
              onChange={(e) => setProvider(e.target.value)}
            >
              {sources.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.provider} · {s.name}
                </option>
              ))}
            </select>
          </Field>
          {Object.entries({
            latitude: "Latitude",
            longitude: "Longitude",
            start: "Start date (UTC)",
            end: "End date (UTC)",
          }).map(([key, label]) => {
            const coord = ["latitude", "longitude"].includes(key),
              limit = key === "latitude" ? 90 : 180;
            return (
              <Field key={key} label={label}>
                <input
                  required
                  type={coord ? "number" : "date"}
                  step={coord ? "any" : undefined}
                  min={coord ? -limit : key === "end" ? query.start : undefined}
                  max={coord ? limit : undefined}
                  value={query[key]}
                  onChange={(e) =>
                    setQuery({ ...query, [key]: e.target.value })
                  }
                />
              </Field>
            );
          })}
        </div>
        <div className="actions">
          <button>Collect hourly rainfall</button>
          <a href={source.url} target="_blank" rel="noreferrer">
            Provider documentation ↗
          </a>
        </div>
      </form>
    </section>
  );
}
