import React, { useEffect, useRef, useState } from "react";
import {
  api,
  Empty,
  Field,
  Heading,
  Icon,
  MapView,
  num,
  Plot,
  Tag,
} from "./common";

export function Review({ records, accept, discard }) {
  return (
    records.length > 0 && (
      <section className="panel">
        <h2>Review import</h2>
        {records.map((r) => (
          <details key={r.id} open={records.length === 1}>
            <summary>
              {r.name} <Tag>{r.kind}</Tag>
            </summary>
            <p>{r.note || r.columns?.join(", ")}</p>
            <small>
              {num(r.rows, 0)} rows or features · {r.duplicates || 0} duplicate
              timestamps · {r.missing || 0} missing targets · {r.gaps || 0} gaps
            </small>
            {r.preview && <Sample item={r} />}
          </details>
        ))}
        <div className="actions">
          <button onClick={accept}>
            Import {records.length} dataset{records.length === 1 ? "" : "s"}
          </button>
          <button className="secondary" onClick={discard}>
            Dismiss
          </button>
        </div>
      </section>
    )
  );
}
function Sample({ item }) {
  return (
    <div className="scroll">
      <table>
        <thead>
          <tr>
            {item.columns.map((k) => (
              <th key={k}>{k}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {item.preview.map((r, i) => (
            <tr key={i}>
              {item.columns.map((k) => (
                <td key={k}>{r[k]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
export function Sources({ sources, onCollect, attach }) {
  return (
    <>
      <Heading
        title="Data sources"
        desc="Collect public data or attach files delivered by a provider."
      />
      <div className="sourceGrid">
        {sources.map((s) => (
          <section className="panel source" key={s.id}>
            <div className="sourceIcon">
              <Icon name={s.action === "Collect" ? "down" : "source"} />
            </div>
            <small>{s.provider}</small>
            <h2>{s.name}</h2>
            <p>{s.kind}</p>
            <small>
              <Icon name="pin" size={14} /> {s.area}
            </small>
            <div className="actions">
              {s.action === "Collect" ? (
                <button onClick={() => onCollect(s)}>Collect dataset</button>
              ) : (
                <a
                  className="button secondary"
                  target="_blank"
                  rel="noreferrer"
                  href={s.url}
                >
                  {s.action} ↗
                </a>
              )}
              <button className="textButton" onClick={() => attach(s)}>
                Attach file
              </button>
            </div>
          </section>
        ))}
      </div>
    </>
  );
}
const columns = [
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
];
function Mapping({ dataset, library, onMap }) {
  const [cfg, set] = useState(() => ({
    name: dataset.name + " · normalized",
    mapping: Object.fromEntries(
      columns.filter((k) => dataset.columns.includes(k)).map((k) => [k, k]),
    ),
    site: "C1",
    timezone: "Asia/Manila",
    provenance: "observed",
    factors: {},
    rain_dataset: "",
    rain_column: "rain_mm",
    rain_time_column: "timestamp",
  }));
  const change = (k, v) => set((s) => ({ ...s, [k]: v }));
  return (
    <details className="panel">
      <summary>Map columns and join observations</summary>
      <div className="formGrid">
        <Field label="Prepared dataset name">
          <input
            value={cfg.name}
            onChange={(e) => change("name", e.target.value)}
          />
        </Field>
        <Field label="Site if absent">
          <input
            value={cfg.site}
            onChange={(e) => change("site", e.target.value)}
          />
        </Field>
        <Field label="Input time zone">
          <input
            value={cfg.timezone}
            onChange={(e) => change("timezone", e.target.value)}
          />
        </Field>
        <Field label="Stage provenance">
          <select
            value={cfg.provenance}
            onChange={(e) => change("provenance", e.target.value)}
          >
            <option value="observed">Measured</option>
            <option value="synthetic">Synthetic warm start</option>
          </select>
        </Field>
        {columns.map((k) => (
          <Field key={k} label={k}>
            <select
              value={cfg.mapping[k] || ""}
              onChange={(e) =>
                change("mapping", { ...cfg.mapping, [k]: e.target.value })
              }
            >
              <option value="">Unmapped</option>
              {dataset.columns.map((v) => (
                <option key={v}>{v}</option>
              ))}
            </select>
          </Field>
        ))}
        {["stage_cm", "rain_mm", "pressure_hpa"].map((k) => (
          <Field key={k} label={k + " unit multiplier"}>
            <input
              type="number"
              step="any"
              value={cfg.factors[k] ?? 1}
              onChange={(e) =>
                change("factors", {
                  ...cfg.factors,
                  [k]: Number(e.target.value),
                })
              }
            />
          </Field>
        ))}
        <Field label="Join separate rainfall file">
          <select
            value={cfg.rain_dataset}
            onChange={(e) => change("rain_dataset", e.target.value)}
          >
            <option value="">Use mapped rain</option>
            {library
              .filter((d) => d.columns && d.id !== dataset.id)
              .map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name}
                </option>
              ))}
          </select>
        </Field>
        {cfg.rain_dataset &&
          ["rain_time_column", "rain_column"].map((k) => (
            <Field key={k} label={k}>
              <input
                value={cfg[k]}
                onChange={(e) => change(k, e.target.value)}
              />
            </Field>
          ))}
      </div>
      <small>
        Unflagged readings default to VALID. Use calibrated interval totals and
        a common vertical datum. The join matches timestamps exactly.
      </small>
      <button onClick={() => onMap(cfg)}>Preview prepared table</button>
    </details>
  );
}
export function Library({
  library,
  selected,
  select,
  upload,
  onMap,
  onTrain,
  pickFiles,
}) {
  const [item, setItem] = useState(null),
    [site, setSite] = useState(""),
    [overlay, setOverlay] = useState(null);
  const folders = useRef();
  const [fail, setFail] = useState("");
  useEffect(() => {
    let live = true;
    setItem(null);
    setFail("");
    if (selected)
      api("library/" + selected)
        .then((x) => {
          if (live) {
            setItem(x);
            setSite(x.sites?.[0] || "");
          }
        })
        .catch((e) => setFail(e.message));
    return () => {
      live = false;
    };
  }, [selected]);
  const rows =
    item?.series
      ?.filter((r) => !site || r[3] === site)
      .map((r) => ({
        timestamp: r[0],
        [item.kind === "rainfall" ? "Rainfall" : "Stage"]: r[1],
        Rainfall: item.kind === "rainfall" ? r[1] : r[2],
      })) || [];
  return (
    <>
      <Heading
        title="Dataset library"
        desc="Inspect coverage, observations and training eligibility."
      >
        <button className="secondary" onClick={() => folders.current.click()}>
          Import folder
        </button>
        <button onClick={() => pickFiles()}>
          <Icon name="upload" />
          Import files
        </button>
      </Heading>
      <input
        ref={folders}
        type="file"
        webkitdirectory=""
        multiple
        hidden
        onChange={(e) => {
          upload(e.target.files);
          e.target.value = "";
        }}
      />
      <div
        className="drop"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          upload(e.dataTransfer.files);
        }}
      >
        <Icon name="upload" /> Drop CSV, Excel or GIS files here
      </div>
      <div className="libraryGrid">
        <section className="panel">
          <h2>
            Datasets <small>{library.length}</small>
          </h2>
          {!library.length && <Empty>Import a source file to begin.</Empty>}
          {library.map((r) => (
            <button
              className={"listItem " + (r.id === selected ? "selected" : "")}
              key={r.id}
              onClick={() => select(r.id)}
            >
              <Icon name={r.kind === "map" ? "pin" : "file"} />
              <span>
                {r.name}
                <small>
                  {r.kind} · {num(r.rows, 0)} records
                </small>
              </span>
            </button>
          ))}
        </section>
        <div>
          {fail && <p>{fail}</p>}
          {item && (
            <>
              <section className="panel">
                <div className="panelHead">
                  <h2>{item.name}</h2>
                  <Tag>{item.kind}</Tag>
                </div>
                <p>{item.note || item.source}</p>
                <div className="miniStats">
                  <span>
                    Records<b>{num(item.rows, 0)}</b>
                  </span>
                  <span>
                    Duplicates<b>{item.duplicates ?? "—"}</b>
                  </span>
                  <span>
                    Missing targets<b>{item.missing ?? "—"}</b>
                  </span>
                  <span>
                    Gaps<b>{item.gaps ?? "—"}</b>
                  </span>
                </div>
                {item.series?.length > 0 && (
                  <>
                    <Field label="Monitoring site">
                      <select
                        value={site}
                        onChange={(e) => setSite(e.target.value)}
                      >
                        {item.sites.map((s) => (
                          <option key={s}>{s}</option>
                        ))}
                      </select>
                    </Field>
                    <Plot
                      rows={rows}
                      fields={
                        item.kind === "rainfall"
                          ? ["Rainfall"]
                          : ["Stage", "Rainfall"]
                      }
                      unit={
                        item.kind === "rainfall"
                          ? "Rainfall (mm)"
                          : "Stage (cm) · rainfall (mm)"
                      }
                    />
                  </>
                )}
                {item.kind === "map" && <MapView layer={item} />}
                {item.kind === "stage" && (
                  <>
                    <Field label="Hazard overlay">
                      <select
                        defaultValue=""
                        onChange={async (e) =>
                          setOverlay(
                            e.target.value
                              ? await api("library/" + e.target.value)
                              : null,
                          )
                        }
                      >
                        <option value="">Site coordinates</option>
                        {library
                          .filter((d) => d.kind === "map")
                          .map((d) => (
                            <option key={d.id} value={d.id}>
                              {d.name}
                            </option>
                          ))}
                      </select>
                    </Field>
                    <MapView layer={overlay} locations={item.locations} />
                  </>
                )}
                {item.columns && (
                  <details>
                    <summary>Column and row preview</summary>
                    <Sample item={item} />
                  </details>
                )}
                <details>
                  <summary>Source and integrity</summary>
                  <p className="wrap">
                    {item.source} · {item.created}
                  </p>
                  <code className="wrap">{item.sha256}</code>
                </details>
                {["stage", "rainfall"].includes(item.kind) && (
                  <button onClick={() => onTrain(item)}>
                    Configure training <Icon name="arrow" />
                  </button>
                )}
              </section>
              {item.columns && (
                <Mapping
                  key={item.id}
                  dataset={item}
                  library={library}
                  onMap={(cfg) => onMap(item.id, cfg)}
                />
              )}
            </>
          )}
        </div>
      </div>
    </>
  );
}
