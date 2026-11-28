import React, { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { api, Empty, Heading, Icon, labels, num, Plot, Tag } from "./common";
import { Library, Review, Sources } from "./library";
import { defaultDataset, Monitor, preset, Training } from "./training";

const pages = [
  ["Overview", "home"],
  ["Datasets", "data"],
  ["Sources", "source"],
  ["Training", "training"],
  ["Runs", "runs"],
];
function Overview({
  library,
  runs,
  go,
  configure,
  openRun,
  openData,
  pickFiles,
}) {
  const [series, setSeries] = useState(null),
    [selected, setSelected] = useState("");
  const [finished, setFinished] = useState(null);
  const recent = runs.find((r) => r.status === "completed");
  useEffect(() => {
    let live = true;
    setFinished(null);
    if (recent)
      api("runs/" + recent.id)
        .then((v) => {
          if (live) setFinished(v);
        })
        .catch(() => {});
    return () => {
      live = false;
    };
  }, [recent?.id]);
  const model =
    finished?.results.find((j) => j.arm === "qt") || finished?.results[0];
  const unit = finished?.config.mode === "rainfall" ? "mm" : "cm";
  const tables = library.filter((d) => ["rainfall", "stage"].includes(d.kind));
  const dataset =
    tables.find((d) => d.id === selected) ||
    tables.find((d) => d.kind === "rainfall" && d.rows > 1000) ||
    tables[0];
  useEffect(() => {
    let ok = true;
    setSeries(null);
    if (dataset)
      api("library/" + dataset.id)
        .then((d) => ok && setSeries(d))
        .catch(() => {});
    return () => {
      ok = false;
    };
  }, [dataset?.id]);
  const rows = (series?.series || [])
    .filter((r) => r[3] === series.sites[0])
    .map((r) => ({
      timestamp: r[0],
      [dataset.kind === "rainfall" ? "Rainfall" : "Stage"]: r[1],
    }));
  return (
    <>
      <Heading
        title="Forecast workbench"
        desc="Barangay Wawa, Tanauan City · Batangas"
      >
        <button className="secondary" onClick={pickFiles}>
          <Icon name="upload" /> Import data
        </button>
        <button onClick={() => configure(dataset)}>
          <Icon name="plus" /> New experiment
        </button>
      </Heading>
      <div className="kpis">
        {[
          ["Datasets", library.length, "Datasets"],
          ["Experiments", runs.length, "Runs"],
          [
            "Active runs",
            runs.filter((r) => ["running", "pausing"].includes(r.status))
              .length,
            "Runs",
          ],
          [
            "Queued runs",
            runs.filter((r) => r.status === "queued").length,
            "Runs",
          ],
        ].map(([label, n, dest]) => (
          <button className="kpi" key={label} onClick={() => go(dest)}>
            <span>{label}</span>
            <strong>{num(n, 0)}</strong>
          </button>
        ))}
      </div>
      <div className="overviewGrid">
        <section className="panel observation">
          <div className="panelHead">
            <div>
              <h2>
                {dataset?.kind === "stage" ? "Stage record" : "Rainfall record"}
              </h2>
              <small>
                {series?.sites?.[0] || "Station observations"}
                {rows.length > 0
                  ? " · " +
                    rows[0].timestamp.slice(0, 7) +
                    " — " +
                    rows.at(-1).timestamp.slice(0, 7)
                  : ""}
              </small>
            </div>
          </div>
          {tables.length > 0 ? (
            <>
              <FieldSelect
                value={dataset.id}
                change={setSelected}
                items={tables}
              />
              <Plot
                rows={rows}
                fields={[dataset.kind === "rainfall" ? "Rainfall" : "Stage"]}
                unit={
                  dataset.kind === "rainfall"
                    ? "Batangas dekadal rainfall · mm"
                    : "Site water level · cm"
                }
                height={240}
              />
            </>
          ) : (
            <Empty>
              Import your rainfall or gauge records to explore the observation
              history.
              <button className="secondary" onClick={pickFiles}>
                Choose files
              </button>
            </Empty>
          )}
        </section>
        <section className="panel instrument">
          <div className="panelHead">
            <h2>Last completed run</h2>
            <Icon name="chip" />
          </div>
          {model ? (
            <>
              <h3>{finished.state.name}</h3>
              <p className="context">
                {labels[model.arm]} · {finished.config.device.toUpperCase()}
              </p>
              <dl className="resultReadings">
                <div className="major">
                  <dt>Held-out MAE</dt>
                  <dd>
                    {num(model.metrics[model.arm].mae)} <small>{unit}</small>
                  </dd>
                </div>
                <div>
                  <dt>Trainable parameters</dt>
                  <dd>{num(model.trainable, 0)}</dd>
                </div>
                <div>
                  <dt>Exported weights</dt>
                  <dd>{num(model.exported, 0)}</dd>
                </div>
                <div>
                  <dt>Best epoch</dt>
                  <dd>
                    {model.best_epoch} / {finished.config.epochs}
                  </dd>
                </div>
              </dl>
              <button
                className="secondary wide"
                onClick={() => openRun(finished.state.id)}
              >
                Inspect forecasts <Icon name="arrow" />
              </button>
            </>
          ) : (
            <>
              <p>
                Complete an experiment to see its held-out error and model size.
              </p>
              <button onClick={() => configure(dataset)}>
                Set up training <Icon name="arrow" />
              </button>
            </>
          )}
        </section>
      </div>
      <div className="overviewGrid bottom">
        <section className="panel">
          <div className="panelHead">
            <h2>Recent experiments</h2>
            <button className="textButton" onClick={() => go("Runs")}>
              View all →
            </button>
          </div>
          {runs.length ? (
            runs.slice(0, 4).map((r) => (
              <button
                key={r.id}
                className="rowLink"
                onClick={() => openRun(r.id)}
              >
                <span className="lightIcon">
                  <Icon name="runs" />
                </span>
                <span>
                  {r.name}
                  <small>
                    {r.created.slice(0, 10)} · {r.jobs} jobs
                  </small>
                </span>
                <Tag>{r.status}</Tag>
              </button>
            ))
          ) : (
            <Empty>Your first experiment will appear here.</Empty>
          )}
        </section>
        <section className="panel">
          <div className="panelHead">
            <h2>Data library</h2>
            <button className="textButton" onClick={() => go("Datasets")}>
              Browse →
            </button>
          </div>
          {library.slice(0, 3).map((d) => (
            <button
              key={d.id}
              className="rowLink"
              onClick={() => openData(d.id)}
            >
              <Icon name={d.kind === "map" ? "pin" : "file"} />
              <span>
                {d.name}
                <small>{d.kind}</small>
              </span>
            </button>
          ))}
          {!library.length && (
            <button className="secondary" onClick={() => go("Sources")}>
              Explore sources
            </button>
          )}
        </section>
      </div>
    </>
  );
}
function FieldSelect({ value, change, items }) {
  return (
    <select
      aria-label="Observation dataset"
      value={value}
      onChange={(e) => change(e.target.value)}
    >
      {items.map((d) => (
        <option key={d.id} value={d.id}>
          {d.name}
        </option>
      ))}
    </select>
  );
}
function App() {
  const [page, setPage] = useState("Overview"),
    [openPanel, setOpenPanel] = useState(false),
    [library, setLibrary] = useState([]),
    [runs, setRuns] = useState([]),
    [sources, setSources] = useState([]),
    [devices, setDevices] = useState({ cpu: "Ready" }),
    [selected, setSelected] = useState(""),
    [active, setActive] = useState(""),
    [detail, setDetail] = useState(null),
    [cfg, setCfg] = useState(preset("rainfall")),
    [stage, setStage] = useState([]),
    [err, setErr] = useState(""),
    [busy, setBusy] = useState(false),
    [online, setOnline] = useState(true);
  const fileRef = useRef(),
    provider = useRef("Local import");
  const go = (p, present = false) => {
    setOpenPanel(present);
    setPage(p);
    setErr("");
    window.scrollTo(0, 0);
  };
  const refresh = useCallback(async () => {
    const [lib, rs] = await Promise.all([api("library"), api("runs")]);
    setLibrary(lib);
    setRuns(rs);
    setSelected((s) => s || lib[0]?.id || "");
    setActive((s) => s || rs[0]?.id || "");
    setOnline(true);
  }, []);
  useEffect(() => {
    refresh().catch((e) => setErr(e.message));
    api("sources")
      .then(setSources)
      .catch((e) => setErr(e.message));
    api("devices")
      .then(setDevices)
      .catch((e) => setErr(e.message));
    const clock = setInterval(
      () =>
        api("runs")
          .then((rs) => {
            setRuns(rs);
            setOnline(true);
          })
          .catch(() => setOnline(false)),
      3000,
    );
    return () => clearInterval(clock);
  }, [refresh]);
  useEffect(() => {
    let ok = true;
    setDetail(null);
    if (!active || !["Training", "Runs"].includes(page)) return;
    const poll = () =>
      api("runs/" + active)
        .then((r) => {
          if (ok) setDetail(r);
        })
        .catch((e) => ok && setErr(e.message));
    poll();
    const id = setInterval(poll, 1500);
    return () => {
      ok = false;
      clearInterval(id);
    };
  }, [active, page]);
  const act = async (fn) => {
    setBusy(true);
    setErr("");
    try {
      await fn();
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  };
  const configure = (d) => {
    const mode =
        d?.kind === "stage"
          ? d.interval_minutes === 60
            ? "hourly"
            : "local"
          : "rainfall",
      item = d || defaultDataset(library, mode);
    setCfg({
      ...preset(mode, item?.id || ""),
      ...(mode !== "rainfall" && item?.sites?.length
        ? { site: item.sites[0] }
        : {}),
    });
    go("Training");
  };
  const upload = (files) =>
    act(async () => {
      const form = new FormData();
      for (const f of files)
        form.append("files", f, f.webkitRelativePath || f.name);
      form.append("source", provider.current);
      setStage(await api("stage", form));
      provider.current = "Local import";
      go("Datasets");
    });
  const pickFiles = (src) => {
    provider.current = typeof src === "string" ? src : "Local import";
    fileRef.current.click();
  };
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="logo">
            <Icon name="wave" size={28} />
          </div>
          <div>
            Q-Sense <small>Studio</small>
          </div>
        </div>
        <nav>
          {pages.map(([p, i]) => (
            <button
              key={p}
              className={page === p ? "on" : ""}
              onClick={() =>
                p === "Training" && !cfg.dataset ? configure() : go(p)
              }
            >
              <Icon name={i} />
              {p}
            </button>
          ))}
        </nav>
        <div className="sidebarFoot">
          <i className={online ? "dot" : "off"} />
          <div>
            {online ? "Local runtime" : "Connection lost"}
            <small>Q-Sense Studio</small>
          </div>
        </div>
      </aside>
      <main>
        <div className="topbar">
          <span>
            Workspace <b>/</b> {page}
          </span>
          <span className="localStatus">Tanauan City · PH</span>
        </div>
        <article aria-busy={busy}>
          <input
            type="file"
            multiple
            ref={fileRef}
            hidden
            onChange={(e) => {
              if (e.target.files.length) upload(e.target.files);
              e.target.value = "";
            }}
          />
          {err && (
            <div role="alert" className="notice">
              {err}
              <button className="textButton" onClick={() => setErr("")}>
                Dismiss
              </button>
            </div>
          )}
          {busy && (
            <div role="status" className="working">
              Working…
            </div>
          )}
          <Review
            records={stage}
            discard={() =>
              act(async () => {
                for (const r of stage) {
                  await api("stage/" + r.id, undefined, "DELETE");
                  setStage((list) => list.filter((x) => x.id !== r.id));
                }
              })
            }
            accept={() =>
              act(async () => {
                let first;
                for (const r of stage) {
                  const saved = await api("stage/" + r.id + "/commit", {});
                  first ??= saved.id;
                  setStage((s) => s.filter((v) => v.id !== r.id));
                }
                setSelected(first);
                await refresh();
                go("Datasets");
              })
            }
          />
          {page === "Overview" && (
            <Overview
              {...{ library, runs, go, configure, pickFiles }}
              openRun={(v) => {
                setActive(v);
                go("Runs");
              }}
              openData={(v) => {
                setSelected(v);
                go("Datasets");
              }}
            />
          )}
          {page === "Datasets" && (
            <Library
              {...{ library, selected, upload, pickFiles }}
              select={setSelected}
              onTrain={configure}
              onMap={(id, c) =>
                act(async () =>
                  setStage([await api("library/" + id + "/map", c)]),
                )
              }
            />
          )}
          {page === "Sources" && (
            <Sources
              sources={sources}
              onCollect={(s, query) =>
                act(async () =>
                  setStage(
                    await api(
                      "sources/" + s.id + (query ? "/hourly" : "/collect"),
                      query || {},
                    ),
                  ),
                )
              }
              attach={(s) => pickFiles(s.url)}
            />
          )}
          {page === "Training" && (
            <Training
              openPanel={openPanel}
              {...{
                library,
                devices,
                cfg,
                setCfg,
                act,
                runs,
                active,
                setActive,
                detail,
              }}
              start={(c) =>
                act(async () => {
                  const s = await api("runs", c);
                  await refresh();
                  setActive(s.id);
                  go("Runs");
                })
              }
            />
          )}
          {page === "Runs" && (
            <Monitor
              present={() => go("Training", true)}
              {...{ runs, active, setActive, detail, act, refresh, busy }}
              duplicate={(c) => {
                setCfg({ ...c, name: c.name + " · copy" });
                go("Training");
              }}
            />
          )}
        </article>
      </main>
    </div>
  );
}
createRoot(document.getElementById("root")).render(<App />);
