import { PanelView } from "./panel";
import {
  LearningMonitor,
  ForecastEvidence,
  ThesisEvaluation,
} from "./evidence";
import { Trace, TrainingView } from "./diagnostics";
import React, { useEffect, useState } from "react";
import {
  api,
  Circuit,
  Empty,
  Field,
  Heading,
  Icon,
  labels,
  num,
  Plot,
  Tag,
} from "./common";
export function defaultDataset(library, mode) {
  if (mode === "rainfall") {
    return library
      .filter((d) => d.kind === "rainfall")
      .sort((a, b) => b.rows - a.rows)[0];
  }
  return library.find(
    (d) =>
      d.kind === "stage" && d.interval_minutes === (mode === "local" ? 10 : 60),
  );
}

export function preset(mode, dataset = "") {
  const rain = mode === "rainfall";
  return {
    name: rain
      ? "Batangas rainfall pilot"
      : mode === "local"
        ? "Wawa · primary comparison"
        : "Hourly basin experiment",
    mode,
    dataset,
    site: rain ? "PH04010" : mode === "local" ? "C2" : "basin_gauge",
    features: [],
    arms: rain
      ? ["qt", "classical"]
      : ["qt", "classical", "compact", "softmax"],
    horizon: rain ? 1 : 3,
    hidden: rain ? 16 : 32,
    lookback: rain ? 12 : 36,
    depth: 2,
    width: 24,
    epochs: 40,
    lr: 0.003,
    batch: 64,
    seeds: rain ? [42] : [42, 43, 44, 45, 46],
    optimizer: "Adam",
    device: "cpu",
    patience: 0,
    folds: mode === "local" ? 3 : 1,
    val_start: rain ? "2014-01-01" : "",
    test_start: rain ? "2020-01-01" : "",
    events: [],
    threshold: null,
    warm_dataset: "",
    warm_site: "",
  };
}
export function Training({
  library,
  devices,
  cfg,
  setCfg,
  start,
  act,
  runs,
  active,
  setActive,
  detail,
  openPanel,
}) {
  const [view, setView] = useState(() =>
    openPanel
      ? "panel"
      : runs.some((r) => r.id === active && r.status === "running")
        ? "trace"
        : "setup",
  );
  const monitor = view === "trace";
  const [plan, setPlan] = useState(null),
    [seedText, setSeeds] = useState(cfg.seeds.join(", ")),
    [eventText, setEvents] = useState(
      cfg.events.map((e) => [e.name, e.start, e.end].join(",")).join("\n"),
    );
  const [observations, setObservations] = useState([]);
  useEffect(() => {
    let live = true;
    setObservations([]);
    if (cfg.dataset)
      api("library/" + cfg.dataset)
        .then((d) => {
          if (live)
            setObservations(
              (d.series || [])
                .filter((r) => r[3] === cfg.site)
                .map((r) => ({ timestamp: r[0], observed: r[1] })),
            );
        })
        .catch(() => {});
    return () => {
      live = false;
    };
  }, [cfg.dataset, cfg.site]);
  const put = (k, v) => setCfg((c) => ({ ...c, [k]: v }));
  useEffect(() => setPlan(null), [cfg, seedText, eventText]);
  const config = () => ({
    ...cfg,
    seeds: seedText.split(",").map((s) => Number(s.trim())),
    events: eventText.trim()
      ? eventText
          .trim()
          .split("\n")
          .map((line) => {
            const [name, start, end] = line.split(",").map((s) => s.trim());
            return { name, start, end };
          })
      : [],
  });
  const eligible = library.filter(
    (d) => d.kind === (cfg.mode === "rainfall" ? "rainfall" : "stage"),
  );
  const field = (k, title, type = "number", props = {}) => (
    <Field key={k} label={title}>
      <input
        type={type}
        step={k === "lr" || k === "threshold" ? "any" : undefined}
        value={cfg[k] ?? ""}
        {...props}
        onChange={(e) =>
          put(
            k,
            type === "number"
              ? e.target.value === ""
                ? null
                : Number(e.target.value)
              : e.target.value,
          )
        }
      />
    </Field>
  );
  const heading = (
    <Heading
      title="Training"
      desc={
        view === "panel"
          ? "Replay quantum training and compare matched forecasts."
          : monitor
            ? "Inspect the network as its weights change."
            : "Configure a reproducible, local experiment."
      }
    >
      <button
        className={view === "setup" ? "" : "secondary"}
        aria-pressed={view === "setup"}
        onClick={() => setView("setup")}
      >
        Experiment setup
      </button>
      <button
        className={monitor ? "" : "secondary"}
        aria-pressed={monitor}
        onClick={() => setView("trace")}
      >
        Inside the model
      </button>
      <button
        className={view === "panel" ? "" : "secondary"}
        aria-pressed={view === "panel"}
        onClick={() => setView("panel")}
      >
        Panel view
      </button>
    </Heading>
  );
  if (view === "panel")
    return (
      <>
        {heading}
        <PanelView key={active} {...{ runs, active, setActive, detail }} />
      </>
    );
  if (monitor)
    return (
      <>
        {heading}
        <TrainingView {...{ runs, active, setActive, detail }} />
      </>
    );
  return (
    <>
      {heading}
      <div className="trainGrid">
        <div>
          <section className="panel">
            <h2>Experiment</h2>
            <div className="formGrid">
              <Field label="Training preset">
                <select
                  value={cfg.mode}
                  onChange={(e) => {
                    const m = e.target.value,
                      p = preset(m, defaultDataset(library, m)?.id || "");
                    setCfg(p);
                    setSeeds(p.seeds.join(", "));
                    setEvents("");
                  }}
                >
                  {["rainfall", "hourly", "local"].map((m) => (
                    <option value={m} key={m}>
                      {labels[m]}
                    </option>
                  ))}
                </select>
              </Field>
              {field("name", "Run name", "text")}
              <Field label="Training dataset">
                <select
                  value={cfg.dataset}
                  onChange={(e) => put("dataset", e.target.value)}
                >
                  <option value="">Select a dataset</option>
                  {eligible.map((d) => (
                    <option value={d.id} key={d.id}>
                      {d.name}
                    </option>
                  ))}
                </select>
              </Field>
              {field("site", "Site or gauge ID", "text")}
            </div>
            {!eligible.length && (
              <p className="notice">
                {cfg.mode === "rainfall"
                  ? "Import the Batangas dekadal rainfall CSV."
                  : "Import calibrated gauge stage and time-aligned rainfall first. Local training also requires ten-minute weather observations."}
              </p>
            )}
          </section>
          <section className="panel">
            <h2>Models and features</h2>
            <div className="checks">
              {["qt", "classical", "compact", "softmax"].map((a) => (
                <label key={a}>
                  <input
                    type="checkbox"
                    checked={cfg.arms.includes(a)}
                    onChange={(e) =>
                      put(
                        "arms",
                        e.target.checked
                          ? [...cfg.arms, a]
                          : cfg.arms.filter((x) => x !== a),
                      )
                    }
                  />
                  {labels[a]}
                </label>
              ))}
            </div>
            <div className="formGrid">
              {field("lookback", "Lookback · steps")}
              {field("hidden", "LSTM hidden units")}
              <Field label="Forecast horizon">
                <select
                  value={cfg.horizon}
                  disabled={cfg.mode === "rainfall"}
                  onChange={(e) => put("horizon", Number(e.target.value))}
                >
                  {(cfg.mode === "rainfall" ? [1] : [1, 3, 6]).map((h) => (
                    <option key={h} value={h}>
                      {h} {cfg.mode === "rainfall" ? "dekad" : "hours"}
                    </option>
                  ))}
                </select>
              </Field>
              {field("depth", "Circuit blocks")}
              {field("width", "Mapping MLP width")}
              <Field label="Feature columns · blank uses all">
                <input
                  value={cfg.features.join(",")}
                  placeholder="stage_cm,rain_mm"
                  onChange={(e) =>
                    put(
                      "features",
                      e.target.value
                        ? e.target.value.split(",").map((s) => s.trim())
                        : [],
                    )
                  }
                />
              </Field>
            </div>
          </section>
          <section className="panel">
            <h2>Optimization</h2>
            <div className="formGrid">
              {field("epochs", "Epoch limit")}
              {field("batch", "Batch size")}
              {field("lr", "Learning rate")}
              {field("patience", "Early stopping · 0 disables")}
              <Field label="Optimizer">
                <select
                  value={cfg.optimizer}
                  onChange={(e) => put("optimizer", e.target.value)}
                >
                  {["Adam", "AdamW", "SGD"].map((a) => (
                    <option key={a}>{a}</option>
                  ))}
                </select>
              </Field>
              <Field label="Device">
                <select
                  value={cfg.device}
                  onChange={(e) => put("device", e.target.value)}
                >
                  {Object.entries(devices).map(([k, v]) => (
                    <option key={k} value={k} disabled={v !== "Ready"}>
                      {k.toUpperCase()} · {v}
                    </option>
                  ))}
                </select>
              </Field>
            </div>
          </section>
          <section className="panel">
            <h2>Evaluation</h2>
            <div className="formGrid">
              <Field label="Random seeds">
                <input
                  value={seedText}
                  onChange={(e) => setSeeds(e.target.value)}
                />
              </Field>
              {field("folds", "Chronological test folds")}
              {field("val_start", "First validation date", "date")}
              {field("test_start", "First test date", "date")}
              {cfg.mode !== "rainfall" &&
                field("threshold", "Approved site threshold · cm")}
            </div>
            {cfg.mode !== "rainfall" && (
              <Field label="Verified event intervals · name,start,end (one per line)">
                <textarea
                  rows="3"
                  value={eventText}
                  onChange={(e) => setEvents(e.target.value)}
                  placeholder="Storm name,2027-01-01T00:00,2027-01-02T23:50"
                />
              </Field>
            )}
            {cfg.mode === "local" && (
              <div className="formGrid">
                <Field label="Hourly source for transfer comparison">
                  <select
                    value={cfg.warm_dataset}
                    onChange={(e) => put("warm_dataset", e.target.value)}
                  >
                    <option value="">Scratch only</option>
                    {library
                      .filter((d) => d.kind === "stage")
                      .map((d) => (
                        <option value={d.id} key={d.id}>
                          {d.name}
                        </option>
                      ))}
                  </select>
                </Field>
                {field("warm_site", "Historical gauge ID", "text")}
              </div>
            )}
          </section>
        </div>
        <aside>
          <section className="panel sticky">
            <h2>Run preview</h2>
            <p>
              {plan
                ? "Validated against the selected observations."
                : "Check the data and experiment size before adding it to the queue."}
            </p>
            {plan && (
              <>
                <div className="miniStats">
                  <span>
                    Jobs<b>{plan.jobs}</b>
                  </span>
                  <span>
                    Qubits<b>{plan.qubits}</b>
                  </span>
                </div>
                <p>
                  {num(plan.qt_trainable)} QT parameters → {num(plan.weights)}{" "}
                  exported weights
                </p>
                <small>
                  Compact budget difference{" "}
                  {plan.compact_difference > 0 ? "+" : ""}
                  {plan.compact_difference} parameters
                </small>
                {plan.folds.map((f, i) => (
                  <p key={i}>
                    Fold {i + 1} · {f.windows.join(" / ")} train, validation and
                    test windows
                    <br />
                    <small>
                      {f.boundaries.map((s) => s.slice(0, 10)).join(" → ")}
                    </small>
                  </p>
                ))}
                <small>{plan.features.join(" · ")}</small>
                {observations.length > 0 && (
                  <Plot
                    rows={observations}
                    fields={["observed"]}
                    unit="First fold · validation and test boundaries"
                    height={150}
                    filter={false}
                    boundaries={[
                      ["Validation", plan.folds[0].boundaries[1]],
                      ["Test", plan.folds[0].boundaries[2]],
                    ]}
                  />
                )}
              </>
            )}
            <button
              className="wide secondary"
              onClick={() =>
                act(async () => setPlan(await api("preview", config())))
              }
            >
              Check configuration
            </button>
            <button
              className="wide"
              disabled={!plan}
              onClick={() => start(config())}
            >
              Queue experiment <Icon name="arrow" />
            </button>
          </section>
          <section className="panel">
            <h2>Quantum circuit</h2>
            <Circuit qubits={plan?.qubits} depth={cfg.depth} />
          </section>
        </aside>
      </div>
    </>
  );
}
export function Monitor({
  runs,
  active,
  setActive,
  detail,
  act,
  refresh,
  duplicate,
  present,
  busy,
}) {
  const [job, setJob] = useState(0);
  const state = detail?.state,
    prog = detail?.progress || {},
    res = detail?.results || [],
    current = res.find((r) => r.job === job) || res[0],
    id = state?.id;
  const control = (action) =>
    act(async () => {
      await api("runs/" + id + "/" + action, {});
      await refresh();
    });
  const link = (file, label) => (
    <a
      className="button secondary"
      href={`/api/runs/${id}/download/${file}?job=${current?.job || 0}`}
    >
      {label}
    </a>
  );
  return (
    <>
      <Heading
        title="Experiment runs"
        desc="Monitor training, inspect forecasts and export results."
      />
      <div className="libraryGrid">
        <section className="panel">
          <h2>Run history</h2>
          {runs.map((s) => (
            <button
              key={s.id}
              className={"listItem " + (s.id === active ? "selected" : "")}
              onClick={() => setActive(s.id)}
            >
              <Icon name="runs" />
              <span>
                {s.name}
                <small>
                  {s.status} · {s.created.slice(0, 10)}
                </small>
              </span>
            </button>
          ))}
          {!runs.length && (
            <Empty>Your queued experiments will appear here.</Empty>
          )}
        </section>
        <div>
          {state && (
            <>
              <section className="panel">
                <div className="panelHead">
                  <h2>{state.name}</h2>
                  <Tag>{state.status}</Tag>
                </div>
                <p>
                  {labels[detail.config.mode]} · {detail.config.site} ·{" "}
                  {detail.config.device.toUpperCase()} · {res.length}/
                  {state.jobs} jobs complete
                </p>
                {state.started && (
                  <small>
                    Elapsed wall time{" "}
                    {num(
                      (Date.parse(state.ended || new Date()) -
                        Date.parse(state.started)) /
                        60000,
                      1,
                    )}{" "}
                    min
                  </small>
                )}
                {state.error && <p className="notice">{state.error}</p>}
                <div className="actions">
                  {["running", "queued"].includes(state.status) && (
                    <button onClick={() => control("pause")}>Pause</button>
                  )}
                  {["paused", "failed"].includes(state.status) && (
                    <button onClick={() => control("resume")}>Resume</button>
                  )}
                  {["running", "queued", "paused", "pausing"].includes(
                    state.status,
                  ) && (
                    <button
                      className="secondary"
                      onClick={() => control("cancel")}
                    >
                      Cancel
                    </button>
                  )}
                  <button
                    className="secondary"
                    onClick={() => duplicate(detail.config)}
                  >
                    Duplicate and edit
                  </button>
                  {["completed", "failed", "cancelled"].includes(
                    state.status,
                  ) && (
                    <button
                      disabled={busy}
                      onClick={() =>
                        act(async () => {
                          const next = await api("runs/" + id + "/retrain", {});
                          await refresh();
                          setActive(next.id);
                        })
                      }
                    >
                      Retrain
                    </button>
                  )}
                  {detail.config.arms.includes("qt") && (
                    <button className="secondary" onClick={present}>
                      Present this run
                    </button>
                  )}
                  {link("config.json", "Configuration")}
                </div>
              </section>
              {prog.diagnostics && (
                <details className="panel trace">
                  <summary>Neural map and epoch diagnostics</summary>
                  <Trace d={prog.diagnostics} caption={prog.phase} />
                </details>
              )}
              {prog.history && (
                <LearningMonitor
                  history={prog.history}
                  epochs={detail.config.epochs}
                  unit={detail.config.mode === "rainfall" ? "mm" : "cm"}
                  status={state.status}
                  caption={
                    labels[prog.arm] +
                    " · " +
                    prog.phase +
                    " · fold " +
                    (prog.fold + 1) +
                    " · seed " +
                    prog.seed +
                    " · best epoch " +
                    prog.best_epoch
                  }
                />
              )}
              {current && (
                <ForecastEvidence
                  key={id + "-" + current.job}
                  {...{ id, current }}
                  unit={detail.config.mode === "rainfall" ? "mm" : "cm"}
                >
                  <Field label="Comparison job">
                    <select
                      value={current.job}
                      onChange={(e) => setJob(+e.target.value)}
                    >
                      {res.map((r) => (
                        <option key={r.job} value={r.job}>
                          {labels[r.arm]} · {r.init} · fold {r.fold + 1} · seed{" "}
                          {r.seed}
                        </option>
                      ))}
                    </select>
                  </Field>
                </ForecastEvidence>
              )}
              {detail.statistics?.jobs > 0 && (
                <ThesisEvaluation
                  unit={detail.config.mode === "rainfall" ? "mm" : "cm"}
                  id={id}
                  statistics={detail.statistics}
                />
              )}
            </>
          )}
        </div>
      </div>
    </>
  );
}
