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
export function Training({ library, devices, cfg, setCfg, start, act }) {
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
  return (
    <>
      <Heading
        title="Training setup"
        desc="Configure a reproducible, local experiment."
      />
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
                      p = preset(
                        m,
                        library.find(
                          (d) =>
                            d.kind ===
                            (m === "rainfall" ? "rainfall" : "stage"),
                        )?.id || "",
                      );
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
}) {
  const [job, setJob] = useState(0),
    [points, setPoints] = useState([]),
    [metric, setMetric] = useState("mae");
  const state = detail?.state,
    prog = detail?.progress || {},
    res = detail?.results || [],
    current = res.find((r) => r.job === job) || res[0],
    id = state?.id;
  useEffect(() => {
    let ok = true;
    setPoints([]);
    if (current && id)
      api("runs/" + id + "/series/" + current.job)
        .then((r) => ok && setPoints(r))
        .catch(() => {});
    return () => {
      ok = false;
    };
  }, [id, current?.job]);
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
                  {link("config.json", "Configuration")}
                </div>
              </section>
              {prog.history && (
                <section className="panel">
                  <div className="panelHead">
                    <h2>Learning curves</h2>
                    <Tag>Epoch {prog.history.at(-1)?.epoch}</Tag>
                  </div>
                  <p>
                    {labels[prog.arm]} · {prog.phase} · fold {prog.fold + 1} ·
                    seed {prog.seed} · best epoch {prog.best_epoch}
                  </p>
                  <Plot
                    rows={prog.history}
                    xkey="epoch"
                    fields={["validation"]}
                    unit="Validation MAE · physical units"
                    height={145}
                    filter={false}
                  />
                  <Plot
                    rows={prog.history}
                    xkey="epoch"
                    fields={["loss"]}
                    unit="Training MSE · normalized"
                    height={135}
                    filter={false}
                  />
                </section>
              )}
              {current && (
                <section className="panel">
                  <h2>Held-out forecasts</h2>
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
                  <Plot
                    rows={points}
                    fields={[
                      "observed",
                      current.arm,
                      ...Object.keys(current.metrics).filter(
                        (k) => k !== current.arm,
                      ),
                    ]}
                    unit={
                      detail.config.mode === "rainfall"
                        ? "Rainfall (mm)"
                        : "Stage (cm)"
                    }
                  />
                  <Field label="Metric">
                    <select
                      value={metric}
                      onChange={(e) => setMetric(e.target.value)}
                    >
                      {["mae", "rmse", "nse"].map((k) => (
                        <option value={k} key={k}>
                          {k.toUpperCase()}
                        </option>
                      ))}
                    </select>
                  </Field>
                  <div className="miniStats">
                    {Object.entries(current.metrics).map(([k, v]) => (
                      <span key={k}>
                        {labels[k]}
                        <b>{num(v?.[metric])}</b>
                      </span>
                    ))}
                  </div>
                  <div className="actions">
                    {link("forecast.csv", "Forecast CSV")}
                    {link("figure.svg", "Figure SVG")}
                    {link("inference.pt", "Inference model")}
                    {link("result.json", "Job metrics")}
                  </div>
                  <p>
                    {num(current.trainable)} trainable · {num(current.exported)}{" "}
                    exported · {num(current.bytes / 1024, 1)} KiB ·{" "}
                    {num(current.latency_ms, 3)} ms per forecast on{" "}
                    {current.benchmark_device}
                  </p>
                  {current.circuit && (
                    <Circuit
                      qubits={current.circuit.qubits}
                      depth={current.circuit.depth}
                      info={current.circuit}
                    />
                  )}
                </section>
              )}
              {detail.statistics?.jobs > 0 && (
                <section className="panel">
                  <div className="panelHead">
                    <h2>Thesis evaluation</h2>
                    <Tag>{detail.statistics.warning_tier}</Tag>
                  </div>
                  <p>
                    {detail.statistics.event_metrics?.length || 0} event
                    summaries · {detail.statistics.comparisons?.length || 0}{" "}
                    paired comparisons
                  </p>
                  {detail.statistics.comparisons?.length > 0 && (
                    <div className="scroll">
                      <table>
                        <thead>
                          <tr>
                            <th>Hypothesis</th>
                            <th>Contrast</th>
                            <th>Metric</th>
                            <th>Δ [95% CI]</th>
                          </tr>
                        </thead>
                        <tbody>
                          {detail.statistics.comparisons.map((r, i) => (
                            <tr key={i}>
                              <td>{r.hypothesis}</td>
                              <td>
                                {r.left} / {r.right}
                              </td>
                              <td>{r.metric}</td>
                              <td>
                                {num(r.delta)} [
                                {(r.ci95 || []).map((v) => num(v)).join(", ")}]
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                  {link("statistics.json", "Export evaluation")}
                  {link("plan.json", "Split manifest")}
                </section>
              )}
            </>
          )}
        </div>
      </div>
    </>
  );
}
