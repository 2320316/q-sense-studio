import React, { useEffect, useRef, useState } from "react";
import {
  ForecastEvidence,
  ThesisEvaluation,
  LearningMonitor,
} from "./evidence";
import { ReferenceEvidence } from "./reference";
import { api, Field, labels, num, Tag, Plot } from "./common";

function Comparison({ qt, classical, ablation, compact, plan, unit }) {
  const count = qt?.trainable ?? plan.qt_trainable,
    control = classical?.trainable ?? plan.weights;
  const errors = [qt, classical, ablation, compact]
    .filter(Boolean)
    .map((r) => [labels[r.arm], r.metrics[r.arm]?.mae]);
  for (const [key, metric] of Object.entries(qt?.metrics || {})) {
    if (key !== "qt" && metric?.mae != null)
      errors.push([labels[key] || key, metric.mae]);
  }
  const delta =
    qt && classical
      ? qt.metrics.qt.mae - classical.metrics.classical.mae
      : null;
  const saving = 100 * (1 - count / control);
  return (
    <section className="panel comparison">
      <h2>What changed with quantum training?</h2>
      <div className="compareGrid">
        <div>
          <h3>Parameters optimized during training</h3>
          <Bars
            rows={[
              ["QT-LSTM", count],
              ["Classical LSTM", control],
            ]}
            digits={0}
          />
          <strong>
            {num(Math.abs(saving), 1)}% {saving >= 0 ? "fewer" : "more"}{" "}
            trainable parameters
          </strong>
          <small>
            Both export a classical LSTM with {num(plan.weights, 0)} parameters.
            Counts follow the configured architecture.
          </small>
        </div>
        <div>
          <h3>Held-out forecast error · MAE ({unit})</h3>
          {errors.length > 0 && <Bars rows={errors} />}
          {delta == null ? (
            <p>
              A paired error comparison appears when both the QT and
              dimension-matched classical jobs finish for this fold, seed and
              initialization.
            </p>
          ) : (
            <>
              <strong>
                {delta === 0
                  ? "Equal held-out MAE"
                  : `QT error is ${num(Math.abs(delta))} ${unit} ${delta > 0 ? "higher" : "lower"}`}
              </strong>
              <small>
                Same test fold, seed and initialization · lower MAE is better
              </small>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
function Bars({ rows, digits = 2 }) {
  const available = rows.filter(([, v]) => Number.isFinite(v)),
    max = Math.max(1e-10, ...available.map(([, v]) => v));
  return (
    <div className="compareBars">
      {available.map(([name, v]) => (
        <div key={name}>
          <span>{name}</span>
          <span className="barTrack">
            <i
              style={{
                width: (v / max) * 100 + "%",
                background: name === labels.qt ? "var(--accent)" : "#94a3a1",
              }}
            />
          </span>
          <b>{num(v, digits)}</b>
        </div>
      ))}
    </div>
  );
}

function CircuitFrame({ frame, block, setBlock }) {
  const n = frame.qubits,
    angles = frame.angles[block] || frame.angles[0],
    height = 44 + n * 26;
  const x = (q) => 176 + ((q + 0.5) * 210) / n,
    y = (q) => 34 + 26 * q,
    right = frame.one_probabilities ? 530 : 404;
  return (
    <figure className="wireDiagram">
      <div className="panelHead">
        <h3>Simulated quantum circuit</h3>
        <Field label="Rotation block">
          <select value={block} onChange={(e) => setBlock(+e.target.value)}>
            {frame.angles.map((_, i) => (
              <option value={i} key={i}>
                {i + 1} of {frame.angles.length}
              </option>
            ))}
          </select>
        </Field>
      </div>
      <svg
        role="img"
        aria-label={`Block ${block + 1}, learned RY angles and ordered CNOT ring on ${n} qubits`}
        viewBox={`0 0 ${right} ${height}`}
      >
        <text x="44" y="12">
          RY rotation
        </text>
        <text x="200" y="12">
          CNOT ring →
        </text>
        {frame.one_probabilities && (
          <text x="401" y="12">
            Final P(1)
          </text>
        )}
        {angles.map((angle, q) => (
          <g key={q}>
            <text x="0" y={y(q) + 4}>
              q{q}
            </text>
            <path d={`M34 ${y(q)}H390`} stroke="#495859" />
            <rect
              x="40"
              y={y(q) - 10}
              width="103"
              height="20"
              rx="3"
              fill="#25332f"
            />
            <circle cx="54" cy={y(q)} r="7" stroke="#69877d" fill="none" />
            <line
              x1="54"
              y1={y(q)}
              x2={54 + 6 * Math.sin(angle)}
              y2={y(q) - 6 * Math.cos(angle)}
              stroke="#77dfce"
              strokeWidth="2"
            />
            <text x="69" y={y(q) + 4}>
              {num(angle, 3)} rad
            </text>
            {frame.one_probabilities && (
              <g>
                <rect
                  x="401"
                  y={y(q) - 3}
                  width="70"
                  height="6"
                  fill="#354043"
                />
                <rect
                  x="401"
                  y={y(q) - 3}
                  width={70 * frame.one_probabilities[q]}
                  height="6"
                  fill="#77dfce"
                />
                <text x="478" y={y(q) + 4}>
                  {num(100 * frame.one_probabilities[q], 1)}%
                </text>
              </g>
            )}
          </g>
        ))}
        {angles.map((_, q) => {
          const target = (q + 1) % n;
          return (
            <g key={q} stroke="#eab67b" fill="#eab67b">
              <title>
                Control q{q} → target q{target}
              </title>
              <line x1={x(q)} y1={y(q)} x2={x(q)} y2={y(target)} />
              <circle cx={x(q)} cy={y(q)} r="2.7" />
              <circle cx={x(q)} cy={y(target)} r="4.3" fill="#181e20" />
              <path d={`M${x(q) - 3} ${y(target)}h6m-3 -3v6`} />
            </g>
          );
        })}
      </svg>
      <figcaption>
        Each dial shows one learned rotation. Each dot controls a flip on the
        connected wire.{" "}
        {frame.one_probabilities &&
          "P(1) is the probability of measuring 1 after all circuit blocks."}
      </figcaption>
    </figure>
  );
}

function QubitHistory({ frames, block }) {
  const [wire, setWire] = useState(0);
  const last = frames.at(-1);
  if (!last?.one_probabilities) return null;
  const rows = frames.map((f) => ({
    epoch: f.epoch,
    "P(1)": 100 * f.one_probabilities[wire],
    "RY angle": f.angles[block][wire],
  }));
  return (
    <details className="qubitHistory">
      <summary>Qubit evolution across epochs</summary>
      <Field label="Inspect wire">
        <select value={wire} onChange={(e) => setWire(+e.target.value)}>
          {last.one_probabilities.map((_, i) => (
            <option key={i} value={i}>
              q{i}
            </option>
          ))}
        </select>
      </Field>
      <div className="compareGrid">
        <Plot
          rows={rows}
          fields={["P(1)"]}
          xkey="epoch"
          unit={"q" + wire + " · final measurement probability (%)"}
          filter={false}
          height={150}
        />
        <Plot
          rows={rows}
          fields={["RY angle"]}
          xkey="epoch"
          unit={"Block " + (block + 1) + " · learned rotation (rad)"}
          filter={false}
          height={150}
        />
      </div>
    </details>
  );
}

function QuantumReplay({
  run,
  job,
  warm,
  ongoing,
  live,
  result,
  unit,
  epochs,
  status,
}) {
  const [frames, setFrames] = useState([]),
    [index, setIndex] = useState(null),
    [playing, setPlaying] = useState(false),
    [block, setBlock] = useState(0),
    [error, setError] = useState("");
  const cursor = useRef(0);
  useEffect(() => {
    let disposed = false,
      timer;
    async function poll() {
      try {
        let batch;
        do {
          batch = await api(
            `runs/${run}/quantum/${job}?after=${cursor.current}&warm=${warm}`,
          );
          if (disposed) return;
          if (batch.length) {
            cursor.current = batch.at(-1).epoch;
            const next = batch;
            setFrames((prev) => prev.concat(next));
          }
        } while (batch.length === 64);
        setError("");
      } catch (e) {
        if (!disposed) setError(e.message);
      }
      if (!disposed && ongoing) timer = setTimeout(poll, 1200);
    }
    poll();
    return () => {
      disposed = true;
      clearTimeout(timer);
    };
  }, [run, job, warm, ongoing]);
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(
      () => setIndex((i) => Math.min((i ?? 0) + 1, frames.length - 1)),
      750,
    );
    return () => clearInterval(timer);
  }, [playing, frames.length]);
  useEffect(() => {
    if (playing && index >= frames.length - 1) setPlaying(false);
  }, [index, frames.length, playing]);
  const fallback = live?.diagnostics?.quantum
    ? {
        ...live.diagnostics.quantum,
        epoch: live.diagnostics.epoch,
        validation: live.history?.at(-1)?.validation,
      }
    : !warm && result?.circuit
      ? { ...result.circuit, epoch: result.best_epoch }
      : null;
  const selected =
    index == null ? frames.length - 1 : Math.min(index, frames.length - 1);
  const frame = frames[selected] || fallback;
  const v = frame?.validation,
    initial = frames[0]?.validation;
  return (
    <section className="panel replay">
      <div className="panelHead">
        <div>
          <h2>Circuit replay</h2>
          <small>
            {warm ? "Hourly pretraining" : "Target-fold training"} · epoch-end
            measurements
          </small>
        </div>
        {frame && <Tag>Epoch {frame.epoch}</Tag>}
      </div>
      {error && <p role="alert">{error}</p>}
      {!frame ? (
        <p>
          {ongoing
            ? "Waiting for the first quantum epoch."
            : "No circuit has been recorded for this phase."}
        </p>
      ) : (
        <>
          <div className="playback">
            <button
              disabled={frames.length < 2}
              onClick={() => {
                if (playing) {
                  setPlaying(false);
                  return;
                }
                setIndex(selected >= frames.length - 1 ? 0 : selected);
                setPlaying(true);
              }}
            >
              {playing ? "Pause replay" : "Play epochs"}
            </button>
            <label>
              Epoch{" "}
              <input
                aria-label="Recorded quantum epoch"
                type="range"
                min="0"
                max={Math.max(0, frames.length - 1)}
                value={Math.max(0, selected)}
                disabled={frames.length < 2}
                onChange={(e) => {
                  setPlaying(false);
                  setIndex(+e.target.value);
                }}
              />
            </label>
            <button
              className="secondary"
              disabled={!frames.length}
              onClick={() => {
                setPlaying(false);
                setIndex(null);
              }}
            >
              Follow latest
            </button>
          </div>
          {!frames.length && (
            <small>
              {live ? "Latest recorded state" : "Saved best-validation circuit"}{" "}
              · retrain to capture epoch replay.
            </small>
          )}
          <div className="quantumGrid">
            <CircuitFrame {...{ frame, block, setBlock }} />
            <div className="basisPanel">
              <h3>Basis states</h3>
              <small>
                Most probable outcomes among {num(2 ** frame.qubits, 0)} states
              </small>
              <Bars
                rows={frame.states
                  .slice(0, 6)
                  .map((id, i) => [
                    id.toString(2).padStart(frame.qubits, "0"),
                    100 * frame.probabilities[i],
                  ])}
              />
              <small>
                Measurement probability (%) · bars scaled to the leading state
              </small>
              {v != null && (
                <div className="validationRead">
                  <h3>Validation MAE at epoch {frame.epoch}</h3>
                  <strong>
                    {num(v)} {unit}
                  </strong>
                  {initial != null && (
                    <small>
                      Epoch {frames[0].epoch} measured {num(initial)} {unit}.
                    </small>
                  )}
                </div>
              )}
            </div>
          </div>
          <p className="circuitFlow">
            {frame.qubits} simulated qubits → basis probabilities → learned
            weight mapper → classical LSTM forecast
          </p>
          {selected >= 0 && (
            <QubitHistory
              frames={frames.slice(0, selected + 1)}
              block={block}
            />
          )}
          {selected >= 0 && (
            <LearningMonitor
              history={frames.slice(0, selected + 1)}
              epochs={epochs}
              unit={unit}
              status={
                playing
                  ? "Replaying"
                  : index != null
                    ? "Selected epoch"
                    : status
              }
              caption={
                "QT-LSTM · " +
                (warm ? "hourly pretraining" : "target fold") +
                " · through epoch " +
                frame.epoch
              }
            />
          )}
        </>
      )}
    </section>
  );
}

export function PanelView({ runs, active, setActive, detail }) {
  const [choice, setChoice] = useState("follow"),
    [phase, setPhase] = useState("auto"),
    [focus, setFocus] = useState(false),
    [section, setSection] = useState("Circuit"),
    [reference, setReference] = useState(null);
  useEffect(() => {
    let ok = true;
    api("presentation/reference")
      .then((v) => ok && setReference(v))
      .catch(() => {});
    return () => {
      ok = false;
    };
  }, []);
  const shell = useRef(null);
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") setFocus(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  const plan = detail?.plan,
    p = detail?.progress,
    results = detail?.results || [];
  const jobs = (plan?.specs || [])
    .map(([fold, seed, arm, init], job) => ({ fold, seed, arm, init, job }))
    .filter((j) => j.arm === "qt");
  const matched = (a, b) =>
    a && b && a.fold === b.fold && a.seed === b.seed && a.init === b.init;
  const target =
    jobs.find((j) => String(j.job) === choice) ||
    jobs.find((j) => matched(j, p)) ||
    jobs[0];
  const qt = results.find((j) => j.job === target?.job);
  const classical = results.find(
    (j) => j.arm === "classical" && matched(j, target),
  );
  const ablation = results.find(
    (j) => j.arm === "softmax" && matched(j, target),
  );
  const compact = results.find(
    (j) => j.arm === "compact" && matched(j, target),
  );
  const warm =
    target?.init === "transfer" &&
    (phase === "warm" ||
      (phase === "auto" &&
        p?.job === target?.job &&
        p?.phase === "hourly pretraining"));
  const live =
    p?.job === target?.job && (p?.phase === "hourly pretraining") === warm
      ? p
      : null;
  const ongoing = ["queued", "running", "pausing", "paused"].includes(
    detail?.state.status,
  );
  const unit = detail?.config.mode === "rainfall" ? "mm" : "cm";
  return (
    <div
      ref={shell}
      className={"panelPresentation" + (focus ? " projection" : "")}
    >
      <div className="panelHead">
        <div>
          <h2>Quantum training comparison</h2>
          <small>
            {detail?.state.name || "Select an experiment"} ·{" "}
            {labels[detail?.config.mode]} {detail?.config.site}
          </small>
        </div>
        <button
          className="secondary"
          onClick={() => {
            setFocus(!focus);
            if (!focus) shell.current?.scrollTo(0, 0);
          }}
        >
          {focus ? "Exit panel view" : "Expand for panel"}
        </button>
      </div>
      <div
        className="actions presentationNav"
        role="group"
        aria-label="Panel evidence"
      >
        {[
          "Circuit",
          "Forecasts & verification",
          "Thesis evaluation",
          ...(reference ? ["Wawa screenshots"] : []),
        ].map((label) => (
          <button
            key={label}
            className={section === label ? "" : "secondary"}
            aria-pressed={section === label}
            onClick={() => setSection(label)}
          >
            {label}
          </button>
        ))}
      </div>
      {section !== "Wawa screenshots" && (
        <div className="formGrid presentationInputs">
          <Field label="Experiment">
            <select value={active} onChange={(e) => setActive(e.target.value)}>
              {runs.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.name} · {r.created.slice(0, 16).replace("T", " ")} UTC
                </option>
              ))}
            </select>
          </Field>
          {jobs.length > 0 && (
            <Field label="Matched experiment jobs">
              <select
                value={choice}
                onChange={(e) => {
                  setChoice(e.target.value);
                  setPhase("auto");
                }}
              >
                <option value="follow">Follow active comparison</option>
                {jobs.map((j) => (
                  <option value={j.job} key={j.job}>
                    Fold {j.fold + 1} · seed {j.seed} · {j.init}
                  </option>
                ))}
              </select>
            </Field>
          )}
          {target?.init === "transfer" && (
            <Field label="Phase">
              <select value={phase} onChange={(e) => setPhase(e.target.value)}>
                <option value="auto">Follow current phase</option>
                <option value="warm">Hourly pretraining</option>
                <option value="fit">Target-fold training</option>
              </select>
            </Field>
          )}
        </div>
      )}
      {section === "Wawa screenshots" && reference ? (
        <ReferenceEvidence reference={reference} />
      ) : !target ? (
        <p>
          This view needs a run with QT-LSTM enabled. Include the
          dimension-matched classical LSTM for a paired accuracy comparison.
        </p>
      ) : (
        <>
          <p className="pairContext">
            Fold {target.fold + 1} · seed {target.seed} · {target.init} ·{" "}
            {detail.state.status}
          </p>
          {section === "Circuit" && (
            <>
              <QuantumReplay
                key={active + "-" + target.job + "-" + warm}
                run={active}
                job={target.job}
                result={qt}
                epochs={detail.config.epochs}
                status={detail.state.status}
                {...{ warm, live, unit, ongoing }}
              />
              <Comparison
                {...{ qt, classical, ablation, compact, plan, unit }}
              />
            </>
          )}
          {section === "Forecasts & verification" && (
            <>
              <Comparison
                {...{ qt, classical, ablation, compact, plan, unit }}
              />
              {qt ? (
                <ForecastEvidence
                  key={qt.job}
                  id={active}
                  current={qt}
                  unit={unit}
                >
                  <p>
                    QT-LSTM · best validation epoch {qt.best_epoch} · held-out
                    test fold {qt.fold + 1}
                  </p>
                </ForecastEvidence>
              ) : (
                <p>
                  Held-out forecasts become available when the selected QT job
                  finishes.
                </p>
              )}
            </>
          )}
          {section === "Thesis evaluation" &&
            (detail.statistics?.jobs ? (
              <ThesisEvaluation
                unit={unit}
                id={active}
                statistics={detail.statistics}
              />
            ) : (
              <p>
                The run-wide thesis evaluation becomes available when all
                comparison jobs finish.
              </p>
            ))}
        </>
      )}
    </div>
  );
}
