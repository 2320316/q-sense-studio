import { ConnectionRevision } from "./common";
import React, { useEffect, useState } from "react";
import { api, Field, labels, num, Tag } from "./common";

const mint = "#77dfce",
  amber = "#eab67b";
function Network({ d }) {
  const n = d.edges.length,
    m = d.features.length;
  const height = Math.max(152, Math.max(n, m) * 19 + 30);
  const y = (i, count) => 16 + ((height - 32) * (i + 0.5)) / count;
  const strength = Math.max(
    1e-6,
    ...d.edges.flat().map(Math.abs),
    ...d.readout.map(Math.abs),
  );
  const line = (x1, y1, x2, y2, w, key, label) => (
    <line
      key={key}
      {...{ x1, y1, x2, y2 }}
      stroke={w < 0 ? amber : mint}
      strokeOpacity={0.14 + (0.82 * Math.abs(w)) / strength}
      strokeWidth={0.7 + (1.8 * Math.abs(w)) / strength}
    >
      <title>
        {label} · {num(w, 4)}
      </title>
    </line>
  );
  return (
    <figure>
      <figcaption>Neural map</figcaption>
      <svg
        role="img"
        aria-label="Signed weights from features into LSTM candidates and from hidden units to the forecast"
        viewBox={`0 0 440 ${height}`}
      >
        {d.edges.map((row, i) =>
          row.map((w, j) =>
            line(
              130,
              y(j, m),
              286,
              y(i, n),
              w,
              i + "-" + j,
              d.features[j] + " → h" + i,
            ),
          ),
        )}
        {d.readout.map((w, i) =>
          line(286, y(i, n), 383, height / 2, w, i, "h" + i + " → forecast"),
        )}
        {d.features.map((name, j) => (
          <g key={name}>
            <text x="122" y={y(j, m) + 4} textAnchor="end">
              {name}
            </text>
            <circle cx="130" cy={y(j, m)} r="3" fill="#c3ccca" />
          </g>
        ))}
        {d.edges.map((row, i) => (
          <circle
            key={i}
            cx="286"
            cy={y(i, n)}
            r="4"
            fill={d.activations.at(-1)[i] < 0 ? amber : mint}
          >
            <title>
              h{i} · {num(d.activations.at(-1)[i], 4)}
            </title>
          </circle>
        ))}
        <circle cx="383" cy={height / 2} r="5" fill="#e8edeb" />
        <text x="383" y={height / 2 + 22} textAnchor="middle">
          Forecast
        </text>
      </svg>
      <small>
        {n} of {d.hidden} units · teal positive, amber negative · stronger
        weights draw brighter edges
      </small>
    </figure>
  );
}

function Heatmap({ title, rows, names, caption }) {
  const width = 440,
    height = names.length * 18 + 18,
    left = 90;
  const cell = (width - left) / rows.length;
  return (
    <figure>
      <figcaption>{title}</figcaption>
      <svg
        role="img"
        aria-label={title + " by time step or circuit layer"}
        viewBox={`0 0 ${width} ${height}`}
      >
        {names.map((name, i) => (
          <g key={name}>
            <text x={left - 8} y={i * 18 + 14} textAnchor="end">
              {name}
            </text>
            {rows.map((values, j) => (
              <rect
                key={j}
                x={left + j * cell}
                y={i * 18 + 2}
                width={Math.max(1, cell - 0.7)}
                height="14"
                fill={values[i] < 0 ? amber : mint}
                opacity={0.1 + 0.9 * Math.min(1, Math.abs(values[i]))}
              >
                <title>
                  {name} · step {j + 1} · {num(values[i], 4)}
                </title>
              </rect>
            ))}
          </g>
        ))}
        <text x={left} y={height - 1}>
          1
        </text>
        <text x={width} y={height - 1} textAnchor="end">
          {rows.length}
        </text>
      </svg>
      <small>{caption}</small>
    </figure>
  );
}

export function Trace({ d, caption }) {
  if (!d)
    return (
      <p>
        No epoch trace is available. A new training run records the first
        snapshot after epoch 1.
      </p>
    );
  const q = d.quantum;
  return (
    <>
      <div className="traceMeta">
        <Tag>Epoch {d.epoch}</Tag>
        <strong>{labels[d.arm]}</strong>
        <small>{caption}</small>
      </div>
      <div className="traceGrid">
        <Network d={d} />
        <div>
          <Heatmap
            title="LSTM gate activity"
            rows={d.gates}
            names={["Input", "Forget", "Candidate", "Output"]}
            caption="Mean across hidden units at each step of the first training window"
          />
          <Heatmap
            title="Hidden state"
            rows={d.activations}
            names={d.readout.map((_, i) => "h" + i)}
            caption="Unit activation across the same window · −1 to +1"
          />
        </div>
      </div>
      {q && (
        <div className="quantumReadout">
          <div>
            <h3>Quantum state vector</h3>
            <small>
              {q.qubits} simulated qubits · {q.angles.length} RY / CNOT blocks ·
              most probable basis states
            </small>
            {q.states.map((id, i) => (
              <div key={id} className="basisRow">
                <code>{id.toString(2).padStart(q.qubits, "0")}</code>
                <progress
                  value={q.probabilities[i]}
                  max={q.probabilities[0] || 1}
                />
                <span>{num(100 * q.probabilities[i], 2)}%</span>
              </div>
            ))}
          </div>
          <div>
            <h3>Rotation angles</h3>
            <small>Learned RY angle per qubit, in radians</small>
            <div className="scroll">
              <table>
                <thead>
                  <tr>
                    <th>Wire</th>
                    {q.angles.map((_, i) => (
                      <th key={i}>Block {i + 1}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {q.angles[0].map((_, j) => (
                    <tr key={j}>
                      <td>q{j}</td>
                      {q.angles.map((row, i) => (
                        <td key={i}>{num(row[j], 3)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export function TrainingView({ runs, active, setActive, detail }) {
  const [job, setJob] = useState("latest"),
    [saved, setSaved] = useState(null),
    [error, setError] = useState("");
  useEffect(() => {
    setJob("latest");
  }, [active]);
  useEffect(() => {
    let ok = true;
    setSaved(null);
    setError("");
    if (job !== "latest" && detail?.state?.id === active)
      api(`runs/${active}/jobs/${job}/inspection`)
        .then((v) => ok && setSaved(v.diagnostics))
        .catch((e) => ok && setError(e.message));
    return () => {
      ok = false;
    };
  }, [active, detail?.state?.id, job]);
  const p = detail?.progress || {},
    results = detail?.results || [];
  const selected = job === "latest" ? p : results.find((r) => r.job === +job);
  return (
    <section className="panel trace">
      <div className="panelHead">
        <h2>Inside the model</h2>
        {detail && <Tag>{detail.state.status}</Tag>}
      </div>
      {runs.length > 0 && (
        <div className="formGrid">
          <Field label="Experiment">
            <select value={active} onChange={(e) => setActive(e.target.value)}>
              {runs.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.name} · {r.created.slice(0, 16).replace("T", " ")} UTC ·{" "}
                  {r.status}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Training snapshot">
            <select value={job} onChange={(e) => setJob(e.target.value)}>
              <option value="latest">Follow latest epoch</option>
              {results.map((r) => (
                <option key={r.job} value={r.job}>
                  {labels[r.arm]} · {r.init} · fold {r.fold + 1} · seed {r.seed}
                </option>
              ))}
            </select>
          </Field>
        </div>
      )}
      {error && <p role="alert">{error}</p>}
      <Trace
        d={job === "latest" ? p.diagnostics : saved}
        caption={
          selected && selected.fold != null
            ? `Fold ${selected.fold + 1} · seed ${selected.seed} · ${selected.init} · ${selected.phase || "last fit epoch"}`
            : ""
        }
      />
    </section>
  );
}

export function Verification({ id, job, unit }) {
  const revision = React.useContext(ConnectionRevision);
  const [limit, setLimit] = useState(""),
    [rounds, setRounds] = useState(20),
    [seed, setSeed] = useState(42),
    [report, setReport] = useState(null);
  const [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const endpoint = `runs/${id}/jobs/${job}`;
  useEffect(() => {
    let ok = true;
    setError("");
    api(endpoint + "/inspection")
      .then((v) => ok && setReport(v.verification))
      .catch((e) => ok && setError(e.message));
    return () => {
      ok = false;
    };
  }, [endpoint, revision]);
  const run = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      setReport(
        await api(endpoint + "/verify", {
          rounds: +rounds,
          seed: +seed,
          max_mae: limit === "" ? null : +limit,
        }),
      );
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };
  const checks = report?.rounds || [],
    stale = checks.map((r) => r.stale_mae).filter(Number.isFinite);
  return (
    <div className="verification">
      <h3>Verify exported model</h3>
      <p>
        Replay the held-out fold. Each seeded round selects up to 32 consecutive
        test windows, repeats inference, then simulates a delayed reading by
        holding the last input step at its previous value.
      </p>
      <form onSubmit={run} className="weatherTools">
        <Field label="Rounds">
          <input
            type="number"
            min="1"
            max="100"
            required
            value={rounds}
            onChange={(e) => setRounds(e.target.value)}
          />
        </Field>
        <Field label="Trial seed">
          <input
            type="number"
            min="0"
            max="4294967295"
            required
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
          />
        </Field>
        <Field label={`Delayed MAE limit (${unit})`}>
          <input
            type="number"
            min="0"
            step="any"
            placeholder="Optional"
            value={limit}
            onChange={(e) => setLimit(e.target.value)}
          />
        </Field>
        <button disabled={busy}>
          {busy ? "Verifying…" : "Run stress test"}
        </button>
      </form>
      {error && (
        <p role="alert" className="notice">
          {error}
        </p>
      )}
      {report && (
        <div aria-live="polite">
          <div className="panelHead">
            <h3>
              {report.passed ? "Replay checks passed" : "Replay checks failed"}
            </h3>
            <Tag>
              {checks.length} rounds · seed {report.seed}
            </Tag>
          </div>
          <p>
            {report.count} held-out targets · archived predictions{" "}
            {report.exported_matches ? "match" : "differ"} · labels{" "}
            {report.labels_match ? "match" : "differ"} ·{" "}
            {checks.filter((r) => r.repeatable && r.finite).length}/
            {checks.length} repeatable and finite trials
          </p>
          <div className="miniStats">
            <span>
              Held-out MAE
              <b>
                {num(report.mae)} {report.unit}
              </b>
            </span>
            <span>
              Persistence MAE
              <b>
                {num(report.baseline_mae)} {report.unit}
              </b>
            </span>
            <span>
              Worst delayed-block MAE
              <b>
                {stale.length ? num(Math.max(...stale)) : "—"} {report.unit}
              </b>
            </span>
          </div>
          <p>
            {report.paired_mae != null && report.baseline_mae != null
              ? `Model MAE ${num(report.paired_mae)} ${report.unit} is ${report.paired_mae < report.baseline_mae ? "below" : report.paired_mae > report.baseline_mae ? "above" : "equal to"} persistence on ${report.paired_count} paired targets.`
              : "No paired baseline available."}{" "}
            Blocks can overlap.
          </p>
          {report.max_mae != null && (
            <p>
              <strong>
                {report.within_limit}/{checks.length} delayed-input trials
              </strong>{" "}
              met your MAE limit of {num(report.max_mae)} {report.unit}.
            </p>
          )}
          <a href={`/api/runs/${id}/download/verification.json?job=${job}`}>
            Download trial results
          </a>
        </div>
      )}
    </div>
  );
}
