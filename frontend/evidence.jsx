import React, { useState, useEffect } from "react";
import { api, Circuit, Field, Plot, labels, num, Tag } from "./common";
import { Verification } from "./diagnostics";

export function ForecastEvidence({ id, current, unit, children }) {
  const [points, setPoints] = useState([]),
    [metric, setMetric] = useState("mae"),
    [error, setError] = useState("");
  useEffect(() => {
    let ok = true;
    setPoints([]);
    setError("");
    if (current && id)
      api("runs/" + id + "/series/" + current.job)
        .then((r) => ok && setPoints(r))
        .catch((e) => ok && setError(e.message));
    return () => {
      ok = false;
    };
  }, [id, current?.job]);
  const link = (file, label) => (
    <a
      className="button secondary"
      href={`/api/runs/${id}/download/${file}?job=${current?.job || 0}`}
    >
      {label}
    </a>
  );
  return (
    <section className="panel">
      <h2>Held-out forecasts</h2>
      {children}
      {error && <p role="alert">{error}</p>}
      <Plot
        rows={points}
        fields={[
          "observed",
          current.arm,
          ...Object.keys(current.metrics).filter((k) => k !== current.arm),
        ]}
        unit={unit === "mm" ? "Rainfall (mm)" : "Stage (cm)"}
      />
      <Field label="Metric">
        <select value={metric} onChange={(e) => setMetric(e.target.value)}>
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
        {num(current.trainable)} trainable · {num(current.exported)} exported ·{" "}
        {num(current.bytes / 1024, 1)} KiB · {num(current.latency_ms, 3)} ms per
        forecast on {current.benchmark_device}
      </p>
      <Verification
        key={id + "-" + current.job}
        id={id}
        job={current.job}
        unit={unit}
      />
      {current.arm === "qt" && current.circuit && (
        <Circuit
          qubits={current.circuit.qubits}
          depth={current.circuit.depth}
          info={current.circuit}
        />
      )}
    </section>
  );
}

export function ThesisEvaluation({ id, statistics, unit }) {
  const link = (file, label) => (
    <a
      className="button secondary"
      href={`/api/runs/${id}/download/${file}?job=${0}`}
    >
      {label}
    </a>
  );
  return (
    <section className="panel">
      <div className="panelHead">
        <h2>Thesis evaluation</h2>
        <Tag>{statistics.warning_tier}</Tag>
      </div>
      <p>
        {statistics.event_metrics?.length || 0} event summaries across this run
        · {statistics.comparisons?.length || 0} paired comparisons
      </p>
      {statistics.comparisons?.length > 0 && (
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
              {statistics.comparisons.map((r, i) => (
                <tr key={i}>
                  <td>{r.hypothesis}</td>
                  <td>
                    {r.left} / {r.right}
                  </td>
                  <td>{r.metric}</td>
                  <td>
                    {r.ci95
                      ? num(r.delta) +
                        " [" +
                        r.ci95.map((v) => num(v)).join(", ") +
                        "]"
                      : "Insufficient matched events"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {statistics.seed_means?.length > 0 && (
        <details>
          <summary>Held-out error by arm and fold</summary>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th>Arm</th>
                  <th>Fold</th>
                  <th>Initialization</th>
                  <th>Seeds</th>
                  <th>MAE ({unit})</th>
                  <th>RMSE ({unit})</th>
                </tr>
              </thead>
              <tbody>
                {statistics.seed_means.map((r) => (
                  <tr key={[r.arm, r.init, r.fold].join("/")}>
                    <td>{labels[r.arm] || r.arm}</td>
                    <td>{r.fold + 1}</td>
                    <td>{r.init}</td>
                    <td>{r.seeds}</td>
                    <td>{num(r.mae)}</td>
                    <td>{num(r.rmse)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
      <p>
        Differences are left minus right. Event-level intervals require at least
        two matched events. A negative MAE difference favors the left model.
      </p>
      {link("statistics.json", "Export evaluation")}
      {link("plan.json", "Split manifest")}
    </section>
  );
}

export function LearningMonitor({ history, epochs, unit, status, caption }) {
  const last = history.at(-1);
  if (!last) return null;
  const hasLoss = Number.isFinite(last.loss),
    initial = history[0].loss;
  const reduction =
    hasLoss && initial > 0 ? (100 * (initial - last.loss)) / initial : null;
  return (
    <section className="panel learningMonitor">
      <div className="panelHead">
        <h2>Learning curves &amp; training monitor</h2>
        <Tag>
          Epoch {last.epoch} / {epochs} · {num((100 * last.epoch) / epochs, 0)}%
        </Tag>
      </div>
      <progress
        aria-label="Training epoch progress"
        value={last.epoch}
        max={epochs}
      />
      <p>{caption}</p>
      <div className="miniStats">
        {hasLoss && (
          <span>
            Training MSE<b>{num(last.loss, 5)}</b>
          </span>
        )}
        <span>
          Validation MAE
          <b>
            {num(last.validation, 3)} {unit}
          </b>
        </span>
        {reduction != null && (
          <span>
            Loss reduction since epoch 1<b>{num(reduction, 1)}%</b>
          </span>
        )}
        <span>
          Status<b>{status}</b>
        </span>
      </div>
      <Plot
        rows={history}
        xkey="epoch"
        fields={["validation"]}
        unit={"Validation MAE (" + unit + ")"}
        filter={false}
        height={160}
      />
      {hasLoss && (
        <Plot
          rows={history}
          xkey="epoch"
          fields={["loss"]}
          unit="Training MSE · normalized"
          filter={false}
          height={145}
        />
      )}
    </section>
  );
}
