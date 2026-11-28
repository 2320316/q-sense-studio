import React, { useState, useEffect, useRef } from "react";

export const ConnectionRevision = React.createContext(0);
export class ConnectionError extends Error {
  constructor() {
    super("Cannot reach the local Q-Sense Studio server.");
  }
}

export async function api(path, body, method) {
  const form = body instanceof FormData;
  const options = {
    method: method || (body === undefined ? "GET" : "POST"),
    headers: {
      "X-QSense": "studio",
      ...(!form && body !== undefined
        ? { "Content-Type": "application/json" }
        : {}),
    },
    body: body === undefined ? undefined : form ? body : JSON.stringify(body),
  };
  let response;
  try {
    response = await fetch("/api/" + path, options);
  } catch (error) {
    if (error instanceof TypeError) throw new ConnectionError();
    throw error;
  }
  if (!response.ok) {
    const error = await response
      .json()
      .catch(() => ({ detail: response.statusText }));
    throw Error(
      typeof error.detail === "string"
        ? error.detail
        : JSON.stringify(error.detail),
    );
  }
  return response.json();
}
const icons = {
  home: "M3 10l9-7 9 7v10H3z M9 20v-7h6v7",
  data: "M3 6c0-4 18-4 18 0s-18 4-18 0 M3 6v12c0 4 18 4 18 0V6 M3 12c0 4 18 4 18 0",
  source: "M5 7h9m-4-4 4 4-4 4 M19 17h-9m4-4-4 4 4 4",
  training: "M3 6h18 M3 12h18 M3 18h18 M8 3v6 M16 9v6 M10 15v6",
  runs: "M3 19h18 M5 15l4-5 4 3 6-9",
  upload: "M12 16V3m-5 5 5-5 5 5 M4 15v5h16v-5",
  plus: "M12 5v14 M5 12h14",
  arrow: "M4 12h15m-6-6 6 6-6 6",
  chip: "M6 6h12v12H6z M9 2v4m6-4v4M9 18v4m6-4v4M2 9h4m-4 6h4m12-6h4m-4 6h4",
  pin: "M12 22s7-8 7-13a7 7 0 1 0-14 0c0 5 7 13 7 13z M10 9a2 2 0 1 0 4 0 2 2 0 1 0-4 0",
  clock: "M12 3a9 9 0 1 0 .1 0 M12 6v6l4 2",
  check: "M4 12l5 5L20 6",
  wave: "M2 15h3l3-9 4 14 4-17 3 12h3",
  file: "M5 3h9l5 5v13H5z M14 3v6h5",
  down: "M12 3v12m-5-5 5 5 5-5 M4 17v4h16v-4",
};
export function Icon({ name, size = 18 }) {
  return (
    <svg
      aria-hidden="true"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d={icons[name]} />
    </svg>
  );
}
export const labels = {
  qt: "QT-LSTM",
  classical: "Classical LSTM",
  compact: "Compact LSTM",
  softmax: "Softmax generator",
  observed: "Observed",
  persistence: "Persistence",
  seasonal_mean: "Seasonal mean",
  arx: "ARX",
  rate_of_rise: "Rate of rise",
  rainfall: "Rainfall pilot",
  hourly: "Stage A · hourly",
  local: "Stage B · local",
};
export function Heading({ title, desc, children }) {
  return (
    <header className="pageHead">
      <div>
        <h1>{title}</h1>
        {desc && <p>{desc}</p>}
      </div>
      <div className="actions">{children}</div>
    </header>
  );
}
export function Field({ label, children }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}
export function Empty({ children }) {
  return <div className="empty">{children}</div>;
}
export function Tag({ children }) {
  return <span className="tag">{children}</span>;
}
export const num = (n, d = 2) =>
  n == null
    ? "—"
    : Number(n).toLocaleString(undefined, { maximumFractionDigits: d });
const colors = [
  "#77dfce",
  "#f0b56e",
  "#a4b6df",
  "#d3a4c8",
  "#bdca87",
  "#73bad6",
];
export function Plot({
  rows = [],
  fields = [],
  xkey = "timestamp",
  unit = "",
  height = 225,
  filter = true,
  boundaries = [],
}) {
  const [hover, setHover] = useState(null),
    [range, setRange] = useState("all");
  const plotRef = useRef(null),
    [width, setWidth] = useState(720);
  useEffect(() => {
    const el = plotRef.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) =>
      setWidth(Math.max(280, entry.contentRect.width)),
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [rows.length > 0]);
  if (!rows.length || !fields.length)
    return <Empty>No observations to plot.</Empty>;
  const n =
    range === "all" ? rows.length : Math.min(Number(range), rows.length);
  const list = rows.slice(-n),
    W = width,
    H = height,
    L = 42,
    R = 12,
    B = 32,
    T = 16;
  const value = (r, k) =>
    r[k] !== null && r[k] !== "" && Number.isFinite(+r[k]) ? +r[k] : null;
  const vals = list
    .flatMap((r) => fields.map((k) => value(r, k)))
    .filter((v) => v !== null);
  if (!vals.length) return <Empty>No valid plotted values.</Empty>;
  const low = Math.min(0, ...vals),
    high = Math.max(low + 0.01, ...vals) * 1.04 || 1;
  const t = list.map((r) =>
    xkey === "epoch" ? Number(r[xkey]) : Date.parse(r[xkey]),
  );
  const span = Math.max(1, t.at(-1) - t[0]);
  const xpos = (time) => L + ((W - L - R) * (time - t[0])) / span;
  const x = (i) => xpos(t[i]),
    y = (v) => T + (H - B - T) * (1 - (v - low) / (high - low));
  const ticks = [0, 1, 2, 3, 4];
  const xticks = W < 430 ? [0, 2, 4] : ticks;
  const idx = hover == null ? null : Math.min(hover, list.length - 1);
  return (
    <div className="plot" ref={plotRef}>
      <div className="plotTools">
        <div className="legend">
          {fields.map((k, i) => (
            <span key={k}>
              <i style={{ background: colors[i % colors.length] }} />
              {labels[k] || k}
            </span>
          ))}
        </div>
        {filter && (
          <select
            aria-label="Plot range"
            value={range}
            onChange={(e) => setRange(e.target.value)}
          >
            <option value="all">Full record</option>
            <option value="100">Last 100 points</option>
            <option value="36">Last 36</option>
          </select>
        )}
      </div>
      <svg
        className="trace"
        role="img"
        aria-label={fields.join(" and ") + " time series"}
        viewBox={`0 0 ${W} ${H}`}
        onMouseLeave={() => setHover(null)}
        onMouseMove={(e) => {
          const box = e.currentTarget.getBoundingClientRect();
          const pos =
            (((e.clientX - box.left) / box.width) * W - L) / (W - L - R);
          const at = t[0] + pos * span;
          setHover(
            t.reduce(
              (best, v, i) =>
                Math.abs(v - at) < Math.abs(t[best] - at) ? i : best,
              0,
            ),
          );
        }}
      >
        {ticks.map((i) => {
          const v = low + ((high - low) * i) / 4;
          return (
            <g key={i}>
              <path
                d={`M${L} ${y(v)}H${W - R}`}
                stroke="var(--grid)"
                strokeDasharray="2 5"
              />
              <text x={L - 8} y={y(v) + 4} textAnchor="end">
                {num(v, 1)}
              </text>
            </g>
          );
        })}
        {fields.map((k, j) => {
          let gap = true;
          const d = list
            .map((r, i) => {
              const v = value(r, k);
              if (v === null) {
                gap = true;
                return "";
              }
              const cmd = gap ? "M" : "L";
              gap = false;
              return cmd + x(i).toFixed(1) + " " + y(v).toFixed(1);
            })
            .join(" ");
          return (
            <path
              key={k}
              d={d}
              stroke={colors[j % colors.length]}
              fill="none"
              strokeWidth={j ? 1.35 : 1.8}
              vectorEffect="non-scaling-stroke"
              strokeLinejoin="round"
            />
          );
        })}
        {boundaries.map(([name, iso]) => {
          const pos = xpos(Date.parse(iso));
          return (
            pos >= L &&
            pos <= W - R && (
              <g key={name}>
                <path
                  d={`M${pos} ${T}V${H - B}`}
                  stroke="#d4ae77"
                  strokeDasharray="4 3"
                />
                <text x={Math.min(W - 64, pos + 4)} y={T + 11}>
                  {name}
                </text>
              </g>
            )
          );
        })}
        {xticks.map((i) => {
          const j = Math.round(((list.length - 1) * i) / 4);
          return (
            <text
              key={i}
              x={x(j)}
              y={H - 11}
              textAnchor={i === 0 ? "start" : i === 4 ? "end" : "middle"}
            >
              {String(list[j][xkey]).slice(0, 10)}
            </text>
          );
        })}
        {idx != null && (
          <path
            d={`M${x(idx)} ${T}V${H - B}`}
            stroke="var(--muted)"
            strokeDasharray="4 3"
          />
        )}
      </svg>
      <small className="readout">
        {idx != null
          ? String(list[idx][xkey]) +
            "  ·  " +
            fields
              .map(
                (k) =>
                  (labels[k] || k) +
                  " " +
                  num(value(list[idx], k)) +
                  " " +
                  unit,
              )
              .join("  ·  ")
          : unit}
      </small>
    </div>
  );
}
export function MapView({ layer, locations = {} }) {
  const points = Object.entries(locations);
  if (!layer?.bounds && !points.length)
    return (
      <Empty>
        Import a GIS layer or site coordinates to see the study area.
      </Empty>
    );
  const b = layer?.bounds || [
    Math.min(...points.map((p) => p[1][0])) - 0.005,
    Math.min(...points.map((p) => p[1][1])) - 0.005,
    Math.max(...points.map((p) => p[1][0])) + 0.005,
    Math.max(...points.map((p) => p[1][1])) + 0.005,
  ];
  const X = (p) => 25 + ((p - b[0]) * 650) / Math.max(1e-7, b[2] - b[0]),
    Y = (p) => 245 - ((p - b[1]) * 220) / Math.max(1e-7, b[3] - b[1]);
  const grid = layer?.grid,
    values = grid?.flat().filter((v) => v != null) || [],
    lo = Math.min(...values),
    hi = Math.max(...values);
  return (
    <>
      <svg
        className="map"
        role="img"
        aria-label="Geographic coverage in longitude and latitude"
        viewBox="0 0 700 270"
      >
        <defs>
          <pattern
            id="graticule"
            width="35"
            height="27"
            patternUnits="userSpaceOnUse"
          >
            <path d="M35 0H0V27" fill="none" stroke="#353d3e" />
          </pattern>
        </defs>
        <rect width="700" height="270" fill="url(#graticule)" />
        {grid?.map((r, i) =>
          r.map(
            (v, j) =>
              v != null && (
                <rect
                  key={i + "-" + j}
                  x={25 + (j * 650) / r.length}
                  y={25 + (i * 220) / grid.length}
                  width={650 / r.length + 1}
                  height={220 / grid.length + 1}
                  fill={`hsl(161 30% ${80 - ((v - lo) / Math.max(hi - lo, 0.01)) * 55}%)`}
                />
              ),
          ),
        )}
        {layer?.paths?.map((ring, i) =>
          ring.length === 1 ? (
            <circle
              key={i}
              cx={X(ring[0][0])}
              cy={Y(ring[0][1])}
              r="2"
              fill="#77dfce"
            />
          ) : (
            <polyline
              key={i}
              points={ring.map((p) => X(p[0]) + "," + Y(p[1])).join(" ")}
              fill="#66cabb44"
              stroke="#70c6b8"
              strokeWidth=".7"
            />
          ),
        )}
        {points.map(([k, [lon, lat]]) => (
          <g key={k}>
            <circle cx={X(lon)} cy={Y(lat)} r="4" fill="#f0b56e" />
            <text x={X(lon) + 6} y={Y(lat)}>
              {k}
            </text>
          </g>
        ))}
        <text x="15" y="19">
          N ↑
        </text>
      </svg>
      <small>
        {b.map((v) => num(v, 3)).join(" / ")} · WGS84 bounds · sampled preview
      </small>
    </>
  );
}
export function Circuit({ qubits = 0, depth = 2, info }) {
  const n = Math.min(qubits || 5, 5);
  return (
    <div>
      <svg
        className="circuit"
        role="img"
        aria-label="Rotation gates and CNOT ring schematic"
        viewBox="0 0 470 140"
      >
        {Array.from({ length: n }, (_, q) => (
          <g key={q}>
            <text x="7" y={20 + q * 24}>
              q{q}
            </text>
            <path d={`M32 ${16 + q * 24}H449`} stroke="#657372" />
            {[66, 250].map((start) => (
              <g key={start}>
                <rect
                  x={start}
                  y={6 + q * 24}
                  width="37"
                  height="20"
                  rx="4"
                  fill="#21403a"
                  stroke="#4b9786"
                />
                <text x={start + 9} y={20 + q * 24}>
                  Ry
                </text>
              </g>
            ))}
          </g>
        ))}
        {[0, 1, 2, 3].slice(0, n - 1).map((q) => (
          <g key={q} stroke="#d7ac70" fill="#d7ac70">
            <path d={`M${126 + q * 22} ${16 + q * 24}v24`} />
            <circle cx={126 + q * 22} cy={16 + q * 24} r="3" />
            <circle cx={126 + q * 22} cy={40 + q * 24} r="5" fill="#171e20" />
            <path d={`M${122 + q * 22} ${40 + q * 24}h8m-4-4v8`} />
          </g>
        ))}
      </svg>
      <small>
        RY rotations → CNOT ring → basis probabilities → mapping MLP → LSTM
        weights · {qubits || "—"} qubits · {depth} blocks. Representative wires
        shown.
      </small>
      {info?.states && (
        <details>
          <summary>Learned basis probabilities</summary>
          {info.states.map((s, i) => (
            <div key={s}>
              <code>{s.toString(2).padStart(info.qubits, "0")}</code>　
              {num(info.probabilities[i], 6)}
            </div>
          ))}
        </details>
      )}
    </div>
  );
}
