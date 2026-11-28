import React, { useState } from "react";
import { Field, Tag } from "./common";
export function ReferenceEvidence({ reference }) {
  const [index, setIndex] = useState(0);
  const item = reference.figures[index];
  if (!item) return null;
  const src = "/api/presentation/reference/image/" + index;
  return (
    <section className="panel referenceEvidence">
      <div className="panelHead">
        <div>
          <h2>{reference.title}</h2>
          <small>{reference.context}</small>
        </div>
        <Tag>Supplied screenshots</Tag>
      </div>
      <div className="miniStats">
        {reference.metrics.map(([label, value]) => (
          <span key={label}>
            {label}
            <b>{value}</b>
          </span>
        ))}
      </div>
      <p>{reference.interpretation}</p>
      <Field label="Source figure">
        <select value={index} onChange={(e) => setIndex(+e.target.value)}>
          {reference.figures.map((f, i) => (
            <option key={f.title} value={i}>
              {f.title}
            </option>
          ))}
        </select>
      </Field>
      <figure>
        <img src={src} alt={item.title + " from the supplied Wawa run"} />
        <figcaption>{item.caption}</figcaption>
      </figure>
      <a href={src} target="_blank" rel="noreferrer">
        Open full-size figure ↗
      </a>
    </section>
  );
}
