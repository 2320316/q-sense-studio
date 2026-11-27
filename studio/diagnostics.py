"""Epoch telemetry from the first training window, using the current effective weights."""

import torch
from torch.nn.functional import linear


@torch.no_grad()
def snapshot(weights, probe, features, model, arm, epoch):
    w = weights["lstm.weight_ih_l0"]
    hidden = w.shape[0] // 4
    h = w.new_zeros(hidden)
    c = torch.zeros_like(h)
    gates = []
    activations = []
    for x in probe:
        joined = linear(x, w, weights["lstm.bias_ih_l0"]) + linear(
            h, weights["lstm.weight_hh_l0"], weights["lstm.bias_hh_l0"]
        )
        i, f, g, o = joined.chunk(4)
        i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
        c = f * c + i * g
        h = o * c.tanh()
        gates.append(torch.stack([i.mean(), f.mean(), g.mean(), o.mean()]))
        activations.append(h[:12])
    result = dict(
        epoch=epoch,
        features=features,
        hidden=hidden,
        arm=arm,
        gates=torch.stack(gates).cpu().tolist(),
        activations=torch.stack(activations).cpu().tolist(),
        # Candidate-state input edges. Readout weights are from the same hidden units.
        edges=w[2 * hidden : 2 * hidden + min(12, hidden)].cpu().tolist(),
        readout=weights["head.weight"][0, :12].cpu().tolist(),
        output=linear(h, weights["head.weight"], weights["head.bias"]).item(),
    )
    if arm == "qt":
        probs = model.probabilities()
        vals, ids = probs.topk(min(8, len(probs)))
        result["quantum"] = dict(
            qubits=model.qubits,
            angles=model.angles.cpu().tolist(),
            states=ids.cpu().tolist(),
            probabilities=vals.cpu().tolist(),
        )
    return result
