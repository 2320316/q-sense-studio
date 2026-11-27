import copy
import csv
import os
import platform
import time

import numpy as np
import torch
from torch.func import functional_call

from .diagnostics import snapshot
from .experiment import prepare, shape_info
from .qt_model import Forecast, QuantumWeights, SoftmaxWeights, generated_parameters
from .storage import read, save


def metrics(y, p):
    diff = y - p
    denom = np.square(y - y.mean()).sum()
    return dict(
        mae=float(abs(diff).mean()),
        rmse=float(np.sqrt(np.mean(diff**2))),
        nse=float(1 - np.square(diff).sum() / denom) if denom else None,
    )


def atomic_weights(file, value):
    tmp = file.with_suffix(".tmp")
    torch.save(value, tmp)
    os.replace(tmp, file)


def tensors(prepared, device):
    raw = prepared["raw"]
    x, y, _ = raw[0]
    xm = x.mean(axis=(0, 1))
    xs = x.std(axis=(0, 1))
    xs[xs < 1e-8] = 1
    ym, ys = float(y.mean()), float(y.std()) or 1.0
    batches = [
        (
            torch.tensor((a - xm) / xs, dtype=torch.float32, device=device),
            torch.tensor((b - ym) / ys, dtype=torch.float32, device=device),
        )
        for a, b, _ in raw
    ]
    return batches, dict(
        feature_mean=xm.tolist(), feature_std=xs.tolist(), target_mean=ym, target_std=ys
    )


def baselines(c, p):
    x, y, idx = p["raw"][0]
    test, _, ids = p["raw"][-1]
    pred = {"persistence": p["origin"][-1]}
    if c.mode == "rainfall":
        seasons = p["ticks"][idx] % 36
        avg = [
            y[seasons == i].mean() if (seasons == i).any() else y.mean()
            for i in range(36)
        ]
        pred["seasonal_mean"] = np.take(avg, p["ticks"][ids] % 36)
    else:
        origin = ids - p["horizon"]

        # Regression on lagged observations; all fit statistics use training windows.
        def design(a, label_ids):
            lagged = a.reshape(len(a), -1)
            if "stage_cm" not in p["names"]:
                end = label_ids[:, None] - p["horizon"]
                stage = p["feature_stage"][end - np.arange(c.lookback - 1, -1, -1)]
                lagged = np.column_stack((stage, lagged))
            return np.column_stack([np.ones(len(a)), lagged])

        train_design = design(x, idx)
        fit_rows = np.isfinite(train_design).all(axis=1)
        if not fit_rows.any():
            raise ValueError("ARX requires valid training stage history")
        coef = np.linalg.lstsq(train_design[fit_rows], y[fit_rows], rcond=1e-8)[0]
        pred["arx"] = design(test, ids) @ coef
        lag = min(c.lookback - 1, 6 if c.mode == "local" else 1)
        old = p["y"][origin - lag]
        pred["rate_of_rise"] = (
            pred["persistence"] + (pred["persistence"] - old) * p["horizon"] / lag
        )
    return pred


class Halt(Exception):
    pass


def train_arm(c, p, arm, seed, dest, notify, halt, initial=None):
    torch.manual_seed(seed)
    shuffle = torch.Generator().manual_seed(seed)
    batches, scale = tensors(p, c.device)
    (x, y), (vx, vy), _ = batches
    size = shape_info(c, len(p["names"]))
    h = size["compact_hidden"] if arm == "compact" else c.hidden
    target = Forecast(x.shape[-1], h).to(c.device)
    generated = arm in ("qt", "softmax")
    if generated:
        target.requires_grad_(False)
        cls = QuantumWeights if arm == "qt" else SoftmaxWeights
        model = cls(size["weights"], c.depth, c.width).to(c.device)
    else:
        model = target
    head_offset = {}
    if initial:
        initial_state = torch.load(initial, map_location=c.device, weights_only=True)
        if generated:
            # Preserve the pretrained generator's encoder and reset its generated head.
            model.load_state_dict(initial_state)
            weights = generated_parameters(target, model)
            head_offset = {
                k: (v - weights[k]).detach()
                for k, v in target.named_parameters()
                if k.startswith("head.")
            }
        else:
            target.lstm.load_state_dict(
                {k[5:]: v for k, v in initial_state.items() if k.startswith("lstm.")}
            )

    def parameters():
        weights = generated_parameters(target, model)
        for k, v in head_offset.items():
            weights[k] = weights[k] + v
        return weights

    def predict(batch):
        if generated:
            return functional_call(target, parameters(), (batch,))
        return target(batch)

    optimizer = getattr(torch.optim, c.optimizer)(model.parameters(), lr=c.lr)
    ck = dest / "checkpoint.pt"
    start, best, best_epoch, best_state = 1, float("inf"), 0, None
    history = []
    if ck.exists():
        state = torch.load(ck, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        shuffle.set_state(state["shuffle"])
        torch.set_rng_state(state["rng"])
        start, best, best_epoch, best_state, history = [
            state[k] for k in ("next", "best", "epoch", "best_state", "history")
        ]
    for epoch in range(start, c.epochs + 1):
        if c.patience and epoch - 1 - best_epoch >= c.patience:
            break
        halt()
        total = 0.0
        for indices in torch.randperm(len(x), generator=shuffle).split(c.batch):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(predict(x[indices]), y[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(indices)
        with torch.no_grad():
            val = torch.cat([predict(chunk) for chunk in vx.split(512)])
            score = (val - vy).abs().mean().item()
        if not np.isfinite(score):
            raise ValueError(
                "Non-finite validation error; check input scale and learning rate"
            )
        if score < best:
            best, best_epoch, best_state = (
                score,
                epoch,
                copy.deepcopy(model.state_dict()),
            )
        history.append(
            dict(
                epoch=epoch, loss=total / len(x), validation=score * scale["target_std"]
            )
        )
        atomic_weights(
            ck,
            dict(
                next=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                shuffle=shuffle.get_state(),
                rng=torch.get_rng_state(),
                best=best,
                epoch=best_epoch,
                best_state=best_state,
                history=history,
            ),
        )
        with torch.no_grad():
            weights = parameters() if generated else dict(target.named_parameters())
            save(
                dest / "diagnostics.json",
                snapshot(weights, x[0], p["names"], model, arm, epoch),
            )
        notify(history, best_epoch)
        halt()
    model.load_state_dict(best_state)
    atomic_weights(dest / "trained.pt", model.state_dict())
    with torch.no_grad():
        reference = predict(vx[:8]).cpu()
        if generated:
            target.load_state_dict(parameters())
        target.cpu()
        bundle = dict(
            features=p["names"],
            hidden=h,
            scale=scale,
            state_dict=target.state_dict(),
            mode=c.mode,
            site=c.site,
            horizon_steps=p["horizon"],
            lookback=c.lookback,
            unit="mm" if c.mode == "rainfall" else "cm",
        )
        atomic_weights(dest / "inference.pt", bundle)
        restored = Forecast(len(p["names"]), h).eval()
        restored.load_state_dict(
            torch.load(dest / "inference.pt", weights_only=True)["state_dict"]
        )
        torch.testing.assert_close(
            restored(vx[:8].cpu()), reference, atol=2e-4, rtol=2e-4
        )
        sample = vx[:1].cpu()
        restored(sample)
        clock = time.perf_counter()
        for _ in range(30):
            restored(sample)
        latency = (time.perf_counter() - clock) * 1000 / 30
        preds = (
            torch.cat([restored(t.cpu()) for t in batches[-1][0].split(512)])
            .numpy()
            .astype(float)
        )
        circuit = None
        if generated:
            probs = model.probabilities().cpu()
            values, indices = probs.topk(min(12, len(probs)))
            circuit = dict(
                qubits=size["qubits"],
                depth=c.depth,
                states=indices.tolist(),
                probabilities=values.tolist(),
                angles=model.angles.cpu().tolist() if arm == "qt" else [],
            )
    details = dict(
        best_epoch=best_epoch,
        trainable=sum(t.numel() for t in model.parameters() if t.requires_grad),
        exported=sum(t.numel() for t in restored.parameters()),
        bytes=(dest / "inference.pt").stat().st_size,
        latency_ms=latency,
        benchmark_device="CPU " + platform.machine(),
        circuit=circuit,
    )
    return preds * scale["target_std"] + scale["target_mean"], details


def run(root, c, plan, notify, halt):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(c.device == "cpu")
    for number, (fold, seed, arm, init) in enumerate(plan["specs"]):
        job = root / f"job-{number:03}"
        job.mkdir(exist_ok=True)
        if (job / "result.json").exists():
            continue
        halt()
        p = prepare(c, fold)

        def progress(history, best, phase="fit"):
            source = job if phase == "fit" else warm
            notify(
                dict(
                    job=number,
                    fold=fold,
                    seed=seed,
                    arm=arm,
                    init=init,
                    phase=phase,
                    history=history,
                    diagnostics=read(source / "diagnostics.json"),
                    best_epoch=best,
                )
            )

        initial = None
        if init == "transfer":
            warm = root / f"warm-{seed}-{arm}"
            warm.mkdir(exist_ok=True)
            if not (warm / "done.json").exists():
                w = prepare(c, warm=True)
                train_arm(
                    c,
                    w,
                    arm,
                    seed,
                    warm,
                    lambda a, b: progress(a, b, "hourly pretraining"),
                    halt,
                )
                save(
                    warm / "done.json",
                    dict(features=w["available"], source=c.warm_dataset),
                )
            initial = warm / "trained.pt"
        prediction, details = train_arm(c, p, arm, seed, job, progress, halt, initial)
        forecast = baselines(c, p)
        forecast[arm] = prediction
        _, actual, ids = p["raw"][-1]
        with (job / "forecast.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp", "observed", "origin_stage", "origin_6h_ago", *forecast]
            )
            for j, i in enumerate(ids):
                past = i - 36
                prior = (
                    p["y"][past]
                    if c.mode == "local"
                    and past >= 0
                    and p["ticks"][i] - p["ticks"][past] == 36
                    else float("nan")
                )
                writer.writerow(
                    [
                        p["dates"][i].isoformat(),
                        actual[j],
                        p["origin"][-1][j],
                        prior,
                        *[v[j] for v in forecast.values()],
                    ]
                )
        summary = {}
        for k, v in forecast.items():
            valid = np.isfinite(v)
            summary[k] = metrics(actual[valid], v[valid]) if valid.any() else None
        save(
            job / "result.json",
            dict(
                job=number,
                arm=arm,
                fold=fold,
                seed=seed,
                init=init,
                metrics=summary,
                **details,
            ),
        )
