"""Paired comparisons on the same timestamps, with verified storms as bootstrap units."""

import csv
import math
from collections import defaultdict
from datetime import datetime

import numpy as np

from .experiment import source
from .storage import read


def indicators(actual, forecast):
    tp = int(np.sum(actual & forecast))
    fn = int(np.sum(actual & ~forecast))
    fp = int(np.sum(~actual & forecast))
    tn = int(np.sum(~actual & ~forecast))
    ratio = lambda a, b: a / b if b else None
    return dict(
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        mwr=ratio(fn, fn + tp),
        far=ratio(fp, tp + fp),
        pofd=ratio(fp, fp + tn),
        precision=ratio(tp, tp + fp),
        f1=ratio(2 * tp, 2 * tp + fp + fn),
    )


def dm(loss_diff, lag):
    d = np.asarray(loss_diff, dtype=float)
    if len(d) < max(3, lag + 2):
        return None
    centered = d - d.mean()
    variance = np.dot(centered, centered) / len(d)
    for k in range(1, lag + 1):
        variance += (
            2 * (1 - k / (lag + 1)) * np.dot(centered[k:], centered[:-k]) / len(d)
        )
    if variance <= 0:
        return None
    z = d.mean() / math.sqrt(variance / len(d))
    return math.erfc(abs(z) / math.sqrt(2))


def holm(pvalues):
    order = sorted(pvalues, key=pvalues.get)
    out = {}
    ceiling = 0.0
    for i, k in enumerate(order):
        ceiling = max(ceiling, min(1.0, (len(order) - i) * pvalues[k]))
        out[k] = ceiling
    return out


def report(root, c):
    jobs = []
    cache = {}
    for path in sorted(root.glob("job-*/result.json")):
        j = read(path)
        with (path.parent / "forecast.csv").open() as f:
            rows = list(csv.DictReader(f))
        key = (j["fold"], j["seed"], j["init"], j["arm"])
        cache[key] = rows
        jobs.append(j)
    out = dict(
        warning_tier="Regression only",
        jobs=len(jobs),
        comparisons=[],
        warnings=[],
        event_metrics=[],
        seed_means=[],
    )
    groups = defaultdict(list)
    for j in jobs:
        groups[(j["arm"], j["init"], j["fold"])].append(j["metrics"][j["arm"]])
    for (arm, init, fold), rows in groups.items():
        out["seed_means"].append(
            dict(
                arm=arm,
                init=init,
                fold=fold,
                seeds=len(rows),
                **{
                    k: float(np.mean([r[k] for r in rows]))
                    if all(r[k] is not None for r in rows)
                    else None
                    for k in ("mae", "rmse", "nse")
                },
            )
        )
    if c.mode == "rainfall" or not cache:
        return out

    def event(t):
        dt = datetime.fromisoformat(t)
        return next((e.name for e in c.events if e.start <= dt <= e.end), None)

    cut = c.threshold
    tier = "A"
    exemplars = {}
    for (fold, _, _, _), rows in cache.items():
        exemplars[fold] = rows

    def evidence(threshold, rise):
        hits = set()
        for fold, rows in exemplars.items():
            for r in rows:
                baseline = float(r["origin_6h_ago"]) if rise else 0.0
                e = event(r["timestamp"])
                if e and float(r["observed"]) - baseline >= threshold:
                    hits.add((fold, e))
        return len({e for _, e in hits}) >= 3 and len({f for f, _ in hits}) >= 2

    if cut is None or not evidence(cut, False):
        cut = None
        if c.mode == "local":
            _, ticks, _, _, stage = source(c)
            changes = (stage[36:] - stage[:-36])[ticks[36:] - ticks[:-36] == 36]
            changes = changes[np.isfinite(changes)]
            if len(changes):
                rapid = float(np.percentile(changes, 95))
                if rapid > 0 and evidence(rapid, True):
                    cut, tier = rapid, "B"
    if cut is not None:
        out.update(
            warning_tier="Flood exceedance" if tier == "A" else "Rapid-rise detection",
            threshold_cm=cut,
        )
    # Per-configuration paired data; seed means at each timestamp prevent pseudoreplication.
    errors = defaultdict(lambda: defaultdict(list))
    misses = defaultdict(lambda: defaultdict(list))
    for (fold, seed, init, arm), rows in cache.items():
        cols = [arm] + (
            [k for k in ("persistence", "arx", "rate_of_rise") if k in rows[0]]
            if arm == c.arms[0]
            else []
        )
        for col in cols:
            label = (col, init)
            actual = np.array([float(r["observed"]) for r in rows])
            pred = np.array([float(r[col]) for r in rows])
            prior = (
                np.array([float(r["origin_6h_ago"]) for r in rows])
                if tier == "B"
                else np.zeros(len(rows))
            )
            valid = np.isfinite(pred) & np.isfinite(actual)
            wvalid = valid & np.isfinite(prior)
            if cut is not None:
                out["warnings"].append(
                    dict(
                        fold=fold,
                        seed=seed,
                        arm=col,
                        init=init,
                        **indicators(
                            actual[wvalid] - prior[wvalid] >= cut,
                            pred[wvalid] - prior[wvalid] >= cut,
                        ),
                    )
                )
            for i, r in enumerate(rows):
                if valid[i]:
                    errors[label][(fold, r["timestamp"])].append(
                        abs(actual[i] - pred[i])
                    )
                    if cut is not None and wvalid[i] and actual[i] - prior[i] >= cut:
                        misses[label][(fold, r["timestamp"])].append(
                            float(pred[i] - prior[i] < cut)
                        )
    for (arm, init), points in errors.items():
        by = defaultdict(list)
        for (fold, t), v in points.items():
            e = event(t)
            if e:
                by[fold, e].append(np.mean(v))
        for (fold, e), values in by.items():
            out["event_metrics"].append(
                dict(arm=arm, init=init, fold=fold, event=e, mae=float(np.mean(values)))
            )
    rng = np.random.default_rng(42)
    # Equal weight per event after averaging matched random seeds. This describes
    # uncertainty across storms rather than treating every ten-minute sample as independent.
    event_groups = defaultdict(list)
    for row in out["event_metrics"]:
        event_groups[row["arm"], row["init"]].append(row["mae"])
    out["event_bootstrap"] = []
    for (arm, init), values in event_groups.items():
        ci = None
        if len(values) >= 2:
            draws = rng.choice(values, (2000, len(values)), replace=True).mean(axis=1)
            ci = np.percentile(draws, [2.5, 97.5]).tolist()
        out["event_bootstrap"].append(
            dict(
                arm=arm,
                init=init,
                events=len(values),
                mean_event_mae=float(np.mean(values)),
                ci95=ci,
            )
        )
    comparisons = [
        ("H01", ("qt", "scratch"), ("classical", "scratch")),
        ("H02", ("qt", "scratch"), ("softmax", "scratch")),
        ("H03", ("qt", "scratch"), ("compact", "scratch")),
        ("H04", ("qt", "transfer"), ("qt", "scratch")),
    ]
    comparisons += [
        ("H05", (a, "scratch"), (b, "scratch"))
        for a in c.arms
        for b in ("persistence", "arx", "rate_of_rise")
    ]
    for hypothesis, left, right in comparisons:
        for metric, group in (("mae", errors), ("mwr", misses)):
            if left not in group or right not in group:
                continue
            by = defaultdict(list)
            fold_losses = defaultdict(list)
            for key in sorted(group[left].keys() & group[right].keys()):
                diff = float(np.mean(group[left][key]) - np.mean(group[right][key]))
                fold_losses[key[0]].append(diff)
                e = event(key[1])
                if e:
                    by[e].append(diff)
            vals = [np.mean(x) for x in by.values()]
            entry = dict(
                hypothesis=hypothesis,
                left="/".join(left),
                right="/".join(right),
                metric=metric,
                events=len(vals),
            )
            if len(vals) >= 2:
                bs = rng.choice(vals, (2000, len(vals)), replace=True).mean(axis=1)
                entry.update(
                    delta=float(np.mean(vals)),
                    ci95=np.percentile(bs, [2.5, 97.5]).tolist(),
                )
            if metric == "mae":
                lag = c.horizon * (6 if c.mode == "local" else 1) - 1
                entry["dm_p_by_fold"] = [dm(v, lag) for v in fold_losses.values()]
                # Bonferroni across disjoint folds before Holm across secondary hypotheses.
                ps = [p for p in entry["dm_p_by_fold"] if p is not None]
                if ps:
                    entry["dm_p"] = min(1.0, min(ps) * len(fold_losses))
            out["comparisons"].append(entry)
    families = defaultdict(list)
    for row in out["comparisons"]:
        if "dm_p" in row and row["hypothesis"] != "H01":
            families[row["hypothesis"]].append(row["dm_p"])
    out["secondary_holm_p"] = holm(
        {
            h: min(1.0, len(ps) * min(ps)) if ps else 1.0
            for h, ps in [(h, families[h]) for h in ("H02", "H03", "H04", "H05")]
        }
    )
    out["primary_settings"] = (
        c.mode == "local"
        and (c.lookback, c.horizon, c.hidden) == (36, 3, 32)
        and len(c.seeds) == 5
    )
    return out
