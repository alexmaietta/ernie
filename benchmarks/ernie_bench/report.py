# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime
from html import escape
from pathlib import Path

from ernie_bench.analysis import summarize_run

LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
DARK = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767")
SHAPES = ("circle", "square", "triangle", "diamond", "circle", "square", "triangle", "diamond")
FACET = ("scene", "width", "height", "tier", "spp", "sm_percent", "cameras_per_world", "outputs")
PHASES = (("startup", "process start"), ("load", "scene load"), ("first_frame", "first frame"))
LOWER_IS_BETTER = {"flip", "lpips", "mean_absolute_error"}
WIDTH, HEIGHT = 560, 300
LEFT, RIGHT, TOP, BOTTOM = 64, 120, 16, 44

STYLE = """
.viz-root {
  color-scheme: light;
  --surface-1: #fcfcfb; --page: #f9f9f7; --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  %(light)s
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19; --page: #0d0d0d; --text-primary: #f0efec; --text-secondary: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    %(dark)s
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1: #1a1a19; --page: #0d0d0d; --text-primary: #f0efec; --text-secondary: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  %(dark)s
}
body { margin: 0; }
.viz-root { background: var(--page); color: var(--text-primary); font: 14px/1.45 system-ui, sans-serif; padding: 24px 32px; min-height: 100vh; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 17px; margin: 32px 0 8px; } h3 { font-size: 14px; margin: 0 0 4px; font-weight: 600; }
.lede { color: var(--text-secondary); margin: 0 0 16px; }
.grid { display: flex; flex-wrap: wrap; gap: 16px; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; }
.card p { color: var(--text-secondary); margin: 0 0 8px; font-size: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; margin: 8px 0 0; font-size: 12px; color: var(--text-secondary); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
svg text { fill: var(--muted); font-size: 11px; }
svg .label { fill: var(--text-secondary); font-size: 11px; }
svg .gridline { stroke: var(--grid); stroke-width: 1; }
svg .baseline { stroke: var(--axis); stroke-width: 1; }
svg .mark { stroke: var(--surface-1); stroke-width: 2; }
svg [data-value] { cursor: default; outline: none; }
svg [data-value]:hover .mark, svg [data-value]:focus .mark { filter: brightness(1.15); stroke: var(--text-primary); }
table { border-collapse: collapse; font-size: 12px; background: var(--surface-1); }
th, td { border-bottom: 1px solid var(--grid); padding: 4px 8px; text-align: right; white-space: nowrap; }
th { color: var(--text-secondary); font-weight: 600; } td:first-child, th:first-child { text-align: left; }
details { margin: 8px 0; } summary { cursor: pointer; color: var(--text-secondary); }
.warn { color: var(--text-secondary); font-size: 12px; }
#tip { position: fixed; pointer-events: none; display: none; background: var(--surface-1); color: var(--text-primary);
  border: 1px solid var(--border); border-radius: 6px; padding: 6px 8px; font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
#tip strong { display: block; font-size: 13px; } #tip span { color: var(--text-secondary); }
"""

SCRIPT = """
const tip = document.getElementById("tip");
function show(event) {
  const mark = event.target.closest("[data-value]");
  if (!mark) { tip.style.display = "none"; return; }
  const value = document.createElement("strong");
  value.textContent = mark.dataset.value;
  const label = document.createElement("span");
  label.textContent = mark.dataset.label;
  tip.replaceChildren(value, label);
  const box = event.clientX === undefined ? mark.getBoundingClientRect() : { right: event.clientX, top: event.clientY };
  tip.style.left = Math.min(box.right + 12, window.innerWidth - 240) + "px";
  tip.style.top = box.top + 12 + "px";
  tip.style.display = "block";
}
document.addEventListener("pointermove", show);
document.addEventListener("focusin", show);
document.addEventListener("focusout", () => { tip.style.display = "none"; });
"""


def ticks(low: float, high: float, values: list[float]) -> list[float]:
    found = []
    for exponent in range(math.floor(math.log10(low)) - 1, math.ceil(math.log10(high)) + 1):
        for step in (1, 2, 5):
            value = step * 10.0**exponent
            if low <= value <= high:
                found.append(value)
    return found if len(found) >= 2 else sorted(set(values))


def number(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.3g}G"
    if value >= 1e6:
        return f"{value / 1e6:.3g}M"
    if value >= 1e3:
        return f"{value / 1e3:.3g}k"
    return f"{value:.3g}"


class LogScale:
    def __init__(self, values: list[float], start: float, end: float, pad: float = 1.15):
        low, high = min(values) / pad, max(values) * pad
        if low == high:
            low, high = low / 2, high * 2
        self.low, self.high, self.start, self.end = math.log10(low), math.log10(high), start, end
        self.ticks = ticks(low, high, values)

    def __call__(self, value: float) -> float:
        return self.start + (math.log10(value) - self.low) / (self.high - self.low) * (self.end - self.start)


def marker(shape: str, x: float, y: float, slot: int) -> str:
    fill = f'fill="var(--series-{slot + 1})" class="mark"'
    if shape == "square":
        return f'<rect x="{x - 4.5:.1f}" y="{y - 4.5:.1f}" width="9" height="9" rx="1.5" {fill}/>'
    if shape == "triangle":
        return f'<path d="M{x:.1f},{y - 5.5:.1f} L{x + 5.5:.1f},{y + 4.5:.1f} L{x - 5.5:.1f},{y + 4.5:.1f} Z" {fill}/>'
    if shape == "diamond":
        return f'<path d="M{x:.1f},{y - 6:.1f} L{x + 6:.1f},{y:.1f} L{x:.1f},{y + 6:.1f} L{x - 6:.1f},{y:.1f} Z" {fill}/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" {fill}/>'


def axes(x: LogScale, y: LogScale, x_title: str, y_title: str) -> list[str]:
    parts = []
    for value in y.ticks:
        position = y(value)
        parts.append(f'<line class="gridline" x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{position:.1f}" y2="{position:.1f}"/>')
        parts.append(f'<text x="{LEFT - 6}" y="{position + 4:.1f}" text-anchor="end">{number(value)}</text>')
    for value in x.ticks:
        parts.append(f'<text x="{x(value):.1f}" y="{HEIGHT - BOTTOM + 16}" text-anchor="middle">{number(value)}</text>')
    parts.append(f'<line class="baseline" x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{HEIGHT - BOTTOM}" y2="{HEIGHT - BOTTOM}"/>')
    parts.append(f'<text x="{(LEFT + WIDTH - RIGHT) / 2}" y="{HEIGHT - 8}" text-anchor="middle">{escape(x_title)}</text>')
    parts.append(f'<text transform="translate(14 {(TOP + HEIGHT - BOTTOM) / 2}) rotate(-90)" text-anchor="middle">{escape(y_title)}</text>')
    return parts


def svg(parts: list[str], title: str) -> str:
    return f'<svg width="{WIDTH}" height="{HEIGHT}" role="img" aria-label="{escape(title)}">{"".join(parts)}</svg>'


def legend(series: list[str], colors: dict[str, int], kind: str) -> str:
    items = []
    for name in series:
        slot = colors[name]
        key = (
            f'<svg width="16" height="10"><line x1="0" x2="16" y1="5" y2="5" stroke="var(--series-{slot + 1})" stroke-width="2"/></svg>'
            if kind == "line"
            else f'<svg width="12" height="12">{marker(SHAPES[slot], 6, 6, slot)}</svg>'
        )
        items.append(f"<span>{key}{escape(name)}</span>")
    return f'<div class="legend">{"".join(items)}</div>'


def facet_title(facet: dict) -> str:
    sm = f" · {facet['sm_percent']}% SMs" if facet.get("sm_percent", 100) != 100 else ""
    return f"{facet['scene']} · {facet['width']}x{facet['height']} · {facet['tier']} · {facet['spp']} spp · {facet['outputs']}{sm}"


def scaling_chart(facet: dict, records: list[dict], colors: dict[str, int]) -> str:
    lines = defaultdict(list)
    for record in records:
        lines[record["series"]].append(record)
    worlds = [record["worlds"] for record in records]
    rates = [record["images_per_second"] for record in records]
    x = LogScale(worlds, LEFT, WIDTH - RIGHT)
    y = LogScale(rates, HEIGHT - BOTTOM, TOP)
    parts = axes(x, y, "worlds", "images per second")
    for name, found in sorted(lines.items()):
        slot = colors[name]
        found.sort(key=lambda record: record["worlds"])
        points = " ".join(f"{x(r['worlds']):.1f},{y(r['images_per_second']):.1f}" for r in found)
        parts.append(f'<polyline points="{points}" fill="none" stroke="var(--series-{slot + 1})" stroke-width="2" stroke-linejoin="round"/>')
        for r in found:
            px, py = x(r["worlds"]), y(r["images_per_second"])
            label = f"{name} · {r['worlds']} worlds · {r['median_seconds'] * 1e3:.3g} ms/frame"
            parts.append(
                f'<g tabindex="0" data-value="{number(r["images_per_second"])} images/s" data-label="{escape(label)}">'
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="12" fill="transparent"/>{marker(SHAPES[slot], px, py, slot)}</g>'
            )
        if len(lines) <= 4:
            last = found[-1]
            parts.append(f'<text class="label" x="{x(last["worlds"]) + 10:.1f}" y="{y(last["images_per_second"]) + 4:.1f}">{escape(name)}</text>')
    title = facet_title(facet)
    return f'<div class="card"><h3>{escape(title)}</h3>{svg(parts, "images per second by worlds, " + title)}{legend(sorted(lines), colors, "line")}</div>'


def pareto_front(points: list[tuple[float, float]], lower: bool) -> list[tuple[float, float]]:
    front, best = [], math.inf if lower else -math.inf
    for seconds, quality in sorted(points):
        if (quality < best) if lower else (quality > best):
            front.append((seconds, quality))
            best = quality
    return front


def pareto_chart(scene: str, metric: str, points: list[dict], colors: dict[str, int]) -> str:
    lower = metric in LOWER_IS_BETTER
    x = LogScale([p["median_seconds"] * 1e3 for p in points], LEFT, WIDTH - RIGHT)
    values = [p[metric] for p in points]
    y = LogScale([max(value, 1e-6) for value in values], HEIGHT - BOTTOM, TOP) if lower else LogScale(values, HEIGHT - BOTTOM, TOP, 1.02)
    parts = axes(x, y, "ms per frame", f"{metric} ({'lower' if lower else 'higher'} is better)")
    front = pareto_front([(p["median_seconds"] * 1e3, max(p[metric], 1e-6) if lower else p[metric]) for p in points], lower)
    if len(front) > 1:
        path = " ".join(f"{x(seconds):.1f},{y(quality):.1f}" for seconds, quality in front)
        parts.append(f'<polyline points="{path}" fill="none" stroke="var(--text-secondary)" stroke-width="1" stroke-dasharray="4 3"/>')
    series = sorted({p["series"] for p in points})
    for p in points:
        slot = colors[p["series"]]
        px, py = x(p["median_seconds"] * 1e3), y(max(p[metric], 1e-6) if lower else p[metric])
        label = f"{p['series']} · {p['worlds']} worlds · {p['width']}x{p['height']} · {p['spp']} spp · {p['median_seconds'] * 1e3:.3g} ms/frame"
        parts.append(
            f'<g tabindex="0" data-value="{metric} {p[metric]:.4g}" data-label="{escape(label)}">'
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="12" fill="transparent"/>{marker(SHAPES[slot], px, py, slot)}</g>'
        )
    title = f"{scene} · rgb {metric} against frame time"
    note = "<p>The dashed line joins the configurations that no other configuration beats on both time and quality.</p>"
    return f'<div class="card"><h3>{escape(title)}</h3>{note}{svg(parts, title)}{legend(series, colors, "marker")}</div>'


def phase_chart(facet: dict, records: list[dict]) -> str:
    largest = max(record["worlds"] for record in records)
    shown = sorted((r for r in records if r["worlds"] == largest), key=lambda r: r["series"])
    totals = [sum(r["phases"].values()) for r in shown]
    row, gap = 26, 8
    height = TOP + len(shown) * (row + gap)
    scale = (WIDTH - RIGHT - 140) / max(totals)
    parts = []
    for index, (r, total) in enumerate(zip(shown, totals)):
        top = TOP + index * (row + gap)
        parts.append(f'<text class="label" x="132" y="{top + row / 2 + 4}" text-anchor="end">{escape(r["series"])}</text>')
        left = 140.0
        for slot, (phase, title) in enumerate(PHASES):
            seconds = r["phases"].get(phase, 0.0)
            width = seconds * scale
            if width <= 0:
                continue
            drawn = max(width - 2, 1)
            parts.append(
                f'<g tabindex="0" data-value="{seconds:.3g} s" data-label="{escape(r["series"] + " · " + title)}">'
                f'<rect x="{left:.1f}" y="{top}" width="{drawn:.1f}" height="{row}" rx="2" fill="var(--series-{slot + 1})"/></g>'
            )
            left += width
        parts.append(f'<text class="label" x="{left + 6:.1f}" y="{top + row / 2 + 4}">{total:.2f} s</text>')
    keys = "".join(
        f'<span><svg width="12" height="12"><rect width="12" height="12" rx="2" fill="var(--series-{slot + 1})"/></svg>{title}</span>'
        for slot, (_, title) in enumerate(PHASES)
    )
    title = f"time to the first image · {largest} worlds · {facet_title(facet)}"
    body = f'<svg width="{WIDTH}" height="{height}" role="img" aria-label="{escape(title)}">{"".join(parts)}</svg>'
    return f'<div class="card"><h3>{escape(title)}</h3>{body}<div class="legend">{keys}</div></div>'


def table(columns: list[tuple[str, str]], rows: list[dict]) -> str:
    head = "".join(f"<th>{escape(title)}</th>" for title, _ in columns)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(row.get(key, '')))}</td>" for _, key in columns) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def primary_gpu(manifest: dict) -> dict:
    return (manifest["machine"].get("gpus") or [{}])[0]


def machine_label(manifest: dict) -> str:
    return f"{manifest['machine']['os'].get('hostname', '?')} {primary_gpu(manifest).get('name', 'no gpu')}"


def load(runs: list[Path], metric: str) -> tuple[list[dict], list[dict], list[dict]]:
    manifests, records, quality = [], [], []
    for run in runs:
        manifest, found = summarize_run(run)
        manifests.append(manifest | {"directory": str(run)})
        for record in found:
            if record["frames"]:
                record["phases"] = {"startup": record["startup_seconds"], "load": record["load"].get("seconds", 0.0), "first_frame": record.get("first_frame_seconds", 0.0)}
                records.append(record | {"run": manifest["run_id"], "machine": machine_label(manifest)})
        for path in sorted(Path(run).glob("quality-*.json")):
            summary = json.loads(path.read_text())
            quality += [row | {"run": manifest["run_id"], "machine": machine_label(manifest)} for row in summary["summary"] if row["output"] == "rgb" and row.get(metric) is not None]
    machines = {record["machine"] for record in records}
    for record in records + quality:
        record["series"] = record["backend"] if len(machines) == 1 else f"{record['backend']} · {record['machine']}"
    return manifests, records, quality


def render(runs: list[Path], metric: str = "flip") -> str:
    manifests, records, quality = load(runs, metric)
    series = sorted({record["series"] for record in records + quality})
    if len(series) > len(LIGHT):
        raise ValueError(f"{len(series)} series is more than {len(LIGHT)} colours, pick fewer backends or runs")
    colors = {name: slot for slot, name in enumerate(series)}
    style = STYLE % {
        "light": " ".join(f"--series-{slot + 1}: {color};" for slot, color in enumerate(LIGHT)),
        "dark": " ".join(f"--series-{slot + 1}: {color};" for slot, color in enumerate(DARK)),
    }

    facets = defaultdict(list)
    for record in records:
        facets[tuple(record[name] for name in FACET)].append(record)
    ordered = [(dict(zip(FACET, key)), found) for key, found in sorted(facets.items(), key=lambda item: str(item[0]))]
    scaling = [scaling_chart(facet, found, colors) for facet, found in ordered]
    phases = [phase_chart(facet, found) for facet, found in ordered]

    timing = {(r["backend"], r["machine"], r["scene"], r["worlds"], r["width"], r["height"], r["tier"], r["spp"]): r for r in records}
    joined = defaultdict(list)
    for row in quality:
        match = timing.get((row["backend"], row["machine"], row["scene"], row["worlds"], row["width"], row["height"], row["tier"], row.get("spp", 1)))
        if match:
            joined[row["scene"]].append(match | {metric: row[metric]})
    pareto = [pareto_chart(scene, metric, points, colors) for scene, points in sorted(joined.items())]

    machine_rows = []
    for manifest in manifests:
        tuning = manifest.get("tuning", {})
        system_prep = tuning.get("system_prep") or {}
        machine_rows.append(
            {
                "run": manifest["run_id"],
                "mode": manifest.get("mode", "timing"),
                "machine": machine_label(manifest),
                "cpu": manifest["machine"]["cpu"].get("model"),
                "driver": primary_gpu(manifest).get("driver"),
                "system-prep": "applied" if system_prep else "not applied",
                "gpu clock": system_prep.get("APPLIED_GPU_CLOCK") or "unlocked",
                "governors": ",".join(tuning.get("cpu", {}).get("governors", [])),
                "warnings": len(manifest.get("warnings", [])),
            }
        )
    result_rows = [
        r
        | {
            "res": f"{r['width']}x{r['height']}",
            "p50 ms": f"{r['median_seconds'] * 1e3:.4g}",
            "p95 ms": f"{r['p95_seconds'] * 1e3:.4g}",
            "p99 ms": f"{r['p99_seconds'] * 1e3:.4g}",
            "gpu ms": f"{r['device_median_seconds'] * 1e3:.4g}" if "device_median_seconds" in r else "-",
            "ci": f"±{r['relative_ci'] * 100:.2f}%",
            "images/s": number(r["images_per_second"]),
            "Mpix/s": f"{r['pixels_per_second'] * 1e-6:.1f}",
            "images/J": f"{r['images_per_joule']:.1f}" if r.get("images_per_joule") else "-",
        }
        for r in sorted(records, key=lambda r: (r["scene"], r["tier"], r["width"], r["series"], r["worlds"]))
    ]
    columns = [(name, name) for name in ("series", "scene", "worlds", "res", "tier", "spp", "sm_percent", "p50 ms", "p95 ms", "p99 ms", "gpu ms", "ci", "images/s", "Mpix/s", "images/J", "run")]
    warnings = "".join(f"<li>{escape(m['run_id'])}: {escape(w)}</li>" for m in manifests for w in m.get("warnings", []))

    sections = [
        f"<h1>Renderer benchmark</h1><p class=\"lede\">{len(manifests)} run{'s' if len(manifests) != 1 else ''}, {len(records)} measured points, generated {datetime.now():%Y-%m-%d %H:%M}.</p>",
        "<h2>Scaling</h2><p class=\"lede\">Images per second against worlds, on log axes. Higher is better.</p>",
        f'<div class="grid">{"".join(scaling)}</div>',
    ]
    if pareto:
        sections += ["<h2>Quality against time</h2>", f'<div class="grid">{"".join(pareto)}</div>']
    sections += [
        "<h2>Time to the first image</h2><p class=\"lede\">Process start, scene load and the first frame, at the largest world count.</p>",
        f'<div class="grid">{"".join(phases)}</div>',
        "<h2>Results</h2>",
        table(columns, result_rows),
        "<h2>Machines</h2>",
        table([(name, name) for name in machine_rows[0]], machine_rows) if machine_rows else "",
    ]
    if warnings:
        sections.append(f'<details><summary>Warnings recorded during the runs</summary><ul class="warn">{warnings}</ul></details>')
    return (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Renderer benchmark</title><style>{style}</style></head>'
        f'<body><div class="viz-root">{"".join(sections)}</div><div id="tip" role="tooltip"></div><script>{SCRIPT}</script></body></html>'
    )
