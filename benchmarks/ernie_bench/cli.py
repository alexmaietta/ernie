# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from ernie_bench.analysis import summarize_run
from ernie_bench.datasets import build_scene
from ernie_bench.harness import check as harness_check
from ernie_bench.config import Profile, Workload, load_suite, scene_spec
from ernie_bench.harness import ROOT, environment, run_suite
from ernie_bench.system import machine, tuning


def selected_suite(args):
    suite = load_suite(args.suite)
    return replace(suite, backends=tuple(args.backend)) if args.backend else suite


def backend_width(rows: list[dict]) -> int:
    return max([7, *(len(row["backend"]) for row in rows)])


def print_failures(manifest: dict) -> None:
    for outcome in manifest["outcomes"]:
        if outcome["status"] != "ok":
            w = outcome["workload"]
            print(f"{outcome['status']}: {outcome['backend']} {outcome['scene']} worlds={w['worlds']} {w['width']}x{w['height']}: {outcome['reason']}")


def run(args) -> None:
    directory = run_suite(
        selected_suite(args), args.out, cpus=args.cpus, nice=args.nice, isolation="inline" if args.inline else "process", realtime=args.realtime
    )
    print(directory)
    show(directory, args.counters)


PROFILE_COLUMNS = [
    ("occ%", "sm__warps_active.avg.pct_of_peak_sustained_active"),
    ("sm%", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("dram%", "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("L1 hit%", "l1tex__t_sector_hit_rate.pct"),
    ("L2 hit%", "lts__t_sector_hit_rate.pct"),
    ("regs", "launch__registers_per_thread"),
]


def profile(args) -> None:
    from ernie_bench.profile import summarize

    directory = run_suite(selected_suite(args), args.out, cpus=args.cpus, nice=args.nice, realtime=args.realtime, profiler=Profile(args.tool, args.frames))
    print(directory)
    records = summarize(directory)
    shown = [(title, metric) for title, metric in PROFILE_COLUMNS if any(metric in r for r in records)]
    width = backend_width(records)
    print(f"{'backend':<{width}} {'worlds':>7} {'ms/frame':>9} {'calls':>6} " + " ".join(f"{title:>8}" for title, _ in shown) + "  kernel")
    seen = defaultdict(int)
    for r in records:
        point = (r["backend"], r["worlds"], r["width"], r["tier"], r["sm_percent"])
        seen[point] += 1
        if seen[point] > args.top:
            continue
        cells = " ".join(f"{r[metric]:>8.1f}" if metric in r else f"{'-':>8}" for _, metric in shown)
        print(f"{r['backend']:<{width}} {r['worlds']:>7} {r['seconds_per_frame'] * 1e3:>9.4f} {r['launches_per_frame']:>6.1f} {cells}  {r['kernel'][:60]}")
    print_failures(json.loads((directory / "manifest.json").read_text()))


def scene(args) -> None:
    spec = scene_spec(args.name)
    spec = replace(spec, seed=args.seed if args.seed is not None else spec.seed)
    start = perf_counter()
    built = build_scene(spec, args.worlds, args.cameras)
    elapsed = perf_counter() - start
    print(f"{built.name}: built in {elapsed * 1000:.1f} ms")
    print(f"  worlds     {built.world_count}")
    print(f"  instances  {len(built.instances)} ({len(built.motion.instance)} moving)")
    print(f"  meshes     {len(built.meshes)}")
    print(f"  triangles  {built.triangle_count}")
    print(f"  cameras    {len(built.cameras)}")


COUNTER_COLUMNS = [
    ("cpu%", "process.cpu_utilization", 100, ".0f"),
    ("vcs/f", "process.voluntary_switches", 1, ".2f"),
    ("ivcs/f", "process.involuntary_switches", 1, ".2f"),
    ("flt/f", "process.minor_faults", 1, ".1f"),
    ("rss MB", "process.rss_bytes", 1e-6, ".0f"),
    ("Δrss MB", "process.rss_growth_bytes", 1e-6, ".2f"),
    ("gpu MB", "gpu0.memory_used_bytes_max", 1e-6, ".0f"),
    ("proc MB", "gpu0.process_memory_bytes_max", 1e-6, ".0f"),
    ("Δgpu MB", "gpu0.memory_growth_bytes", 1e-6, ".2f"),
    ("gpu%", "gpu0.utilization_mean", 1, ".0f"),
    ("ipc", "perf.ipc", 1, ".2f"),
    ("miss%", "perf.cache_miss_rate", 100, ".1f"),
    ("llc%", "perf.llc_miss_rate", 100, ".1f"),
    ("migr/f", "perf.cpu_migrations", 1, ".2f"),
    ("sys%", "system.cpu_utilization", 100, ".0f"),
    ("cores%", "system.benchmark_cpus_utilization", 100, ".0f"),
    ("fgn ms/f", "system.foreign_cpu_seconds", 1e3, ".3f"),
    ("load", "system.load_average", 1, ".1f"),
]


def counters(records: list[dict]) -> None:
    print()
    width = backend_width(records)
    print(f"{'backend':<{width}} {'worlds':>7} {'res':>9} {'outputs':<16} {'start s':>8} {'load MB':>8} " + " ".join(f"{title:>8}" for title, *_ in COUNTER_COLUMNS))
    for r in records:
        if "counters" not in r:
            continue
        load = r["load"].get("process.rss_growth_bytes")
        cells = [f"{r['startup_seconds']:>8.3f}", f"{load * 1e-6:>8.1f}" if load is not None else f"{'-':>8}"]
        for _, metric, scale, form in COUNTER_COLUMNS:
            value = r["counters"].get(metric)
            cells.append(f"{value * scale:>8{form}}" if value is not None else f"{'-':>8}")
        resolution = f"{r['width']}x{r['height']}"
        print(f"{r['backend']:<{width}} {r['worlds']:>7} {resolution:>9} {r['outputs']:<16} " + " ".join(cells))


def show(directory: Path, with_counters: bool) -> None:
    manifest, records = summarize_run(directory)
    width = backend_width(records)
    print(
        f"{'backend':<{width}} {'scene':<16} {'worlds':>7} {'cams':>5} {'res':>9} {'outputs':<16} {'tier':<12} {'spp':>4} {'sm%':>4} "
        f"{'p50 ms':>9} {'p95 ms':>9} {'p99 ms':>9} {'gpu ms':>9} {'quiet ms':>9} {'pre%':>5} {'ci ±%':>6} {'iqr ms':>9} {'outl':>5} "
        f"{'drift%':>7} {'spread%':>7} {'clean':>6} {'images/s':>12} {'Mpix/s':>9} {'images/J':>10}"
    )
    for r in records:
        resolution = f"{r['width']}x{r['height']}"
        clean = f"{r['clean_repeats']}/{r['repeats']}"
        point = (
            f"{r['backend']:<{width}} {r['scene']:<16} {r['worlds']:>7} {r['cameras_per_world']:>5} "
            f"{resolution:>9} {r['outputs']:<16} {r['tier']:<12} {r['spp']:>4} {r['sm_percent']:>4}"
        )
        if not r["frames"]:
            print(f"{point}  every repeat was disturbed")
            continue
        per_joule = f"{r['images_per_joule']:>10.1f}" if r["images_per_joule"] else f"{'-':>10}"
        quiet = f"{r['quiet_median_seconds'] * 1e3:.4g}" if r["quiet_median_seconds"] is not None else "-"
        device = f"{r['device_median_seconds'] * 1e3:.4g}" if "device_median_seconds" in r else "-"
        print(
            f"{point} {r['median_seconds'] * 1e3:>9.4g} {r['p95_seconds'] * 1e3:>9.4g} {r['p99_seconds'] * 1e3:>9.4g} {device:>9} {quiet:>9} {r['preempted_fraction'] * 100:>5.0f} {r['relative_ci'] * 100:>6.2f} {r['iqr_seconds'] * 1e3:>9.4g} "
            f"{r['outliers']:>5} {r['drift'] * 100:>7.2f} {r['process_spread'] * 100:>7.2f} {clean:>6} {r['images_per_second']:>12.0f} {r['pixels_per_second'] * 1e-6:>9.1f} {per_joule}"
        )

    if with_counters:
        counters(records)
        for moment in ("start", "end"):
            busiest = ", ".join(f"{p['name']} {p['cpu_percent']:.0f}%" for p in manifest.get(f"processes_{moment}", [])[:5])
            print(f"busiest other processes at {moment}: {busiest or 'none'}")

    for warning in manifest.get("warnings", []):
        print(f"warning: {warning}")
    print_failures(manifest)


def summary(args) -> None:
    show(args.run, args.counters)


def report(args) -> None:
    from ernie_bench.report import render

    args.out.write_text(render(args.runs, args.metric))
    print(args.out)


def images(args) -> None:
    from ernie_bench import media

    written = media.export_images(args.run, args.exposure)
    print(f"wrote {len(written)} images under {args.run / 'media' / 'images'}")


def movie(args) -> None:
    from ernie_bench import media

    for path in media.export_movies(args.run, args.fps, args.exposure):
        print(path)


def panel(args) -> None:
    from ernie_bench import media

    for path in media.export_panels(args.runs, args.output, args.image, args.movie, args.fps, args.exposure):
        print(path)


def montage(args) -> None:
    from ernie_bench import media

    sequences = [media.read_media(path) for path in args.inputs]
    labels = args.label or [Path(path).stem for path in args.inputs]
    if len(labels) != len(sequences):
        raise SystemExit(f"got {len(labels)} labels for {len(sequences)} inputs")
    frames = media.montage(sequences, labels, args.title, args.columns)
    if len(frames) > 1 or args.out.suffix.lower() in media.VIDEO_SUFFIXES:
        print(media.write_movie(args.out, frames, args.fps))
    else:
        print(media.save_image(args.out, frames[0]))


THRESHOLDS = {"hit_agreement": 0.98, "depth_agreement": 0.98, "segmentation_agreement": 0.97, "rgb_agreement": 0.95}


def quality(args) -> None:
    from ernie_bench.quality import evaluate, summarize

    rows = evaluate(args.runs, args.reference)
    summary = summarize(rows)
    destination = args.runs[0] / f"quality-{args.reference}.json"
    destination.write_text(json.dumps({"reference": args.reference, "rows": rows, "summary": summary}, indent=2))
    columns = ("psnr", "ssim", "flip", "lpips", "mean_absolute_error", "hit_agreement", "absolute_relative", "rmse", "mean_iou")
    width = backend_width(summary)
    print(f"{'backend':<{width}} {'worlds':>6} {'res':>9} {'tier':<10} {'output':<12} {'n':>4} " + " ".join(f"{name[:9]:>9}" for name in columns))
    for row in summary:
        resolution = f"{row['width']}x{row['height']}"
        cells = " ".join(f"{row[name]:>9.4g}" if row.get(name) is not None else f"{'-':>9}" for name in columns)
        print(f"{row['backend']:<{width}} {row['worlds']:>6} {resolution:>9} {row['tier']:<10} {row['output']:<12} {row['samples']:>4} {cells}")
    print(destination)


def check(args) -> None:
    python = environment(args.backend)
    if python != sys.executable:
        raise SystemExit(subprocess.run([python, "-m", "ernie_bench.cli", *sys.argv[1:]]).returncode)
    spec = scene_spec(args.scene)
    outputs = ("rgb", "depth", "segmentation") if args.tier in ("unlit", "direct") else ("depth", "segmentation")
    workload = Workload(worlds=args.worlds, cameras_per_world=args.cameras, width=args.width, height=args.height, outputs=outputs, tier=args.tier)
    scene = build_scene(spec, workload.worlds, workload.cameras_per_world)
    images = list(range(min(workload.image_count, args.images)))
    scores, frames, depth, classes = harness_check(args.backend, scene, workload, images)
    workload = frames.pop("workload")
    reference_rgb = frames.pop("reference_rgb")
    failed = [name for name, minimum in THRESHOLDS.items() if scores.get(name, 1.0) < minimum]
    print(json.dumps(scores | {"passed": not failed}, indent=2))
    if args.save:
        from ernie_bench import media

        tiles, labels = [], []
        expected = {"depth": depth[..., None], "segmentation": classes[..., None], "rgb": reference_rgb}
        shown = [output for output in ("rgb", "depth", "segmentation") if output in frames and expected[output] is not None]
        for slot, image in enumerate(images):
            for source, arrays in ((args.backend, frames), ("reference", expected)):
                for output in shown:
                    exposure = 1.0 if output == "rgb" else None
                    tiles.append(media.upscale(media.displayable(output, arrays[output][slot : slot + 1], exposure)[0]))
                    labels.append(f"{source} {output} {image}")
        title = f"{args.backend} vs reference · {args.scene} · {workload.width}x{workload.height}"
        print(media.save_image(args.save, media.grid(tiles, labels, title, columns=2 * len(shown))))
    if failed:
        raise SystemExit(f"conventions check failed: {', '.join(failed)}")


def system(args) -> None:
    state = tuning.current()
    print(json.dumps({"machine": machine.snapshot(), "tuning": state, "warnings": tuning.check(state)}, indent=2, default=str))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ernie-bench")
    commands = parser.add_subparsers(required=True)

    controls = argparse.ArgumentParser(add_help=False)
    controls.add_argument("--out", type=Path, default=ROOT / "data" / "results")
    controls.add_argument("--cpus", default="performance", help="performance, isolated, all, or a cpu list like 0-5,8")
    controls.add_argument("--nice", type=int, help="process priority, negative values need CAP_SYS_NICE")
    controls.add_argument("--realtime", type=int, metavar="PRIORITY", help="run under SCHED_FIFO at this priority, needs CAP_SYS_NICE")
    suite = argparse.ArgumentParser(add_help=False, parents=[controls])
    suite.add_argument("suite", type=Path)
    suite.add_argument("--backend", action="append", help="only run these backends")

    command = commands.add_parser("run", parents=[suite], help="run a suite")
    command.add_argument("--counters", action="store_true", help="also print cpu, memory, cache and gpu counters")
    command.add_argument("--inline", action="store_true", help="run backends in this process instead of fresh workers")
    command.set_defaults(handler=run)

    command = commands.add_parser("profile", parents=[suite], help="run a suite under ncu or nsys and list the slowest kernels")
    command.add_argument("--tool", default="nsys", choices=["nsys", "ncu"])
    command.add_argument("--frames", type=int, default=10, help="frames to profile after warmup")
    command.add_argument("--top", type=int, default=10, help="kernels to list per point")
    command.set_defaults(handler=profile)

    command = commands.add_parser("scene", help="build a scene and print its size")
    command.add_argument("name")
    command.add_argument("--worlds", type=int, default=1)
    command.add_argument("--cameras", type=int, default=1)
    command.add_argument("--seed", type=int)
    command.set_defaults(handler=scene)

    command = commands.add_parser("summary", help="summarise a run directory")
    command.add_argument("run", type=Path)
    command.add_argument("--counters", action="store_true", help="also print cpu, memory, cache and gpu counters")
    command.set_defaults(handler=summary)

    command = commands.add_parser("report", help="write an html report with scaling, quality and startup charts from one or more runs")
    command.add_argument("runs", type=Path, nargs="+")
    command.add_argument("--metric", default="flip", help="quality metric for the quality against time chart, from a quality-*.json in the runs")
    command.add_argument("-o", "--out", type=Path, default=Path("report.html"))
    command.set_defaults(handler=report)

    command = commands.add_parser("images", help="export captured frames as png")
    command.add_argument("run", type=Path)
    command.add_argument("--exposure", type=float, help="fixed exposure for rgb instead of auto")
    command.set_defaults(handler=images)

    command = commands.add_parser("movie", help="encode each captured sequence as an mp4")
    command.add_argument("run", type=Path)
    command.add_argument("--fps", type=float, default=30)
    command.add_argument("--exposure", type=float, help="fixed exposure for rgb instead of auto")
    command.set_defaults(handler=movie)

    command = commands.add_parser("panel", help="put backends side by side with labels, from one or more runs")
    command.add_argument("runs", type=Path, nargs="+")
    command.add_argument("--output", action="append", help="only these outputs")
    command.add_argument("--image", type=int, action="append", help="only these image indices")
    command.add_argument("--movie", action="store_true", help="animate sequences instead of one png per step")
    command.add_argument("--fps", type=float, default=30)
    command.add_argument("--exposure", type=float, help="fixed exposure for rgb instead of auto")
    command.set_defaults(handler=panel)

    command = commands.add_parser("montage", help="merge images, frame folders or videos into one labelled panel")
    command.add_argument("inputs", type=Path, nargs="+")
    command.add_argument("--label", action="append", help="one per input, defaults to file names")
    command.add_argument("--title")
    command.add_argument("--columns", type=int)
    command.add_argument("--fps", type=float, default=30)
    command.add_argument("-o", "--out", type=Path, required=True)
    command.set_defaults(handler=montage)

    command = commands.add_parser("quality", help="per-pixel image quality of captured frames against a reference")
    command.add_argument("runs", type=Path, nargs="+")
    command.add_argument("--reference", default="analytic", help="analytic, or a backend whose captures are in the runs, such as mitsuba")
    command.set_defaults(handler=quality)

    command = commands.add_parser("check", help="compare a backend's depth and segmentation against the reference ray caster")
    command.add_argument("backend")
    command.add_argument("--scene", default="hide_and_seek")
    command.add_argument("--worlds", type=int, default=2)
    command.add_argument("--cameras", type=int, default=2)
    command.add_argument("--width", type=int, default=64)
    command.add_argument("--height", type=int, default=48)
    command.add_argument("--tier", default="unlit")
    command.add_argument("--images", type=int, default=4, help="how many images to compare")
    command.add_argument("--save", type=Path, help="write a side-by-side png")
    command.set_defaults(handler=check)

    command = commands.add_parser("system", help="print the machine description and performance settings")
    command.set_defaults(handler=system)

    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
