# Ernie Benchmarks

A framework for comparing GPU batch renderers on throughput, scaling and image quality. It renders the same scenes through under the same timing rules, then reports speed and accuracy side by side.

## Capabilities

- **Shared scenes.** Scenes are generated in Python and passed to every backend through one interface, so each renderer draws the same geometry, cameras and lights.
- **Per-backend environments.** Each backend runs in its own virtual environment and a separate worker process. Backends with conflicting CUDA, USD or Python versions can run in the same suite, and a crash in one backend doesn't stop the run.
- **Adaptive timing.** Warmup continues until frame times settle. Repeats continue until the confidence interval reaches its target. Each result records its spread, the machine, the driver and the clocks it ran at.
- **Machine setup.** `scripts/system-prep` locks GPU and memory clocks, pins the CPU governor, moves services and interrupts off benchmark cores, allows realtime priority and starts CUDA MPS. `restore` undoes all of it.
- **Parameter sweeps.** A suite sweeps any combination of worlds, cameras, resolution, outputs, lighting tier, samples per pixel and the share of the GPU a backend may use.
- **Reference checks.** `check` compares a backend's depth and segmentation against a reference ray caster. `quality` scores captured frames with SSIM, FLIP and LPIPS against an analytic reference or another backend such as Mitsuba.
- **Kernel profiling.** `profile` runs a suite under Nsight Systems or Nsight Compute and lists the slowest kernels at each point.
- **Hardware counters.** `--counters` adds CPU, memory, cache and GPU counters to each result.
- **Startup timing.** Startup is timed separately from steady-state frames, and cold runs drop the page cache first.
- **Reports and media.** Results are stored as Parquet. `report` writes an HTML page with scaling, quality and startup charts, and `images`, `movie`, `panel` and `montage` turn captures into labelled comparisons.

## Quick Start

Install the harness into its own environment.

```bash
cd benchmarks
pip install -e ".[media,test]"
```

It's recommended to set up a [uv]() environment in the `benchmarks` itself.

### Smoke test

The smoke suite needs only the Ernie environment, which installs `ernie-renderer` from PyPI.

```bash
scripts/make-env ernie
ernie-bench run configs/suites/smoke.toml
ernie-bench summary data/results/<time>-smoke
```

### Compare

The compare suite needs Newton, Ernie and Madrona. Madrona is built from source and needs Python 3.12 and CUDA 12.8. Set `CUDA_ROOT` if CUDA isn't in `/usr/local/cuda-12.8`.

```bash
scripts/make-env ernie
scripts/make-env newton
scripts/make-env madrona --python python3.12
scripts/setup-madrona
```

Check each backend against the reference before timing it.

```bash
ernie-bench check ernie_cuda_software
ernie-bench check newton
ernie-bench check madrona_rt
```

Lock the clocks for the run and release them afterwards. `system-prep status` shows what is applied.

```bash
sudo scripts/system-prep apply --gpu-clock 2100
ernie-bench run configs/suites/compare.toml
sudo scripts/system-prep restore
```

Each run is written to `data/results/<time>-<suite>/`. Summarise it or build a report from one or more runs.

```bash
ernie-bench summary data/results/<time>-compare
ernie-bench report data/results/<time>-compare -o report.html
```

`compare-full` also needs `scripts/make-env ovrtx`, and `quality` needs `scripts/make-env mitsuba` too.

## Suites

| Suite | Backends | Purpose |
| --- | --- | --- |
| `smoke` | Ernie | A quick check that the harness and Ernie work end to end |
| `compare` | Newton, Ernie CUDA software, Madrona RT | The headline comparison |
| `compare-full` | Every GPU backend and mode | The full comparison across 16 to 1024 worlds |
| `quality` | Every backend including Mitsuba | Captures frames for `ernie-bench quality` |

Run only some backends from a suite with `--backend`.

```bash
ernie-bench run configs/suites/compare-full.toml --backend newton --backend madrona_rt
```

## Framework

Configs live in three folders under `configs/`.

- **`scenes/`** names a dataset from `ernie_bench/datasets` with its seed and parameters.
- **`workloads/`** holds the base render settings and timing rules that suites start from.
- **`suites/`** picks backends, scenes and a workload, then lists the values to sweep. Each sweep combination becomes one point. A suite's `[measurement]` overrides the workload's timing and `[capture]` saves frames.

The harness in `ernie_bench/harness.py` builds the scene and runs each point in a worker process, using the Python from that backend's environment. The worker loads the scene, warms up, times frames, captures images and writes its samples back. The run directory holds a `manifest.json` with the suite, machine and settings, plus the samples as Parquet.

## Backends

| Name | Renderer | Environment |
| --- | --- | --- |
| `ernie`, `ernie_<api>_<mode>` | Ernie, from the `ernie-renderer` package on PyPI. `api` is `vulkan` or `cuda`, and `mode` is `hardware` or `software`. | `envs/ernie` |
| `newton` | Newton's tiled camera renderer | `envs/newton` |
| `madrona_rt`, `madrona_rast` | Madrona's ray tracer and rasteriser, built from `external/madrona_benchmark` with the patches in `native/madrona/patches` | `envs/madrona` |
| `ovrtx`, `ovrtx_tiled` | NVIDIA OVRTX | `envs/ovrtx` |
| `mitsuba`, `mitsuba_cpu` | Mitsuba 3 on CUDA or LLVM, used mainly as a quality reference | `envs/mitsuba` |
| `null` | Does nothing, to measure the harness's own overhead | any |

A backend that can't run a point reports why, for example an unsupported tier or too many worlds, and the run continues.

### Adding a Backend

1. Implement the `Backend` protocol in `ernie_bench/backends/base.py` in a new module under `ernie_bench/backends/`. It provides `info`, `capabilities`, `load`, `update`, `render`, `sync`, `readback`, `stream` and `close`.
2. Declare what it supports in `capabilities()` so the harness skips points it can't run.
3. Register a name in `BACKENDS` in `ernie_bench/backends/__init__.py`. The part before the first underscore picks the environment, so `foo_fast` runs in `envs/foo`.
4. Add `envs/<name>/requirements.txt`, plus `envs/<name>/paths.txt` for local builds that must be on the Python path, then run `scripts/make-env <name>`.
5. Verify it against the reference with `ernie-bench check <name> --save check.png` before timing it.
