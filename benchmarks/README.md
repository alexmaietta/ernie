# Benchmarks

`ernie_benchmarks` runs Ernie's benchmarks through one fixed lifecycle, after Isaac Sim's `isaacsim.performance.benchmarking`, with Isaac Lab's autotuning and comparison method on top. A case is a scenario (a scene and what changes each step) and a value for each of its parameters; suites are named lists of cases. Build it with `-DERNIE_BUILD_BENCHMARKS=ON`:

```sh
cmake -S . -B build -DERNIE_BUILD_BENCHMARKS=ON
cmake --build build --target ernie_benchmarks -j 4
```

## Running

```sh
build/bin/ernie_benchmarks --list                     # scenarios, parameters, presets, output sets, suites
build/bin/ernie_benchmarks                            # the smoke suite
build/bin/ernie_benchmarks --suite scaling --label before-lod
build/bin/ernie_benchmarks --scenario house,dense --set preset=minimal,lit --set resolution=64,128
build/bin/ernie_benchmarks --suite ladder --filter 'dense/.*resolution=128' --images sheets/
build/bin/ernie_benchmarks --scenario shapes --autotune worlds --limit 'duration_ms<=16.7'
build/bin/ernie_benchmarks --compare RUN_A1,RUN_A2,RUN_A3 RUN_B1,RUN_B2,RUN_B3
```

- `--suite NAME` runs a registered suite; `--scenario A,B` runs scenarios at their defaults instead. `--set NAME=V1,V2` sets a parameter in every case whose scenario has it, one case per value, so repeated `--set`s sweep their product. `--filter REGEX` keeps the cases whose names match; `--cases` lists them without running.
- `--warmup`, `--samples`, `--steps` (0 calibrates), `--completion fenced|pipelined` and `--profiles process,gpu|none` override each scenario's sampling.
- `--images DIR` writes a contact sheet of up to 16 views per case and render var, from the validation step.
- Each case gets a fresh renderer, so none inherits another's render targets, pools or caches; `--reuse-renderer` shares one.

## The lifecycle

Each case runs through the same phases (runner.h):

1. **Setup**: the scenario's `Build`, its `Setup`, and a first fenced step (the renderer's first sync of the scene), each kept as a setup region (`build`, `setup`, `first_step`).
2. **Calibration**: unless the steps per sample are given, they double from 1 until a sample takes 50 ms (Isaac Sim's calibrated batching), so short steps are timed over enough of them.
3. **Warm-up**: untimed samples (3 by default).
4. **Measurement**: independent samples (20 by default), each timed whole. Each step changes the scene (`Scenario::Step`, such as new camera poses on CUDA), is submitted, and has its outputs mapped to CUDA without a copy, as a learner reads them. The completion boundary is part of the result: `fenced` waits for every step (latency), `pipelined` only for a sample's last (throughput).
5. **Validation**: one more fenced step, its outputs checked by the scenario (`Validate`; batches check that views do not render flat) and drawn as contact sheets.
6. **Teardown**.

A phase that throws ends the case as invalid, its error kept with the phase's name. Statistics are over every sample, none dropped as outliers: mean, median, standard deviation, min, max, p50, p95, p99.

Scopes follow Isaac Sim's: `workflow` (whole frames: the batch scenarios), `subsystem` (a part of the renderer, such as `scene_sync`), `operation` (small calls).

## Metrics

Every sample records, besides its time per step (`duration_ms`):

| Metric | Profile | |
|---|---|---|
| `steps_per_s`, `<unit>_per_s` | always | Rates in the scenario's units, such as `views_per_s`, `worlds_per_s`, `Mpixels_per_s` |
| `gpu_ms`, `gpu/<zone>_ms` | gpu | GPU time per step from timestamp queries, and per pass (`gpu/rendering/views_ms`...) |
| `gpu_busy` | gpu | GPU time over wall time per step: near 1 is GPU bound, low is CPU bound or stalled |
| `device_memory_used_mib`, `device_memory_budget_mib`, `device_memory_fraction` | gpu | This process's device-local memory and its budget, from VK_EXT_memory_budget |
| `process_cpu_utilization`, `process_rss_mib` | process | CPU time over wall time, and resident memory |
| `renderer_memory_mib`, `triangles` | always | The renderer's own count of its memory, and the triangles of a frame |

Every metric is vendor-neutral, from Vulkan and the OS. A scenario can time its own regions (`Measure region("name")`) and record its own metrics (`Record(name, value, unit)`) inside its steps.

## Results

Each run is a directory, `<output>/<UTC time>-<label>`, where the output is `--output`, else `$ERNIE_BENCHMARK_OUTPUT_DIR`, else `~/.cache/ernie/benchmarks`. It holds:

- a JSON result per case, schema `ernie.benchmark.result.v1`, written atomically. It keeps the benchmark's identity, parameters, scope and completion; the sampling; the status and errors; the lifecycle's times and validation; the scene's stats; every metric's samples and statistics; every region's observations; and the environment (host, OS, CPU, memory, GPU, driver, Vulkan version, commit and build).
- `run.json`, the run and its cases.
- `summary.csv`, a row per case of its parameters and means.

The program also prints a line per case as it finishes, and a table at the end.

## Autotuning

`--autotune NAME` finds the largest value of an integer parameter (worlds, views...) whose run stays valid and within the limits, as Isaac Lab's camera benchmark finds how many cameras fit. Probes start at the case's value and grow by `--autotune-factor` (2) until one fails, a limit crossed or the run failing (out of memory, say), then bisect to within 5%. Each probe is a short run (1 warm-up and 5 measured samples). The best value then gets a full run.

Limits are `--limit METRIC<=VALUE` or `METRIC>=VALUE` on a metric's mean, such as `duration_ms<=16.7` (60 Hz), `views_per_s>=100000` or `device_memory_fraction<=0.8`; without one, `device_memory_fraction<=0.9`. The search is written to `autotune__<case>.json` in the run's directory.

## Comparing

`--compare BASE HEAD` compares runs: each side is a comma-separated list of run directories, and `--metric` picks the metric (`duration_ms` by default). For each case in both, it prints the change with a 95% Welch interval and a verdict: better or worse only when the interval leaves out zero, else inconclusive. It exits 1 if any case is conclusively worse.

With several runs a side the interval is over the runs' means. That is the regression screening Isaac Lab's method asks for: at least 3 runs of each, in separate processes, alternating base and head. With one run a side it is over that run's samples, which share the process and understate the uncertainty.

## Batch scenarios

`BatchScenario` (batch.h) renders `worlds` worlds, `views` views of each, through one batched camera at `resolution` squared. Worlds overlap at the origin and are told apart by their world index; their content is PointInstancer instances with per-instance worlds, so prototypes are shared across worlds. The parameters every batch scenario has:

| Parameter | Default | |
|---|---|---|
| `preset` | `minimal` | A registered preset: `minimal` (one forward pass, sun, no shadows), `lit` (full pipeline, RGB bands, one cascade of hard shadows), `lit-noshadow`, `native` (16 bands, soft cascades, sensor noise) |
| `outputs` | `rgb` | A registered output set: `depth`, `rgb`, `rgbd`, `gbuffer` (colour, distance, normals, prim and instance ids), `hdr` |
| `resolution` | 128 | Pixels along each side |
| `worlds` | 1024 | Worlds in the batch |
| `views` | 1 | Views of each world |

The scenarios, simplest first:

| Scenario | Scene | Own parameters |
|---|---|---|
| `cartpole` | A cart balancing a pole on a rail, seen from the front | `spread` |
| `shapes` | A walled arena of boxes, spheres, capsules and cylinders laid out per world, seen by agents in it | `objects`, `arena` |
| `tabletop` | A finely tessellated robot arm posed per world at a textured table, cubes, a cabinet | `detail`, `cubes` |
| `house` | Furnished houses of textured rooms, one of `layouts` per world, seen from inside | `layouts`, `objects`, `detail` |
| `dense` | Rooms crowded with a library of detailed textured objects, millions of triangles a world | `layouts`, `objects`, `library`, `triangles` |

Every scene is generated in code; nothing is downloaded.

## Scene scenarios

`scene_sync` measures keeping the renderer in step with a large scene rather than rendering it: a grid of `prims` cubes in groups of `fanout`, each under `depth` nested transforms, seen whole from above at `resolution` squared with the minimal preset. Each step makes one kind of `edit`, so its time against `none` is the CPU's cost of what changed:

| `edit` | Each step |
|---|---|
| `none` | Nothing: the cost of a frame with no change |
| `colour` | One cube's displayColor |
| `xform` | One cube's translate |
| `parent` | One group's translate, moving its cubes |
| `batch` | Every cube's translate, rows of one CPU tensor (`Scene::Write` of a `PrimSet`), as a simulation writes |
| `add` | A cube added and the last step's removed |
| `walk` | No edit; every cube's world transform and visibility read, as a full sync's draw items are |

## Suites

| Suite | Cases |
|---|---|
| `smoke` | Every batch scenario, minimal and lit, 64 px, 64 worlds |
| `scaling` | Resolution 64 to 256 against 256 to 4096 worlds |
| `outputs` | Each output set at 128 px, 1024 worlds |
| `presets` | Each preset at 128 px, 1024 worlds |
| `views` | 1 to 8 views per world |
| `envs` | Cartpole, 16 to 4096 worlds at 128 to 512 px |
| `ladder` | Rising complexity at 1024 worlds, 64 to 256 px: shapes seen by 5 agents, a depth-only tabletop, houses, dense rooms |
| `scene` | `scene_sync`, every edit at 1k, 10k and 100k cubes (about 10 minutes, mostly building the 100k scenes) |

## Adding to them

- A scenario derives from `Scenario` (bench.h), or `BatchScenario` for a batch of views, in a file of its own under `scenarios/`, and registers itself with `ERNIE_BENCH_SCENARIO(Class, "name", "description")`. It declares its parameters with defaults (`Params`, or `WorldParams` for a batch) and builds its scene (`Build`, or `BuildWorlds` and `Poses`). It may override `Describe` (scope, completion, sampling, profiles), `Setup`, `Step`, `Validate`, `Teardown`, `Products` and `Units`. `AddStat` and `DefineInstances` record what it built. New files under `scenarios/` are picked up on the next configure.
- A preset is a function changing the scene's camera, product, settings or sun (`kPaths`), registered with `ERNIE_BENCH_PRESET` in presets.cpp; an output set, a list of render vars, with `ERNIE_BENCH_OUTPUTS`.
- A suite returns a list of cases, usually `Sweep{scenarios, axes}.Cases()`, registered with `ERNIE_BENCH_SUITE` in suites.cpp.
