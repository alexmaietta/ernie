# Efficient Renderer for Natural and Interplanetary Environments (Ernie)

Ernie is a modular, real-time, physically-accurate, spectral renderer designed for interplanetary environments. It provides a Hydra-like API along with efficient batched GPU transfers to integrate into physics and reinforcement learning workflows.




## Getting Started

### Python

```sh
pip install ernie
```

```python
import ernie

ernie.Scene()...
```

### C++



## Installation

### Precomplied binaries

### Build from source

## Profiling

Ernie can be built with zones for the [Tracy](https://github.com/wolfpld/tracy) profiler: CPU
zones for each step's phases (scene sync, frame setup, recording, every wait) and Vulkan GPU
zones for its passes (uploads, culling, shadows, G-buffer, shading, sensor filters), on one
timeline, with a frame mark per step.

```sh
cmake -S . -B build -DERNIE_TRACY=ON
cmake --build build -j
```

Nothing is collected until a profiler connects, so a profiling build runs at full speed until
then; without `ERNIE_TRACY` the zones compile to nothing. Run any program that uses Ernie,
including Python scripts against `build/python`, and connect the Tracy profiler GUI (version
0.14.1, the one Ernie fetches), or capture headless and export statistics:

```sh
# Tracy's command-line tools, from the source Ernie fetched
cmake -S build/_deps/tracy-src/capture -B build/tracy-tools/capture -DCMAKE_BUILD_TYPE=Release
cmake --build build/tracy-tools/capture -j
cmake -S build/_deps/tracy-src/csvexport -B build/tracy-tools/csvexport -DCMAKE_BUILD_TYPE=Release
cmake --build build/tracy-tools/csvexport -j

PYTHONPATH=build/python python demos/python/demo_worlds_gpu_poses.py 1024 2000 &
build/tracy-tools/capture/tracy-capture -o steps.tracy -s 5    # 5 seconds of steps
build/tracy-tools/csvexport/tracy-csvexport steps.tracy        # CPU zone statistics
build/tracy-tools/csvexport/tracy-csvexport -g steps.tracy     # each GPU zone
```

GPU zones line up with the CPU's when the device supports `VK_KHR_calibrated_timestamps`.
Instrument new code with `ERNIE_ZONE` (`ernie/util/profile.h`) and `ERNIE_GPU_ZONE`
(`ernie/device/gpu_profile.h`).
