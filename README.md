# SR-RT_GPU_engine

A fresh GPU-focused companion project to `SR-RT_engine`.

This project starts from a clean renderer core instead of trying to force the CPU path tracer architecture onto Metal. The goal is to keep the parts of the original project that are still useful at the product/interface layer, while building a new GPU-native renderer backend from scratch.

## Current Status

The project currently includes:
- a minimal Metal compute renderer starter
- a simple command-line executable, `sr_rt_gpu`
- copied `presets/video/` files from the CPU project for future GUI/video work
- a small host-side scene builder layer
- a minimal options-schema JSON output for future frontend integration

The starter renderer is intentionally small, but it is no longer just a one-off demo. It now has:
- explicit camera / sphere / plane GPU buffers
- named starter scenes
- a simple ground plane with checker shading
- a tiny backend contract we can grow over time

## Build

```bash
/Applications/CMake.app/Contents/bin/cmake -S . -B build
/Applications/CMake.app/Contents/bin/cmake --build build -j
```

## Run

```bash
./build/sr_rt_gpu
```

Default output:

- `outputs/metal_starter.ppm`

Custom output and size:

```bash
./build/sr_rt_gpu --scene wide --width 1280 --height 720 --output outputs/wide.ppm
```

Print the current minimal options schema:

```bash
./build/sr_rt_gpu --print-options-schema
```

## Why This Project Exists

The CPU renderer in `SR-RT_engine` is already useful and stable. Rather than destabilizing it, this project is a sandbox for GPU-native design decisions such as:
- flat GPU-friendly scene data
- Metal compute kernels
- GPU-oriented material evaluation
- later reuse of the existing GUI ideas against a new backend contract

## Near-Term Roadmap

1. Add a second primitive type beyond spheres and planes.
2. Split the renderer host code into smaller modules.
3. Expand the options schema so a future GUI can discover legal scenes and defaults.
4. Decide which pieces of the current GUI layer should be copied over once the backend contract exists.
