# GPU Roadmap

## Guiding Principle

Keep `SR-RT_engine` as the working CPU renderer.
Use `SR-RT_GPU_engine` to explore a cleaner GPU-native architecture without inheriting CPU-only constraints.

## Starter Milestones

1. Metal toolchain sanity check
- render one shaded sphere to a file
- prove host/shader/dispatch/output flow works

2. Camera + scene uniforms
- move from hardcoded scene values to explicit GPU buffers

3. Primitive list
- start with spheres only
- move to a flat buffer of sphere structs on the GPU

4. Shared frontend contract
- decide what a future GUI/backend API should look like
- only then bring over GUI code intentionally

## Non-Goals For The Starter

- full parity with the CPU renderer
- complete material library
- video rendering first
- packaging and release automation immediately
