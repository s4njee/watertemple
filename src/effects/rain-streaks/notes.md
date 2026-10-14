## How it works

Rain particles use a bounded CPU store and one instanced quad draw. Each has depth-dependent fall speed, width and opacity; wind changes horizontal velocity. The renderer sorts from far to near. Ground crossings emit expanding elliptical rings plus three upward ballistic droplets. A second fullscreen pass presents the procedural street texture, baked only on scene or size changes. No compute shader is needed.

## Cost

Particle updates and sorting scale with count. The store is capped at 2,048 particles including impact droplets; the rain control exposes 16–1,000 streaks. The instance buffer is 96 KiB. A mipmapped, reduced-resolution `rgba16float` backdrop is shared by the two rendering passes. Local 1080p default and 1,000-streak runs are recorded in [the cost study](../../docs/weather-cost-study.md).

Local Apple M1 Max (Dawn Metal), 1920×1080: conservative complete-frame cost **0.328 ms**, using the larger of median span and workload slope over the measured configurations. Owned effect code is **9.86 kB gzipped**. Shared runtime and WGSL imports bring the transitive load to about 104 kB; standalone packaging remains Epic 12.1. Phones and other GPU classes remain unmeasured.

## Pitfalls

Depth is a presentation attribute rather than a full 3D scene or depth-buffer collision. The ground slopes only by the depth band; DOM cards do not obstruct rain. Extremely narrow streaks may alias at small thumbnail sizes. Stalls are capped at 0.1 seconds before speed scaling.

## When to use it

Tier A ambient rain in a weather hero or backdrop. The Drizzle and Storm presets demonstrate density, wind and fall speed. Keep text on a contrast plate in busy scenes.

## References

- [Iain Cantlay, High-Speed, Off-Screen Particles, GPU Gems 3](https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-23-high-speed-screen-particles) — particle overdraw trade-offs; this small demo renders its crisp particles directly.

- [WebGPU specification: render passes and premultiplied compositing](https://www.w3.org/TR/webgpu/).
