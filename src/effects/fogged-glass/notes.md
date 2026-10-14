## How it works

A ping-pong `rgba16float` render texture stores condensation coverage independently of canvas resolution. Pointer paths, including coalesced movement samples, wipe capsules from that field. Recovery is an exponential approach to the selected condensation level. The display mixes sharp and mip-blurred versions of the authored street scene according to the persistent field, then adds fine fog scattering. Resize recreates the state and bind groups together.

## Cost

Two steady render passes: one reduced-grid memory update and one fullscreen optics pass. The memory grid is up to 256 cells tall with an aspect-matched width capped at 1,024, using 16 bytes per cell for the two textures. The backdrop is baked and mipmapped only on size or scene changes. Default and 16-segment drag configurations are measured in [the cost study](../../docs/weather-cost-study.md).

Local Apple M1 Max (Dawn Metal), 1920×1080: conservative complete-frame cost **1.442 ms**, using the larger of median span and workload slope over the measured configurations. Owned effect code is **9.85 kB gzipped**. Shared runtime and WGSL imports bring the transitive load to about 104 kB; standalone packaging remains Epic 12.1. Phones and other GPU classes remain unmeasured.

## Pitfalls

The wipe is real persistent coverage, but condensation is a coverage model rather than individual microdroplet physics. Half precision may quantize very slow recovery. A stroke is capped at 16 segments per rendered frame; very long high-frequency input paths may be truncated. The canvas captures a wiping gesture, so do not cover scroll-critical touch areas with it.

## When to use it

Tier B wipeable windows and glass cards where interaction is the point. Morning Window recovers slowly; Steamy Glass fills cleared paths sooner. The captured still is the no-WebGPU fallback.

## References

- [GPU Gems, Fast Fluid Dynamics Simulation on the GPU](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu) — texture ping-pong and advection background, rather than an implementation of that fluid solver.

- [WebGPU specification: render passes and premultiplied compositing](https://www.w3.org/TR/webgpu/).
