## How it works

Click or tap creates a deterministic angular burst of independent droplets. Positions use the analytic ballistic update, including the half-gravity displacement term, so partitioning a frame does not change the trajectory. The instanced fragment shader shades each droplet as a small lens with a highlight. Optional ambient bursts start fountains every 1.5 animation seconds; Tap Feedback disables them so presses are the only trigger.

## Cost

Bounded at 2,048 live drops, with 8–128 particles per burst and finite lifetimes. Rendering touches only each sprite’s quad and one cached background pass. Active 128-drop press feedback is measured separately from the default ambient scene in [the study](../../docs/weather-cost-study.md).

Local Apple M1 Max (Dawn Metal), 1920×1080: conservative complete-frame cost **0.197 ms**, using the larger of median span and workload slope over the measured configurations. Owned effect code is **9.86 kB gzipped**. Shared runtime and WGSL imports bring the transitive load to about 104 kB; standalone packaging remains Epic 12.1. Phones and other GPU classes remain unmeasured.

## Pitfalls

Particles are a graphic feedback effect, not a continuous fluid solver, and do not collide with page elements. Repeated presses fill the bounded pool; extra droplets are ignored until old ones expire. Tiny droplets need enough canvas pixels to remain legible.

## When to use it

Tier A click feedback, a small fountain or celebratory water motif. Turn off ambient bursts for button feedback and preserve the real HTML button’s keyboard behavior in an integration.

## References

- [Iain Cantlay, GPU Gems 3 particle rendering discussion](https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-23-high-speed-screen-particles).

- [WebGPU specification: render passes and premultiplied compositing](https://www.w3.org/TR/webgpu/).
