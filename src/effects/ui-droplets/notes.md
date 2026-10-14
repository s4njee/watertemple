## How it works

A bounded CPU model starts with 32 condensation droplets and allows at most 48. Small droplets remain capillary-pinned; larger drops accelerate under gravity with drag. Substeps and swept relative motion detect coalescence. Merging conserves spherical volume (r³), volume-weighted position and velocity; drops drain past the bottom and new condensation seeds arrive at the top. A fragment loop renders each lens over a rounded card with refraction, channel dispersion and edge highlights.

The Source tab exposes the shared renderer, shader and dynamics files. All frame resources and listeners are owned by the effect instance. The gallery host pauses for hidden/offscreen and reduced-motion states, renders a static first frame, and uses the captured poster when WebGPU is unavailable.

## Cost

Local default: **1.049 ms** median GPU workload slope at **1920 × 1080** on **Apple M1 Max (Metal)**. One fullscreen fragment pass, no compute and no required optional GPU features. One 1024 × 640 rgba16float section uses 5 MiB. The linear MSDF atlas adds 216 KiB. Section baking, decoding and upload are startup costs excluded from the steady-state GPU number.

About **54.2 kB gzipped** of effect code, including the shared UI renderer, compressed atlas and local shaders; the tiny entry wrapper alone is not a standalone embed. Cold transitive imports are around 150.8 kB. See the [UI cost study](docs/ui-cost-study.md) for all samples, active-input configurations and limitations. Tier B is a target; mobile and other integrated GPUs are **unmeasured**.

## Pitfalls

Spherical volume is the merge model, rather than contact-angle physics. Growth, condensation and drainage add/remove water explicitly. The shader checks all 48 slots per pixel, and the CPU pairwise collision loop is quadratic but bounded. Drops are clipped to the card. Do not use unbounded populations for a UI effect.

## When to use it

Weather-themed cards, a cold-glass product illustration or a brief condensation vignette. Ship the poster with semantic HTML as the fallback; preserve readable content when motion is reduced.

## References

- [WebGPU external image copies](https://www.w3.org/TR/webgpu/#dom-gpuqueue-copyexternalimagetotexture)
- [Backdrop capture study](docs/backdrop-capture-study.md)
