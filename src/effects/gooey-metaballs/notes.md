## How it works

Four rounded navigation pills and a fifth moving circle are combined using a polynomial smooth minimum. With the pointer outside the canvas the follower travels on a small orbit; moving inside attaches it to the pointer. Merge distance controls the connecting necks; follower radius and water strength change the circular item.

The Source tab exposes the shared renderer, shader and dynamics files. All frame resources and listeners are owned by the effect instance. The gallery host pauses for hidden/offscreen and reduced-motion states, renders a static first frame, and uses the captured poster when WebGPU is unavailable.

## Cost

Local default: **0.175 ms** median GPU workload slope at **1920 × 1080** on **Apple M1 Max (Metal)**. One fullscreen fragment pass, no compute and no required optional GPU features. One 1024 × 640 rgba16float section uses 5 MiB. The linear MSDF atlas adds 216 KiB. Section baking, decoding and upload are startup costs excluded from the steady-state GPU number.

About **54.2 kB gzipped** of effect code, including the shared UI renderer, compressed atlas and local shaders; the tiny entry wrapper alone is not a standalone embed. Cold transitive imports are around 150.8 kB. See the [UI cost study](docs/ui-cost-study.md) for all samples, active-input configurations and limitations. Tier A is a target; mobile and other integrated GPUs are **unmeasured**.

## Pitfalls

Smooth unions change the apparent outline and therefore visual hit areas. Keep actual navigation links in HTML; these four pills are decorative motifs. An SDF union is not a fluid simulation.

## When to use it

Decorative navigation markers, selection indicators and cursor followers with simple geometry. Ship the poster with semantic HTML as the fallback; preserve readable content when motion is reduced.

## References

- [WebGPU external image copies](https://www.w3.org/TR/webgpu/#dom-gpuqueue-copyexternalimagetotexture)
- [Backdrop capture study](docs/backdrop-capture-study.md)
- [msdfgen — multichannel distance fields](https://github.com/Chlumsky/msdfgen)
- [Inigo Quilez — 2D distance functions](https://iquilezles.org/articles/distfunctions2d/)
