## How it works

Tank, horizontal progress bar and circular loader share a rounded/analytic shape mask. A two-wave sloshing surface splits the empty interior from the refracting liquid; a narrow line marks the meniscus. Fill level is live, and explicit zero/full handling prevents waves exposing a gap at the endpoints. The canvas exposes progressbar semantics and a matching 0–100 accessible value.

The Source tab exposes the shared renderer, shader and dynamics files. All frame resources and listeners are owned by the effect instance. The gallery host pauses for hidden/offscreen and reduced-motion states, renders a static first frame, and uses the captured poster when WebGPU is unavailable.

## Cost

Local default: **0.175 ms** median GPU workload slope at **1920 × 1080** on **Apple M1 Max (Metal)**. One fullscreen fragment pass, no compute and no required optional GPU features. One 1024 × 640 rgba16float section uses 5 MiB. The linear MSDF atlas adds 216 KiB. Section baking, decoding and upload are startup costs excluded from the steady-state GPU number.

About **54.2 kB gzipped** of effect code, including the shared UI renderer, compressed atlas and local shaders; the tiny entry wrapper alone is not a standalone embed. Cold transitive imports are around 150.8 kB. See the [UI cost study](docs/ui-cost-study.md) for all samples, active-input configurations and limitations. Tier A is a target; mobile and other integrated GPUs are **unmeasured**.

## Pitfalls

This demonstrates a component driven by a supplied progress value; it does not track an actual network operation. The meniscus is a visual approximation and not conserved fluid volume. Update the value from application state and put a readable percentage in HTML where users need it.

## When to use it

Upload/progress indicators, loading rings and capacity dashboards. Ship the poster with semantic HTML as the fallback; preserve readable content when motion is reduced.

## References

- [WebGPU external image copies](https://www.w3.org/TR/webgpu/#dom-gpuqueue-copyexternalimagetotexture)
- [Backdrop capture study](docs/backdrop-capture-study.md)
- [Inigo Quilez — 2D distance functions](https://iquilezles.org/articles/distfunctions2d/)
