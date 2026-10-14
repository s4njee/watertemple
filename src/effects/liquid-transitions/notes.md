## How it works

Two separately baked sections are blended through a moving drip boundary, a sinusoidal wave or a spatial noise dissolve. A narrow band at the boundary displaces both images and adds a liquid edge highlight. The automatic cycle uses continuous phase; disable it to scrub exact progress. Clicking the canvas in manual mode animates to the opposite section over a smooth eased interval. Progress zero and one return the two original images exactly for every style.

The Source tab exposes the shared renderer, shader and dynamics files. All frame resources and listeners are owned by the effect instance. The gallery host pauses for hidden/offscreen and reduced-motion states, renders a static first frame, and uses the captured poster when WebGPU is unavailable.

## Cost

Local default: **0.109 ms** median GPU workload slope at **1920 × 1080** on **Apple M1 Max (Metal)**. One fullscreen fragment pass, no compute and no required optional GPU features. Two 1024 × 640 rgba16float sections use 10 MiB. The linear MSDF atlas adds 216 KiB. Section baking, decoding and upload are startup costs excluded from the steady-state GPU number.

About **54.2 kB gzipped** of effect code, including the shared UI renderer, compressed atlas and local shaders; the tiny entry wrapper alone is not a standalone embed. Cold transitive imports are around 150.8 kB. See the [UI cost study](docs/ui-cost-study.md) for all samples, active-input configurations and limitations. Tier A is a target; mobile and other integrated GPUs are **unmeasured**.

## Pitfalls

This is a section-image transition prototype, not a router. In a routed application commit content/focus changes when the transition finishes, and keep links usable during loading. Both section textures remain resident. Strong displacement clamps at the image boundary.

## When to use it

Short section reveals, route covers and image galleries with two owned snapshots. Ship the poster with semantic HTML as the fallback; preserve readable content when motion is reduced.

## References

- [WebGPU external image copies](https://www.w3.org/TR/webgpu/#dom-gpuqueue-copyexternalimagetotexture)
- [Backdrop capture study](docs/backdrop-capture-study.md)
