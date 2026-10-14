## How it works

The shared scroll tracker reads window and nested scrolling ancestors, including fixed/sticky backdrops whose rectangle does not move. Eased scroll distance divided by the configured range sets the water level. Pixels below the two-wave surface refract the generated page illustration and receive depth absorption; a meniscus crosses the design as the page submerges. Preview mode exposes an explicit level for testing and screenshots.

The Source tab exposes the shared renderer, shader and dynamics files. All frame resources and listeners are owned by the effect instance. The gallery host pauses for hidden/offscreen and reduced-motion states, renders a static first frame, and uses the captured poster when WebGPU is unavailable.

## Cost

Local default: **0.153 ms** median GPU workload slope at **1920 × 1080** on **Apple M1 Max (Metal)**. One fullscreen fragment pass, no compute and no required optional GPU features. One 1024 × 640 rgba16float section uses 5 MiB. The linear MSDF atlas adds 216 KiB. Section baking, decoding and upload are startup costs excluded from the steady-state GPU number.

About **54.2 kB gzipped** of effect code, including the shared UI renderer, compressed atlas and local shaders; the tiny entry wrapper alone is not a standalone embed. Cold transitive imports are around 150.8 kB. See the [UI cost study](docs/ui-cost-study.md) for all samples, active-input configurations and limitations. Tier B is a target; mobile and other integrated GPUs are **unmeasured**.

## Pitfalls

This submerges the canvas-rendered page illustration, not the surrounding DOM text. Use Backdrop context to exercise nested scrolling, or scroll the effect page while it remains visible. Headless mode has no scroll offset and stays dry unless preview is enabled. Pair a production integration with readable HTML and reduced motion.

## When to use it

An editorial story backdrop, scroll-driven section reveal or narrative landing page. Ship the poster with semantic HTML as the fallback; preserve readable content when motion is reduced.

## References

- [WebGPU external image copies](https://www.w3.org/TR/webgpu/#dom-gpuqueue-copyexternalimagetotexture)
- [Backdrop capture study](docs/backdrop-capture-study.md)
