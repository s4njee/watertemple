## How it works

A rounded-rectangle signed distance clips a convex lens over the pre-rendered section. The lens samples a refracted position, offsets the red and blue samples at its edge, and averages five taps for a small blur. A grazing specular rim and a fixed MSDF label sit above the backdrop. Corner radius, blur, tint and refraction are live controls.

The Source tab exposes the shared renderer, shader and dynamics files. All frame resources and listeners are owned by the effect instance. The gallery host pauses for hidden/offscreen and reduced-motion states, renders a static first frame, and uses the captured poster when WebGPU is unavailable.

## Cost

Local default: **0.306 ms** median GPU workload slope at **1920 × 1080** on **Apple M1 Max (Metal)**. One fullscreen fragment pass, no compute and no required optional GPU features. One 1024 × 640 rgba16float section uses 5 MiB. The linear MSDF atlas adds 216 KiB. Section baking, decoding and upload are startup costs excluded from the steady-state GPU number.

About **54.2 kB gzipped** of effect code, including the shared UI renderer, compressed atlas and local shaders; the tiny entry wrapper alone is not a standalone embed. Cold transitive imports are around 150.8 kB. See the [UI cost study](docs/ui-cost-study.md) for all samples, active-input configurations and limitations. Tier A is a target; mobile and other integrated GPUs are **unmeasured**.

## Pitfalls

The five-tap blur is an inexpensive approximation: large radii can show separate taps. Refraction can clamp at texture edges. This panel refracts its designed content, not arbitrary DOM under the canvas.

## When to use it

Product cards, calm translucent dashboard panels and visual overlays with owned backdrop imagery. Ship the poster with semantic HTML as the fallback; preserve readable content when motion is reduced.

## References

- [WebGPU external image copies](https://www.w3.org/TR/webgpu/#dom-gpuqueue-copyexternalimagetotexture)
- [Backdrop capture study](docs/backdrop-capture-study.md)
- [msdfgen — multichannel distance fields](https://github.com/Chlumsky/msdfgen)
- [Inigo Quilez — 2D distance functions](https://iquilezles.org/articles/distfunctions2d/)
