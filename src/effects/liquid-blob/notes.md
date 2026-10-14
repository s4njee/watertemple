## How it works

Three moving sphere SDFs form a smooth 3D union. A perspective ray marches at most 72 conservative steps to the first surface and estimates its normal by finite differences. A refracted ray marches up to 40 interior steps toward an exit surface, refracts a second time, and uses interior travel for coloured absorption. Schlick Fresnel blends that transmission with an analytic sky/light environment reflection; total internal reflection has a guarded reflection fallback.

The Source tab exposes the shared renderer, shader and dynamics files. All frame resources and listeners are owned by the effect instance. The gallery host pauses for hidden/offscreen and reduced-motion states, renders a static first frame, and uses the captured poster when WebGPU is unavailable.

## Cost

Local default: **0.437 ms** median GPU workload slope at **1920 × 1080** on **Apple M1 Max (Metal)**. One fullscreen fragment pass, no compute and no required optional GPU features. One 1024 × 640 rgba16float section uses 5 MiB. The linear MSDF atlas adds 216 KiB. Section baking, decoding and upload are startup costs excluded from the steady-state GPU number.

About **54.2 kB gzipped** of effect code, including the shared UI renderer, compressed atlas and local shaders; the tiny entry wrapper alone is not a standalone embed. Cold transitive imports are around 150.8 kB. See the [UI cost study](docs/ui-cost-study.md) for all samples, active-input configurations and limitations. Tier B is a target; mobile and other integrated GPUs are **unmeasured**.

## Pitfalls

This is a ray-marched hero lens, without scene-depth integration or a captured HDR environment map. Refraction samples the designed backdrop in screen space; the interior step budget is approximate and can limit tangent rays. Pixel count and march complexity dominate cost.

## When to use it

A focused liquid hero object or campaign illustration, sized to a bounded region of a page. Ship the poster with semantic HTML as the fallback; preserve readable content when motion is reduced.

## References

- [WebGPU external image copies](https://www.w3.org/TR/webgpu/#dom-gpuqueue-copyexternalimagetotexture)
- [Backdrop capture study](docs/backdrop-capture-study.md)
- [Inigo Quilez — 2D distance functions](https://iquilezles.org/articles/distfunctions2d/)
