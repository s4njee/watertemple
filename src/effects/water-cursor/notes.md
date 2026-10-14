## How it works

A half-resolution r16float ping-pong field holds wetness across frames. Up to sixteen segments from the pointer path deposit continuous strokes, including coalesced motion events. A four-neighbour diffusion step spreads the field and exponential decay dries it. The display pass uses wetness gradients for lens distortion, slight darkening and grazing highlights. Entering starts a new stroke, and leaving allows old wetness to decay. Resize clears the wet field.

The Source tab exposes the shared renderer, shader and dynamics files. All frame resources and listeners are owned by the effect instance. The gallery host pauses for hidden/offscreen and reduced-motion states, renders a static first frame, and uses the captured poster when WebGPU is unavailable.

## Cost

Local default: **0.175 ms** median GPU workload slope at **1920 × 1080** on **Apple M1 Max (Metal)**. Two fullscreen fragment passes, no compute and no required optional GPU features. One 1024 × 640 rgba16float section uses 5 MiB. The half-resolution wet field adds about 1.98 MiB for two r16float textures at 1080p. The linear MSDF atlas adds 216 KiB. Section baking, decoding and upload are startup costs excluded from the steady-state GPU number.

About **54.2 kB gzipped** of effect code, including the shared UI renderer, compressed atlas and local shaders; the tiny entry wrapper alone is not a standalone embed. Cold transitive imports are around 150.8 kB. See the [UI cost study](docs/ui-cost-study.md) for all samples, active-input configurations and limitations. Tier A is a target; mobile and other integrated GPUs are **unmeasured**.

## Pitfalls

Up to sixteen segments approximate a very dense coalesced path; finer curves can be simplified. Diffusion is clamped for stability and is a visual wetness model, not a Navier–Stokes solver. Only one pointer is tracked for the trail. The field resets on resize and is paused by the host when hidden/offscreen.

## When to use it

A subtle optional cursor treatment over owned artwork; avoid making it necessary to operate the interface. Ship the poster with semantic HTML as the fallback; preserve readable content when motion is reduced.

## References

- [WebGPU external image copies](https://www.w3.org/TR/webgpu/#dom-gpuqueue-copyexternalimagetotexture)
- [Backdrop capture study](docs/backdrop-capture-study.md)
