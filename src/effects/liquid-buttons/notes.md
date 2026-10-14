## How it works

Hovering the visible button or focusing the canvas drives an exactly integrated damped spring toward one; leaving drives it toward zero. That spring changes the rounded boundary and excites a surface wobble. Pointer press, Enter or Space adds an impulse and starts a decaying travelling radial wave. Frequency and damping tune the response. The canvas has a button role, keyboard input and an accessible label; its original attributes and listeners are restored on teardown.

The Source tab exposes the shared renderer, shader and dynamics files. All frame resources and listeners are owned by the effect instance. The gallery host pauses for hidden/offscreen and reduced-motion states, renders a static first frame, and uses the captured poster when WebGPU is unavailable.

## Cost

Local default: **0.153 ms** median GPU workload slope at **1920 × 1080** on **Apple M1 Max (Metal)**. One fullscreen fragment pass, no compute and no required optional GPU features. One 1024 × 640 rgba16float section uses 5 MiB. The linear MSDF atlas adds 216 KiB. Section baking, decoding and upload are startup costs excluded from the steady-state GPU number.

About **54.2 kB gzipped** of effect code, including the shared UI renderer, compressed atlas and local shaders; the tiny entry wrapper alone is not a standalone embed. Cold transitive imports are around 150.8 kB. See the [UI cost study](docs/ui-cost-study.md) for all samples, active-input configurations and limitations. Tier A is a target; mobile and other integrated GPUs are **unmeasured**.

## Pitfalls

The demonstration action is the visual press response. Connect a real CTA through a native button in a product integration. Only the centred button rectangle responds to pointer hover; keyboard focus responds without a pointer. The mocked hero can cover this rectangle, so use Canvas context to try it.

## When to use it

A small interactive CTA or onboarding action where the motion can acknowledge a real state change. Ship the poster with semantic HTML as the fallback; preserve readable content when motion is reduced.

## References

- [WebGPU external image copies](https://www.w3.org/TR/webgpu/#dom-gpuqueue-copyexternalimagetotexture)
- [Backdrop capture study](docs/backdrop-capture-study.md)
- [msdfgen — multichannel distance fields](https://github.com/Chlumsky/msdfgen)
