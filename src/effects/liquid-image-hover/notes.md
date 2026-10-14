## How it works

Pointer distance launches an analytic radial ripple across the coastal illustration. Its gradient displaces the image and separates red and blue sampling positions. The image region remains still while the pointer is outside. Drop a local image on the canvas to replace the illustration: the bitmap is uploaded to an sRGB texture, sampled in linear light, and fitted with a centred cover crop. Decode/version guards close bitmaps and prevent stale uploads after destruction.

The Source tab exposes the shared renderer, shader and dynamics files. All frame resources and listeners are owned by the effect instance. The gallery host pauses for hidden/offscreen and reduced-motion states, renders a static first frame, and uses the captured poster when WebGPU is unavailable.

## Cost

Local default: **0.175 ms** median GPU workload slope at **1920 × 1080** on **Apple M1 Max (Metal)**. One fullscreen fragment pass, no compute and no required optional GPU features. One 1024 × 640 rgba16float section uses 5 MiB. The linear MSDF atlas adds 216 KiB. Section baking, decoding and upload are startup costs excluded from the steady-state GPU number.

About **54.2 kB gzipped** of effect code, including the shared UI renderer, compressed atlas and local shaders; the tiny entry wrapper alone is not a standalone embed. Cold transitive imports are around 150.8 kB. See the [UI cost study](docs/ui-cost-study.md) for all samples, active-input configurations and limitations. Tier A is a target; mobile and other integrated GPUs are **unmeasured**.

## Pitfalls

The default is a generated coastal illustration. The hover effect is inside the right-hand image rectangle; pointer interaction can be obscured by mock DOM content in Hero context. Very large images can exceed the device texture limit and retain the default with an accessible failure message. Still-image uploads are excluded from steady-state timings.

## When to use it

Editorial cards, photography galleries and product image previews with occasional hover feedback. Ship the poster with semantic HTML as the fallback; preserve readable content when motion is reduced.

## References

- [WebGPU external image copies](https://www.w3.org/TR/webgpu/#dom-gpuqueue-copyexternalimagetotexture)
- [Backdrop capture study](docs/backdrop-capture-study.md)
