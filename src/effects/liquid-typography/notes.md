## How it works

A genuine RGB multi-channel signed-distance atlas is generated from Anton using msdfgen edge colouring and distance/error-correction routines. The shader reconstructs coverage from the median of RGB distances and scales edge anti-aliasing with screen derivatives. Three styles perturb glyph coordinates with a ripple, extend the lower outline into melting drips, or mask water behind the letters. The shipped atlas contains A–Z and 0–9; the display word is LIQUID. Generator source, pinned msdfgen revision and both licenses are included.

The Source tab exposes the shared renderer, shader and dynamics files. All frame resources and listeners are owned by the effect instance. The gallery host pauses for hidden/offscreen and reduced-motion states, renders a static first frame, and uses the captured poster when WebGPU is unavailable.

## Cost

Local default: **0.262 ms** median GPU workload slope at **1920 × 1080** on **Apple M1 Max (Metal)**. One fullscreen fragment pass, no compute and no required optional GPU features. One 1024 × 640 rgba16float section uses 5 MiB. The linear MSDF atlas adds 216 KiB. Section baking, decoding and upload are startup costs excluded from the steady-state GPU number.

About **54.2 kB gzipped** of effect code, including the shared UI renderer, compressed atlas and local shaders; the tiny entry wrapper alone is not a standalone embed. Cold transitive imports are around 150.8 kB. See the [UI cost study](docs/ui-cost-study.md) for all samples, active-input configurations and limitations. Tier B is a target; mobile and other integrated GPUs are **unmeasured**.

## Pitfalls

Fixed cell advances deliberately replace kerning/shaping in this study. The displayed word is not editable, selectable or a substitute for a semantic heading. Large melts can stretch glyph corners; a four-pixel atlas range limits outline effects. The distance atlas must be sampled as linear data, never sRGB.

## When to use it

A display headline or campaign word, paired with an HTML heading and a static-text fallback. Ship the poster with semantic HTML as the fallback; preserve readable content when motion is reduced.

## References

- [WebGPU external image copies](https://www.w3.org/TR/webgpu/#dom-gpuqueue-copyexternalimagetotexture)
- [Backdrop capture study](docs/backdrop-capture-study.md)
- [msdfgen — multichannel distance fields](https://github.com/Chlumsky/msdfgen)
