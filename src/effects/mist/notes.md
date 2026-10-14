## How it works

Two passes, both cheap. `layers.wgsl` runs at a quarter of the canvas size per axis (1/16 of the pixels) and writes three density layers, one per channel of an `rgba16float` target: far, middle, near. Each layer comes from `mist_layer` in `src/wgsl/mist.wgsl`: a 3D fbm whose domain is bent by a warp field. The base pattern slides along the drift direction while the warp slides at 60 percent of that speed, so the wisps billow and fold instead of translating rigidly. Layers differ in scale (far wisps are bigger), drift speed (far is slower) and phase, so they decorrelate. `mist_profile` multiplies in the density profile: full strength at the source edge (bottom, top, left, right, or none), fading to nothing by `height`, over a `falloff` distance. Ground-hugging fog is a low `height`, a horizontal drift and `stretch` above 1; rising steam is a tall `height`, a drift near 90 degrees and `stretch` below 1.

Scroll goes through `trackScroll`. `offset` (pixels scrolled; for a sticky canvas it keeps counting while the canvas stays put on screen) shifts each layer's sample point in proportion to its depth, so near layers move further than far ones: that is the parallax. Fast scrolling stirs the drift a little.

`compose.wgsl` runs at full size. It reconstructs the three densities with a cubic B-spline (`wash_sample_bspline`; four filtered taps), weighs them into one density, and passes it through `mist_opacity`, which is monotonic, smooth and never exceeds the cap: `alpha = cap * (1 - exp(-3 d)) / (1 - exp(-3))` for `d` in [0, 1]. The output is straight premultiplied alpha, `vec4(tint * a, a)`, with no blend mode, and the tint is a little cooler and dimmer where the far layer dominates. Alpha is dithered by one LSB so that a soft veil does not band.

**Opacity cap.** `maxOpacity` is the most opaque any pixel can ever be (default 0.38, hard limit `MIST_ALPHA_LIMIT` = 0.8 in `mist.wgsl`; the control cannot go higher and the shader clamps again). Worst-case text contrast under the default cap, worked out by hand with the default tint: dark text `#0d1b22` on a white page through full-cap mist is about 5.6:1, and light text `#d8eef7` on the dark page about 5.3:1, both above the 4.5:1 AA minimum, and the typical alpha is far below the cap. `tests/decoration.test.ts` checks the cap by rendering the defaults, every preset and the extremes.

## Cost

Indicative HUD readings on an Apple Silicon laptop, 1702x932 canvas (1.6 Mpx, DPR 2), shared GPU, one after the other: default (layers at 1/4 size) about 1.8 ms, near the measurement floor (see docs/effect-authoring.md); layers at full size about 8.5 ms. The cut from rendering the field at a quarter of the size is roughly four to five times. The cost is the layer pass (three layers of five 3D gradient-noise evaluations each, at 1/16 of the pixels) and scales with its area; the compose pass is four filtered taps. Memory: one `rgba16float` target at 1/16 of the canvas (about 0.8 MB at this size). Tier A on this hardware; not measured on an integrated Intel or Mali GPU or a phone. The field-resolution control switches the size.

## Pitfalls

- **A pale mist is invisible on a pale page.** White mist over a white section is, correctly, nothing. The demo tints are mid-light so they show on both site themes; on a real site the mist sits over imagery, or the tint matches the page's fog colour.
- **Narrow canvases.** Features first measured in canvas heights gave less than one wisp across a phone-width hero, so the whole width was one blob. Features are now measured against the shorter side.
- **8-bit banding of alpha.** A smooth veil of 0 to 0.4 alpha has only about 100 steps; the steps show on a dark page. Triangular dither on alpha fixes it.
- **Premultiplied.** `rgb <= alpha` must hold; the tint is multiplied by the dithered alpha, so it does.
- **Reserved word.** `from` is a reserved word in WGSL; the uniform is `source_edge`.
- **Text above or below.** If the mist sits above text, readability depends on the cap (above). Put it under text where you can; over imagery it does not matter.
- **Parallax and a sticky canvas.** A rect-based scroll position never moves for a `position: sticky` backdrop, so the helper also reads the scrolling ancestors' `scrollTop`; reading back the canvas alpha in the `backdrop` context confirms the pattern changes with the inner scroll.

## When to use it

A soft veil that thins toward one edge over a hero photograph, fog hugging the bottom of a landscape, steam over a hot-spring or kitchen image, or depth for a long page, when a static gradient looks flat and a video is too heavy. Tier A. For a static page use two or three pre-rendered transparent PNG or WebP wisps with a slow CSS translate and a `mask-image` gradient, which needs no WebGPU; use the shader when the motion must respond to scroll and billow rather than slide. Keep it `pointer-events: none` and out of the way of text.

## References

- Inigo Quilez, [Domain warping](https://iquilezles.org/articles/warp/), for the billowing warp.
- Christian Sigg, Markus Hadwiger, [Fast Third-Order Texture Filtering](https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-20-fast-third-order-texture-filtering) (GPU Gems 2, ch. 20), for the upsampling filter.
- Andrew Schneider and Nathan Vos, [The Real-time Volumetric Cloudscapes of Horizon Zero Dawn](https://advances.realtimerendering.com/s2015/The%20Real-time%20Volumetric%20Cloudscapes%20of%20Horizon%20-%20Zero%20Dawn%20-%20ARTR.pdf) (SIGGRAPH 2015).
