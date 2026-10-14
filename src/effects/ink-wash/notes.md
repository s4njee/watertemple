## How it works

Watercolour is a threshold of a smooth field plus paper, so the expensive part (the field) is low-frequency and does not need full resolution. Three passes:

1. **bake paper (once).** `paper.wgsl` writes a 256x256 tileable texture, grain in red and thin curved fibres in green, using `wash_paper_tile` from `src/wgsl/wash.wgsl`. It repeats every 200 CSS pixels.
2. **pigment field (1/4 of the size per axis, 1/16 of the pixels).** `field.wgsl` calls `wash_field`: a domain-warped 3D fbm for the bloom shape, a slow tone noise, and a detail noise; nine gradient-noise evaluations, written to an `rgba16float` target.
3. **compose (full size).** `compose.wgsl` reconstructs the field with a cubic B-spline (`wash_sample_bspline`: four bilinear taps, C2 smooth, so no kinks at texel borders even at 4x magnification), thresholds it into blooms, and mixes two pigments over the paper. There is no noise in this pass.

The look comes from a few cheap rules. A bloom is where the field exceeds a level; its outline distance in pixels is `(field - level) / fwidth(field)`, so edge widths are in pixels whatever the field's slope. `wash_pool` adds the darker rim just inside the outline where pigment gathers (edge darkening). The tone field decides per region whether an edge is crisp (painted on dry paper) or feathered (painted wet): that variation is most of what stops it looking like a map. A faint wide wash under the blooms, pigment that deepens toward a bloom's middle, and a mottle from the detail field finish it. `wash_granulate` makes pigment settle into the paper's valleys (more granules where there is more pigment); fibres tint the paper.

Output: opaque, with the paper colour, or with "Transparent paper" on, premultiplied ink over nothing, so the page behind shows through the unpainted paper. Pigments are mixed in sRGB, like paint on a screen. Output is dithered by one LSB. Palettes (indigo, tea, storm, rose, moss) have a light and a dark paper variant and stay close to the paper's brightness so text remains readable; `custom` takes three colours.

## Cost

Indicative HUD readings on an Apple Silicon laptop, 1702x932 canvas (1.6 Mpx, DPR 2), shared GPU, measured one after the other on the same page: default (field at 1/4 size) about 1.3 to 1.8 ms, which is at or near the measurement floor (see docs/effect-authoring.md); field at 1/2 size about 1.7 ms; field at full size about 5.2 ms. `hello` (five octaves of warped fbm at full size) read about 5.1 ms in the same session. So the same kind of field computed per full-resolution pixel costs about what `hello` costs, and computing it at a quarter of the size and upsampling costs roughly a quarter of that, at the floor. The field-resolution control in the panel switches between them. Memory: the field target (about 0.8 MB at this size), the paper (256 KB). Tier A with margin; not measured on an integrated Intel or Mali GPU or a phone. The cost is dominated by the field pass, and scales with the field's area; the compose pass is four filtered taps and one paper tap per pixel.

## Pitfalls

- **Bilinear upsampling of a thresholded field shows kinks.** A low-resolution field read with plain bilinear filtering has gradient discontinuities at texel borders, which become faceted outlines once you threshold it. The cubic B-spline fixes that; the test checks it reproduces linear data exactly.
- **Edge width in field units varies.** A constant width in field units is a thin line where the field is steep and a wide smear where it is flat. Dividing by `fwidth` gives a width in pixels. That needs derivatives in uniform control flow, so the compose pass computes all of them before any branch.
- **First version looked like a map.** One threshold, a flat interior and a uniform edge. Varying edge softness, a mottled interior, a faint wide wash and deeper pigment toward the middle fixed it.
- **Dark paper makes the pooled rim look like a neon outline.** Pigment that is lighter than the paper reads as a glow. The dark presets use a lower edge darkening.
- **Paper in device pixels changes with DPR.** The paper tile is sized in CSS pixels, so the grain keeps its size on a 1x and a 2x screen.
- **Banding.** A slow, low-contrast gradient bands visibly in 8 bits; the one-LSB triangular dither (`sky_dither`) hides it.
- **Legibility.** Contrast is tested, not assumed: for the default palettes, body text (`#0d1b22` on light paper, `#d8eef7` on dark) stays above 7:1 against the worst pixel (measured 8.3 to 10.2) and the darkest-to-lightest swing is under 2:1 (measured 1.5 to 1.9); every non-transparent preset stays above 4.5:1. Raising `contrast` or `edgeDarkening` lowers that.

## When to use it

A calm, restrained background behind long-form text, documentation or a portfolio, or a hero that should read as paper and pigment rather than as a gradient. Put text on a plate or keep `contrast` at its default; the `backdrop` and `hero` contexts show both. "Transparent paper" turns it into an ink layer over a page colour or an image. Tier A. For a static page, a pre-rendered WebP tile (or an SVG `feTurbulence` filter) in the same palette is cheaper still and needs no WebGPU; use the shader when the slow motion matters, or the same field is reused (Epic 6's ink-in-water builds on `wash.wgsl`).

## References

- Cassidy Curtis, Sean Anderson, Joshua Seims, Kurt Fleischer, David Salesin, [Computer-Generated Watercolor](https://grail.cs.washington.edu/projects/watercolor/) (SIGGRAPH 1997), for edge darkening, granulation and paper.
- Christian Sigg, Markus Hadwiger, [Fast Third-Order Texture Filtering](https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-20-fast-third-order-texture-filtering) (GPU Gems 2, ch. 20), for the four-tap cubic B-spline.
- Inigo Quilez, [Domain warping](https://iquilezles.org/articles/warp/), for the warped fbm.
- Inigo Quilez, [Distance to implicit curves](https://iquilezles.org/articles/distance/), for the pixel-space outline distance.
