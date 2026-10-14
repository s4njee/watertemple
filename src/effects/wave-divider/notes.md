## How it works

One fragment pass, no textures. `src/wgsl/waveline.wgsl` describes each wave layer as a curve `y = f(x)`: three sines at incommensurate frequencies plus a little gradient noise (so it never reads as a pure sine), with the analytic slope returned alongside the height. From height and slope it builds a signed distance, `d = (line_y - y) / sqrt(1 + slope^2)`, which has unit gradient at the line. That is what makes the edge cheap and crisp: coverage is `clamp(0.5 - d / px, 0, 1)` where `px` is one device pixel in CSS pixels (`1 / dpr`), so every edge is exactly one pixel wide at any DPR, with no MSAA and no `fwidth` (so the shader may branch freely).

`divider.wgsl` stacks 2 to 5 layers, back to front. Back layers have longer wavelengths, smaller speed and a weaker response to scroll; the front layer is fastest and carries a foam line (a crisp rim plus Voronoi bubbles that shrink with distance from the crest, `waveline_foam`). Layers are composited with premultiplied "over" in sRGB, the space the browser composites the rest of the page in, so the output is `vec4(rgb * a, a)` and the transparent region above the waves is a true alpha 0. `flip` mirrors the whole thing so it can cap the bottom of a section.

Scroll goes through `src/gpu/scroll.ts` (`trackScroll(canvas)`). `offset` (px scrolled) slides each layer sideways, back layers less than front ones, and `progress` (0 as the canvas enters at the bottom of the viewport, 1 as it leaves at the top) nudges the back layers up and down. Smoothed scroll speed swells the amplitude and quickens the phase a little. At rest, and with no scrolling at all, the waves simply run on their own clock; phase is accumulated so changing `speed` never makes them jump.

**Matching a page colour exactly.** The front layer is one flat colour with no dither, no gradient and no noise, and alpha 1, so it is byte-identical to a CSS colour. Parameters are linear RGB, so convert the CSS colour: `srgbToLinear(channel / 255)` per channel (the colour picker in the controls shows the sRGB hex, so you can paste `#0b1c29` there). The shader encodes back to sRGB and rounds to the same byte (`tests/decoration.test.ts` checks five colours byte for byte). The demo's "Front = demo page colour" switch reads `--wt-surface` through a hidden probe element so it follows the site theme; a real site just pastes its own hex into `colorFront` and leaves the switch off.

## Cost

Indicative HUD reading on an Apple Silicon laptop, 1702x257 canvas (a 28 percent band at DPR 2), shared GPU, `waves` pass: about 1.0 ms. That is the measurement floor described in docs/effect-authoring.md (a pass that draws almost nothing reads about 1.2 ms there), so the real cost is below what the HUD can resolve. Per pixel: up to five layers of 6 trig calls and one gradient noise, plus foam work only in the strip around the front crest. Cost scales with canvas area and layer count; no textures. Not measured on an integrated Intel or Mali GPU or a phone. Tier A.

Against the alternative: an inline SVG with the same paths is a few hundred bytes, costs no GPU context and can be animated with CSS transforms or SMIL, but each animated path is re-rasterised on the CPU or compositor, per-pixel foam is not possible, and "waves that react to scroll speed" needs JavaScript driving path data. The shader costs a WebGPU device and canvas (shared across the page if you use it elsewhere), roughly 1 ms of GPU on a laptop, and needs a fallback for browsers without WebGPU. If the only thing you want is a static or slowly translating wave edge, use the SVG.

## Pitfalls

- **Premultiplied alpha.** The canvas wants `rgb * a`. Writing straight colour with alpha makes the antialiased edge dark against a light page (a "dark fringe"). Composite in premultiplied form and clamp `rgb <= a` after dithering; the test renders every preset and checks that for every pixel.
- **Foam on a pale front colour is invisible.** The first version put foam inside the front layer, so on a white section it was white on white. Foam now straddles the crest and reaches over the water behind it, where there is always contrast.
- **Bubbles.** A bubble of radius zero still covered half of its centre pixel (a bare smoothstep ramp), leaving a dot of foam where there should be none; and discs larger than half a Voronoi cell are clipped flat at the cell edge. The radius is now faded in from zero and capped.
- **Dither breaks exact matching.** Dither is needed on the back layers' gradients (8-bit steps show), but the flat front body skips it, otherwise a section edge would show a faint noise boundary.
- **Tall waves clip at the top edge.** Amplitude is limited so the highest crest stays 2 px inside the open edge; otherwise it would draw a straight line at the top of the canvas.
- **Distance is first order.** `waveline_distance` is exact at the curve and underestimates further away; fine for antialiasing and thin bands, not for soft glows more than a few pixels wide.
- **A scroll listener would be wrong here.** The `backdrop` context scrolls an inner container; `trackScroll` reads the canvas rect and the scrolling ancestors once per frame instead.

## When to use it

The break between a hero and the next section, a footer edge, or the bottom of a coloured block (`flip`), when you want layers, per-pixel foam and a scroll response that an SVG cannot give cheaply. Tier A on Apple Silicon at full resolution; the canvas is short and wide, so area is small. Reach for an SVG path when the wave is static or only translates, when the page must render with no WebGPU and no script, or when you need exactly one layer. Place it as an ordinary block between two sections (or absolutely positioned over the seam), with `pointer-events: none` if it overlaps content, and give the section below the same colour as `colorFront`. At band heights from 96 px to 200 px and at phone width the waves keep their proportions because amplitude is a fraction of the canvas height and wavelength is in CSS pixels.

## References

- Inigo Quilez, [Distance to implicit curves](https://iquilezles.org/articles/distance/), for the `f / |grad f|` estimate used here.
- Inigo Quilez, [Smooth Voronoi](https://iquilezles.org/articles/smoothvoronoi/), for the bubble field behind the foam.
- Alex Vlachos, [Water Flow in Portal 2](<https://advances.realtimerendering.com/s2010/Vlachos-Waterflow(SIGGRAPH2010).pdf>) (SIGGRAPH 2010), for foam thickness thresholded by a coverage mask.
- WebGPU spec, [canvas alpha modes](https://www.w3.org/TR/webgpu/#canvas-configuration), for premultiplied compositing.
