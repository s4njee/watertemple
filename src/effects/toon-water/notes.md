## How it works

One fragment pass plus a one-off bake. The island and eight rocks never move, so `sdf-bake.wgsl` draws their signed distance fields once per canvas size into a half-resolution `rg16float` texture (island in r, rocks in g). The main pass `toon-water.wgsl` reads it with one bilinear fetch, so the coast costs no noise per pixel. A distance field is smooth, so half resolution still gives straight, sub-pixel-accurate edges.

The reusable stylisation is `src/wgsl/toon.wgsl`. Its one convention is that every function takes `aa`, the size of one pixel in the function's own unit, and replaces each hard edge with a one pixel linear ramp; because the scene is flat, a pixel in world units is a constant (`zoom / height`), so no derivatives are needed for most edges. The water goes through these steps:

- **Depth bands.** The distance to the nearest land, plus a slow sine wobble so contours are not perfect offset curves, becomes `t`; `toon_band(t, bands, aa)` quantises it into flat levels and `toon_ramp` turns the level into one of four palette colours.
- **Cell foam.** `toon_foam_cells` is F2 minus F1 of the animated Voronoi (`voronoi_2d_anim`): zero on cell borders, with seeds that orbit so cells drift and morph instead of sliding. `toon_cell_edge(F, |grad F|, width_px)` thresholds it into a line whose width is the same number of pixels everywhere: F divided by its screen-space gradient (`dpdx`, `dpdy`) is the distance to the border in pixels. A second, larger and slower layer sits in deep water. Patches of foam are cut off by a hard threshold on one drifting noise field (used for both layers, in opposite senses), so line ends are crisp rather than faded.
- **Shore.** A foam ring whose width breathes (`toon_ring` on the signed distance), a detached swash line behind it, and an ink outline, both sides of the coast.
- **Glints.** Hard-edged swooshes (a thresholded stretched noise) over deep water, and four-pointed star sparkles (`toon_sparkle`, one cell lookup, astroid shape) that bloom and fade.
- **Land.** Sand rim, grass with round bushes (a static Voronoi, one disc per cell with a lit cap), rocks with a lit cap made by sampling the rock field shifted toward the shadow side.

`deepFoam` switches the second Voronoi layer off (a uniform branch, so derivatives stay legal). Palettes are named sets of twelve colours (sRGB hex converted to linear in `index.ts`); `custom` builds a ramp from three colour parameters. The name is a nod to the project: this is the playful end of the group, a generic cel-shading technique with no game's art in it.

## Cost

Indicative, from the HUD on an Apple Silicon laptop, 1702x932 (1.6 Mpx), DPR 2, GPU shared with other tabs: about 2.0 ms for the `toon water` pass with both foam layers, about 1.6 ms with `deepFoam` off, so each animated Voronoi is roughly 0.4 ms there and the rest sits near the ~1.2 ms floor the HUD cannot resolve below. The distance field bake runs once per resize and was not read separately. Memory: 1.5 MB for the field at this size. The HUD reads that as 102 percent of the 2 ms Tier A budget at 1.6 Mpx (80 percent with `deepFoam` off), and 1080p is 2.1 Mpx, so on this machine the default is right at the line and the cheap mode is inside it. On an integrated GPU or a phone expect the second layer to be the thing to switch off; none of that was measured. The first draft evaluated two noise octaves and eight circles per pixel for the coast; baking them did not move the HUD number measurably, which says the two Voronoi evaluations dominate.

What scales the cost: pixels, then the foam layers (one Voronoi each), then everything else, which is small.

## Pitfalls

- **Line width is the whole look.** F2 minus F1 is not a distance; dividing by its gradient in pixels is what keeps a line two pixels wide at every angle. Without it, lines swell where cells are flat and vanish where they are steep.
- **Derivatives need uniform control flow.** The foam must be evaluated for every pixel, land included, before the land branch; a Voronoi inside `if (land)` gives garbage `dpdx`. The `deepFoam` skip is a uniform condition for that reason.
- **Constant-size units.** Widths are given for a 600 px tall canvas and scaled by height, so the picture looks the same at DPR 1 and 2 and in a small card.
- **Narrow canvases.** The scene zooms out when the aspect is below 1.25 so the island fits a portrait phone.
- **Smooth fades are not toon.** A `smoothstep` mask on glints looked like airbrushing; every mask here is a one pixel ramp.
- **Shimmer.** Checked by rendering two frames a quarter of a frame apart: 1.3 percent of pixels changed by more than 8 levels and 0.02 percent by more than 120, i.e. edges move by sub-pixel amounts and nothing flickers.
- **Reading the numbers.** The HUD floor is about 1.2 ms on this machine; do not conclude the foam costs "0.4 ms" on other hardware.

## When to use it

Playful or illustrative sites, games, kids and education products, a footer or hero that is meant to look drawn. It is cheap, resolution-independent and survives being a still image, which also makes it the right fallback story: ship an SVG or PNG of the same palette where WebGPU is missing. Prefer a realistic effect (refraction, detail normals, flow map) where the photograph is the point. For a subtler painted look see the ink-wash and mist effects.

## References

- Steven Worley, A Cellular Texture Basis Function (1996): the Voronoi cells behind the foam network.
- Inigo Quilez, [Distance functions](https://iquilezles.org/articles/distfunctions2d/) and [Smooth Voronoi](https://iquilezles.org/articles/smoothvoronoi/).
- Alex Vlachos, [Water Flow in Portal 2](<https://advances.realtimerendering.com/s2010/Vlachos-Waterflow(SIGGRAPH2010).pdf>), SIGGRAPH 2010, for shoreline foam shaping.
- `src/wgsl/toon.wgsl`, `voronoi.wgsl` and `sdf.wgsl` in this repository.
