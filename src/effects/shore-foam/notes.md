## How it works

Foam appears where the water is shallow or touches something, so the shared chunk `src/wgsl/foam.wgsl` keys everything off one number, `depth`, the thickness of water at the pixel. It builds the look in three steps that can be changed separately:

1. `foam_mask(depth, width, noise, breakup)` is the coverage: 1 at depth 0, 0 at `width`, with the reach of the foam shortened where a noise value is low, so it comes out in fingers instead of a band.
2. `foam_texture(p, time)` is the thickness: two scales of animated Voronoi, with lace on the cell walls and bubbles in the cells, under a slow patchiness. The seeds orbit, so the lace boils instead of sliding.
3. `foam_dissolve(thickness, coverage, softness)` thresholds one by the other. As coverage falls, thin lace goes first and the thick clumps last, so foam breaks up into holes instead of fading to grey.

The same depth drives `foam_edge_fade` (0 at the intersection, 1 beyond a fade distance), which hides the hard line where a water surface meets geometry, and `foam_wetness` and `foam_darken_wet` for the dark sand a wave leaves behind.

Two passes. `ground.wgsl` bakes the static beach once per resize into two targets: dry sand and rocks with the terrain height in alpha (`rgba16float`), and the signed distance to the nearest rock plus a smooth noise used to offset the timing along the shoreline (`rg16float`). `surf.wgsl` runs every frame. It computes the water level from `foam_swash`, a run-up cycle that rushes up the beach fast and drains slowly, offset per position so the lines are not in step and so the swell reaches the beach later the further out it starts. `depth = level - terrain height` goes through `absorb_water` and `absorb_composite` for the colour of the water over the sand, through `foam_edge_fade` for a soft contact, and through the foam steps for the whitewater at the waterline and a ring around each rock. The foam texture is carried a little with the water level, so it moves with the wave. The foam work is skipped for every pixel that is not near a waterline or a rock.

The sandbar offshore catches the swell too, so a foam line appears over it when the water is high.

## Cost

Indicative HUD readings on an Apple Silicon laptop, 1702x724 (1.2 Mpx), DPR 2, shared GPU: about 1.1 ms for the `surf` pass with the calm lagoon preset and about 1.4 ms with lively surf. The first version computed the foam everywhere and read 2.45 ms; branching it out of the open water and the dry beach roughly halved that. The bake pass is not repeated. Two textures the size of the canvas (`rgba16float` plus `rg16float`, about 15 MB at this size). Tier A on this hardware; not measured on an integrated Intel or Mali GPU or a phone, where a heavy foam band across the whole frame (large `foam width`) will cost more.

## Pitfalls

- **Full coverage is a white sheet.** With coverage at 1 the thresholded lace is solid. The demo caps the coverage it hands to `foam_dissolve` at 0.85 of the `foam amount` slider so holes always show.
- **Foam only in water.** `foam_mask` is 1 at any depth at or below 0, including the dry beach, so the caller must multiply by "is there water here".
- **Rocks must stand above the swell.** A rock whose top was below the water level still got a foam ring but no visible rock. Rock tops are set above the highest run-up.
- **Rigid lines.** One wave clock for the whole beach makes every foam line slide in parallel. The per-position phase noise and the delay with distance from the shore fix that.
- **Depth range.** A linear slope read as muddy sand-coloured water; a shelf that drops away makes the deep water read as water.
- **Timing the baked beach.** Changing a bake-time constant needs a resize; the beach is not a live parameter.

## When to use it

Shoreline, pond edge and river bank in a hero, section background or card, where the view is top-down or near it. Tier A on Apple Silicon at full resolution. It does not simulate waves, only the run-up level and foam; for a 3D view with geometry, feed `foam_mask` and `foam_edge_fade` the difference between the scene depth and the water depth (stories 8.7, 8.8) and reuse the same texture and dissolve steps.

## References

- Alex Vlachos, [Water Flow in Portal 2](<https://advances.realtimerendering.com/s2010/Vlachos-Waterflow(SIGGRAPH2010).pdf>), SIGGRAPH 2010, for soft depth-based foam and edge fade.
- Steven Worley, A Cellular Texture Basis Function (1996), the Voronoi basis of the lace.
- Inigo Quilez, [Smooth Voronoi](https://iquilezles.org/articles/smoothvoronoi/), for the cell-wall and distance tricks.
- NVIDIA, [Soft Particles](https://developer.download.nvidia.com/whitepapers/2007/SDK10/SoftParticles_hi.pdf) (2007), the depth-difference fade that `foam_edge_fade` follows.
