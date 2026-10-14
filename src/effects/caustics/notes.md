## How it works

Caustics are what a wavy surface does to sunlight: it focuses the light into a net of bright, connected filaments on the floor and leaves the rest dark. The shared chunk `src/wgsl/caustics.wgsl` builds that net four ways, all as pure functions of a tile coordinate and a time in radians. Every one repeats exactly every 1.0 in x and y (so it can be baked or tiled) and every procedural one loops after TAU in time. The demo is one fullscreen pass (`overlay.wgsl`) that evaluates the chosen function, exposes it with `caustics_expose`, tints it, and writes it as premultiplied alpha so the page shows through.

**Energy is conserved, not added.** Each function returns light with a mean of about 1 for any `sharp`. Sharper caustics put the same energy into thinner lines (peaks of 5 to 30 times the mean) instead of adding white, so the picture does not get brighter as it gets crisper. The mean is held by a fitted normaliser per construction and checked on the GPU in `tests/water-light.test.ts`: within 2% for the procedural methods across sharpness 0 to 1, within 15% for the baked ones.

The four methods:

1. **Warped cellular** (`caustics_cellular`). F2 minus F1 of a tileable, animated Voronoi (`voronoi_2d_tile_anim`): zero on cell borders, so the light collects there. The input is bent by low sine harmonics of the whole tile so the cells curve like a wavy surface, and each cell scales its border by a hash so some edges are brighter. The sharpness sets the line width.
2. **Iterated sine** (`caustics_sine`). The classic water-turbulence idea written from scratch: four rounds of `q = p + warp(q, time)`, each adding a Lorentzian peak at the zeros of the warped phase. Three changes make it usable: integer harmonics (it tiles and loops), a different scale per round of `(f + j) / f` (the argument is still an integer number of periods per tile, but the rounds are not commensurate, so no lattice shows), and a low-frequency bend of the whole tile. No hashes and no cell search.
3. **Baked, two taps** (`caustics_baked`, `src/gpu/bake-caustics.ts`). One of the two functions above is rendered once into a 512x512 `rgba16float` tile whose four channels are four time slices of one loop. Each frame is one tap, or two drifting taps (the second 1.41x finer, turned 45 degrees, half a loop out of phase) combined by average or by `min`. The slices are blended cyclically, so the net boils instead of just sliding. A higher mip `lod` blurs it, which is the cheap stand-in for lower sharpness and keeps the mean.
4. **Stacking** (`caustics_stack`) and **chromatic fringing** (`caustics_rgb`, `caustics_baked_rgb`) are options on top: two layers (the second turned 45 degrees and finer, averaged) give the denser look of a real pool, and fringing samples the pattern at three positions about a tenth of a cell apart, red one way and blue the other.

The demo's **View** control: `overlay` writes premultiplied alpha, `opaque` shows the same light over a deep-water colour, and `seam check` shifts the pattern half a tile and draws the real tile seams in red, so a seam would show as a break in the filaments across a line. There is none.

## Cost

Indicative HUD readings on an Apple Silicon laptop (Metal), canvas 1702x828 (1.4 Mpx, DPR 2), one fullscreen pass, other tabs sharing the GPU. The HUD floor on this machine is about 1 ms for a pass that draws almost nothing, so anything near 1 ms means "too cheap to measure here".

| Setting                          | `caustics` pass |
| -------------------------------- | --------------- |
| cellular, 1 layer                | about 1.1 ms    |
| cellular, 2 layers               | about 2.1 ms    |
| cellular, 2 layers + fringing    | about 2.6 ms    |
| sine, 1 layer                    | below 1 ms      |
| sine, 2 layers                   | about 1.3 ms    |
| sine, 2 layers + fringing        | about 2.1 ms    |
| baked, average, 2 taps           | about 1.1 ms    |
| baked, min, 6 taps with fringing | about 1.0 ms    |

One reading of 18.8 ms for sine with 2 layers and fringing was taken while another tab was loading the GPU and is not representative; the table is from quiet readings. Texture memory is two baked maps (one per source) of 2.7 MB each with their mips, rendered once at startup. Cost scales with pixels times evaluations: cellular is about 130 ALU plus 9 hashes per evaluation, sine about 100 ALU plus 40 sin/cos, baked two taps plus about 20 ALU. Nothing was measured on an integrated Intel or Mali GPU or on a phone; sine and baked are the safe ones to try there.

### The comparison

| Approach          | Per-pixel work                                 | Indicative cost (1.4 Mpx, M-series) | Look                                                                    | Tiling and loop                        | Artefacts                                                                                                 |
| ----------------- | ---------------------------------------------- | ----------------------------------- | ----------------------------------------------------------------------- | -------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Warped cellular   | 9 cell hashes + 4 sin; x2 for two layers       | 1.1 ms (1 layer), 2.1 ms (2)        | Crisp polygonal net with bright borders; the classic look               | Exact, period 1 and TAU                | Cell shapes show at low `cells` and with one layer; uniform line width                                    |
| Iterated sine     | about 40 sin/cos, no hashes; x2 for two layers | below 1 ms (1 layer), 1.3 ms (2)    | Long flowing filaments, soft; the most water-like of the four to my eye | Exact, period 1 and TAU                | Regular lattice if the scales are commensurate (fixed in the chunk)                                       |
| Baked, single tap | 1 texture tap (4 slices) + 10 ALU              | at the floor (about 1 ms)           | The baked net itself, flowing strands, boiling through its slices       | Exact (repeat sampler); slice loop TAU | Needs 2.7 MB and a bake; the four slices cross-fade; sharpness only via mips                              |
| Baked, average    | 2 texture taps (4 slices each) + 20 ALU        | about 1.1 ms                        | Whatever it was baked from, softly layered                              | Exact (repeat sampler); slice loop TAU | Needs 2.7 MB and a bake; the four slices cross-fade (not examined at high speed); sharpness only via mips |
| Baked, min        | the same two taps, combined by `min`           | about 1.0 ms (also with 6 taps)     | Sparse bright slivers and nodes; a sparkle, not a net                   | As above                               | Throws away 85% of the light and rescales (peaks of 70); looks like glitter, not water                    |

**Recommendation.** For an ambient overlay on phones or under other effects, bake the sine net and read it with a single tap: one texture read per pixel, at the measurement floor, 2.7 MB, and the boiling slices keep it from looking like a scrolling image. (Averaging two taps of the sine map was tried first and looked like fibre or hair: two long-line nets overlapped make a mesh. Averaging suits the cellular map, where it fills in the polygon look.) For a hero where it is the star and you want to tune sharpness live, evaluate `caustics_sine` per pixel with one layer: it costs about the same as the bake lookup here and has no memory, no seam and no slider limits. Use `caustics_cellular` when you want the cell-net look, and expect to pay two layers (about 2 ms here) for it to stop looking like a Voronoi diagram. Skip `min` unless you want glitter. Fringing triples the cost of the procedural methods and is nearly free with the baked ones at this size, so it belongs on the baked path.

## Pitfalls

- **Mean 1 is not "looks medium bright".** A sharp net has most pixels below 1 and filaments far above. With the mean preserved, `intensity` (the exposure) is the one knob that makes it look right; start near 0.5.
- **A pale tint cannot lighten a light page.** Screen-style light over white is white. The `light page` preset darkens instead, with a saturated blue at low opacity. The overlay is ordinary premultiplied alpha with `alpha = max(light) * opacity`, so on a dark page it adds light and on a light page it tints.
- **The sine construction repeats at its own period.** With one scale for every round the whole function repeats every half cell and a regular lattice shows. The per-round scales `(f + j) / f` and the low-frequency bend are what break it; take them out and you get fabric.
- **A Jacobian "fold" variant was tried and dropped.** Reading the intensity from the determinant of the warp's Jacobian (light piles up where the map folds) is physically neat and cheap, but it came out as broad satin sheets with a few thin lines, not filaments.
- **Product of two layers is too sparse.** Multiplying two nets leaves only the crossings. Averaging keeps the mean and gives a denser net; that is what `caustics_stack` does.
- **`min` needs a rescale that depends on blur.** The mean of the minimum of two mean-1 nets is about 0.16 at lod 0 and rises toward 1 as the mips blur. `_caustics_min_gain` is a fit (about 15%).
- **Fringing is easy to overdo.** A tenth of a cell is already a visible rainbow; the slider tops out there.
- **Drift is not periodic.** Only the four slices loop; the two baked taps drift continuously, so do not rely on `time + TAU` for the baked path.

## When to use it

- A shimmering light layer over a hero image, a card or a pool-coloured section, as an overlay (Tier A on this hardware).
- As the light source in other scenes: the pool floor and the underwater seabed in this epic read the same baked map.
- Not for a physically correct pool: the caustics are independent of the surface above them. For caustics that follow a simulated surface, project a light mesh (Epic 5, story 5.4).
- If the page cannot run WebGPU or you want zero cost, a looping WebP or MP4 of the baked map with `mix-blend-mode: screen` on a dark section, or `multiply` of a blue tint on a light one, is almost indistinguishable.

## References

- Steven Worley, A Cellular Texture Basis Function (1996), for the F2 - F1 distance.
- Evan Wallace, [WebGL Water](https://madebyevan.com/webgl-water/), for the projected-mesh caustics this overlay does not do.
- Juan Guardado and Daniel Sanchez-Crespo, [Rendering Water Caustics](https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-2-rendering-water-caustics) (GPU Gems, ch. 2), for the physical picture and the baked, animated caustic texture.
- Inigo Quilez, [Voronoi edges](https://iquilezles.org/articles/voronoilines/), on why F2 - F1 is not a true border distance.
- The Shadertoy "water turbulence" family of tileable sine-warp caustics, for the iterated-sine idea; the code here is a rewrite.
