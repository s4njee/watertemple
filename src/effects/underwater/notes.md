## How it works

One fullscreen pass (`scene.wgsl`), no offscreen target, built from two new shared chunks and one baked map.

- **Wobble.** The whole picture is looked at through a slow, low-frequency displacement (a few sines, about 1% of the frame at the default setting). It is applied to the coordinates every layer uses, so shafts, seabed and particles move together; the scene never shears apart.
- **Depth fog.** A vertical gradient from lit, bright water at the top to the deep colour: `1 - exp(-y * density)`. The deep colour is a parameter; the lit colour is derived from it, so one colour picker moves the whole palette. A faint sky lightening sits at the very top.
- **Light shafts** (`src/wgsl/shafts.wgsl`). An analytic construction in screen space, not a ray march. `shafts_fan` gives each pixel the angle of the ray from a virtual light source above the frame (constant along each ray); `shafts_beams` is two octaves of animated gradient noise along that angle, so bands fan out from the source and drift sideways; `shafts_falloff` fades them with the distance travelled below the surface and ripples them slowly along each ray. The source's height sets how parallel they are, its horizontal position sets the angle. Two noise evaluations, a `sqrt`, an `exp` and a `sin` per pixel.
- **Marine snow** (`src/wgsl/motes.wgsl`). A grid of cells with at most one particle each; each particle is confined to its cell so one lookup is enough (one hash per layer, not nine). Two layers: a far one (small, slow, dim) and a near one (bigger, faster, brighter), which gives parallax for free. Each mote sways and twinkles on its own phase.
- **Seabed** (optional). Sand under a wavy horizon, lit by two taps of the baked caustic map (the same chunk as the `caustics` overlay) and hazed toward the deep colour with distance. The map is read only below the horizon, so with `seabed = 0` it costs nothing.

The scene is calm by design: low contrast between bands, sparse small particles, no pointer input, and a speed that defaults to a slow drift. Put text over it with a modest overlay or text shadow as you would over any photo.

## Cost

Indicative HUD readings on an Apple Silicon laptop (Metal), canvas 1702x828 (1.4 Mpx), DPR 2, other tabs sharing the GPU: about 1.0 ms for the single `underwater` pass with the shallow reef preset (seabed on), which is at the measurement floor (about 1 ms for a pass that draws almost nothing), so treat it as "too cheap to measure here". Texture memory is 683 KB: only the 256x256 caustic map, baked once. Per pixel: 2 noise evaluations for the shafts, 2 hashes for the particles, a handful of sines for the wobble, and 2 texture taps in the seabed strip. Not measured on an integrated Intel or Mali GPU or on a phone.

## Pitfalls

- **Beams that are too crisp fight the text.** `shaft contrast` above about 0.7 gives hard-edged bands with dark water between them; the demo fixes it at 0.55 and lets the intensity slider do the rest.
- **A source inside the frame is singular.** The fan coordinate blows up at the source, so it must sit above the top edge. The chunk clamps the denominator, and the demo keeps the source at least 0.12 frame heights above it.
- **One cell per mote.** A particle is kept inside its own cell so one lookup is enough, which means motes cannot overlap a cell border and the grid can show at very high density; the demo keeps the density modest.
- **Wobble moves the horizon of the seabed.** Because the same displacement is applied to the seabed height it undulates a little.
- **Colour picked by eye is not linear.** The deep colour parameter is linear RGB; the picker converts, but a mid-blue in sRGB becomes quite a dark linear value. The presets use values near (0, 0.03 to 0.2, 0.2).
- **Long-running pages.** The particles' time is a plain accumulated float; after many hours drift precision degrades, so wrap the clock if a page stays open all day.

## When to use it

A calm full-bleed hero or section background for ocean, diving, aquarium, wellness or water-technology pages, behind headings and body text. Tier A on this hardware: one cheap pass, no pointer, no compute. If you need real volumetric lighting (shadows from objects, a moving camera), that needs a depth buffer and a march; if you only need a gradient and a few particles, CSS will do. For a top-down view of water use `pool-floor`; for the caustics alone use the `caustics` overlay.

## References

- Kenny Mitchell, [Volumetric Light Scattering as a Post-Process](https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-13-volumetric-light-scattering-post-process) (GPU Gems 3, ch. 13), the standard screen-space light-shaft technique; the shafts here replace its radial blur with noise along the ray angle.
- Beer-Lambert absorption as in `src/wgsl/absorption.wgsl`; the fog here is its single-colour cheap form.
- Hash-based cell particles as popularised in procedural shaders; the hash here is PCG (Jarzynski and Olano, 2020) from `src/wgsl/hash.wgsl`.
