## How it works

Two parts: a scene to look at, and a pane of glass with drops on it. Neither keeps state between frames.

**The scene** (`scene.wgsl`) is a procedural night street, dusk or grey day: skylines with lit windows, neon signs, and four layers of hashed point lights, drawn in HDR (`rgba16float`) at no more than half the canvas size and at most 1024 texels on the long edge. It is drawn once, then its mip chain is built (the sharp picture, used inside drops and along trails) and it is shrunk twice with the 13-tap downsample and blurred with a small Gaussian into an eighth-size copy (the fogged picture). All of that is redone only when the scene, the fog slider (it sets the blur radius) or the canvas size changes. The lights are far brighter than 1, so the blur leaves glowing blobs instead of grey haze.

**The glass** (`glass.wgsl`) runs every frame and reads `droplet_field` from the shared chunk `src/wgsl/droplets.wgsl`. Per pixel it gets back a drop coverage, the position inside the drop, the drop's radius and a "wiped" amount, and then:

- outside any drop the pixel is one bilinear tap of the blurred copy, desaturated a little with a milky veil;
- along the trail of a running drop the glass is wiped: the sharp picture shows through, more strongly near the head and tapering away behind it, so the trail re-fogs from the old end;
- inside a drop the sharp picture is sampled at `uv - local * radius * strength`, which flips it, as a real drop does, and shows a different slice of the scene in each drop. Drops also get a darker rim, a bright crescent along the lower edge (light collected from behind) and a glint.

The drop field is the point of the effect. The glass is cut into columns, each shifted up or down by its own hashed amount so cell borders never line up in rows. Each cell holds one run per cycle (6 to 14 seconds): a drop forms at the top, runs down a path that wobbles sideways as a function of height only, stops and starts (progress is a line minus two sines whose amplitudes sum to under 1, so it never reverses), sheds beads that shrink with age, and shrinks away at the bottom. Because the path is a function of height, the trail can be rebuilt from the head's position alone: that is how a stateless field can wipe the fog. Radius, beads and wipe all start and end at zero, so nothing pops at a cycle boundary (`tests/rain.test.ts` runs 400 points over 83 seconds and bounds the largest one-step change in coverage). Static beads and an optional mist of tiny ones are separate, cheaper layers.

Layers per pixel, at the default quality 3: two running-drop layers (a big one and a 0.62x one), one layer of static beads and one of fine mist. Each is a single cell lookup (two integer hashes and about a hundred ALU operations), so no loops. Quality 1 drops the second runner layer and the mist. Texture taps: one on pixels in plain fog, two on drops and trails.

## Cost

Indicative HUD readings on an Apple Silicon laptop, 1702x662 canvas (1.1 Mpx, DPR 2), other agents' tabs sharing the GPU: about 1.1 to 1.2 ms for the default preset and about 1.5 ms for `downpour`. That is the floor for this machine (a near-empty pass reads about 1.2 ms), so the effect is "too cheap to measure here"; the quoted figure for the heaviest preset is an upper estimate. The scene and blur chain run only on change and are not in the steady-state figure. Memory: about 3.8 MB at this size (the HDR scene with mips plus three small targets). Not measured on an integrated Intel or Mali GPU or on a phone. Cost scales with canvas pixels and with the number of drop layers; it does not scale with how many drops are showing, because every pixel looks up its cells whether or not they hold a drop. Tier A on this hardware; mobile is untested, and the scorecard says so. If it ever proves too heavy, cut quality to 1 or render the glass into a half-size canvas.

## Pitfalls

- **Drops that pop.** The first version let a drop's centre pixel keep 0.5 coverage as its radius went to zero, and picked between a drop and a bead by distance instead of coverage, so a speck beat a bead and a drop vanished in a single step. Coverage now fades with the radius when a drop is under half a pixel, and the larger coverage wins. The time-continuity test found both.
- **Dark drops.** With a dark scene, small drops showed almost black: they only see the few pixels around them. A drop has to show a big slice of the world, so refraction defaults to 3 (the sampled slice is four times the drop's size) and the scene has many lights and lit windows.
- **Trails that look like bars.** A wipe as wide as the drop, with a hard start, reads as a rectangle. The wipe is now narrower than the drop, tapers away from the head, and fades with age.
- **Visible grid.** One cell size gives rows of drops. Staggering each column, two cell sizes and per-cell periods and phases hide it; at very low quality (one layer) the columns can still be seen at a glance.
- **Banding and blockiness.** The scene is drawn at half size and read through bilinear taps, so windows in a wiped trail look a little blocky on a large screen. Raise the size of the scene target if that matters.
- **Resolution units.** Cell sizes are in image heights, so on a tall phone screen drops are large relative to the width. Scale `dropSize` down there.
- **Colour.** The canvas has to be sRGB-encoded by the shader; the offscreen HDR targets stay linear.

## When to use it

A mood background behind text or a login form, a window overlay on a photo or video you already have (swap the procedural scene for that texture and keep the blur chain), or a weather or music app. It is a good fit for Tier A on desktop and likely on recent phones, but nothing here has been timed on one. If you need a fogged window the user can wipe with a finger, that is story 10.5 and needs per-frame state (a wipe texture); this effect only wipes where a drop has run. If you need rain falling outside rather than on the glass, see story 10.1. For the lowest cost, play a short looped video of the same scene.

## References

- Martijn Steinrucken ("BigWings"), [Rainy window shader tutorial and Shadertoy "Heartfelt"](https://www.shadertoy.com/view/ltffzl): the well-known drop-grid-with-trails idea this builds on; this implementation is written from scratch and differs in structure (columns, stop-go curve, wipe from the path).
- Jorge Jimenez et al., "Next Generation Post Processing in Call of Duty: Advanced Warfare" (SIGGRAPH 2014): the 13-tap downsample used for the fog.
- Inigo Quilez, [Hash functions and Voronoi articles](https://iquilezles.org/articles/): background for the hashed-cell approach.
- Mark Jarzynski and Marc Olano, "Hash Functions for GPU Rendering" (JCGT 2020): the integer hash behind every random choice here.
