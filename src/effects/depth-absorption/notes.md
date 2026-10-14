## How it works

Water absorbs light exponentially with the distance the light travels through it, and absorbs the colours unevenly: red is gone within a few metres, green lasts tens of metres, blue longest. That is Beer-Lambert: `T = exp(-sigma * path)` per colour channel. `src/wgsl/absorption.wgsl` turns it into functions that take the path length as a parameter, so this demo feeds them an analytic depth and a later 3D scene can feed them depth-buffer differences.

Designer-friendly input: `absorb_coefficients_from_color(color, depth)` returns the `sigma` for which white light has become `color` after `depth` metres, so "clear tropical water turns (0.06, 0.5, 0.72) at 6 m" is the whole specification. The seven named water types (`absorb_water`) are defined that way, each with a scatter colour and a turbidity.

Fog: `absorb_composite(bed, sigma, turbidity, scatter, path)` is `bed * T + scatter * (turbidity / extinction) * (1 - T)` with `extinction = sigma + turbidity`. At path 0 it returns the bed exactly; at infinite path it returns `absorb_deep_color`. Because the in-scattered term is weighted by the same extinction, red water light dies first and the deep colour comes out teal or blue by itself, without a separate gradient.

Slant paths: a view ray crosses more water than the vertical depth. `absorb_path_length` divides by the cosine of the refracted angle (Snell), and `absorb_path_length_two_way` adds the sun's descent, because the light that reached the bed was absorbed on the way down too.

Two passes. `bed.wgsl` runs once per resize or bed choice into an `rgba16float` target: rgb is the dry bed colour (one of the shared backdrops plus shaded rocks and a chest), alpha is the normalised depth of a sloping shelf with a drop-off and submerged objects (domes and a box that rise by a fraction of the local depth). `water.wgsl` then runs every frame: it looks up the bed at a position shifted by the tilt (two fixed-point steps, a cheap parallax so submerged objects shift with depth), applies the sun and view paths, adds a Schlick sky reflection and animated caustics that wash out with depth, and draws a depth ramp (what a white tile looks like at each depth) along the bottom. The **Debug view** shows raw depth with contour lines each tenth of the depth range, or the pure transmittance with no bed.

## Cost

Indicative HUD reading on an Apple Silicon laptop, 1702x724 (1.2 Mpx), DPR 2, shared GPU: about 0.8 ms for the `water` pass; the bake pass is not repeated. One `rgba16float` texture the size of the canvas (about 10 MB at this size). Per pixel: 3 texture taps, one gnoise, about 10 `exp` and 3 `log` for the water type, so the water type and the debug view cost the same. Not measured on an integrated Intel or Mali GPU or a phone.

## Pitfalls

- **Units are the whole game.** Coefficients are per metre; a depth in uv or pixels gives either no tint or black. `depthScale` is in metres on purpose.
- **Brightness of the scatter colour.** The deep colour is `scatter * turbidity / extinction`, and it is easy to land on a washed-out slate blue. Tune the scatter colour's ratio between green and blue, not only its brightness: the first tropical colour came out grey.
- **Tea lake.** Absorption eats blue first there, so the usual blue-green shift reverses; a depth ramp that goes brown is correct.
- **Parallax is a visual aid.** The shift is `tan(refracted angle) * normalised depth * 0.12` of the frame, not a real projection. Steep angles stretch the shore.
- **Derivatives in branches.** The contour lines and ramp ticks use `fwidth`; WGSL only allows it in uniform control flow, so the derivatives are taken at the top of the shader before any per-pixel branch.
- **Reserved words** again: `crate` is reserved in WGSL.

## When to use it

Anywhere water should get deeper in colour as it gets deeper: pool floors, lakes, shorelines, a basin or a river bed. It is Tier A as a texture lookup plus a few `exp`. Use `absorb_composite` for a tint and fog; it is single-scatter and uniformly lit, so it does not do sun shafts or colour of light that changes with depth, and it cannot show the surface-to-bed scattering of milky water in detail. For real volumetrics you need a ray-marched or froxel approach.

## References

- Beer-Lambert law; Jerlov water types for realistic absorption ranges.
- Jerry Tessendorf, [Simulating Ocean Water](https://people.computing.clemson.edu/~jtessen/reports/papers_files/coursenotes2004.pdf), for the practical absorption and scattering model.
- Secchi depth and the rule of thumb `extinction = 1.7 / Secchi depth` (Poole and Atkins, 1929), used in `absorb_turbidity_from_visibility`.
- Mark Finch, [Effective Water Simulation from Physical Models](https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-1-effective-water-simulation-physical-models), GPU Gems ch. 1.
