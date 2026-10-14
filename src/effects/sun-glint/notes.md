## How it works

A low sun over open water, built from two terms that are added to the same Fresnel-weighted reflection `fresnel-sky` uses. `sun-glint.wgsl` leaves the sun out of the reflected sky (`sun_gain` 0) and adds it back analytically, so it is counted once.

1. **The lobe, `spec_sun`.** A GGX (Trowbridge-Reitz) microfacet lobe with Smith visibility and water Fresnel. GGX rather than Blinn-Phong because its long tail gives the soft glow around a hard core that a sun reflection on water has; a Phong lobe is a Gaussian and looks like a plastic dot. The sun is not a point: the lobe is widened by half the sun's angular radius and scaled by the sun's solid angle, so `spec_sun` returns the fraction of the sun disc's radiance that reaches the eye. It equals the Fresnel reflectance at a perfect mirror alignment and can never exceed 1, so multiplying by the disc radiance (30) cannot blow out.
2. **Sparkle, `spec_sparkle`.** The waves have normals down to a few centimetres, but the facets that make real water glitter are smaller than a pixel. Each cell of a world-space grid (`sparkle size`, 6 cm by default) holds a glint with probability `density`. A glint is a round patch with its own facet normal, tilted away from the wave normal by a random slope up to 0.3 that slowly circles with time. Its brightness is a Gaussian of the slope distance between the half vector and that facet normal. Because cells are tied to world positions, glints stay put when the camera moves, and because the facet normals circle, they flash on and off as the waves move without ever popping.

```wgsl
let lobe = spec_sun(normal, view, sun, params.roughness, SKY_SUN_ANGULAR_RADIUS);
let sparkle = spec_sparkle(p.xz, normal, view, sun, time, footprint, cell, density, sharpness)
            * fresnel_water(saturate(dot(view, half_vector)));
colour = base + sun_radiance * lobe + sun_radiance * sparkle * intensity;
```

**Anti-aliasing is the part that matters.** If a cell is smaller than a few pixels the glints are sampled below their size and shimmer as noise. `spec_sparkle` takes the pixel footprint on the water (`waterplane_footprint`) and never lets a cell get narrower than three pixels: it blends two power-of-two cell levels, so glints stay about pixel-sized in the distance, and the mean brightness does not depend on the level because the area fraction and facet statistics are scale-free. The test suite checks that the mean brightness holds within a factor of 1.6 across footprints and that the output has no seam where the level changes.

The `view` parameter shows the lobe alone or the sparkle alone. `speed` and `drift` move the waves and the camera, which is the quickest way to see the glints stay put.

## Cost

One pass, no textures. The sparkle is two cell evaluations, each with two integer hashes and about 30 ALU operations, on top of the wave surface the three optics demos share. The lobe is about 35 operations.

Indicative readings, Apple Silicon laptop GPU, shared with other work: the HUD showed a median of about 0.9 ms at 1662x662 (1.1 Mpx, DPR 2); headless 1920x1080 timestamp readings ranged from 0.5 to 1.4 ms across repeated runs, lowest repeated value about 0.6 ms. Not measured on an integrated PC GPU or a phone. Setting `sparkle density` to exactly 0 skips the sparkle (an early return), which is the quickest way to see what it costs.

## Pitfalls

- **The sun counted twice.** The reflected sky must not contain the sun disc when the lobe is added, or the reflection is double bright. The demo reflects the sky with `sun_gain` 0.
- **Energy.** A mirror-like lobe for a true sun is enormously bright per steradian. Treating the sun as an irradiance gave peaks in the hundreds; treating it as a disc of finite angular size gives a lobe whose peak is at most the Fresnel-weighted disc radiance. Always multiply `spec_sun` by the disc radiance, not by a free "sun intensity".
- **Glints do not twinkle where they are centred.** A facet normal circles around the wave normal at a fixed radius. If the half vector sits exactly on the wave normal the distance never changes and the glint is steady; it twinkles where the half vector is displaced by about that radius. The first twinkle test missed this and measured nothing.
- **Round dots up close.** With a camera a metre or two above the water the cells are large in pixels and the glints read as round discs in a loose grid. Raise `sparkle size` less and `density` more, or add a second, finer sparkle layer, if that is visible in your use.
- **Cell boundaries.** A glint mask that reaches the cell edge shows square cut-offs; the mask here ends at 0.42 of a cell around a centre jittered by at most 0.08, so it is zero at every edge.
- **Reflected directions below the horizon.** As in `fresnel-sky`, a steep wave normal can reflect the ray into the water; the reflection is folded upward.
- **Hashing everywhere was twice the cost.** The first version ran the cell hashes for every water pixel and the HUD read about 1.9 ms at 1.1 Mpx. Glints exist only where the half vector is within the facet spread of the wave normal, so `spec_sparkle` now returns early outside that region (0.3 plus four sigma in slope), and the same view read about 0.9 ms. Wide glitter paths (high wave strength, high sun) cost more because more of the picture is inside the region.
- **GPU time is noisy** on a shared machine: single runs differed by a factor of two. Quote minimums of repeated runs.

## When to use it

Tier A. Use the lobe on anything shiny with a sun; use the sparkle where flat specular looks dead: sea and lake heroes, glittering snow, wet sand, metallic paint flake. It is a cheap per-pixel term and works on phones in principle (not measured).

It is not a replacement for a bloom pass: the glints are HDR values well above 1, and they look best when the next story's HDR post chain (3.10) spreads them a little. For a calm mirror where the exact sun reflection matters, the plain lobe at low roughness is enough; for large specular highlights with real shadows, use ray-traced or shadow-mapped sun light.

## References

- Bruce Walter et al., [Microfacet Models for Refraction through Rough Surfaces](https://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf) (2007), GGX.
- Eric Heitz, [Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs](https://jcgt.org/published/0003/02/03/) (2014).
- Steve Hammon, PBR Diffuse Lighting for GGX+Smith Microsurfaces (GDC 2017), the visibility approximation used in `specular.wgsl`.
- `spec_sparkle` is a hashed-cell shortcut, not a glint BRDF. Search the literature on rendering glinty appearances for properly derived, more expensive versions.
