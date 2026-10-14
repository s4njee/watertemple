## How it works

A two-sided Gerstner surface is drawn with above-water sky and an underwater floor. A screen pass samples wave height two meters along each camera ray and compares it with that ray's world height. A derivative-scaled transition avoids a hard aliasing line; a narrow darkened meniscus marks the interface. Pixels on the submerged side receive blue-green attenuation and scattering tint.

Camera height crosses zero continuously through the control or the above water, half submerged and underwater presets. Near the surface the projected lattice uses a minimum reference distance so a camera exactly at y=0 cannot collapse the whole mesh into one point. Shader code and the numerical wave mirror are shared with the ocean study.

## Cost

One scene render plus one full-resolution grading pass. The intermediate uses the supplied output format, so grading is applied to encoded display color. The default 1080p cost is measured on the M1 Max; mobile and browser-specific thermal behavior remain unmeasured.

## Pitfalls

The two-meter wave test is a near-camera meniscus approximation; it does not trace the nearest refractive surface. The grading operates in display space for this illustrative transition. This does not resolve a physical Snell window, total internal reflection, multiple interfaces or underwater volumetric shadows. Extreme wave heights and camera pitch can reveal the projected grid's finite coverage.

## When to use it

Use an immersive above/below transition or a waterline composition study. For physically accurate underwater optics, add ray intersections, dielectric refraction and a linear HDR compositing path.

## References

- Mark Finch, [Effective Water Simulation from Physical Models](https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-1-effective-water-simulation-physical-models), GPU Gems, 2004.
- Jerry Tessendorf, [Simulating Ocean Water](https://evasion.inrialpes.fr/Membres/Fabrice.Neyret/NaturalScenes/fluids/water/waves/fluids-nuages/waves/Jonathan/articlesCG/simulating-ocean-water-01.pdf), course notes, 1999–2004.
