## How it works

A camera mirrored across y=0 renders procedural mountain ridges, tree silhouettes and the sky into a half-resolution linear `rgba16float` reflection target. The water projects its world position through that mirrored camera, distorts the sample by its ripple normal, and combines it with the lake body through Fresnel weighting. Gentle Gerstner ripples keep the reflection nearly still; sun glints and horizon fog complete the lighting.

Morning mirror and wind ripples presets compare a near-flat reflector with light surface detail. The reflection target is separate from the main scene and rebuilt on resize. Its pass binds a dummy reflection source so it cannot read its own render attachment.

## Cost

One half-resolution environment pass and one main scene pass. The reflection target uses about 3.96 MiB at a 1920×1080 output, alongside the depth buffer and mesh. Default GPU time and built bundle cost are recorded in the ocean study. Timestamp queries are optional; mobile is untested.

## Pitfalls

The reflection describes a flat reference plane; strong waves make the projection less accurate. Reflections at the target edge are clamped, and there is no oblique near-plane clipping because the reflected scene consists of procedural distant scenery. The mountains and trees are directional silhouettes, not imported meshes. Add clipping and actual scene geometry for nearby buildings or moving objects.

## When to use it

Use a calm landscape hero, a reflective environment backdrop or a low-wind planar-reflection study. Use the ocean renderer's procedural sky reflection when reflecting local scenery is unnecessary.

## References

- Mark Finch, [Effective Water Simulation from Physical Models](https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-1-effective-water-simulation-physical-models), GPU Gems, 2004.
- Arul Asirvatham and Hugues Hoppe, [Terrain Rendering Using GPU-Based Geometry Clipmaps](https://developer.nvidia.com/gpugems/gpugems2/part-i-geometric-complexity/chapter-2-terrain-rendering-using-gpu-based-geometry), GPU Gems 2, 2005. The study uses a simplified concentric ring mesh.
