## How it works

A fragment shader checks a 3×3 neighbourhood of hashed rain-event cells. Each cell produces a new jittered impact position per cycle, then an expanding radial wave with a fading envelope. Screen derivatives reconstruct the height gradient, which offsets the procedural pool tiles and changes the highlight. Every ring expands and fades; no history texture or compute pass is used.

## Cost

One fullscreen optics pass after the cached backdrop bake. The main cost is nine event candidates per pixel, independent of the apparent number of rings. The 1080p default run is in [the weather cost study](../../docs/weather-cost-study.md).

Local Apple M1 Max (Dawn Metal), 1920×1080: conservative complete-frame cost **0.786 ms**, using the larger of median span and workload slope over the measured configurations. Owned effect code is **9.86 kB gzipped**. Shared runtime and WGSL imports bring the transitive load to about 104 kB; standalone packaging remains Epic 12.1. Phones and other GPU classes remain unmeasured.

## Pitfalls

The waves are independent visual rings: there is no wave-equation interaction or reflection from geometry. Derivative normals need enough pixels per ring to avoid shimmer, and the finite 3×3 neighbourhood bounds ring radius. Density zero removes disturbances while retaining the water surface.

## When to use it

Tier A rain on a tiled pool, wet card or quiet backdrop. Quiet Rain slows sparse rings; Silver Rings increases density and highlight strength.

## References

- [Heightfield ripples in this project](?effect=ripples) — use the simulated solver when waves should interact with boundaries.

- [WebGPU specification: render passes and premultiplied compositing](https://www.w3.org/TR/webgpu/).
