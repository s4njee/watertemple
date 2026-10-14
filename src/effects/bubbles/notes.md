## How it works

Rising particles integrate a sinusoidal horizontal wobble analytically. Instanced quads reconstruct sphere normals, use the existing scene texture for screen-space refraction, add a Fresnel-like rim and a compact highlight. Underwater Air uses an underwater scene; Soap Film changes the backdrop and evaluates angle- and thickness-dependent interference at representative RGB wavelengths (650, 510 and 475 nm). Changing film thickness changes colour rather than merely rotating a rainbow palette.

## Cost

The live count is bounded at 140 bubbles. The same 96 KiB particle store and cached mipmapped backdrop are reused. Default underwater and 140-soap-bubble configurations are measured in [the cost study](../../docs/weather-cost-study.md). Pixel overdraw grows with bubble radius and overlap.

Local Apple M1 Max (Dawn Metal), 1920×1080: conservative complete-frame cost **0.328 ms**, using the larger of median span and workload slope over the measured configurations. Owned effect code is **9.85 kB gzipped**. Shared runtime and WGSL imports bring the transitive load to about 104 kB; standalone packaging remains Epic 12.1. Phones and other GPU classes remain unmeasured.

## Pitfalls

This is a sphere-impostor lens and a three-wavelength thin-film approximation, not full spectral multiple-reflection optics or bubble-fluid coupling. Bubbles do not collide or merge; overlaps use depth-ordered alpha compositing. The transparent centre samples the authored backdrop rather than arbitrary live DOM.

## When to use it

Tier A submerged atmosphere and soap-bubble decoration. Try Underwater Air for a calm hero, or Soap Film for an iridescent card backdrop. Avoid placing fine text directly behind strong refraction.

## References

- [Belcour and Barla, A Practical Extension to Microfacet Theory for the Modeling of Varying Iridescence (2017)](https://belcour.github.io/blog/research/publication/2017/05/01/brdf-thin-film.html) — reference for more complete thin-film optics; this demo uses a simpler phase approximation.

- [WebGPU specification: render passes and premultiplied compositing](https://www.w3.org/TR/webgpu/).
