## How it works

A top-down swimming pool built entirely from the shared optics chunks, in two passes of which only one runs per frame.

1. **Floor, on resize** (`floor.wgsl`). `backdrop_sample` draws the pool tiles (`backdrop.wgsl`: aqua tiles, bevel, grout, a dark lane stripe) into an `rgba8unorm-srgb` texture the size of the canvas, with a full mip chain. This stands in for the image or photo a real site would refract. It is redrawn only when the canvas size changes.
2. **Pool, every frame** (`pool.wgsl`), per pixel:
   - **Surface.** Two scrolling taps of the baked ripple normal map (`createWaterNormalMap`, read from mip 1.6 because its finest octaves are too busy for a pool), blended with `normal_blend_whiteout`, plus the pointer: a lens bump under the cursor and rings that run out of it, built analytically.
   - **Refraction.** `refraction_offset_cheap` turns the normal and the depth into where the ray reaches the floor, and `refraction_sample` reads the floor texture there, through `refraction_safe_uv` so the frame edges never smear. A small mip blur grows with depth.
   - **Caustics.** `caustics_baked` reads the baked caustic map (the iterated-sine net, 512x512, four time slices) at the _refracted_ floor position, so the light bends with the surface. Deeper water gives larger, blurrier, fainter caustics: the lookup scale, the mip level and the sun's transmittance (`absorb_transmittance`) all depend on depth. The floor is lit by `ambient + sun * caustic`, so the average brightness stays put and the caustics only redistribute it.
   - **Water.** `absorb_water` gives the coefficients for the chosen water type and `absorb_composite` tints and fogs the lit floor for the slanted path the eye's ray takes (`absorb_path_length`).
   - **Surface highlight.** `spec_sun` (GGX) with the sun almost overhead gives the glints where the ripples tilt toward it. Then `tonemap_reinhard_white` and `linear_to_srgb`.

The pointer is ambient: `trackPointer` without capture, so touch scrolling still works on a page with this behind it. While the cursor is over the canvas the lens follows it and moving it feeds the rings; otherwise a slow wander keeps a little motion in the water.

## Cost

Indicative HUD readings on an Apple Silicon laptop (Metal), canvas 1702x828 (1.4 Mpx), DPR 2, other tabs sharing the GPU: the `pool` pass reads about 1.0 ms, which is at the measurement floor (about 1 ms for a pass that draws almost nothing), so it is "too cheap to measure here" rather than a precise figure. The floor pass runs once per resize. Texture memory is about 11.2 MB in three textures: the floor with mips (about 7.5 MB at this size), the ripple normals (about 1.4 MB) and the caustic map (about 2.7 MB). Per pixel there are about five texture reads (two ripple taps, one caustic lookup of two taps, one floor tap), no noise and no hashes, which is why it stays near the floor. What scales it: canvas pixels, and a lower mip level on the ripple normals if you want it sharper. Not measured on an integrated Intel or Mali GPU or a phone.

## Pitfalls

- **Normals that are too busy make jagged tiles.** The first version read the ripple map at its finest level and the grout lines turned to noise. Two things fixed it: reading mip 1.6 and keeping `ripple strength` modest (the whole effect is about slopes under 0.25).
- **Blur radius is in the wrong units fast.** The depth blur is a mip chosen from a radius in pixels; at 14 px it removed the grout entirely. It is now a couple of pixels.
- **Caustics must not be blown out.** Peaks of 20 times the mean over a bright aqua tile clip to white however you tone-map. The map is baked at lower sharpness than the overlay demo, and the lit floor goes through `tonemap_reinhard_white`.
- **The caustics are lit by the refracted position, not the pixel.** Using the pixel's own position would pin the light to the screen while the tiles wobble under it.
- **No real shadow or pool geometry.** There are no walls, steps or underwater objects. The depth is one number for the whole pool.
- **The floor texture is the size of the canvas.** On a 4K display at DPR 2 it is large; render it at half size if memory matters (the mip chain hides it).

## When to use it

A hero or section background for a pool, spa, swimwear or hydration site, or anywhere a calm top-down water surface should sit over a tiled or photographic floor. It is Tier A on this hardware: one cheap pass, no compute, and it responds to the pointer without capturing it. Swap `floor.wgsl` for a texture of your own image (or a photo of tiles) to refract that instead. If you need walls, objects, a moving camera or real caustics from a simulated surface, that is Epic 5 (a 3D pool scene with ray-traced refraction); if you only need the light, use the `caustics` overlay on its own.

## References

- The `caustics` write-up, for the caustic construction and its sources.
- Mark Finch, [Effective Water Simulation from Physical Models](https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-1-effective-water-simulation-physical-models) (GPU Gems, ch. 1), for the lit, refracted look of a water surface.
- Colin Barre-Brisebois and Stephen Hill, Blending in Detail (2012), for the whiteout blend of the two ripple layers.
- Beer-Lambert absorption as used in `src/wgsl/absorption.wgsl`.
