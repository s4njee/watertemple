## How it works

Two passes. `scene.wgsl` draws one of the shared backdrops (`backdrop.wgsl`: pool tiles, a test card, sand) into an `rgba8unorm-srgb` texture with a full mip chain, and only when the backdrop choice or the canvas size changes. This stands in for what a real site has: an image, a video frame or a captured page that already exists as a texture. `water.wgsl` then draws the water over it, and everything interesting lives in the shared chunk `src/wgsl/refraction.wgsl`.

For each pixel the water pass builds a surface slope from two octaves of animated noise (`fbm_3d_d`, so the slope is the analytic derivative, no extra taps), adds a lens bump and outward rings around the pointer, and turns the slope into a unit normal. The normal becomes a uv offset in one of two ways:

- `refraction_offset_physical` bends a view ray with Snell's law (`refract()` at eta = 1/IOR), lets it travel `thickness` through the water, and returns the lateral shift against the unrefracted straight-through point. It handles tilted views.
- `refraction_offset_cheap` is the small-angle form, `-slope * thickness * (1 - 1/ior)`. It is within about 5% of the physical form up to 30 degrees of tilt, and it is what most sites would ship.

Thickness is in image heights, so the aspect ratio is handled once, in `refraction_to_uv`. The offset is exactly zero for a flat surface seen head-on and grows linearly with thickness and with IOR.

Dispersion gives each colour channel its own IOR (red lowest, so red bends least). Because the offset is proportional to `1 - 1/ior`, the three channels are the green offset times `refraction_dispersion_scales`: three taps in `refraction_sample_rgb`, or `refraction_sample_spectral`, which takes up to 16 taps along the spectrum with overlapping red, green and blue weights, normalised so a flat colour stays exact and an edge blurs into a rainbow instead of three ghosts. `dispersion = 0` falls back to one tap.

Edges: every tap goes through `refraction_safe_uv`. It fades out the part of the offset that points at a nearby border (so the image never smears or wraps), then mirror-folds whatever is left, so the uv is always inside [0, 1]. Roughness is a mip hook: `refraction_blur_lod` turns a blur radius in pixels into a level for `textureSampleLevel`.

The **View** control shows the pieces. `split` puts the untouched backdrop on the left half, `normals` and `offset` show the intermediate fields. The tile lines are the proof of coherence: they stay straight lines that bend smoothly as a whole, which noise wobble would not do.

The pointer is ambient: `trackPointer` without capture, so touch still scrolls the page. A lens follows the cursor while it is over the canvas and wanders slowly when it is not; moving the cursor feeds the rings.

## Cost

Indicative, read from the HUD on an Apple Silicon laptop (Metal), 1702x724 (1.2 Mpx), DPR 2, with other tabs sharing the GPU, so treat it as a rough guide:

| Setting                 | `water` pass                                      |
| ----------------------- | ------------------------------------------------- |
| 1 tap (dispersion 0)    | about 1.3 ms                                      |
| 3 taps, physical offset | about 1.3 ms                                      |
| spectral, 6 taps        | not read separately; 8 taps measured about 2.5 ms |

The backdrop pass runs only when the backdrop or canvas size changes. A video backdrop would add an upload plus mip generation every frame, which this demo does not measure. Memory is one backdrop texture with mips (about 6 MB at this size). The HUD marks 3 taps as within the Tier A budget and 8 spectral taps as over it, so the 3-tap path is the one to call Tier A; spectral is a quality option. Nothing was measured on an integrated Intel or Mali GPU, or on a phone.

What scales the cost: pixels, and then the tap count. The noise (2 octaves with derivatives) and the physical `refract()` are a small fixed share.

## Pitfalls

- **Frame and signs.** The normal frame is the screen's (x right, y down, z toward the viewer). The ray bends toward the normal, so a surface tilted toward +x shows the backdrop from the -x side. Getting this backwards still looks like water until you compare against the split view, then the lens magnifies where it should shrink.
- **Units.** `thickness` is in image heights and the offset must be divided by the aspect ratio for x. Skip that and the distortion is stretched on wide canvases.
- **Edges.** Clamp-to-edge smears the border pixel across the frame, repeat shows the opposite edge, and clamping the offset gives a visible seam. Fading the outward-pointing part of the offset and then mirroring avoids all three, at the price of calmer water in a thin band (default 0.05 of the image) at the borders.
- **Specular.** A tight highlight on a noisy normal turns into round blobs at low exponents. The demo uses an exponent of 700 and a small gain.
- **Spectral cost.** More than three taps buys a smoother fringe but the pass cost grows with every tap. Prefer the 3-tap form unless the backdrop has hard black and white edges.
- **Sand backdrop.** The first version had small, dark, spherically shaded pebbles that read as terrazzo or bubbles. Larger, lower-contrast stones with contact shadows read as a stream bed.
- **WGSL reserved words** bite shared shaders: `patch` and `crate` are reserved, as are many other ordinary English words.

## When to use it

Water or glass distortion over an image, a video or any texture you already have: hero images, card hover, pool floors, liquid glass. It is Tier A with one to three taps. For a backdrop that is the live page you first need a way to get page pixels into a texture (story 9.1); this demo does not answer that. For a real 3D scene with objects at different depths, a screen-space offset is wrong at depth discontinuities (objects in front of the water get refracted too) and you want a depth-aware version or ray tracing instead.

## References

- Snell's law and `refract()`: the WGSL specification, built-in functions.
- Valve, [Water Flow in Portal 2](<https://advances.realtimerendering.com/s2010/Vlachos-Waterflow(SIGGRAPH2010).pdf>), SIGGRAPH 2010, for the practical refraction and tint setup in a game.
- Inigo Quilez, [Domain warping](https://iquilezles.org/articles/warp/), for the noise-with-derivatives slope.
