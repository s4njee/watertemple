## How it works

One fullscreen fragment pass shows open water under a procedural sky. For each pixel `fresnel-sky.wgsl` builds a camera ray (`waterplane_ray_dir`), intersects it with the plane y = 0 (`waterplane_hit`), reads a wave normal at the hit point (`waterplane_wave`), and then does the three things this effect exists to prove:

1. **Reflect.** The ray is mirrored about the wave normal (`waterplane_reflect`) and the result is looked up in the sky.
2. **Weight by Fresnel.** `fresnel_schlick(dot(normal, view), f0)` says how much of that reflection the surface really returns: about 2% looking straight down (F0 from the index of refraction, 0.0204 for water at 1.333), rising to 100% at the horizon. The rest of the light goes into the water, which is approximated by a flat deep-water `body` colour lit by the sky's ambient colour.
3. **Tonemap.** The sky is HDR, with the sun at 30 times diffuse white, so the sum goes through `tonemap_aces`, then `linear_to_srgb`, then a one-LSB triangular dither (`sky_dither`) so the smooth sky gradient does not band in 8 bits.

```wgsl
let fresnel = fresnel_schlick(dot(normal, view), params.f0);
colour = mix(body, reflected, fresnel);
```

**Where the sky comes from** is the `source` parameter, so the two ways to get a reflection can be compared on identical waves. `procedural` evaluates `sky_radiance_ex` per pixel. `cubemap` renders the same function once into a 256-pixel `rgba16float` cube texture with `src/gpu/cubemap.ts` and samples it with `textureSampleLevel`. The cubemap is redrawn only when the sun or haze changes, not every frame. The `view` parameter shows the Fresnel term alone (grey, scaled by 2), the reflection alone, or the body colour alone.

Distant water stays clean because `waterplane.wgsl` fades each wave out as its wavelength approaches a few pixels (`waterplane_footprint`). Far away the surface is flat, F is about 1, and it reflects the horizon colour: the seam with the real horizon disappears without any fog.

## Cost

One pass, no textures sampled in procedural mode; one cube texture sample in cubemap mode. Per pixel: five swell waves (a `sin`, `cos` and `exp` each), two 2D noise ripples, one sky evaluation, a few smoothsteps for the body lighting.

Indicative readings, Apple Silicon laptop GPU, shared with other work, so treat them as rough: the HUD showed about 1.0 ms at 1662x662 (1.1 Mpx, DPR 2). Timestamp readings from a headless 1920x1080 run ranged from 0.5 to 1.3 ms; the lowest repeated value was about 0.5 ms. Not measured on an integrated PC GPU or a phone. Cost scales with pixel count; waves that have faded out at distance are skipped, so a lower camera is cheaper than a high one.

The cubemap pass is six 256x256 faces; the HUD showed about 0.7 ms for the first render (includes warm-up) and it then does not run until the sun moves. Texture memory: 256x256x6 at 8 bytes per texel, about 3 MB.

## Pitfalls

- **Schlick is not exact at grazing angles.** Against the exact Fresnel equations it reads 0.58 where the exact value is 0.53 at 84 degrees, a worst-case error of 0.06; it is within 0.01 up to 70 degrees. On a moving wavy surface nobody can tell. `tests/optics-surface.test.ts` states and checks both bounds.
- **A wave normal can face away from the eye.** `dot(normal, view)` goes negative on steep back faces, and the reflected ray can point into the water. Fresnel clamps the cosine (so the pixel reads as a grazing mirror), and `waterplane_reflect` folds a downward reflection upward (`y = |y|`). Without both, dark specks appear on wave backs.
- **The cubemap cannot hold the sun.** The sun disc is 0.0093 rad in radius, smaller than a texel at 256 pixels per face, so in `cubemap` mode the sun is a soft blob with a lower peak. Use a cubemap for the smooth sky and add the sun analytically (see `sun-glint`).
- **Cubemap mips are plain per-face downsamples.** `generateMipmaps` does not filter across face edges, so blurry lookups of a mipped cube show faint seams. This demo samples mip 0 only.
- **8-bit banding.** A clear-sky gradient shows contour bands once it reaches an 8-bit canvas; the dither removes them at the price of one hash per pixel.
- **ACES desaturates.** The first sunset was a grey-lilac sky with a white sun. The twilight horizon colour and the sunset glow in `sky.wgsl` were pushed more orange to survive the tonemapper. Check any new sky colour after tonemapping, not before.

## When to use it

Tier A. It is the reflection model for any water scene that has a sky and no geometry to reflect: a lake or sea background, a pool in daylight, the base layer of an ocean. It is cheap enough for a hero or backdrop, including on phones (not measured on one yet).

Reach for something else when the surface must reflect real objects: that needs planar reflections or screen-space reflections (story 3.9). When the sky is a photograph rather than a function, bake it into a cubemap through the same helper and keep everything else.

## References

- Christophe Schlick, [An Inexpensive BRDF Model for Physically-based Rendering](https://onlinelibrary.wiley.com/doi/10.1111/1467-8659.1330233) (1994).
- Sebastien Lagarde, [Image-based lighting approaches and parallax-corrected cubemap](https://seblagarde.wordpress.com/2012/09/29/image-based-lighting-approaches-and-parallax-corrected-cubemap/) (2012), the roughness-aware Fresnel used in `fresnel_schlick_roughness`.
- Krzysztof Narkowicz, [ACES filmic tone mapping curve](https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/).
- The WebGPU and Direct3D cube map face layout is documented in `src/gpu/cubemap.ts`.
