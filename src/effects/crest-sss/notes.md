## How it works

Big swell, with the camera looking toward a low sun. The base image is the same Fresnel-weighted sky reflection plus a small sun lobe as the other optics demos; the new term is `sss_crest_color`, added on top.

The idea comes from the translucency model of Barre-Brisebois and Bouchard (GDC 2011). Light that enters the back of a thin wave leaves the front, bent upward toward the eye by the surface. The shader takes the direction the light travels, `-light`, bends it outward along the wave normal by `distortion`, and measures how directly that points at the viewer, raised to `power`:

```wgsl
let leaving = normal * clamp(distortion, 0.0, 1.0) - light;
let through = dot(view, leaving) / length(leaving);
let forward = pow(saturate(through), power);
```

That alone would light every wave face that looks toward the sun, so two weights make it read as a crest effect. A **thickness** weight uses the crest factor `waterplane_crest(height, strength, scale)`: 0 at the mean level, 1 near the tops of the tallest waves, so thin tall water glows and troughs do not. A **facing** weight favours normals turned away from the sun. Finally the result is multiplied by `(1 - fresnel)`, since light that scatters out of the water competes with the surface reflection, and by the sun's colour, which turns orange as the sun sinks.

The big waves also use `waterplane_hit_waves`, one fixed-point step that moves the ray hit onto the displaced surface before the waves are read, which matters when the water is a few metres high and the camera is at 5 m.

To see what the term does, turn `subsurface scattering` off (the `sss off` preset), or set `view` to `sss only` to see the term by itself, or `crest mask` to see the thickness weight on its own. `scatter colour`, `SSS strength`, `SSS distortion` and `SSS power` shape it.

## Cost

One pass, no textures. The SSS term itself is one normalize and one `pow`; the rest is the same wave surface and sky as the other demos, and `waterplane_hit_waves` adds the five swell waves a second time.

Indicative readings, Apple Silicon laptop GPU, shared with other work: the HUD showed 0.85 to 1.1 ms at 1662x662 (1.1 Mpx, DPR 2); headless 1920x1080 timestamp readings between 0.5 and 1.4 ms across repeated runs, lowest repeated value about 0.5 ms. Not measured on an integrated PC GPU or a phone.

## Pitfalls

- **The sign of the normal bend.** The textbook form steers the light by adding the normal, which for an eye above a water surface cancels the effect exactly where it should appear: a ray travelling down from a low sun never points up at the eye. Bending it outward, `normal * d - light`, lifts it into view. The distortion is clamped to 1 because above 1 a viewer on the sunny side could see the term; a test checks it is exactly 0 there.
- **The crest mask was nearly empty at first.** Normalising wave height by the largest possible height (all five waves lined up) left nearly everything below the threshold, because the surface's standard deviation is about a third of that. The mapping in `waterplane_crest` now saturates at 40% of the amplitude, so about one point in five reads as crest.
- **It is a height cue, not thickness.** The term does not know about breaking waves or foam; it lights any tall steep face that looks toward the sun, and a swell with a heightfield wave function never folds over. For breaking waves, drive `crest` from a Jacobian or foam mask (Epic 8) instead.
- **View dependent by design.** With the sun behind the camera the term is exactly 0. A flat low-angle picture of the sun side of a wave looks plain, as it does in life.
- **Tonemapping.** The scatter colour is added in linear HDR before ACES, which desaturates bright greens toward cyan-white. Pick the scatter colour looking at the final picture, not at the colour swatch.
- **Speed.** Swell with the default `wave size` of 2.2 travels at about 9 m/s (deep-water dispersion), which looks fast from 5 m up; `speed` slows it without changing the shape.

## When to use it

Tier A. It is a drop-in glow term for ocean crests, backlit waterfalls and sheets of water, and anything translucent where you have a height or thickness estimate and a light direction. It needs a low or medium sun in front of the camera to show up at all, so it suits golden-hour scenes and moving cameras that sweep across the sun.

Where real transmittance matters (a swimmer, a glass of water, thick ice) use a proper depth-based absorption term (story 3.3) or a thickness texture instead.

## References

- Colin Barre-Brisebois and Marc Bouchard, [Approximating Translucency for a Fast, Cheap and Convincing Subsurface Scattering Look](https://colinbarrebrisebois.com/2011/03/07/gdc-2011-approximating-translucency-for-a-fast-cheap-and-convincing-subsurface-scattering-look/) (GDC 2011).
- The height-weighted variant for oceans appears in open-source ocean renderers such as [Crest](https://github.com/wave-harmonic/crest).
- Krzysztof Narkowicz, [ACES filmic tone mapping curve](https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/).
