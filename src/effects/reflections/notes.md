## How it works

A calm square pool in a small 3D scene: a red pillar and a yellow box that cross the waterline, a blue sphere floating half in the water, a tall thin green pole far away, an orange lintel on two posts that hangs over the water (its underside is the interesting part), and a few objects on the banks. The water reflects all of it, weighted by Fresnel, with a light animated normal wobble. The `strategy` parameter chooses where the reflection comes from; `split` shows planar on the left of the divider and SSR on the right.

Three passes per frame, each timed separately in the HUD:

1. **`reflection`** (planar and split only). The scene is drawn again with the camera mirrored in the plane `y = 0` (`mirrorCamera` in `src/gpu/scene/camera.ts`) into an `rgba16float` target at 1, 1/2 or 1/4 of the canvas size. The sky is drawn first, then the objects.
2. **`scene`**. The normal forward pass (`src/gpu/scene/forward.ts`) into `rgba16float` colour plus a `depth32float` buffer. There is no water mesh in it.
3. **`water`**. One fullscreen pass (`water.wgsl`). For every pixel it rebuilds the camera ray from the camera uniform, intersects it with the plane analytically, and calls the pixel water if that happens in front of the depth buffer. The scene depth behind the water gives the depth of water along the ray, which drives a Beer-Lambert tint of what is seen through the surface. Fresnel (Schlick, F0 = 0.02) blends that with the reflection. Tone mapping and the sRGB encode happen here.

The reflection comes from one of:

- **planar**: the pixel's world position is projected with the mirrored camera and the reflection image is sampled there. The mirrored camera sees points on the plane at exactly the uv the real camera does, so no flip or offset is needed. The ripple normal then pushes the lookup by `normal.xz * distortion * 0.25` in uv. This is a hack: it moves the picture rather than bending a ray.
- **ssr**: `ssr_trace` in `src/wgsl/ssr.wgsl` marches the main depth buffer along `reflect(view, normal)`. The ray's start and end are projected to the screen once; 1/z is affine along a projected line, so each step costs one lerp, one reciprocal and one `textureLoad`, with no matrix multiply. A step that ends up behind the depth buffer by less than `thickness` plus the depth the ray covers in one step is a hit, refined by 5 bisections. The hit's colour comes from the `scene` target; on a miss the sky model is used. Confidence fades over the last 10% of the screen and the last 40% of `max distance`. Per-pixel jitter (interleaved gradient noise) shifts the samples by up to 3/4 of a step.
- **sky**: `rf_sky(reflect(view, normal))` only. The baseline.

**Clipping below the plane.** The mirrored pass must not draw the pool floor or the submerged part of anything, or the floor, which sits between the mirrored eye and the surface, would cover the reflection. I use an oblique near plane (Lengyel): row 2 of the projection is replaced so that the near plane is the water plane. It costs nothing per fragment, keeps early-z, and clips triangles exactly at the plane. Its costs: the depth range is skewed (so the mirrored pass cannot use `camera_linear_depth`; it does not need to), and it is only valid when the mirrored eye is below the plane, so the orbit control's minimum pitch keeps the eye above the water. The alternative is `discard` in the fragment shader against a plane (a cost on every fragment of every object, and no early-z for the clipped part), or the `clip-distances` feature, which would clip in the vertex stage with a normal projection. `clip-distances` is optional: Chrome 131+, Safari 27, but absent on Mali and on about half of Android reports (docs/support-matrix.md section 2), so it cannot be the only path. The oblique matrix is unit-tested to put points on the plane at depth 0 and points below it at negative depth.

Reflecting flips handedness, so the mirrored pass draws with clockwise front faces. A test renders a box above the plane, plus a slab below it, through the mirrored camera and checks that the box lands where the real camera would see its reflection and that no slab pixel appears.

The scene plumbing (camera, orbit control, meshes, forward pass) lives in `src/gpu/scene/` and is meant to be reused by the pool, ocean and particle stories. Conventions are fixed in the header of `camera.ts` and `camera.wgsl`: right-handed, +Y up, view looks down -Z, depth 0..1 (not reversed), uv origin top-left.

## Cost

**Read this first: the table's "GPU total" column is too high.** It was made by adding up the per-pass times, and on this GPU the pass intervals overlap, so their sum overstates the frame. Measured afterwards as the time from the first pass starting to the last one ending, on the same machine with nothing else using the GPU (canvas 1502 x 636, 0.96 Mpx, DPR 2, HUD averages):

| Strategy                    | GPU time per frame |
| --------------------------- | ------------------ |
| sky only                    | 0.9 ms             |
| SSR, 48 steps               | 1.2 ms             |
| planar, full resolution     | 1.1 ms             |
| planar, 1/2 resolution      | 1.2 ms             |
| split (planar 1/2 + SSR 48) | 1.3 ms             |

A pass that draws almost nothing reads about 1.2 ms on this machine, so every row is at or near that floor: **this scene is too light to separate the strategies by GPU time on Apple Silicon.** The differences below are real in kind (an extra scene render, a march per water pixel) but their size needs the Epic 13 benchmark on slower hardware. The per-pass columns of the original table are kept because they still show which pass grows when a setting changes.

Measured with the HUD on Apple Silicon (adapter `apple` / `metal-3`) in Chrome 152, canvas 1702 x 828 (1.41 Mpx, DPR 2), 14 to 16 HUD samples per cell. **Indicative only**: the GPU was shared with other tabs and other work during these runs, and repeating a configuration moved its total by up to a factor of two (the sky-only total came out at 1.75, 2.67 and 2.75 ms in three runs). Compare passes within a row, not totals across rows. These are not benchmark results and are not in the Scorecard.

| Strategy                    | `reflection` | `scene`    | `water`    | GPU total  | Extra texture memory   |
| --------------------------- | ------------ | ---------- | ---------- | ---------- | ---------------------- |
| sky only                    | none         | 0.6 to 1.2 | 1.2 to 1.6 | 1.8 to 2.8 | none                   |
| SSR, 16 steps               | none         | 0.7        | 1.65       | 2.3        | none                   |
| SSR, 48 steps (default)     | none         | 0.5 to 0.8 | 1.8 to 2.0 | 2.3 to 2.8 | none                   |
| SSR, 128 steps              | none         | 0.4        | 2.1        | 2.5        | none                   |
| planar, full resolution     | 0.9 to 1.35  | 1.0 to 1.4 | 1.5 to 1.7 | 3.4 to 4.5 | 17 MB (colour + depth) |
| planar, 1/2 resolution      | 0.7          | 0.8        | 1.4        | 2.9        | 4 MB                   |
| planar, 1/4 resolution      | 0.6 to 1.3   | 0.7 to 1.4 | 0.9 to 1.9 | 2.2 to 4.6 | 1 MB                   |
| split (planar 1/2 + SSR 48) | 0.7          | 0.8        | 1.6        | 3.2        | 4 MB                   |

What the numbers do say, because they hold within each run:

- The planar pass costs roughly as much as the scene pass at full resolution (0.9 against 1.0 ms), because it is the same scene drawn again. It scales with the scene: this one has 15 objects, so it is planar's best case. A scene with ten times the geometry pays ten times.
- Dropping the planar target to 1/4 resolution (1/16 of the pixels) barely dropped its time (0.56 against 0.71 ms for the scene pass in the same run). On this tiny scene the pass is dominated by fixed work (the fullscreen sky with four fbm octaves, 15 draws, the pass boundary), not by fill. The resolution scale saves memory and bandwidth reliably and saves time only when the scene is fill-bound.
- SSR adds no pass. Its cost sits in the `water` pass, roughly 0.2 to 0.6 ms over sky-only at 48 steps, and about +0.4 ms more going from 16 to 128 steps (one run). It is independent of scene complexity and proportional to the number of water pixels.
- The `water` pass in sky-only mode is not free: it still evaluates five sines for the normal and the cloud fbm for every water pixel.
- Texture memory: scene colour 11.3 MB plus depth 5.6 MB at this size is common to all strategies; the planar target adds its row above. SSR needs the scene target to be `rgba16float` and the depth to be `depth32float` and sampled, which the forward pass already provides.
- Bundle: the effect's chunk is 57 kB (16.8 kB gzipped); most of it is wgpu-matrix's `mat4` and `vec3`, which do not tree-shake to individual functions because each is one API object. `camera.wgsl` and `ssr.wgsl` are shared chunks.

Tier: **B** (multi-pass 3D: two scene renders and a fullscreen composite). Every strategy is well inside the 6 ms Tier B budget on this device, but the budget is for 1080p on an integrated GPU and phones are untested.

## Pitfalls

What each strategy gets wrong, from what I saw and from the construction:

**SSR**

- **Underside and occluded content is missing (seen).** The lintel's underside faces the water but never faces the camera, so the depth buffer has no data for it. Where planar shows a dark band across the water (the underside, then the posts), SSR shows sky and clouds, plus a small stray patch of the lintel's orange front face, which is the nearest thing the ray passes behind in screen space. The same holds for the far sides of the pillars and the pool walls.
- **Most rays miss (seen in the mask).** In the `mask` debug view only the water close to objects is green; most of the pool is red, i.e. the sky fallback. SSR only reflects what is on screen and in front of the ray, so a pool seen from above, mostly reflecting open sky, gets little from it. Content outside the frame is absent: the ray leaves the screen and falls back. The 10% edge fade is there to hide that, and it is exact in the numerical test, but I did not isolate an edge artefact in a screenshot.
- **Step size and banding (seen).** At 8 steps with jitter off, the reflection of the box and the pole breaks into dashed horizontal bands; with jitter on the bands become fine noise, which the ripples hide. Thin objects (the 0.45-wide far pole) can slip between samples at low step counts. A thickness that is too small loses hits behind thin occluders; too large puts ghost hits behind them (the unit test shows a pillar's silhouette being hit with a thickness of 50).
- **Grazing angles.** The ray covers a long path in few pixels, so reflections of surfaces met at a grazing angle are stretched, and the water at very low camera pitch is a thin sliver. I did not capture a clear stretching artefact in this scene; I am listing it from the construction, not from an observation.
- **Cost model.** Cost is per water pixel, so a lake filling the screen costs what a puddle does not. Misses cost the full march.
- **Depth must exclude the water.** Here the water is composited after the fact, so the depth buffer holds only the scene. If the water were a mesh writing depth, the trace would have to read a depth copy from before the water.

**Planar**

- **A whole extra scene render.** Every vertex and every pixel of the reflected geometry is paid for again. Resolution scale cuts the pixels but not the vertices or the pass overhead (see Cost).
- **Exact only for one flat plane.** Two water levels, a tilted surface or a curved one need another pass each; waves are faked by moving the lookup. The distortion hack shears the reflection: at `distortion = 1` and `ripple = 2` the lower-left of the planar half shows bright sky patches probably where the offset lookup runs off the edge of the target and the sampler clamps (I did not confirm that in the mask view). Expect that failure mode with any offset wider than a few pixels.
- **Quarter resolution looks soft, not wrong (seen).** At 1/4 the reflection is visibly blocky at the edges of objects but sits correctly in the right place; the ripples and Fresnel weighting hide most of it.
- **Clipping subtleties.** The oblique near plane skews the depth range and the far plane (reflected geometry above the plane is fine; the far end of the depth range loses a little precision), it needs the mirrored eye below the plane (so the camera must stay above the water), and winding flips. Missing any one gives holes or a covered reflection, not subtle error. Reflected fog uses the real path length, which is the distance to the mirrored eye.
- **Distortion is not physical.** Planar and SSR share one `distortion` slider, but planar moves the image by a small offset while SSR tilts the ray. At the same setting SSR's sky reflection is sheared far more near the horizon, where a small change of angle moves the lookup a long way. With the same ripple, the clouds in the planar half stay crisp and those in the SSR half and the sky-only mode turn into hazy streaks.

**Both**

- Colours are linear `rgba16float`; the sun disc reaches 60 in the sky model, so it survives being reflected through ripples and shows up in the reflection as a soft glint.
- HUD timings are quantised to about 100 microseconds in Chrome and, here, noisy because the GPU was shared.
- The orbit control captures the pointer and sets `touch-action: none` on the canvas, so on a phone the canvas will not scroll the page by touch. Mount it that way only where dragging is the point.

## When to use it

- **Tier B or C scenes with one flat water surface (a pool, pond, puddle, a hero set piece): planar**, at 1/2 resolution (1/4 if the ripples are strong). It is correct for anything in the scene, including content off screen and undersides, and its cost is predictable: one more render of your scene. Cull the reflection pass hard (distance, smaller LOD, no small props) and skip it when the water is off screen or the camera is under it.
- **Scenes where the reflecting surface is arbitrary or many (wet floors, several pools, windows, rooms): SSR**, with a sky or probe fallback and 24 to 48 steps with jitter. It costs a fixed amount per reflecting pixel and no second scene render, but accept that it will miss undersides and off-screen content, and design the camera so that is not the story (a low, wide view is worse than a high one).
- **Ambient water behind UI, Tier A: sky only.** One pass, no second render; a light normal wobble and Fresnel carry it. Add a few static sprite reflections if the page needs them.
- **Best of both:** planar for the one hero surface in a showpiece, with SSR kept for the rest, is what the `split` mode compares. Use `split` to judge for your own scene.
- Not measured: phones, integrated GPUs, scenes of a realistic size, and any scene with its own water mesh. The cost split above should be re-measured with Epic 13 before it is quoted.

## References

- Eric Lengyel, "Oblique View Frustum Depth Projection and Clipping": http://terathon.com/lengyel/Lengyel-Oblique.pdf (derived again for 0..1 depth in `obliqueProjection`).
- Morgan McGuire and Michael Mara, "Efficient GPU Screen-Space Ray Tracing", Journal of Computer Graphics Techniques 3(4), 2014: https://jcgt.org/published/0003/04/04/ (screen-space stepping with 1/z interpolation).
- Tomasz Stachowiak, "Stochastic Screen-Space Reflections", Advances in Real-Time Rendering, SIGGRAPH 2015: https://advances.realtimerendering.com/ (jitter and confidence fades).
- Jorge Jimenez, "Next Generation Post Processing in Call of Duty: Advanced Warfare", SIGGRAPH 2014 (interleaved gradient noise).
- Christophe Schlick, "An Inexpensive BRDF Model for Physically-based Rendering", 1994 (the Fresnel approximation).
- docs/support-matrix.md, sections 2 and 6, for `clip-distances`, `timestamp-query` and default limits.
- Links were written from memory and not re-checked this session.
