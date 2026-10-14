// refraction.wgsl -- screen-space refraction: turn a surface normal into a uv offset, sample a
// backdrop TEXTURE through it, with chromatic dispersion and safe edge handling.
//
// The production pattern: the scene behind the water is a texture (an image, a video frame, a captured
// page, a previous pass). The water pass finds where each pixel's ray ends up and samples the texture
// THERE instead of at the pixel itself. Nothing is ray-traced; the offset is a closed-form function of
// the normal, the index of refraction and how much water the ray crosses.
//
// FRAME (read this once). Normals, slopes and offsets live in the SCREEN frame: x to the right, y DOWN
// (the uv direction), z toward the viewer. Heights and slopes are taken with respect to a uv-aligned
// coordinate whose unit is the image HEIGHT (p = vec2f(uv.x * aspect, uv.y)). `thickness` is measured
// in the same unit: 0.1 means the backdrop is a tenth of the image height behind the surface. A flat
// surface has normal (0, 0, 1). Offsets named *_uv are in uv (x already divided by aspect).
//
// Normals
//   refraction_normal_from_slope(slope: vec2f) -> vec3f   unit normal of a height field with gradient
//                                    `slope` (dh/dx, dh/dy); use the _d noise functions for the slope
// Offsets, in world (image-height) units. Both are exactly 0 for a flat surface seen head-on.
//   refraction_offset_cheap(n, ior, thickness) -> vec2f
//                                    small-angle form: -(n.xy / n.z) * thickness * (1 - 1/ior). What most
//                                    sites ship. Within ~5% of the physical form up to ~30 degrees tilt.
//   refraction_offset_physical(n, view, ior, thickness) -> vec2f
//                                    Snell's law: bends `view` (unit-ish vector from the eye INTO the
//                                    scene, view.z < 0; (0, 0, -1) is looking straight down) with the
//                                    built-in refract() at eta = 1/ior, travels `thickness` down, and
//                                    returns the lateral shift relative to the unrefracted straight
//                                    continuation. Handles tilted views. ior is clamped to >= 1 (so no
//                                    total internal reflection); grazing rays are clamped at cos 0.05.
//   refraction_to_uv(offset: vec2f, aspect: f32) -> vec2f    world offset -> uv offset (x / aspect)
// Dispersion. Normal dispersion: blue light has the higher IOR, so red bends least.
//   refraction_dispersion_iors(ior, dispersion) -> vec3f     (ior - d, ior, ior + d) for (r, g, b), >= 1.
//                                    Real water: d ~ 0.003. Crown glass ~ 0.006. Designers use 0.01-0.05.
//   refraction_dispersion_scales(ior, dispersion) -> vec3f   (1 - 1/ior_c) / (1 - 1/ior): the factor by
//                                    which the green offset must be scaled for each channel. Exact for
//                                    the cheap form, accurate to a few percent for the physical one.
//   struct RefractionRGB { r: vec2f, g: vec2f, b: vec2f }    one uv offset per channel
//   refraction_rgb_offsets(offset_g_uv: vec2f, scales: vec3f) -> RefractionRGB
// Edge handling. Samples may land outside the texture; clamping smears the border pixel across the
// frame and wrapping shows the opposite edge. These keep every uv inside [0, 1] with neither.
//   refraction_mirror_uv(uv: vec2f) -> vec2f                 triangle-wave fold into [0, 1]: never wraps
//   refraction_fade_offset(uv, offset_uv, fade) -> vec2f     shrinks the part of the offset that points
//                                    at a border to 0 over `fade` (uv units, ~0.03-0.08) before reaching it
//   refraction_safe_uv(uv, offset_uv, fade) -> vec2f         mirror(uv + fade_offset(..)); always in [0, 1]
// Roughness / depth blur hook (needs a backdrop with a mip chain and a filtering sampler)
//   refraction_blur_lod(radius_px: f32, max_lod: f32) -> f32 mip level that blurs by about `radius_px`
//                                    texels; use radius_px = k * thickness for murky or deep water
// Sampling. `tex` / `samp` are function parameters (this chunk declares no bindings). Colours are
// whatever the texture holds; sample an `-srgb` format to get linear light. Uses textureSampleLevel, so
// it is legal in non-uniform control flow.
//   refraction_sample(tex, samp, uv, offset_uv, lod, fade) -> vec3f                  1 tap, no dispersion
//   refraction_sample_rgb(tex, samp, uv, offsets: RefractionRGB, lod, fade) -> vec3f 3 taps: r from the
//                                    red offset, g from green, b from blue
//   refraction_sample_spectral(tex, samp, uv, offset_g_uv, ior, dispersion, lod, fade, taps: i32) -> vec3f
//                                    `taps` (clamped 2..16) offsets spread over the spectrum, weighted by
//                                    overlapping red / green / blue responses and normalised, so flat colour
//                                    stays exact and edges blur into a smooth rainbow instead of three ghosts
//
// Cost (per pixel): offset_cheap ~8 ALU; offset_physical ~35 ALU (one refract); safe_uv ~15 ALU; every
// tap is one textureSampleLevel (bilinear; trilinear when lod has a fraction, so ~2x the bandwidth)
// plus safe_uv. refraction_sample_rgb = 3 taps; refraction_sample_spectral = `taps` taps and ~25 ALU each.
// Pitfalls: never feed uv + offset to a clamp-to-edge sampler directly (border smear); a backdrop that is
// itself the canvas needs a copy first (a texture cannot be read and written in the same pass).

const REFRACTION_MIN_COS: f32 = 0.05;

struct RefractionRGB {
  r: vec2f,
  g: vec2f,
  b: vec2f,
}

fn refraction_normal_from_slope(slope: vec2f) -> vec3f {
  return normalize(vec3f(-slope, 1.0));
}

fn refraction_offset_cheap(n: vec3f, ior: f32, thickness: f32) -> vec2f {
  let slope = n.xy / max(n.z, REFRACTION_MIN_COS);
  return -slope * thickness * (1.0 - 1.0 / max(ior, 1.0));
}

fn refraction_offset_physical(n: vec3f, view: vec3f, ior: f32, thickness: f32) -> vec2f {
  let incident = normalize(view);
  let refracted = refract(incident, n, 1.0 / max(ior, 1.0));
  let down_in = max(-incident.z, REFRACTION_MIN_COS);
  let down_out = max(-refracted.z, REFRACTION_MIN_COS);
  return thickness * (refracted.xy / down_out - incident.xy / down_in);
}

fn refraction_to_uv(offset: vec2f, aspect: f32) -> vec2f {
  return vec2f(offset.x / max(aspect, 1e-4), offset.y);
}

fn refraction_dispersion_iors(ior: f32, dispersion: f32) -> vec3f {
  return max(vec3f(ior) + vec3f(-1.0, 0.0, 1.0) * dispersion, vec3f(1.0));
}

fn refraction_dispersion_scales(ior: f32, dispersion: f32) -> vec3f {
  let iors = refraction_dispersion_iors(ior, dispersion);
  let denom = max(1.0 - 1.0 / max(ior, 1.0), 1e-4);
  return (vec3f(1.0) - vec3f(1.0) / iors) / denom;
}

fn refraction_rgb_offsets(offset_g_uv: vec2f, scales: vec3f) -> RefractionRGB {
  return RefractionRGB(offset_g_uv * scales.x, offset_g_uv, offset_g_uv * scales.z);
}

fn refraction_mirror_uv(uv: vec2f) -> vec2f {
  let t = uv - 2.0 * floor(uv * 0.5); // [0, 2)
  return vec2f(1.0) - abs(vec2f(1.0) - t);
}

fn refraction_fade_offset(uv: vec2f, offset_uv: vec2f, fade: f32) -> vec2f {
  // Distance to the border the offset points at, per axis.
  let toward = vec2f(
    select(uv.x, 1.0 - uv.x, offset_uv.x > 0.0),
    select(uv.y, 1.0 - uv.y, offset_uv.y > 0.0),
  );
  let k = smoothstep(vec2f(0.0), vec2f(max(fade, 1e-4)), toward);
  return offset_uv * k;
}

fn refraction_safe_uv(uv: vec2f, offset_uv: vec2f, fade: f32) -> vec2f {
  return refraction_mirror_uv(uv + refraction_fade_offset(uv, offset_uv, fade));
}

fn refraction_blur_lod(radius_px: f32, max_lod: f32) -> f32 {
  return clamp(log2(max(2.0 * radius_px, 1.0)), 0.0, max(max_lod, 0.0));
}

fn refraction_sample(
  tex: texture_2d<f32>, samp: sampler, uv: vec2f, offset_uv: vec2f, lod: f32, fade: f32,
) -> vec3f {
  return textureSampleLevel(tex, samp, refraction_safe_uv(uv, offset_uv, fade), lod).rgb;
}

fn refraction_sample_rgb(
  tex: texture_2d<f32>, samp: sampler, uv: vec2f, offsets: RefractionRGB, lod: f32, fade: f32,
) -> vec3f {
  let r = textureSampleLevel(tex, samp, refraction_safe_uv(uv, offsets.r, fade), lod).r;
  let g = textureSampleLevel(tex, samp, refraction_safe_uv(uv, offsets.g, fade), lod).g;
  let b = textureSampleLevel(tex, samp, refraction_safe_uv(uv, offsets.b, fade), lod).b;
  return vec3f(r, g, b);
}

fn refraction_sample_spectral(
  tex: texture_2d<f32>, samp: sampler, uv: vec2f, offset_g_uv: vec2f,
  ior: f32, dispersion: f32, lod: f32, fade: f32, taps: i32,
) -> vec3f {
  let count = clamp(taps, 2, 16);
  let denom = max(1.0 - 1.0 / max(ior, 1.0), 1e-4);
  var sum = vec3f(0.0);
  var weights = vec3f(0.0);
  for (var k = 0; k < count; k++) {
    // t runs over -1.5 .. 1.5: -1 is where the red response peaks, +1 the blue, 0 the green.
    let t = ((f32(k) + 0.5) / f32(count) * 2.0 - 1.0) * 1.5;
    let ior_t = max(ior + dispersion * t, 1.0);
    let scale = (1.0 - 1.0 / ior_t) / denom;
    let w = max(vec3f(1.0) - abs(vec3f(t) - vec3f(-1.0, 0.0, 1.0)), vec3f(0.0));
    let colour = textureSampleLevel(tex, samp, refraction_safe_uv(uv, offset_g_uv * scale, fade), lod).rgb;
    sum += w * colour;
    weights += w;
  }
  return sum / max(weights, vec3f(1e-4));
}
