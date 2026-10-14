// wash.wgsl -- watercolour and ink-wash building blocks: a cheap low-frequency pigment field, bicubic
// upsampling of it, soft blooms with pooled (darker) edges, granulation, and a tileable paper texture.
//
// The idea: pigment in water has soft, wandering blooms whose EDGES are crisp and darker (pigment is
// carried to the edge as the bloom dries), over a paper whose grain shows through. All of that is a
// threshold of a smooth field plus a texture, so the expensive part (the field) is low-frequency and
// can be rendered at a fraction of the output resolution, then reconstructed smoothly:
//
//   field pass (small target)   wash_field(p, time, warp)            -> vec3f  (bloom, tone, detail)
//   compose pass (full size)    wash_sample_bspline(tex, samp, uv)   -> the field, C2-smooth
//                               wash_body / wash_pool                -> bloom coverage, pooled edge
//                               wash_granulate(density, grain, k)    -> grain shows where pigment is
//   once, at creation           wash_paper_tile(uv)                  -> vec2f (grain, fibre) to bake
//
// Functions
//   wash_field(p: vec2f, time: f32, warp: f32) -> vec3f
//       Three smooth fields at one point, each in about [-0.9, 0.9]: x = bloom shape (domain-warped
//       3D fbm, 3 octaves; its level sets are the bloom outlines), y = tone (one slow noise at a coarser
//       scale: pick between two pigments), z = detail (finer noise at 2.2x the frequency: roughens an
//       outline). `p` is in feature units (about one blob per unit); `time` in seconds-ish (the field
//       evolves by one feature in about 8 units of time); `warp` in [0, 1.5] is how far the domain is
//       bent (0 = plain fbm blobs, 1 = swirled tendrils). Nine 3D gradient-noise evaluations in total.
//   wash_sample_bspline(tex: texture_2d<f32>, samp: sampler, uv: vec2f) -> vec4f
//       The texture reconstructed with a cubic B-spline (C2: no kinks at texel borders even when the
//       texture is 4x smaller than the output), using four hardware-bilinear taps instead of sixteen.
//       `samp` must be a linear, clamp-to-edge sampler. The result is slightly smoother than the
//       texels (B-spline smoothing), which is what you want for a low-frequency field.
//   wash_body(inside_px: f32, softness_px: f32) -> f32
//       Bloom coverage in [0, 1] from the signed distance in pixels to the bloom outline, positive
//       inside: a ramp `softness_px` wide on each side. Get inside_px = (field - level) / fwidth(field),
//       which is the distance in pixels to the field's level set to first order.
//   wash_pool(inside_px: f32, width_px: f32, softness_px: f32) -> f32
//       The darker rim where pigment gathers, in [0, 1]: 1 just inside the outline, falling to 0 by
//       `width_px` inside it, and 0 outside. Add to the pigment density, scaled by an edge strength.
//   wash_granulate(density: f32, grain: f32, strength: f32) -> f32
//       Pigment settles into the paper's valleys: density modulated by the grain (`grain` in [0, 1],
//       0.5 neutral), more where there is more pigment. strength 0 returns `density` unchanged.
//   wash_paper_tile(uv: vec2f) -> vec2f
//       Paper at one point of a tile, for baking once into a texture that is repeated: x = grain height
//       in [0, 1] (mean 0.5), y = fibre in [0, 1] (mostly 0; thin curved streaks). Seamless: the value
//       at uv and uv + 1 are equal, on both axes. Sample the baked texture with a repeating sampler.
//
// Cost: wash_field about 9 gnoise_3d (run it at 1/16 of the pixels); wash_sample_bspline 4 filtered
// fetches; body, pool and granulate a few ALU each; wash_paper_tile 5 tiled gnoise_2d (bake it once).
// Credits: cubic B-spline with bilinear taps from C. Sigg and M. Hadwiger, "Fast Third-Order Texture
// Filtering" (GPU Gems 2, ch. 20); domain warping from I. Quilez; the pooled-edge model follows the
// "edge darkening" of Curtis et al., "Computer-Generated Watercolor" (SIGGRAPH 1997), simplified to a
// rim at the level set of a smooth field.

#include "noise.wgsl"
#include "fbm.wgsl"

fn wash_field(p: vec2f, time: f32, warp: f32) -> vec3f {
  let q = domain_warp_2d_anim(p * 0.8, time * 0.35, warp * 0.9, 2, 2.0, 0.5);
  let bloom = fbm_3d(vec3f(q * 1.1, time * 0.22), 3, 2.0, 0.5) * 1.5;
  let tone = gnoise_3d(vec3f(q * 0.45 + vec2f(11.0, 3.0), time * 0.15)) * 1.4;
  let detail = gnoise_3d(vec3f(q * 2.4 + vec2f(5.0, 17.0), time * 0.3));
  return vec3f(bloom, tone, detail);
}

fn wash_sample_bspline(tex: texture_2d<f32>, samp: sampler, uv: vec2f) -> vec4f {
  let size = vec2f(textureDimensions(tex, 0));
  let coord = uv * size - 0.5;
  let base = floor(coord);
  let f = coord - base;
  let f2 = f * f;
  let f3 = f2 * f;
  let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
  let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
  let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
  let w3 = f3 / 6.0;
  let g0 = w0 + w1;
  let g1 = w2 + w3;
  // Each pair of neighbouring texels becomes one bilinear tap placed so the hardware weights match.
  let p0 = (base - 1.0 + w1 / g0 + 0.5) / size;
  let p1 = (base + 1.0 + w3 / g1 + 0.5) / size;
  let t00 = textureSampleLevel(tex, samp, vec2f(p0.x, p0.y), 0.0);
  let t10 = textureSampleLevel(tex, samp, vec2f(p1.x, p0.y), 0.0);
  let t01 = textureSampleLevel(tex, samp, vec2f(p0.x, p1.y), 0.0);
  let t11 = textureSampleLevel(tex, samp, vec2f(p1.x, p1.y), 0.0);
  return g0.y * (g0.x * t00 + g1.x * t10) + g1.y * (g0.x * t01 + g1.x * t11);
}

fn wash_body(inside_px: f32, softness_px: f32) -> f32 {
  return smoothstep(-softness_px, softness_px, inside_px);
}

fn wash_pool(inside_px: f32, width_px: f32, softness_px: f32) -> f32 {
  let rise = smoothstep(-softness_px, softness_px, inside_px);
  let fall = 1.0 - smoothstep(0.0, max(width_px, 1e-3), inside_px - softness_px);
  return rise * fall;
}

fn wash_granulate(density: f32, grain: f32, strength: f32) -> f32 {
  let valley = 0.5 - grain;
  return max(density * (1.0 + strength * 1.6 * valley * (0.35 + density)), 0.0);
}

fn wash_paper_tile(uv: vec2f) -> vec2f {
  // Grain: two scales of tiled gradient noise, a texel or two across.
  let g1 = gnoise_2d_tile(uv * 128.0 + vec2f(17.0, 3.0), vec2i(128, 128));
  let g2 = gnoise_2d_tile(uv * 48.0 + vec2f(5.0, 29.0), vec2i(48, 48));
  let grain = clamp(0.5 + 0.55 * (0.62 * g1 + 0.38 * g2), 0.0, 1.0);

  // Fibres: noise stretched 8x along one axis, thinned to the lines where it crosses zero; one set
  // runs along each axis, and a slow patchiness makes them clump the way real paper does.
  let along_y = gnoise_2d_tile(vec2f(uv.x * 40.0, uv.y * 5.0) + vec2f(2.0, 1.0), vec2i(40, 5));
  let along_x = gnoise_2d_tile(vec2f(uv.x * 5.0, uv.y * 40.0) + vec2f(3.0, 7.0), vec2i(5, 40));
  let thin_y = pow(1.0 - min(abs(along_y) * 2.2, 1.0), 12.0);
  let thin_x = pow(1.0 - min(abs(along_x) * 2.2, 1.0), 12.0);
  let clump = clamp(0.5 + 0.9 * gnoise_2d_tile(uv * 6.0 + vec2f(1.0, 4.0), vec2i(6, 6)), 0.0, 1.0);
  let fibre = clamp(max(thin_y, thin_x * 0.8) * clump, 0.0, 1.0);
  return vec2f(grain, fibre);
}
