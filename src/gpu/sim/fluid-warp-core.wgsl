// fluid-warp-core.wgsl -- reading a DISPLACEMENT FIELD carried by the stable-fluids solver
// (src/gpu/sim/fluid-warp.ts), and turning it into the place to sample the content.
//
// THE FIELD. One rgba16float texel per sample, xy = D, the displacement of the content at that point
// from where it rests, in IMAGE HEIGHTS (canvas height = 1, x right, y DOWN, the unit of the fluid's
// velocity). The picture a visitor sees at uv shows the content that REST at  uv - D:
//
//     colour(uv) = content(uv - D(uv) / (aspect, 1))
//
// That is a BACKWARD map (every output pixel asks where its colour comes from), so there are no holes
// and no overlaps to resolve, and D = 0 returns the source unchanged. A forward map (push each source
// texel to uv + D) would leave gaps where the flow stretches the content and pile texels up where it
// compresses. The field itself is advected by the flow and accumulates it (fluid-warp-advect.wgsl).
//
// All of it assumes `uv` has its origin top-left, like fullscreen.wgsl. Functions are prefixed fwarp_.
//
//   fwarp_fetch(field, samp, uv) -> vec2f
//       D at uv by one bilinear tap (`samp` must be a clamp-to-edge linear sampler). Piecewise linear:
//       its slope jumps at texel boundaries, which shows as faint creases when the field is coarse.
//   fwarp_fetch_cubic(field, samp, uv, size: vec2f) -> vec2f
//       D at uv through a cubic B-spline in four bilinear taps (Sigg and Hadwiger, GPU Gems 2 ch. 20).
//       C2 smooth, exact for a constant field, and a little softer than bilinear. `size` is the field's
//       size in texels.
//   fwarp_to_uv(d: vec2f, aspect: f32) -> vec2f
//       a displacement in image heights as a uv offset (x / aspect).
//   fwarp_source_uv(uv, d_uv, fade: f32) -> vec2f
//       where to sample the content: uv - d_uv, with the part of the offset that points at a border faded
//       to nothing over `fade` (uv units, 0.02 to 0.08) before it gets there and the rest mirror-folded
//       back into [0, 1], so the picture neither smears the border pixel nor wraps. EXACTLY uv when d_uv
//       is zero or the result stays inside [0, 1] (no round trip through a fold).
//   fwarp_gradient(field, samp, uv, size: vec2f, aspect: f32) -> mat2x2f
//       the Jacobian of D by central differences over one texel (four bilinear taps), per image height:
//       column 0 is dD/dx, column 1 is dD/dy. Zero for a constant field. The measure of local stretch.
//   fwarp_stretch(g: mat2x2f) -> f32
//       the Frobenius norm of that Jacobian: 0 at rest or in a pure translation, about 1 where the
//       content has been stretched by a factor of two.
//   fwarp_film_slope(g: mat2x2f, d: vec2f) -> vec2f
//       the gradient of |D|, scaled to zero as |D| goes to zero: a film that is thick where the content
//       has been moved far. Feed it to a normal for a liquid sheen that is exactly nothing at rest.
//
// Cost: fetch 1 tap, fetch_cubic 4, gradient 4; a few dozen ALU.

#include "refraction.wgsl"

fn fwarp_fetch(field: texture_2d<f32>, samp: sampler, uv: vec2f) -> vec2f {
  return textureSampleLevel(field, samp, uv, 0.0).xy;
}

fn fwarp_fetch_cubic(field: texture_2d<f32>, samp: sampler, uv: vec2f, size: vec2f) -> vec2f {
  let coord = uv * size - vec2f(0.5);
  let base = floor(coord);
  let f = coord - base;
  let f2 = f * f;
  let f3 = f2 * f;
  // Cubic B-spline weights of the texels base-1 .. base+2, merged pairwise into two bilinear taps.
  let w0 = (1.0 - f) * (1.0 - f) * (1.0 - f) / 6.0;
  let w1 = (3.0 * f3 - 6.0 * f2 + 4.0) / 6.0;
  let w2 = (-3.0 * f3 + 3.0 * f2 + 3.0 * f + 1.0) / 6.0;
  let w3 = f3 / 6.0;
  let g0 = w0 + w1;
  let g1 = w2 + w3;
  let p0 = (base - vec2f(1.0) + w1 / g0 + vec2f(0.5)) / size;
  let p1 = (base + vec2f(1.0) + w3 / g1 + vec2f(0.5)) / size;
  let s00 = textureSampleLevel(field, samp, vec2f(p0.x, p0.y), 0.0).xy;
  let s10 = textureSampleLevel(field, samp, vec2f(p1.x, p0.y), 0.0).xy;
  let s01 = textureSampleLevel(field, samp, vec2f(p0.x, p1.y), 0.0).xy;
  let s11 = textureSampleLevel(field, samp, vec2f(p1.x, p1.y), 0.0).xy;
  return g0.y * (g0.x * s00 + g1.x * s10) + g1.y * (g0.x * s01 + g1.x * s11);
}

fn fwarp_to_uv(d: vec2f, aspect: f32) -> vec2f {
  return vec2f(d.x / max(aspect, 1e-4), d.y);
}

fn fwarp_source_uv(uv: vec2f, d_uv: vec2f, fade: f32) -> vec2f {
  // Sampling happens at uv - d_uv, so the offset that matters is -d_uv.
  let offset = refraction_fade_offset(uv, -d_uv, fade);
  let s = uv + offset;
  let inside = all(s >= vec2f(0.0)) && all(s <= vec2f(1.0));
  return select(refraction_mirror_uv(s), s, inside);
}

fn fwarp_gradient(
  field: texture_2d<f32>, samp: sampler, uv: vec2f, size: vec2f, aspect: f32,
) -> mat2x2f {
  let t = vec2f(1.0) / size;
  let r = textureSampleLevel(field, samp, uv + vec2f(t.x, 0.0), 0.0).xy;
  let l = textureSampleLevel(field, samp, uv - vec2f(t.x, 0.0), 0.0).xy;
  let d = textureSampleLevel(field, samp, uv + vec2f(0.0, t.y), 0.0).xy;
  let u = textureSampleLevel(field, samp, uv - vec2f(0.0, t.y), 0.0).xy;
  // d/dx in image heights = (d/du) / aspect; d/dy = d/dv.
  let ddx = (r - l) * (0.5 * size.x / max(aspect, 1e-4));
  let ddy = (d - u) * (0.5 * size.y);
  return mat2x2f(ddx, ddy);
}

fn fwarp_stretch(g: mat2x2f) -> f32 {
  return sqrt(dot(g[0], g[0]) + dot(g[1], g[1]));
}

fn fwarp_film_slope(g: mat2x2f, d: vec2f) -> vec2f {
  let m = length(d);
  let dir = d / max(m, 1e-6) * smoothstep(0.0, 0.01, m);
  return vec2f(dot(g[0], dir), dot(g[1], dir));
}
