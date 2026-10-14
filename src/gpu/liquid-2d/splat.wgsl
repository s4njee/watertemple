// liquid-2d splat pass: one soft blob per live particle, added into the field texture.
//
// The kernel is w(q) = k (1 - q^2)^3 for q = distance / R < 1, with R = blobRadius * particle radius. It
// is smooth (C2 at the rim, so the iso-line has no corners), compact (the quad is exactly R plus one field
// texel) and cheap (two multiplies and a cube). `kernel_k` normalises it so that the field is 1 inside bulk
// liquid at rest density: see liquid-2d/index.ts for the derivation.
//
// Channels written: r = sum of w (density / thickness), g = sum of w * foam_i, with foam_i the per-particle
// scalar (positions.w) mapped through smoothstep(foam_low, foam_high, .). b and a stay free.
//
// The particle position arrives as an instance-step vertex buffer (stride 16), so a solver that ping-pongs
// or reorders its buffers needs no bind group here. A non-finite position parks the quad outside the clip
// volume, so one bad particle cannot put a NaN into the additive field.

#include "liquid2d_params.wgsl"

@group(0) @binding(0) var<uniform> P: Liquid2dParams;

struct SplatVarying {
  @builtin(position) position: vec4f,
  // Offset from the particle centre in units of the blob radius R.
  @location(0) q: vec2f,
  @location(1) @interpolate(flat) foam: f32,
}

@vertex
fn vs(@builtin(vertex_index) vertex: u32, @location(0) particle: vec4f) -> SplatVarying {
  var out: SplatVarying;
  // Parked: outside the clip volume, so the rasteriser drops it.
  out.position = vec4f(2.0, 2.0, 2.0, 1.0);
  out.q = vec2f(0.0);
  out.foam = 0.0;

  let centre = (particle.xy - P.dom_center) / P.dom_half; // normalised device coordinates, y up
  // NaN and infinity fail this comparison; so does a particle absurdly far away.
  if (!(abs(centre.x) < 1e4 && abs(centre.y) < 1e4)) {
    return out;
  }
  let corner = vec2f(f32(vertex & 1u), f32((vertex >> 1u) & 1u)) * 2.0 - 1.0;
  let texel = 2.0 * P.dom_half / P.field_size; // world size of one field texel
  let reach = P.blob_radius + max(texel.x, texel.y); // one texel beyond the rim
  let offset = corner * reach;
  out.position = vec4f(centre + offset / P.dom_half, 0.0, 1.0);
  out.q = offset / P.blob_radius;
  let foam = smoothstep(P.foam_low, P.foam_high, particle.w);
  out.foam = select(0.0, foam, foam >= 0.0 && foam <= 1.0); // a NaN scalar counts as no foam
  return out;
}

@fragment
fn fs(in: SplatVarying) -> @location(0) vec4f {
  let q2 = dot(in.q, in.q);
  if (q2 >= 1.0) { discard; }
  let a = 1.0 - q2;
  let w = P.kernel_k * a * a * a;
  return vec4f(w, w * in.foam, 0.0, 0.0);
}
