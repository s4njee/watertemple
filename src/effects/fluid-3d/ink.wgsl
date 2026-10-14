// ink.wgsl -- what the ink does to light, shared by the ray march (march.wgsl) and the light-volume pass
// (volume-prep.wgsl). Declares the parameter block at group 0 binding 0 (the struct itself is generated from
// index.ts's `MarchParams`).
//
// The scalar field the solver advects has the ink's colour as an OPTICAL DENSITY per unit length in rgb
// (-ln of the colour times the amount, so two inks add: subtractive mixing) and the amount in a. The medium is
// extinction = absorption + scattering:
//     sigma_a = rgb * abs_gain          coloured: it is what makes the ink look like ink rather than smoke
//     sigma_s = a * scatter             grey: a little milkiness, what lets a lit edge glow
// and the colour of the light it scatters is the ink's own colour, exp(-rgb / a).

#include "march_params.wgsl"

@group(0) @binding(0) var<uniform> P: MarchParams;

// Extinction per unit of box length, per channel, of an ink sample.
fn ink_extinction(d: vec4f) -> vec3f {
  return d.rgb * P.abs_gain + vec3f(d.a * P.scatter);
}

// The colour of the ink (linear rgb, 0..1) given a sample: white where there is none.
fn ink_tint(d: vec4f) -> vec3f {
  return exp(-d.rgb / max(d.a, 1e-3));
}

// Distance along the unit ray to the obstacle sphere (P.sphere: world centre in xyz, radius in w), or -1 when
// there is none, it is missed or it is behind the origin. The solver's obstacle is this sphere.
fn sphere_hit(o: vec3f, d: vec3f) -> f32 {
  if (P.sphere.w <= 0.0) {
    return -1.0;
  }
  let oc = o - P.sphere.xyz;
  let b = dot(oc, d);
  let disc = b * b - (dot(oc, oc) - P.sphere.w * P.sphere.w);
  if (disc < 0.0) {
    return -1.0;
  }
  let t = -b - sqrt(disc);
  return select(-1.0, t, t > 0.0);
}
