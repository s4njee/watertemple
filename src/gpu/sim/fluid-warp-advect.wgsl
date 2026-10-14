// fluid-warp-advect.wgsl -- one step of the displacement field: carry it with the flow, add the motion
// the flow gave the content during the step, let it relax, keep it bounded. A fullscreen fragment pass
// over the field texture; it reads the previous field and the solver's staggered velocity and writes
// the new field (every texel).
//
// THE EQUATION. Let D(x, t) be how far the content now at x has been moved from its rest position.
// A parcel of content moves with the fluid, and the content at the parcel was displaced by exactly the
// distance the parcel has travelled from rest, so along its path dD/dt = v. Written for the fixed point
// x (an Eulerian field):
//
//     dD/dt + (v . grad) D = v  -  r D
//
// the left side is the advection of D by the flow (semi-Lagrangian, below), the first term on the
// right is the accumulation of motion, and r (P.relaxation, 1 / s) is the rate at which the content
// springs back: 0 keeps every smear for ever (paint), a large r makes it a short-lived dent (jelly).
// Steady uniform flow v for time t gives D = v t exactly (tests/fluid-warp.test.ts checks it).
//
// THE STEP, in image-height units. Semi-Lagrangian with a midpoint trace, like the solver's own:
//     v1   = v(x - dt/2 v(x))                  velocity at the middle of the parcel's path
//     D'   = D(x - dt v1) + gain dt v1         what the parcel carried, plus what it just moved
//     D'  *= exp(-r dt)                        relaxation (exact for any dt)
//     D'   = D' limited to |D'| <= max_disp    the content cannot be dragged off to infinity
// Where the trace leaves the domain: clamped (walls), or zero displacement for `open` edges, where
// the picture's own content enters.
//
// PRECISION. D is rgba16float (the solver's format, filterable everywhere): about three decimal digits.
// A step adds v dt, e.g. 0.5 x 1/60 = 0.008 image heights, well above the half-float spacing of a
// displacement of a few tenths (2.4e-4 at 0.3), so accumulation works; a flow slower than about 0.02
// heights per second stops adding once |D| is above 0.3. The limit is far below a pixel.

#include "fullscreen.wgsl"
#include "fluid.wgsl"
#include "warp_params.wgsl"

@group(0) @binding(0) var<uniform> P: WarpParams;
@group(0) @binding(1) var field_tex: texture_2d<f32>;
@group(0) @binding(2) var velocity_tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

fn flow_at(uv: vec2f) -> vec2f {
  return fluid_velocity_at(velocity_tex, samp, clamp(uv, vec2f(0.0), vec2f(1.0)), P.grid);
}

fn finite_vec(v: vec2f) -> vec2f {
  let bad = (bitcast<vec2u>(v) & vec2u(0x7f800000u)) == vec2u(0x7f800000u);
  return select(v, vec2f(0.0), bad);
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let uv = in.uv;
  let to_uv = vec2f(1.0 / P.aspect, 1.0); // image heights -> uv
  let v0 = flow_at(uv);
  let v1 = flow_at(uv - 0.5 * P.dt * v0 * to_uv);
  let src = uv - P.dt * v1 * to_uv;
  var carried = textureSampleLevel(field_tex, samp, clamp(src, vec2f(0.0), vec2f(1.0)), 0.0).xy;
  if (P.open != 0u && (any(src < vec2f(0.0)) || any(src > vec2f(1.0)))) {
    carried = vec2f(0.0);
  }
  var d = (carried + P.gain * P.dt * v1) * P.decay;
  let m = length(d);
  if (m > P.max_disp) {
    d = d * (P.max_disp / m);
  }
  return vec4f(finite_vec(d), 0.0, 0.0);
}
