// fluid3d-advect.wgsl -- advection of the velocity field (by itself) and of the scalar field (ink: colour
// as optical density in rgb, amount in a) by the velocity, in two schemes. Every kernel writes one
// rgba16float storage texture and reads others as textures.
//
// Bindings (besides the shared 0 `P`):
//   2 src    the field being advected: velocity ((nx+1) x (ny+1) x (nz+1)) or the scalar
//   3 aux    scheme scratch: the forward-advected prediction
//   4 dst    the output
//   5 flow   scalar entry points only: the velocity that carries the scalar
//   6 samp   linear, clamp-to-edge sampler
//
// SCHEMES.
//   semi-Lagrangian  1 pass   cs_*_sl / cs_*_pred   f' = A f          sample the field where the parcel came from
//   MacCormack       2 passes cs_*_pred, cs_*_mc    f^ = A f;  f' = f^ + (f - A^R f^) / 2, clamped to the
//                                                   range of the eight samples the forward trace interpolates
// (Selle, Fedkiw, Kim, Liu, Rossignac 2008). The limiter keeps the correction from creating new extrema, which
// is also what makes it stable. BFECC, which the 2D solver also has, is not built here: in 2D it kept no
// more detail than MacCormack for an extra pass, and a pass over a 3D grid is dear.
//
// The scalar entry points `cs_dye_sl` and `cs_dye_mc` apply the scalar's dissipation; `cs_dye_pred` does not.

#include "fluid3d-core.wgsl"

@group(0) @binding(2) var src: texture_3d<f32>;
@group(0) @binding(3) var aux: texture_3d<f32>;
@group(0) @binding(4) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var flow: texture_3d<f32>;
@group(0) @binding(6) var samp: sampler;

fn limited(x: f32, lo: f32, hi: f32) -> f32 {
  return select(x, clamp(x, lo, hi), P.limit != 0u);
}

fn limited4(x: vec4f, lo: vec4f, hi: vec4f) -> vec4f {
  return select(x, clamp(x, lo, hi), P.limit != 0u);
}

// ---- Velocity -----------------------------------------------------------------------------------------------

fn vel_pred(axis: u32, t: vec3i) -> f32 {
  let f = face_of(axis, t);
  if (face_blocked(axis, f)) {
    return 0.0;
  }
  return comp_sample(src, samp, axis, trace(src, samp, face_position(axis, f), -1.0));
}

fn vel_mc(axis: u32, t: vec3i) -> f32 {
  let f = face_of(axis, t);
  if (face_blocked(axis, f)) {
    return 0.0;
  }
  let p = face_position(axis, f);
  let x = trace(src, samp, p, -1.0);
  let forward = comp_sample(src, samp, axis, x);
  let bounds = comp_bounds(src, axis, x);
  let original = textureLoad(src, f, 0)[axis];
  let back = comp_sample(aux, samp, axis, trace(src, samp, p, 1.0));
  return limited(forward + 0.5 * (original - back), bounds.x, bounds.y);
}

fn in_velocity_grid(id: vec3u) -> bool {
  return all(id < P.grid + vec3u(1u));
}

@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn cs_vel_sl(@builtin(global_invocation_id) id: vec3u) {
  if (!in_velocity_grid(id)) {
    return;
  }
  let t = vec3i(id);
  textureStore(dst, t, vec4f(finite3(vec3f(vel_pred(0u, t), vel_pred(1u, t), vel_pred(2u, t))), 0.0));
}

@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn cs_vel_mc(@builtin(global_invocation_id) id: vec3u) {
  if (!in_velocity_grid(id)) {
    return;
  }
  let t = vec3i(id);
  textureStore(dst, t, vec4f(finite3(vec3f(vel_mc(0u, t), vel_mc(1u, t), vel_mc(2u, t))), 0.0));
}

// ---- Scalar ---------------------------------------------------------------------------------------------------

// Non-negative, finite and no larger than P.dye_max; zero in the obstacle.
fn finish_dye(v: vec4f, decay: f32) -> vec4f {
  return clamp(finite4(v * decay), vec4f(0.0), vec4f(P.dye_max));
}

fn dye_pred(t: vec3i) -> vec4f {
  return scalar_sample(src, samp, trace(flow, samp, scalar_pos(t), -1.0));
}

@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn cs_dye_sl(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= P.scalar_size)) {
    return;
  }
  let t = vec3i(id);
  if (scalar_solid(t)) {
    textureStore(dst, t, vec4f(0.0));
    return;
  }
  textureStore(dst, t, finish_dye(dye_pred(t), P.dye_decay));
}

@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn cs_dye_pred(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= P.scalar_size)) {
    return;
  }
  let t = vec3i(id);
  if (scalar_solid(t)) {
    textureStore(dst, t, vec4f(0.0));
    return;
  }
  textureStore(dst, t, finish_dye(dye_pred(t), 1.0));
}

@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn cs_dye_mc(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= P.scalar_size)) {
    return;
  }
  let t = vec3i(id);
  if (scalar_solid(t)) {
    textureStore(dst, t, vec4f(0.0));
    return;
  }
  let p = scalar_pos(t);
  let x = trace(flow, samp, p, -1.0);
  let forward = scalar_sample(src, samp, x);
  let bounds = scalar_bounds(src, x);
  let original = textureLoad(src, t, 0);
  let back = scalar_sample(aux, samp, trace(flow, samp, p, 1.0));
  textureStore(dst, t, finish_dye(limited4(forward + 0.5 * (original - back), bounds.lo, bounds.hi), P.dye_decay));
}
