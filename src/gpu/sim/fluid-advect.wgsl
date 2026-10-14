// fluid-advect.wgsl -- advection of the velocity field (by itself) and of the dye (by the velocity), in
// three schemes. Every kernel writes one rgba16float storage texture and reads others as textures.
//
// Bindings (besides the shared 0 `P` and 1 `solid_mask`):
//   2 src   the field being advected: velocity (W+1 x H+1) or dye
//   3 aux   scheme scratch: the forward-advected prediction, or the error-compensated field
//   4 dst   the output
//   5 flow  dye entry points only: the velocity that carries the dye
//   6 flow_samp  dye entry points only: a linear sampler, used when `P.dye_filter` asks for a cubic
//                reconstruction of `flow` (fluid-core.wgsl, `vel_at_cubic`)
//
// SCHEMES (Selle, Fedkiw, Kim, Liu, Rossignac 2008, "An Unconditionally Stable MacCormack Method").
// Write A for one semi-Lagrangian step (sample the field at where the parcel came from) and A^R for
// the reverse step (sample at where the parcel is going):
//   semi-Lagrangian  1 pass    cs_*_sl             f' = A f
//   MacCormack       2 passes  cs_*_pred, cs_*_mc  f^ = A f;  f' = f^ + (f - A^R f^) / 2, clamped
//   BFECC            3 passes  cs_*_pred, cs_*_bfecc_error, cs_*_bfecc_final
//                              f^ = A f;  g = f + (f - A^R f^) / 2;  f' = A g, clamped
// "Clamped" means limited to the smallest and largest value among the source samples the forward
// trace interpolates. The limiter is what keeps the error correction from creating new extrema (and
// makes both schemes unconditionally stable); without it they overshoot and ring.
//
// The `*_sl` and `*_mc`/`*_bfecc_final` entry points of the dye apply the dye dissipation; `*_pred`
// and `*_bfecc_error` never do (they are intermediates).

#include "fluid-core.wgsl"

@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var aux: texture_2d<f32>;
@group(0) @binding(4) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var flow: texture_2d<f32>;
@group(0) @binding(6) var flow_samp: sampler;

// The limiter: the corrected value clamped to the range of the source samples it came from. It is only
// ever off (P.limit == 0) to show what it prevents.
fn limited(x: f32, lo: f32, hi: f32) -> f32 {
  return select(x, clamp(x, lo, hi), P.limit != 0u);
}

fn limited4(x: vec4f, lo: vec4f, hi: vec4f) -> vec4f {
  return select(x, clamp(x, lo, hi), P.limit != 0u);
}

// ---- Velocity ------------------------------------------------------------------------------------

fn vel_pred(axis: u32, c: vec2i) -> f32 {
  let f = face_of(axis, c);
  if (face_state(axis, f) != FACE_FREE) {
    return 0.0;
  }
  return face_sample(src, axis, trace(src, face_position(axis, f), -1.0));
}

fn vel_mc(axis: u32, c: vec2i) -> f32 {
  let f = face_of(axis, c);
  if (face_state(axis, f) != FACE_FREE) {
    return 0.0;
  }
  let p = face_position(axis, f);
  let forward = face_sample_bounds(src, axis, trace(src, p, -1.0));
  let original = face_tap(src, axis, f.x, f.y).x;
  let back = face_sample(aux, axis, trace(src, p, 1.0));
  return limited(forward.x + 0.5 * (original - back), forward.y, forward.z);
}

fn vel_bfecc_error(axis: u32, c: vec2i) -> f32 {
  let f = face_of(axis, c);
  if (face_state(axis, f) != FACE_FREE) {
    return 0.0;
  }
  let original = face_tap(src, axis, f.x, f.y).x;
  let back = face_sample(aux, axis, trace(src, face_position(axis, f), 1.0));
  return original + 0.5 * (original - back);
}

fn vel_bfecc_final(axis: u32, c: vec2i) -> f32 {
  let f = face_of(axis, c);
  if (face_state(axis, f) != FACE_FREE) {
    return 0.0;
  }
  let x = trace(src, face_position(axis, f), -1.0);
  let bounds = face_sample_bounds(src, axis, x);
  return limited(face_sample(aux, axis, x), bounds.y, bounds.z);
}

fn finish_vel(v: vec2f) -> vec4f {
  return vec4f(finite2(v), 0.0, 0.0);
}

@compute @workgroup_size(16, 16)
fn cs_vel_sl(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.grid + vec2u(1u))) {
    return;
  }
  let c = vec2i(id.xy);
  textureStore(dst, c, finish_vel(vec2f(vel_pred(0u, c), vel_pred(1u, c))));
}

// The forward pass of both corrected schemes is the plain semi-Lagrangian step.
@compute @workgroup_size(16, 16)
fn cs_vel_pred(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.grid + vec2u(1u))) {
    return;
  }
  let c = vec2i(id.xy);
  textureStore(dst, c, finish_vel(vec2f(vel_pred(0u, c), vel_pred(1u, c))));
}

@compute @workgroup_size(16, 16)
fn cs_vel_mc(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.grid + vec2u(1u))) {
    return;
  }
  let c = vec2i(id.xy);
  textureStore(dst, c, finish_vel(vec2f(vel_mc(0u, c), vel_mc(1u, c))));
}

@compute @workgroup_size(16, 16)
fn cs_vel_bfecc_error(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.grid + vec2u(1u))) {
    return;
  }
  let c = vec2i(id.xy);
  textureStore(dst, c, finish_vel(vec2f(vel_bfecc_error(0u, c), vel_bfecc_error(1u, c))));
}

@compute @workgroup_size(16, 16)
fn cs_vel_bfecc_final(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.grid + vec2u(1u))) {
    return;
  }
  let c = vec2i(id.xy);
  textureStore(dst, c, finish_vel(vec2f(vel_bfecc_final(0u, c), vel_bfecc_final(1u, c))));
}

// ---- Dye -----------------------------------------------------------------------------------------

fn dye_solid(t: vec2i) -> bool {
  return P.solid != 0u && textureLoad(solid_mask, dye_cell(t), 0).r >= 0.5;
}

fn finish_dye(v: vec4f, decay: f32) -> vec4f {
  let m = clamp(v * decay, vec4f(-P.dye_max), vec4f(P.dye_max));
  return select(m, vec4f(0.0), (bitcast<vec4u>(m) & vec4u(0x7f800000u)) == vec4u(0x7f800000u));
}

// Where a dye texel's parcel was / will be, in dye texels.
fn dye_trace(t: vec2i, way: f32) -> vec2f {
  let ps = dye_to_sim(vec2f(t) + 0.5);
  // The cubic velocity needs the plain domain (see fluid-core.wgsl); anything else keeps the bilinear one.
  if (P.dye_filter != 0u && P.solid == 0u && P.boundary != BOUNDARY_WRAP) {
    return sim_to_dye(trace_cubic(flow, flow_samp, ps, way));
  }
  return sim_to_dye(trace(flow, ps, way));
}

fn dye_pred(t: vec2i) -> vec4f {
  return dye_sample(src, dye_trace(t, -1.0)).v;
}

@compute @workgroup_size(16, 16)
fn cs_dye_sl(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.dye_size)) {
    return;
  }
  let t = vec2i(id.xy);
  if (dye_solid(t)) {
    textureStore(dst, t, vec4f(0.0));
    return;
  }
  textureStore(dst, t, finish_dye(dye_pred(t), P.dye_decay));
}

@compute @workgroup_size(16, 16)
fn cs_dye_pred(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.dye_size)) {
    return;
  }
  let t = vec2i(id.xy);
  if (dye_solid(t)) {
    textureStore(dst, t, vec4f(0.0));
    return;
  }
  textureStore(dst, t, finish_dye(dye_pred(t), 1.0));
}

@compute @workgroup_size(16, 16)
fn cs_dye_mc(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.dye_size)) {
    return;
  }
  let t = vec2i(id.xy);
  if (dye_solid(t)) {
    textureStore(dst, t, vec4f(0.0));
    return;
  }
  let forward = dye_sample(src, dye_trace(t, -1.0));
  let original = textureLoad(src, t, 0);
  let back = dye_sample(aux, dye_trace(t, 1.0)).v;
  let corrected = limited4(forward.v + 0.5 * (original - back), forward.lo, forward.hi);
  textureStore(dst, t, finish_dye(corrected, P.dye_decay));
}

@compute @workgroup_size(16, 16)
fn cs_dye_bfecc_error(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.dye_size)) {
    return;
  }
  let t = vec2i(id.xy);
  if (dye_solid(t)) {
    textureStore(dst, t, vec4f(0.0));
    return;
  }
  let original = textureLoad(src, t, 0);
  let back = dye_sample(aux, dye_trace(t, 1.0)).v;
  textureStore(dst, t, finish_dye(original + 0.5 * (original - back), 1.0));
}

@compute @workgroup_size(16, 16)
fn cs_dye_bfecc_final(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.dye_size)) {
    return;
  }
  let t = vec2i(id.xy);
  if (dye_solid(t)) {
    textureStore(dst, t, vec4f(0.0));
    return;
  }
  let x = dye_trace(t, -1.0);
  let bounds = dye_sample(src, x);
  let corrected = limited4(dye_sample(aux, x).v, bounds.lo, bounds.hi);
  textureStore(dst, t, finish_dye(corrected, P.dye_decay));
}
