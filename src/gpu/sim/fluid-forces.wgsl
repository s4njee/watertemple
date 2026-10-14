// fluid-forces.wgsl -- everything that adds to the fluid between advection and projection: splats of
// momentum and dye, the vorticity (curl) field, body forces with vorticity confinement, viscosity.
//
// Bindings (besides the shared 0 `P` and 1 `solid_mask`); an entry point uses only some of them:
//   2 src       the texture being updated: velocity (W+1 x H+1) or, for cs_splat_dye, the dye
//   3 dst       its rgba16float storage output
//   4 splats    read-only storage buffer of `Splat`s (P.force_count / P.dye_count of them)
//   5 curl_tex  r32float W x H vorticity (physical units, 1 / s), read by cs_forces
//   6 dye_tex   the dye, for buoyancy
//   7 samp      a filtering sampler for dye_tex
//   8 curl_dst  r32float storage output of cs_curl
//   9 base      the un-diffused velocity of the viscosity iteration
//  10 smooth_dst r32float storage output of cs_curl_smooth: |curl| blurred over 3 x 3 cells
//  11 smooth_tex the same, read by cs_forces

#include "fluid-core.wgsl"

@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<storage, read> splats: array<Splat>;
@group(0) @binding(5) var curl_tex: texture_2d<f32>;
@group(0) @binding(6) var dye_tex: texture_2d<f32>;
@group(0) @binding(7) var samp: sampler;
@group(0) @binding(8) var curl_dst: texture_storage_2d<r32float, write>;
@group(0) @binding(9) var base: texture_2d<f32>;
@group(0) @binding(10) var smooth_dst: texture_storage_2d<r32float, write>;
@group(0) @binding(11) var smooth_tex: texture_2d<f32>;

// ---- Splats ----------------------------------------------------------------------------------------

fn splat_velocity(axis: u32, c: vec2i) -> f32 {
  let f = face_of(axis, c);
  if (face_state(axis, f) != FACE_FREE) {
    return 0.0;
  }
  let uv = face_position(axis, f) / vec2f(P.grid);
  var dv = 0.0;
  for (var k = 0u; k < P.force_count; k++) {
    let s = splats[k];
    dv += splat_weight(uv, s) * select(s.value.y, s.value.x, axis == 0u);
  }
  return face_tap(src, axis, f.x, f.y).x + dv;
}

@compute @workgroup_size(16, 16)
fn cs_splat_velocity(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.grid + vec2u(1u))) {
    return;
  }
  let c = vec2i(id.xy);
  let v = vec2f(splat_velocity(0u, c), splat_velocity(1u, c));
  let m = P.vmax;
  textureStore(dst, c, vec4f(clamp(finite2(v), vec2f(-m), vec2f(m)), 0.0, 0.0));
}

@compute @workgroup_size(16, 16)
fn cs_splat_dye(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.dye_size)) {
    return;
  }
  let t = vec2i(id.xy);
  if (P.solid != 0u && textureLoad(solid_mask, dye_cell(t), 0).r >= 0.5) {
    textureStore(dst, t, vec4f(0.0));
    return;
  }
  let uv = (vec2f(t) + 0.5) / vec2f(P.dye_size);
  var d = textureLoad(src, t, 0);
  for (var k = 0u; k < P.dye_count; k++) {
    let s = splats[k];
    d += splat_weight(uv, s) * s.value;
  }
  let m = clamp(d, vec4f(-P.dye_max), vec4f(P.dye_max));
  textureStore(dst, t, select(m, vec4f(0.0), (bitcast<vec4u>(m) & vec4u(0x7f800000u)) == vec4u(0x7f800000u)));
}

// ---- Vorticity ---------------------------------------------------------------------------------------

// Curl at a cell centre in 1 / s: the mean of the four corner circulations of the cell, which
// telescopes to eight face velocities. Positive is dv/dx > du/dy, which is CLOCKWISE on screen (y down).
@compute @workgroup_size(16, 16)
fn cs_curl(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.grid)) {
    return;
  }
  let c = vec2i(id.xy);
  var w = 0.0;
  if (cell_class(c) == CELL_FLUID) {
    let dv = face_tap(src, 1u, c.x + 1, c.y).x - face_tap(src, 1u, c.x - 1, c.y).x
           + face_tap(src, 1u, c.x + 1, c.y + 1).x - face_tap(src, 1u, c.x - 1, c.y + 1).x;
    let du = face_tap(src, 0u, c.x, c.y + 1).x - face_tap(src, 0u, c.x, c.y - 1).x
           + face_tap(src, 0u, c.x + 1, c.y + 1).x - face_tap(src, 0u, c.x + 1, c.y - 1).x;
    w = 0.25 * P.cells_per_h * (dv - du);
  }
  textureStore(curl_dst, c, vec4f(w, 0.0, 0.0, 0.0));
}

// |curl| blurred with the 3 x 3 binomial kernel [1 2 1; 2 4 2; 1 2 1] / 16. Confinement follows the
// GRADIENT of this, not of the raw curl: the raw gradient of a cell-scale ripple points at a different
// cell every step and confinement amplifies whatever it points at, which grows the ripple into a
// lattice of spurious vortices. The blur keeps the vortices it can resolve (three cells and more)
// and turns the force away from the noise.
@compute @workgroup_size(16, 16)
fn cs_curl_smooth(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.grid)) {
    return;
  }
  let c = vec2i(id.xy);
  let centre = abs(textureLoad(curl_tex, c, 0).x);
  var sum = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let w = f32((2 - abs(dx)) * (2 - abs(dy)));
      let q = c + vec2i(dx, dy);
      var value = centre;
      if (cell_class(q) == CELL_FLUID) {
        value = abs(textureLoad(curl_tex, cell_wrapped(q), 0).x);
      }
      sum += w * value;
    }
  }
  textureStore(smooth_dst, c, vec4f(sum / 16.0, 0.0, 0.0, 0.0));
}

// ---- Forces --------------------------------------------------------------------------------------------

// |curl| of a neighbour; a solid or open neighbour borrows the centre's value (zero gradient).
fn curl_or(c: vec2i, fallback: f32) -> f32 {
  if (cell_class(c) != CELL_FLUID) {
    return fallback;
  }
  return textureLoad(curl_tex, cell_wrapped(c), 0).x;
}

fn smooth_or(c: vec2i, fallback: f32) -> f32 {
  if (cell_class(c) != CELL_FLUID) {
    return fallback;
  }
  return textureLoad(smooth_tex, cell_wrapped(c), 0).x;
}

// Vorticity confinement (Fedkiw, Stam, Jensen 2001): an acceleration eps * h * (N x w) that points
// round the nearest vortex centre in the sense it already turns, with N the unit gradient of |w|.
// It puts back the small-scale rotation that interpolation in advection smooths away. h is the cell
// size in image heights, so the force vanishes with the grid spacing.
fn confinement(c: vec2i) -> vec2f {
  let w = curl_or(c, 0.0);
  let a = abs(w);
  let here = smooth_or(c, 0.0);
  let gx = 0.5 * (smooth_or(c + vec2i(1, 0), here) - smooth_or(c + vec2i(-1, 0), here));
  let gy = 0.5 * (smooth_or(c + vec2i(0, 1), here) - smooth_or(c + vec2i(0, -1), here));
  let len = sqrt(gx * gx + gy * gy);
  if (len < 1e-4 || a < 1e-3) {
    return vec2f(0.0);
  }
  let n = vec2f(gx, gy) / len;
  return P.vorticity * w * vec2f(n.y, -n.x) / P.cells_per_h;
}

fn face_confinement(axis: u32, f: vec2i) -> f32 {
  var cells = face_cells(axis, f);
  var sum = 0.0;
  var n = 0.0;
  for (var k = 0; k < 2; k++) {
    if (cell_class(cells[k]) == CELL_FLUID) {
      let a = confinement(cell_wrapped(cells[k]));
      sum += select(a.y, a.x, axis == 0u);
      n += 1.0;
    }
  }
  return sum / max(n, 1.0);
}

fn forced_velocity(axis: u32, c: vec2i) -> f32 {
  let f = face_of(axis, c);
  if (face_state(axis, f) != FACE_FREE) {
    return 0.0;
  }
  let a = select(P.accel.y, P.accel.x, axis == 0u);
  var accel = a;
  if (P.buoyancy != 0.0) {
    let uv = face_position(axis, f) / vec2f(P.grid);
    let density = textureSampleLevel(dye_tex, samp, uv, 0.0).a;
    accel += P.buoyancy * density * select(P.buoy_dir.y, P.buoy_dir.x, axis == 0u);
  }
  if (P.vorticity != 0.0) {
    accel += face_confinement(axis, f);
  }
  let v = (face_tap(src, axis, f.x, f.y).x + accel * P.dt) * P.vel_decay;
  return clamp(finite_or_zero(v), -P.vmax, P.vmax);
}

@compute @workgroup_size(16, 16)
fn cs_forces(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.grid + vec2u(1u))) {
    return;
  }
  let c = vec2i(id.xy);
  textureStore(dst, c, vec4f(forced_velocity(0u, c), forced_velocity(1u, c), 0.0, 0.0));
}

// ---- Viscosity -----------------------------------------------------------------------------------------

// One Jacobi iteration of the implicit diffusion (1 - nu dt laplacian) u = base, with
// a = nu dt / h^2 in P.diffuse_a. Neighbours inside a solid are left out (zero normal gradient).
fn diffused(axis: u32, c: vec2i) -> f32 {
  let f = face_of(axis, c);
  if (face_state(axis, f) != FACE_FREE) {
    return 0.0;
  }
  let b = face_tap(base, axis, f.x, f.y).x;
  let t0 = face_tap(src, axis, f.x - 1, f.y);
  let t1 = face_tap(src, axis, f.x + 1, f.y);
  let t2 = face_tap(src, axis, f.x, f.y - 1);
  let t3 = face_tap(src, axis, f.x, f.y + 1);
  let sum = t0.x * t0.y + t1.x * t1.y + t2.x * t2.y + t3.x * t3.y;
  let n = t0.y + t1.y + t2.y + t3.y;
  return (b + P.diffuse_a * sum) / (1.0 + P.diffuse_a * n);
}

@compute @workgroup_size(16, 16)
fn cs_diffuse(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.grid + vec2u(1u))) {
    return;
  }
  let c = vec2i(id.xy);
  textureStore(dst, c, vec4f(diffused(0u, c), diffused(1u, c), 0.0, 0.0));
}
