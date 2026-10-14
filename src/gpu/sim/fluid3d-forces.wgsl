// fluid3d-forces.wgsl -- everything that adds to the fluid between advection and projection: splats of
// momentum and ink, the cell-centred velocity, the vorticity (curl) field, vorticity confinement, and
// the body forces (gravity, buoyancy from the ink).
//
// Bindings (besides the shared 0 `P`); an entry point uses only some of them:
//   2 src       the texture being updated: velocity ((nx+1) x (ny+1) x (nz+1)), the scalar for cs_splat_dye,
//               or the cell-centred velocity for cs_curl
//   3 dst       its rgba16float storage output
//   4 splats    read-only storage buffer of `Splat`s (P.force_count / P.dye_count of them)
//   5 curl_tex  rgba16float cells: vorticity (rgb, 1 / s) and its magnitude (a)   (cs_blur)
//   6 dens_tex  the scalar field, for buoyancy
//   7 samp      a filtering sampler for dens_tex
//   8 blur_dst  r32float storage output of cs_blur
//   9 blur_tex  |vorticity| blurred over its 7-point neighbourhood (cs_conf)
//  10 conf_dst  rgba16float storage output of cs_conf
//  11 conf_tex  the confinement acceleration at cell centres, read by cs_forces
//  12 curl_dst  rgba16float storage output of cs_curl

#include "fluid3d-core.wgsl"

@group(0) @binding(2) var src: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var<storage, read> splats: array<Splat>;
@group(0) @binding(5) var curl_tex: texture_3d<f32>;
@group(0) @binding(6) var dens_tex: texture_3d<f32>;
@group(0) @binding(7) var samp: sampler;
@group(0) @binding(8) var blur_dst: texture_storage_3d<r32float, write>;
@group(0) @binding(9) var blur_tex: texture_3d<f32>;
@group(0) @binding(10) var conf_dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(11) var conf_tex: texture_3d<f32>;
@group(0) @binding(12) var curl_dst: texture_storage_3d<rgba16float, write>;

fn in_velocity_grid(id: vec3u) -> bool {
  return all(id < P.grid + vec3u(1u));
}

fn in_grid(id: vec3u) -> bool {
  return all(id < P.grid);
}

// ---- Splats -----------------------------------------------------------------------------------------------------

fn splat_velocity(axis: u32, t: vec3i) -> f32 {
  let f = face_of(axis, t);
  if (face_blocked(axis, f)) {
    return 0.0;
  }
  let p = face_position(axis, f) * P.h;
  var dv = 0.0;
  for (var k = 0u; k < P.force_count; k++) {
    let s = splats[k];
    dv += splat_weight(p, s) * s.value[axis];
  }
  return textureLoad(src, f, 0)[axis] + dv;
}

@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn cs_splat_velocity(@builtin(global_invocation_id) id: vec3u) {
  if (!in_velocity_grid(id)) {
    return;
  }
  let t = vec3i(id);
  let v = vec3f(splat_velocity(0u, t), splat_velocity(1u, t), splat_velocity(2u, t));
  textureStore(dst, t, vec4f(clamp(finite3(v), vec3f(-P.vmax), vec3f(P.vmax)), 0.0));
}

@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn cs_splat_dye(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= P.scalar_size)) {
    return;
  }
  let t = vec3i(id);
  if (scalar_solid(t)) {
    textureStore(dst, t, vec4f(0.0));
    return;
  }
  let p = (vec3f(t) + 0.5) / vec3f(P.scalar_size) * P.size;
  var d = textureLoad(src, t, 0);
  for (var k = 0u; k < P.dye_count; k++) {
    let s = splats[k];
    d += splat_weight(p, s) * s.value;
  }
  textureStore(dst, t, clamp(finite4(d), vec4f(0.0), vec4f(P.dye_max)));
}

// ---- Vorticity ------------------------------------------------------------------------------------------------------

// Velocity at a cell centre: the mean of its two faces along each axis. Written to `fields`, the
// texture consumers (debug views) read, and read by cs_curl.
@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn cs_center(@builtin(global_invocation_id) id: vec3u) {
  if (!in_grid(id)) {
    return;
  }
  let c = vec3i(id);
  let u = 0.5 * (textureLoad(src, c, 0).x + textureLoad(src, c + vec3i(1, 0, 0), 0).x);
  let v = 0.5 * (textureLoad(src, c, 0).y + textureLoad(src, c + vec3i(0, 1, 0), 0).y);
  let w = 0.5 * (textureLoad(src, c, 0).z + textureLoad(src, c + vec3i(0, 0, 1), 0).z);
  textureStore(dst, c, vec4f(u, v, w, 0.0));
}

// Centre velocity of a neighbour; outside the box or in the obstacle it borrows the cell's own (zero gradient).
fn centre_or(c: vec3i, fallback: vec3f) -> vec3f {
  if (cell_solid(c)) {
    return fallback;
  }
  return textureLoad(src, c, 0).xyz;
}

// Vorticity (curl of the velocity) at a cell centre in 1 / s from central differences of the centre
// velocities, and its magnitude in the alpha channel.
@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn cs_curl(@builtin(global_invocation_id) id: vec3u) {
  if (!in_grid(id)) {
    return;
  }
  let c = vec3i(id);
  var w = vec3f(0.0);
  if (!cell_solid(c)) {
    let here = textureLoad(src, c, 0).xyz;
    let xp = centre_or(c + vec3i(1, 0, 0), here);
    let xm = centre_or(c + vec3i(-1, 0, 0), here);
    let yp = centre_or(c + vec3i(0, 1, 0), here);
    let ym = centre_or(c + vec3i(0, -1, 0), here);
    let zp = centre_or(c + vec3i(0, 0, 1), here);
    let zm = centre_or(c + vec3i(0, 0, -1), here);
    w = 0.5 * P.inv_h * vec3f(
      (yp.z - ym.z) - (zp.y - zm.y),
      (zp.x - zm.x) - (xp.z - xm.z),
      (xp.y - xm.y) - (yp.x - ym.x),
    );
  }
  textureStore(curl_dst, c, vec4f(w, length(w)));
}

fn curl_mag(c: vec3i, fallback: f32) -> f32 {
  if (cell_solid(c)) {
    return fallback;
  }
  return textureLoad(curl_tex, c, 0).w;
}

// |vorticity| blurred over the cell and its six neighbours. Confinement follows the GRADIENT of this, not
// of the raw magnitude: the raw gradient of a cell-scale ripple points at a different cell every step
// and confinement amplifies whatever it points at, which grows the ripple into a lattice of spurious
// vortices (the 2D solver showed this). The blur keeps the vortices it can resolve and turns the force
// away from the noise.
@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn cs_blur(@builtin(global_invocation_id) id: vec3u) {
  if (!in_grid(id)) {
    return;
  }
  let c = vec3i(id);
  let here = curl_mag(c, 0.0);
  let sum = 2.0 * here
    + curl_mag(c + vec3i(1, 0, 0), here) + curl_mag(c + vec3i(-1, 0, 0), here)
    + curl_mag(c + vec3i(0, 1, 0), here) + curl_mag(c + vec3i(0, -1, 0), here)
    + curl_mag(c + vec3i(0, 0, 1), here) + curl_mag(c + vec3i(0, 0, -1), here);
  textureStore(blur_dst, c, vec4f(sum / 8.0, 0.0, 0.0, 0.0));
}

fn blur_or(c: vec3i, fallback: f32) -> f32 {
  if (cell_solid(c)) {
    return fallback;
  }
  return textureLoad(blur_tex, c, 0).x;
}

// Vorticity confinement (Fedkiw, Stam, Jensen 2001): an acceleration eps * h * (N x w) that points round the
// nearest vortex line in the sense it already turns, with N the unit gradient of |w|. It puts back the
// small-scale rotation that interpolation in advection smooths away, which in 3D is the rolled-up ring of
// a falling drop and the tendrils that come off it. h is the cell size, so the force vanishes with the
// grid spacing.
@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn cs_conf(@builtin(global_invocation_id) id: vec3u) {
  if (!in_grid(id)) {
    return;
  }
  let c = vec3i(id);
  var a = vec3f(0.0);
  if (!cell_solid(c)) {
    let here = blur_or(c, 0.0);
    let g = 0.5 * vec3f(
      blur_or(c + vec3i(1, 0, 0), here) - blur_or(c + vec3i(-1, 0, 0), here),
      blur_or(c + vec3i(0, 1, 0), here) - blur_or(c + vec3i(0, -1, 0), here),
      blur_or(c + vec3i(0, 0, 1), here) - blur_or(c + vec3i(0, 0, -1), here),
    );
    let w = textureLoad(curl_tex, c, 0).xyz;
    let len = length(g);
    if (len > 1e-4 && length(w) > 1e-3) {
      a = P.vorticity * P.h * cross(g / len, w);
    }
  }
  textureStore(conf_dst, c, vec4f(a, 0.0));
}

// ---- Forces ------------------------------------------------------------------------------------------------------------

// The confinement acceleration on a face: the mean of the two cells it separates that are in the box.
fn face_confinement(axis: u32, f: vec3i) -> f32 {
  let g = vec3i(P.grid);
  let a = axis_step(axis);
  var sum = 0.0;
  var n = 0.0;
  let lo = f - a;
  if (!cell_solid(lo)) {
    sum += textureLoad(conf_tex, lo, 0)[axis];
    n += 1.0;
  }
  if (!cell_solid(f)) {
    sum += textureLoad(conf_tex, f, 0)[axis];
    n += 1.0;
  }
  return sum / max(n, 1.0);
}

fn forced_velocity(axis: u32, t: vec3i) -> f32 {
  let f = face_of(axis, t);
  if (face_blocked(axis, f)) {
    return 0.0;
  }
  var accel = P.accel[axis];
  if (P.buoyancy != 0.0) {
    let density = textureSampleLevel(dens_tex, samp, face_position(axis, f) / vec3f(P.grid), 0.0).a;
    accel += P.buoyancy * density * P.buoy_dir[axis];
  }
  if (P.vorticity != 0.0) {
    accel += face_confinement(axis, f);
  }
  let v = (textureLoad(src, f, 0)[axis] + accel * P.dt) * P.vel_decay;
  return clamp(finite_or_zero(v), -P.vmax, P.vmax);
}

@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn cs_forces(@builtin(global_invocation_id) id: vec3u) {
  if (!in_velocity_grid(id)) {
    return;
  }
  let t = vec3i(id);
  textureStore(dst, t, vec4f(forced_velocity(0u, t), forced_velocity(1u, t), forced_velocity(2u, t), 0.0));
}
