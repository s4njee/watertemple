// mpm-g2p.wgsl -- G2P, the last pass of an MLS-MPM step (src/gpu/sim/mpm.ts): every particle gathers its new
// velocity and affine matrix from the 27 nodes it touches, then moves.
//
//     v_p = sum_i w_ip v_i
//     C_p = (4 / h^2) sum_i w_ip v_i (x_i - x_p)^T        (APIC / MLS: the velocity gradient, to second order)
//     x_p += dt v_p
//
// The particle is then pushed out of every collider and clamped to the container [bounds_min, bounds_max]
// (a particle that reaches a wall loses the velocity into it, up to the wall's own). Anything non-finite is
// replaced by zero and the velocity and C are limited, so no input can put a NaN into the state.
//
// Foam, the 0..1 scalar in positions.w: 1 - smoothstep(foam.x, foam.y, r) for a particle whose density ratio
// r (stored by the stress pass in affine0.w) has fallen to spray level, or smoothstep(foam.z, foam.w, speed) for a
// fast one, whichever is larger, and it fades with `foam_decay` per step once the reason is gone. `aux` is a
// debug scalar for the colour views: 1 speed (m/s), 2 density ratio r, 0 foam.
//
// Bindings: 1 uniform and 7 storage buffers.

#include "mpm-params.wgsl"
#include "mpm-core.wgsl"

@group(0) @binding(0) var<uniform> P: MpmParams;
@group(0) @binding(1) var<storage, read> gridVel: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> positions: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> velocities: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> affine0: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> affine1: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> affine2: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> aux: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let p = gid.x;
  if (p >= P.count) {
    return;
  }
  let position = positions[p];
  var x = position.xyz;

  let dims = vec3i(P.dims);
  let g = x * P.inv_h;
  let base = clamp(vec3i(floor(g - vec3f(0.5))), vec3i(0), dims - vec3i(3));
  let fx = g - vec3f(base);
  let W = mpm_weight_matrix(fx);

  var v = vec3f(0.0);
  var B = mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0));
  for (var k = 0; k < 3; k++) {
    for (var j = 0; j < 3; j++) {
      for (var i = 0; i < 3; i++) {
        let w = W[i].x * W[j].y * W[k].z;
        let d = (vec3f(f32(i), f32(j), f32(k)) - fx) * P.h;
        let node = u32(((base.z + k) * dims.y + (base.y + j)) * dims.x + (base.x + i));
        let gv = gridVel[node].xyz;
        v += w * gv;
        B += mat3x3f(gv * (w * d.x), gv * (w * d.y), gv * (w * d.z));
      }
    }
  }
  var C = B * (4.0 * P.inv_h * P.inv_h);
  v = mpm_finite3(v);
  let speed_limit = P.max_speed;
  let speed = length(v);
  if (speed > speed_limit) {
    v *= speed_limit / speed;
  }
  let c_limit = 4.0 * speed_limit * P.inv_h;
  C = mat3x3f(
    clamp(mpm_finite3(C[0]), vec3f(-c_limit), vec3f(c_limit)),
    clamp(mpm_finite3(C[1]), vec3f(-c_limit), vec3f(c_limit)),
    clamp(mpm_finite3(C[2]), vec3f(-c_limit), vec3f(c_limit)),
  );

  x += P.dt * v;

  // Colliders: out along the normal, and the velocity into them removed.
  for (var c = 0u; c < P.collider_count; c++) {
    let a = P.colliders[c * 3u];
    let b = P.colliders[c * 3u + 1u];
    let u = P.colliders[c * 3u + 2u].xyz;
    let s = mpm_collider_sdf(a, b, x);
    if (s.w < 0.0) {
      x -= s.w * s.xyz;
      var rel = v - u;
      let vn = dot(rel, s.xyz);
      if (vn < 0.0) {
        rel -= vn * s.xyz;
      }
      v = u + rel;
    }
  }

  // The container.
  let eps = 1.0e-3 * P.h;
  let lo = P.bounds_min + vec3f(eps);
  let hi = P.bounds_max - vec3f(eps);
  for (var axis = 0; axis < 3; axis++) {
    if (x[axis] <= lo[axis]) {
      x[axis] = lo[axis];
      v[axis] = max(v[axis], P.wall_vmin[axis]);
    } else if (x[axis] >= hi[axis]) {
      x[axis] = hi[axis];
      v[axis] = min(v[axis], P.wall_vmax[axis]);
    }
  }
  x = clamp(x, lo, hi);
  x = select(position.xyz, x, mpm_is_finite(x.x) && mpm_is_finite(x.y) && mpm_is_finite(x.z));

  // Foam.
  let r = affine0[p].w;
  let now = length(v);
  let spray = 1.0 - smoothstep(P.foam.x, P.foam.y, r);
  let fast = smoothstep(P.foam.z, P.foam.w, now);
  let foam = clamp(max(max(spray, fast), position.w * P.foam_decay), 0.0, 1.0);

  positions[p] = vec4f(x, mpm_finite_or_zero(foam));
  velocities[p] = vec4f(v, 0.0);
  affine0[p] = vec4f(C[0], r);
  affine1[p] = vec4f(C[1], 0.0);
  affine2[p] = vec4f(C[2], 0.0);
  var shown = foam;
  if (P.scalar_mode == 1u) {
    shown = now;
  } else if (P.scalar_mode == 2u) {
    shown = r;
  }
  aux[p] = shown;
}
