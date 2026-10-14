// mpm-grid.wgsl -- the grid update of an MLS-MPM step (src/gpu/sim/mpm.ts): decode the fixed-point sums, turn
// momentum into velocity, add gravity and enforce the boundaries. One invocation per node.
//
//     v_i = q_i / m_i        (both are fixed point with the same scale, so the scale cancels)
//     v_i *= exp(-damping dt); v_i += g dt
//     colliders, then the container walls, then |v_i| <= max_speed.
//
// CONTAINER WALLS are the faces of the axis-aligned box [bounds_min, bounds_max] and they move: wall_vmin and
// wall_vmax are the velocities of the six faces. Every node on or beyond a face is a wall node: its velocity
// along the wall normal is kept from going below the wall's own (a retreating wall does not pull the fluid
// along, a wall moving into the fluid pushes it), and its tangential velocity is blended towards the wall's by
// `friction` (0 slip, 1 stick). Particles are clamped to the box in the G2P pass.
//
// COLLIDERS are signed-distance shapes (sphere, box, plane) that may move. A node inside one loses the part of
// its velocity (relative to the collider) that points into it, and the tangential part is damped by the
// collider's friction.
//
// The result goes to gridVel (xyz velocity, w mass in particle units) for the G2P pass; the fixed-point grid is
// cleared by the next step.

#include "mpm-params.wgsl"
#include "mpm-core.wgsl"

// Workgroup size; mpm.ts rewrites this line and @workgroup_size together.
const WG = 64u;

@group(0) @binding(0) var<uniform> P: MpmParams;
@group(0) @binding(1) var<storage, read> grid: array<vec4<i32>>;
@group(0) @binding(2) var<storage, read_write> gridVel: array<vec4f>;

fn wall_axis(v_in: vec3f, axis: i32, is_max: bool, wall_velocity: vec3f) -> vec3f {
  var v = v_in;
  let normal_speed = wall_velocity[axis];
  if (is_max) {
    v[axis] = min(v[axis], normal_speed);
  } else {
    v[axis] = max(v[axis], normal_speed);
  }
  for (var t = 0; t < 3; t++) {
    if (t != axis) {
      v[t] = mix(v[t], wall_velocity[t], P.friction);
    }
  }
  return v;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let node = gid.x + gid.y * (groups.x * WG);
  let nx = P.dims.x;
  let ny = P.dims.y;
  if (node >= nx * ny * P.dims.z) {
    return;
  }
  let cell = grid[node];
  let mass = cell.x;
  // Negative mass means the fixed-point sum wrapped: treat the node as empty rather than trust it.
  if (mass <= 0) {
    gridVel[node] = vec4f(0.0);
    return;
  }

  var v = vec3f(f32(cell.y), f32(cell.z), f32(cell.w)) / f32(mass);
  v *= P.damping_factor;
  v += P.gravity * P.dt;

  let ijk = vec3u(node % nx, (node / nx) % ny, node / (nx * ny));
  let xn = vec3f(ijk) * P.h;

  for (var c = 0u; c < P.collider_count; c++) {
    let a = P.colliders[c * 3u];
    let b = P.colliders[c * 3u + 1u];
    let u = P.colliders[c * 3u + 2u].xyz;
    let s = mpm_collider_sdf(a, b, xn);
    if (s.w < 0.0) {
      var rel = v - u;
      let vn = dot(rel, s.xyz);
      if (vn < 0.0) {
        rel = (rel - vn * s.xyz) * (1.0 - clamp(b.w, 0.0, 1.0));
      }
      v = u + rel;
    }
  }

  for (var axis = 0; axis < 3; axis++) {
    if (xn[axis] <= P.bounds_min[axis]) {
      v = wall_axis(v, axis, false, P.wall_vmin);
    }
    if (xn[axis] >= P.bounds_max[axis]) {
      v = wall_axis(v, axis, true, P.wall_vmax);
    }
  }

  v = mpm_finite3(v);
  let speed = length(v);
  if (speed > P.max_speed) {
    v *= P.max_speed / speed;
  }
  gridVel[node] = vec4f(v, mpm_from_fixed(mass));
}
