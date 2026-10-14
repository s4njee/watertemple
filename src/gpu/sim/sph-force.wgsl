// SPH pass 2: forces, integration and boundaries, fused. Every particle reads the CURRENT positions, velocities and
// (rho, p) of its neighbours and writes its new position and velocity to separate output buffers (the host swaps
// them), so the result does not depend on the order in which particles are processed.
//
// Acceleration of particle i, summed over neighbours j within h (m is the particle mass, d = x_i - x_j, r = |d|):
//   pressure      a -= m (p_i / rho_i^2 + p_j / rho_j^2) grad_i W_spiky                  symmetric: conserves momentum
//   viscosity     a += nu m 2 / (rho_i + rho_j) (v_j - v_i) Laplacian W_visc(r)         laminar, equal and opposite
//   artificial    a -= m Pi_ij grad_i W_spiky, Pi_ij = alpha c h_s |v.d| / (rho_ij (r^2 + 0.01 h_s^2)) for approaching
//                 pairs only (Monaghan): damps the sound waves of the weakly compressible fluid, equal and opposite
//   cohesion      a -= kappa (m / rho0) / h^2 phi(r / h) d / r                            attraction, equal and opposite
//   walls         a += 2 rho0 p_i / rho_i^2 sum F(d / h) / h n                            see sph-core.wgsl
//   gravity, a pointer drag or attractor, and a linear damping.
// Then semi-implicit (symplectic) Euler:  v += a dt;  v /= 1 + damping dt;  |v| <= speed_max;  x += v dt, and a
// hard projection back into the box and out of the obstacles with restitution and friction, which is the safety net
// that guarantees no particle ever leaves the domain whatever the forces did.

#include "sph_params.wgsl"
#include "sph-kernels.wgsl"
#include "neighbor_bindings.wgsl"

@group(0) @binding(0) var<uniform> params: SphParams;
@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read> velocities: array<vec4f>;
@group(0) @binding(3) var<storage, read> state: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> positions_out: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> velocities_out: array<vec4f>;
// The grid's bindings (neighbor_bindings.wgsl) start at binding 6.

#include "sph-core.wgsl"

fn finite2(v: vec2f) -> bool {
  return all(abs(v) < vec2f(1e18));
}

// A pointer is two vec4f: (x, y, vx, vy) then (radius, strength, mode, 0). Mode 0 drags the fluid along with the
// pointer (strength is a rate in 1/s: the fluid is pulled toward the pointer's velocity, with an outward share that
// grows with the pointer's speed so a fast poke parts the water); mode 1 attracts (strength in m/s^2).
fn pointer_accel(p: vec2f, v: vec2f) -> vec2f {
  var a = vec2f(0.0);
  for (var k = 0u; k < params.pointer_count; k++) {
    let head = params.pointers[2u * k];
    let tail = params.pointers[2u * k + 1u];
    let d = p - head.xy;
    let r = length(d);
    let radius = tail.x;
    if (r >= radius || radius <= 0.0) { continue; }
    let q = 1.0 - r / radius;
    let w = q * q;
    if (tail.z < 0.5) {
      let speed = length(head.zw);
      let away = select(vec2f(0.0), d / max(r, 1e-6), r > 1e-6);
      let target_velocity = head.zw + away * (0.5 * speed);
      a += tail.y * w * (target_velocity - v);
    } else {
      let toward = select(vec2f(0.0), -d / max(r, 1e-6), r > 1e-6);
      a += toward * tail.y * w - v * (4.0 * w);
    }
  }
  return a;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.count) { return; }
  let p = positions[i].xy;
  let v = velocities[i].xy;
  let st = state[i];
  let rho_i = st.x;
  let p_i = st.y;
  let pr_i = p_i / (rho_i * rho_i);
  let h = params.h;
  let h2 = h * h;
  let m = params.mass;
  let h_sml = 0.5 * h;

  var acc = params.gravity;
  let c = neighbor_cell_of(neighbor_grid, vec3f(p, 0.0));
  for (var row = 0u; row < neighbor_grid.rows; row++) {
    let span = neighbor_row_span(c, row);
    for (var k = span.x; k < span.y; k++) {
      let j = neighbor_particle(k);
      if (j == i) { continue; }
      var d = p - positions[j].xy;
      let r2 = dot(d, d);
      if (r2 >= h2) { continue; }
      var r = sqrt(r2);
      if (r < 1e-5 * h) {
        r = 1e-5 * h;
        d = sph_coincident_direction(i, j) * r;
      }
      let dir = d / r;
      let sj = state[j];
      let rho_j = sj.x;
      let vj = velocities[j].xy;
      let grad = sph_spiky_grad_2d(d, h);

      acc -= m * (pr_i + sj.y / (rho_j * rho_j)) * grad;

      let rho_mean = 0.5 * (rho_i + rho_j);
      acc += (params.viscosity * m / rho_mean) * (vj - v) * sph_viscosity_laplacian_2d(r, h);

      let approach = dot(v - vj, d);
      if (approach < 0.0) {
        let pi_ij = params.artificial_viscosity * params.sound_speed * h_sml * (-approach)
          / (rho_mean * (r2 + 0.01 * h_sml * h_sml));
        acc -= m * pi_ij * grad;
      }

      acc -= dir * (params.cohesion * sph_cohesion(r, h));
    }
  }
  acc += (2.0 * params.rest_density * pr_i) * sph_wall_gradient(p);
  acc += pointer_accel(p, v);

  // Integrate.
  let dt = params.dt;
  var vel = (v + acc * dt) / (1.0 + params.damping * dt);
  let speed_before = length(vel);
  if (speed_before > params.speed_max) {
    vel *= params.speed_max / speed_before;
  }
  var pos = p + vel * dt;

  // The hard boundary: the box, then the obstacles.
  let lo = params.box_min + vec2f(params.margin);
  let hi = params.box_max - vec2f(params.margin);
  if (pos.x < lo.x) { pos.x = lo.x; if (vel.x < 0.0) { vel.x *= -params.restitution; } vel.y *= 1.0 - params.friction; }
  if (pos.x > hi.x) { pos.x = hi.x; if (vel.x > 0.0) { vel.x *= -params.restitution; } vel.y *= 1.0 - params.friction; }
  if (pos.y < lo.y) { pos.y = lo.y; if (vel.y < 0.0) { vel.y *= -params.restitution; } vel.x *= 1.0 - params.friction; }
  if (pos.y > hi.y) { pos.y = hi.y; if (vel.y > 0.0) { vel.y *= -params.restitution; } vel.x *= 1.0 - params.friction; }
  for (var k = 0u; k < params.obstacle_count; k++) {
    let sd = sph_obstacle_sd(k, pos) - params.margin;
    if (sd < 0.0) {
      let n = sph_obstacle_normal(k, pos);
      pos -= n * sd;
      let vn = dot(vel, n);
      if (vn < 0.0) { vel -= (1.0 + params.restitution) * vn * n; }
      let vt = vel - dot(vel, n) * n;
      vel -= params.friction * vt;
    }
  }
  // Whatever happened above, never hand back a number that is not one.
  if (!finite2(pos) || !finite2(vel)) {
    pos = clamp(p, lo, hi);
    vel = vec2f(0.0);
  }
  pos = clamp(pos, lo, hi);

  let speed = length(vel);
  var scalar = rho_i / params.rest_density;
  if (params.scalar_mode == 1u) {
    scalar = p_i / params.pressure_ref;
  } else if (params.scalar_mode == 2u) {
    scalar = speed;
  }
  positions_out[i] = vec4f(pos, 0.0, scalar);
  velocities_out[i] = vec4f(vel, 0.0, speed);
}
