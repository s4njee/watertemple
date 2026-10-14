// sph-core.wgsl -- the parts of the SPH solver that the density pass and the force pass share: the walls (box sides
// and SDF obstacles), the tables that stand in for a continuum of fluid behind them, and the equation of state.
// Included by sph-density.wgsl and sph-force.wgsl AFTER they declare `var<uniform> params: SphParams`
// (src/gpu/sim/sph.ts generates the struct; see the header of sph.ts for what every field means).
//
// WALLS ARE A CONTINUUM, NOT A LAYER OF PARTICLES. A particle near a wall is missing the neighbours that would be
// behind it, so its summed density is too low and the pressure that should hold it off the wall vanishes: the usual
// SPH artefacts of particles crushed against the floor in a dense crust, or a gap. Instead of ghost or boundary
// particles, the wall is treated as fluid at rest density rho0 filling the half-plane behind it, and the two
// quantities the missing neighbours would have contributed are integrated analytically over that half-plane:
//   - density:  rho_i += rho0 * G(d / h),  G(q) = integral over the half-plane beyond distance d of W_poly6.
//               G(0) = 1/2 (half the kernel is behind a wall you touch), G(1) = 0.
//   - pressure: a_i += 2 rho0 p_i / rho_i^2 * F(d / h) / h * n,  F(q) / h = the integral along a line at distance d of
//               W_spiky (by the divergence theorem the half-plane integral of grad W is exactly that line integral
//               times the normal n, pointing from the wall into the fluid). The factor 2 is the pressure of the
//               mirrored fluid behind the wall taken equal to p_i.
// G and F are tabulated on the CPU (33 samples, linear interpolation) by numerical quadrature: tests/sph.test.ts
// checks them against an independent quadrature. d is the distance from the particle to the wall surface, from
// the box sides or from an obstacle's SDF; the SDF is used as if the wall were locally flat, which is accurate
// while the obstacle's radius of curvature is several kernel radii.

#include "sdf.wgsl"

// Obstacles are two vec4f each: (kind, cx, cy, 0) then (a, b, c, 0). kind 1 = circle of radius a; kind 2 = rounded
// box of half size (a, b) and corner radius c.
fn sph_obstacle_sd(k: u32, p: vec2f) -> f32 {
  let head = params.obstacles[2u * k];
  let tail = params.obstacles[2u * k + 1u];
  let q = p - head.yz;
  if (head.x < 1.5) {
    return sd_circle(q, tail.x);
  }
  return sd_rounded_box_2d(q, tail.xy, tail.z);
}

fn sph_obstacle_normal(k: u32, p: vec2f) -> vec2f {
  let e = 0.05 * params.h;
  let g = vec2f(
    sph_obstacle_sd(k, p + vec2f(e, 0.0)) - sph_obstacle_sd(k, p - vec2f(e, 0.0)),
    sph_obstacle_sd(k, p + vec2f(0.0, e)) - sph_obstacle_sd(k, p - vec2f(0.0, e)),
  );
  let l = length(g);
  return select(vec2f(0.0, 1.0), g / l, l > 1e-12);
}

fn sph_table_gamma(q: f32) -> f32 {
  let x = clamp(q, 0.0, 1.0) * 32.0;
  let k = min(u32(floor(x)), 31u);
  let f = x - f32(k);
  let k1 = k + 1u;
  return mix(params.wall_gamma[k >> 2u][k & 3u], params.wall_gamma[k1 >> 2u][k1 & 3u], f);
}

fn sph_table_force(q: f32) -> f32 {
  let x = clamp(q, 0.0, 1.0) * 32.0;
  let k = min(u32(floor(x)), 31u);
  let f = x - f32(k);
  let k1 = k + 1u;
  return mix(params.wall_force[k >> 2u][k & 3u], params.wall_force[k1 >> 2u][k1 & 3u], f);
}

// Density the walls add at p: rho0 * sum of G over every wall within h.
fn sph_wall_density(p: vec2f) -> f32 {
  let h = params.h;
  let inv_h = 1.0 / h;
  let sides = vec4f(
    p.x - params.box_min.x,
    params.box_max.x - p.x,
    p.y - params.box_min.y,
    params.box_max.y - p.y,
  );
  var g = 0.0;
  for (var w = 0u; w < 4u; w++) {
    if (sides[w] < h) {
      g += sph_table_gamma(max(sides[w], 0.0) * inv_h);
    }
  }
  for (var k = 0u; k < params.obstacle_count; k++) {
    let d = sph_obstacle_sd(k, p);
    if (d < h) {
      g += sph_table_gamma(max(d, 0.0) * inv_h);
    }
  }
  return g * params.rest_density;
}

// The wall term of the pressure force without its pre-factor 2 rho0 p_i / rho_i^2: sum of F(d / h) / h * n.
fn sph_wall_gradient(p: vec2f) -> vec2f {
  let h = params.h;
  let inv_h = 1.0 / h;
  let sides = vec4f(
    p.x - params.box_min.x,
    params.box_max.x - p.x,
    p.y - params.box_min.y,
    params.box_max.y - p.y,
  );
  var a = vec2f(0.0);
  if (sides.x < h) { a.x += sph_table_force(max(sides.x, 0.0) * inv_h); }
  if (sides.y < h) { a.x -= sph_table_force(max(sides.y, 0.0) * inv_h); }
  if (sides.z < h) { a.y += sph_table_force(max(sides.z, 0.0) * inv_h); }
  if (sides.w < h) { a.y -= sph_table_force(max(sides.w, 0.0) * inv_h); }
  for (var k = 0u; k < params.obstacle_count; k++) {
    let d = sph_obstacle_sd(k, p);
    if (d < h) {
      a += sph_obstacle_normal(k, p) * sph_table_force(max(d, 0.0) * inv_h);
    }
  }
  return a * inv_h;
}

// Equation of state (Tait): p = B ((rho / rho0)^gamma - 1), B = rho0 c^2 / gamma, so that the speed of sound at rest
// is c. Negative pressure is clamped to 0: below rest density the fluid has no tensile strength, which is what stops
// particles at a free surface from clumping into pairs. The density ratio is capped at `pressure_cap` so a block
// injected on top of water, or any overlap, pushes hard but not infinitely.
fn sph_pressure(rho: f32) -> f32 {
  let ratio = min(rho / params.rest_density, params.pressure_cap);
  if (ratio <= 1.0) {
    return 0.0;
  }
  return params.tait_b * (pow(ratio, params.gamma) - 1.0);
}
