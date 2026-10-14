// pbf-core.wgsl -- what every pass of the position-based fluid solver shares (story 7.4): the two uniform blocks,
// the container and the colliders, the analytic wall density and the pointer impulses. Included by the pbf-*.wgsl
// pass files, which declare their own storage bindings; `pbf_params.wgsl` is generated from the uniform structs in
// pbf.ts (PbfParams: one block per update, PbfStep: one block per sub-step).
//
// Bindings 0 and 1 are the same in every pass: params (all passes) and tick (the passes that need the container or
// the step length). See pbf.ts for the numbering of the rest.
//
// THE CONTAINER. An axis-aligned box [tick.bounds_min, tick.bounds_max], the inner surface of the glass. The centre of
// a particle may come no closer to a wall than params.wall_offset (half a particle spacing: a lattice of spacing s
// that fills the box has its outer layer s/2 from the glass). The box is part of the tick uniform because story 7.8
// moves it: a wall that advances pushes the particles in front of it by position projection, a wall that retreats
// leaves them to fall. The projection is a clamp, so a particle can never end a step outside the box, however far the
// wall moved or the particle was thrown.
//
// COLLIDERS. Up to 8 static spheres and boxes, two vec4 each: (centre or min, radius) and (max, kind), kind 1 sphere,
// 2 box, 0 unused. A particle inside one (grown by wall_offset) is pushed out along the shortest way.
//
// FRICTION. A contact removes up to `friction * penetration depth` of the particle's tangential displacement over the
// step (Coulomb's law with position-based quantities: the penetration depth stands in for the normal force). A particle
// that merely touches a wall slides freely; one pressed into it, as the bottom of a deep column is, is held back.
//
// WALL DENSITY. A wall hides the fluid that would be behind it, so a particle near a wall sees a density deficit that
// the solver would otherwise read as "free surface" and answer with an attraction towards the fluid. params carries two
// tables (33 samples, d = 0 .. h) computed on the CPU: the density of the missing half-space of fluid at distance d
// from the particle (an integral of Poly6 over the half-space, times the rest number density), and the gradient of the
// same with Spiky. pbf_wall_terms sums them over the walls (2 or 3 axes, both sides). It is exact for a flat wall and
// double counts a little in a corner.

#include "pbf-kernels.wgsl"
#include "pbf_params.wgsl"

@group(0) @binding(0) var<uniform> params: PbfParams;
@group(0) @binding(1) var<uniform> tick: PbfStep;

struct PbfWall {
  // Number density of the missing fluid beyond the walls, in the units of the kernel sums.
  density: f32,
  // Sum over the walls of the gradient of that density with respect to the particle, pointing at the wall.
  grad: vec3f,
}

fn pbf_table_density(d: f32) -> f32 {
  let x = clamp(d / params.h, 0.0, 1.0) * 32.0;
  let k = min(u32(x), 31u);
  let f = x - f32(k);
  let a = params.wall_density[k >> 2u][k & 3u];
  let b = params.wall_density[(k + 1u) >> 2u][(k + 1u) & 3u];
  return mix(a, b, f);
}

fn pbf_table_gradient(d: f32) -> f32 {
  let x = clamp(d / params.h, 0.0, 1.0) * 32.0;
  let k = min(u32(x), 31u);
  let f = x - f32(k);
  let a = params.wall_gradient[k >> 2u][k & 3u];
  let b = params.wall_gradient[(k + 1u) >> 2u][(k + 1u) & 3u];
  return mix(a, b, f);
}

fn pbf_wall_terms(p: vec3f) -> PbfWall {
  var out: PbfWall;
  out.density = 0.0;
  out.grad = vec3f(0.0);
  if (params.boundary_density == 0u) {
    return out;
  }
  let lo = p - tick.bounds_min;
  let hi = tick.bounds_max - p;
  for (var a = 0u; a < params.dimension; a++) {
    // Low wall: the missing fluid lies below, so the density grows as the particle moves down (towards -axis).
    if (lo[a] < params.h) {
      let d = max(lo[a], 0.0);
      out.density += pbf_table_density(d);
      out.grad[a] -= pbf_table_gradient(d);
    }
    if (hi[a] < params.h) {
      let d = max(hi[a], 0.0);
      out.density += pbf_table_density(d);
      out.grad[a] += pbf_table_gradient(d);
    }
  }
  return out;
}

// Pushes x along the unit normal n by `depth`, then removes up to friction * depth of the tangential part of the
// displacement since `prev` (where the particle was when the step began).
fn pbf_contact(x: vec3f, prev: vec3f, n: vec3f, depth: f32, friction: f32) -> vec3f {
  var out = x + n * depth;
  if (friction > 0.0) {
    let moved = out - prev;
    let tangential = moved - n * dot(moved, n);
    let len = length(tangential);
    if (len > 1e-9) {
      out -= tangential * min(1.0, friction * depth / len);
    }
  }
  return out;
}

// Keeps a particle inside the box and outside the colliders. `friction_scale` is 0 for the prediction (the particle
// has not yet been corrected) and 1 for the final position of an iteration.
fn pbf_contain(x_in: vec3f, prev: vec3f, friction_scale: f32) -> vec3f {
  var x = x_in;
  let o = params.wall_offset;
  let friction = params.friction * friction_scale;
  let lo = tick.bounds_min + vec3f(o);
  let hi = tick.bounds_max - vec3f(o);

  for (var c = 0u; c < params.collider_count; c++) {
    let a = params.colliders[c * 2u];
    let b = params.colliders[c * 2u + 1u];
    let kind = u32(b.w + 0.5);
    if (kind == 1u) {
      let d = x - a.xyz;
      let len = length(d);
      let reach = a.w + o;
      if (len < reach) {
        var n = vec3f(0.0, 1.0, 0.0);
        if (len > 1e-6) {
          n = d / len;
        }
        x = pbf_contact(x, prev, n, reach - len, friction);
      }
    } else if (kind == 2u) {
      let below = x - (a.xyz - vec3f(o));
      let above = (b.xyz + vec3f(o)) - x;
      var inside = below.x > 0.0 && above.x > 0.0 && below.y > 0.0 && above.y > 0.0;
      if (params.dimension == 3u) {
        inside = inside && below.z > 0.0 && above.z > 0.0;
      }
      if (inside) {
        // The face of least penetration (the z faces do not exist in 2D).
        var best = below.x;
        var n = vec3f(-1.0, 0.0, 0.0);
        if (above.x < best) { best = above.x; n = vec3f(1.0, 0.0, 0.0); }
        if (below.y < best) { best = below.y; n = vec3f(0.0, -1.0, 0.0); }
        if (above.y < best) { best = above.y; n = vec3f(0.0, 1.0, 0.0); }
        if (params.dimension == 3u) {
          if (below.z < best) { best = below.z; n = vec3f(0.0, 0.0, -1.0); }
          if (above.z < best) { best = above.z; n = vec3f(0.0, 0.0, 1.0); }
        }
        x = pbf_contact(x, prev, n, best, friction);
      }
    }
  }

  // The walls last, so that nothing a collider did can leave the box. A box narrower than twice the offset
  // collapses to its middle instead of inverting.
  let mid = (tick.bounds_min + tick.bounds_max) * 0.5;
  let lo_c = min(lo, mid);
  let hi_c = max(hi, mid);
  for (var a = 0u; a < params.dimension; a++) {
    if (x[a] < lo_c[a]) {
      var n = vec3f(0.0);
      n[a] = 1.0;
      x = pbf_contact(x, prev, n, lo_c[a] - x[a], friction);
    } else if (x[a] > hi_c[a]) {
      var n = vec3f(0.0);
      n[a] = -1.0;
      x = pbf_contact(x, prev, n, x[a] - hi_c[a], friction);
    }
  }
  // Friction moves a particle sideways, possibly past another wall: the clamp is the guarantee.
  x = clamp(x, lo_c, hi_c);
  if (params.dimension == 2u) {
    x.z = 0.0;
  }
  return x;
}

// The pointer impulses: a blend of the velocity towards the pointer's, and a radial burst, inside a smooth falloff.
fn pbf_impulses(p: vec3f, v_in: vec3f) -> vec3f {
  var v = v_in;
  for (var k = 0u; k < params.impulse_count; k++) {
    let a = params.impulses[k * 3u];
    let b = params.impulses[k * 3u + 1u];
    let c = params.impulses[k * 3u + 2u];
    let d = p - a.xyz;
    let q = length(d) / max(a.w, 1e-6);
    if (q < 1.0) {
      let w = (1.0 - q * q) * (1.0 - q * q);
      v += w * (b.w * (b.xyz - v) + c.x * d / max(length(d), 1e-6));
    }
  }
  return v;
}

// A finite value or the fallback: comparisons with NaN are false, so this also catches NaN without isnan.
fn pbf_finite3(v: vec3f, fallback: vec3f) -> vec3f {
  let ok = all(abs(v) < vec3f(1e7));
  return select(fallback, v, ok);
}
