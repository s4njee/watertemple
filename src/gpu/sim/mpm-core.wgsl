// mpm-core.wgsl -- the pure maths of the MLS-MPM solver (src/gpu/sim/mpm.ts): B-spline weights, the fixed-point
// encoding of the grid, collider distance functions and a few guards. No bindings and no module-scope
// variables, so a test can paste this file in front of a one-line `fn f(x: vec4f) -> vec4f` and run it.
//
// WEIGHTS. The quadratic B-spline, nodes at integer multiples of h. A particle at grid coordinate g = x / h
// touches the 3 nodes base .. base + 2 on each axis, base = floor(g - 0.5), fx = g - base in [0.5, 1.5):
//     w0 = 0.5 (1.5 - fx)^2     w1 = 0.75 - (fx - 1)^2     w2 = 0.5 (fx - 0.5)^2
// They sum to 1 and reproduce linear fields: sum_o w_o (base + o) = g, so a particle's position is the weighted
// mean of its nodes' positions (checked in tests/mpm.test.ts). `mpm_weight_matrix` returns them as a matrix
// whose column o is the per-axis weight of offset o: the weight of node (i, j, k) is W[i].x * W[j].y * W[k].z.
//
// FIXED POINT. WGSL atomics are integers only, so mass and momentum are summed as i32 with MPM_FIXED_SCALE
// units per 1.0 (2^20). One unit of mass is ONE PARTICLE (every particle has mass 1; the equation of state is
// written in ratios, so the physical rest density never appears). Range: a node holds up to +-2047.99 units of
// mass or of mass * velocity (m/s). That is 255 particles per cell at 8 per cell at rest (32x the rest
// density) and 100 particles' worth of mass moving at 20 m/s. Resolution: one unit in the last place is 9.5e-7,
// so a lone particle moving at 1e-3 m/s (momentum 7.5e-4 at its central node) is held to 0.07 %, and one at 1e-5
// m/s to 6 %. The velocity of a node is a ratio of two fixed-point numbers, so the scale cancels in the decode.
// An i32 wraps on overflow; the grid pass treats negative mass as empty and clamps every velocity it decodes, so
// the failure mode of a clump beyond the range is a wrong velocity bounded by max_speed, never NaN.
//
// Functions
//   mpm_to_fixed(x) -> i32                       round to nearest unit; saturates at the i32 range
//   mpm_from_fixed(i) -> f32
//   mpm_weight_matrix(fx) -> mat3x3f             the three weights per axis, as above
//   mpm_is_finite(x) -> bool                     false for NaN and +-Inf (tests the exponent bits, which a
//                                                compiler cannot fold away as it can `x != x`)
//   mpm_finite_or_zero(x) / mpm_finite3(v)       0 for NaN and +-Inf
//   mpm_kernel_above(t) -> f32                   share of a node's weight function above a wall t cells away
//   mpm_collider_sdf(a, b, x) -> vec4f           (unit outward normal, signed distance) of a collider at x
//   mpm_collider_kind(a) -> i32                  0 none, 1 sphere, 2 box, 3 plane
//
// Colliders are two vec4f: a = (centre or point on the plane, kind), b = (radius, 0, 0 | half extent |
// outward normal, friction); the third vec4f of a collider (xyz velocity) is used by the passes only.

const MPM_FIXED_SCALE: f32 = 1048576.0;
const MPM_INV_FIXED_SCALE: f32 = 1.0 / 1048576.0;

fn mpm_to_fixed(x: f32) -> i32 {
  // f32 -> i32 saturates in WGSL; the explicit clamp keeps that independent of the driver.
  return i32(clamp(round(x * MPM_FIXED_SCALE), -2147483520.0, 2147483520.0));
}

fn mpm_from_fixed(i: i32) -> f32 {
  return f32(i) * MPM_INV_FIXED_SCALE;
}

fn mpm_weight_matrix(fx: vec3f) -> mat3x3f {
  let a = vec3f(1.5) - fx;
  let b = fx - vec3f(1.0);
  let c = fx - vec3f(0.5);
  return mat3x3f(0.5 * a * a, vec3f(0.75) - b * b, 0.5 * c * c);
}

fn mpm_is_finite(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}

// The integral of the quadratic B-spline w(u) (centred on a node, support +-1.5 cells) over u > t, t in cells.
// With a wall t cells from a node and material on the far side, this is how much of the node's kernel the
// material fills: 1 for t <= -1.5, 0 for t >= 1.5. mpm-p2g2.wgsl divides the density estimate by what a fluid
// at rest density filling the container would give, which removes the deficit at the walls.
fn mpm_kernel_above(t: f32) -> f32 {
  if (t <= -1.5) {
    return 1.0;
  }
  if (t <= -0.5) {
    let a = t + 1.5;
    return 1.0 - a * a * a / 6.0;
  }
  if (t <= 0.5) {
    return 0.5 - 0.75 * t + t * t * t / 3.0;
  }
  if (t <= 1.5) {
    let a = 1.5 - t;
    return a * a * a / 6.0;
  }
  return 0.0;
}

fn mpm_finite_or_zero(x: f32) -> f32 {
  return select(x, 0.0, (bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u);
}

fn mpm_finite3(v: vec3f) -> vec3f {
  return vec3f(mpm_finite_or_zero(v.x), mpm_finite_or_zero(v.y), mpm_finite_or_zero(v.z));
}

fn mpm_collider_kind(a: vec4f) -> i32 {
  return i32(a.w + 0.5);
}

fn mpm_collider_sdf(a: vec4f, b: vec4f, x: vec3f) -> vec4f {
  let kind = mpm_collider_kind(a);
  let d = x - a.xyz;
  if (kind == 1) {
    let len = max(length(d), 1e-6);
    return vec4f(d / len, len - b.x);
  }
  if (kind == 2) {
    let q = abs(d) - b.xyz;
    let outside = max(q, vec3f(0.0));
    let len = length(outside);
    if (len > 0.0) {
      return vec4f(sign(d) * outside / len, len);
    }
    // Inside: leave through the nearest face.
    var n = vec3f(0.0);
    if (q.x >= q.y && q.x >= q.z) {
      n.x = select(-1.0, 1.0, d.x >= 0.0);
    } else if (q.y >= q.z) {
      n.y = select(-1.0, 1.0, d.y >= 0.0);
    } else {
      n.z = select(-1.0, 1.0, d.z >= 0.0);
    }
    return vec4f(n, max(q.x, max(q.y, q.z)));
  }
  if (kind == 3) {
    let n = normalize(b.xyz);
    return vec4f(n, dot(d, n));
  }
  return vec4f(0.0, 1.0, 0.0, 1.0e9);
}
