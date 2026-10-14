// sph-kernels.wgsl -- smoothing kernels for smoothed-particle hydrodynamics (story 7.3), in 2D and 3D, with
// their gradients. Used by src/gpu/sim/sph.ts; the position-based-fluids and MPM solvers can reuse them.
//
// CONVENTIONS. `h` is the SUPPORT radius: every kernel is exactly 0 for r >= h (Muller, Charypar & Gross 2003
// use the same h; Monaghan's "smoothing length" is h / 2 for a cubic spline). `r` is the distance between two
// particles, `d` the vector from the other particle to this one (d = x_i - x_j), so a gradient returned for `d`
// is the gradient with respect to x_i and points toward SMALLER kernel values, i.e. away from the other particle
// for a decreasing kernel. h must be positive. Every function returns 0 outside the support and is finite for
// any finite r and d.
//
// NORMALISATION. Each W integrates to 1 over its support: in 2D  integral W(|x|) dA = 1, in 3D  integral W dV = 1.
// The 3D constants found in most references are WRONG in 2D: the integral of a radial function carries 2 pi r dr
// in 2D and 4 pi r^2 dr in 3D. The constants below are derived from that and checked by numerical quadrature on
// the GPU in tests/sph.test.ts (the integrals come out within 1e-3 of 1).
//
//   poly6    W = c (h^2 - r^2)^3                       2D c = 4 / (pi h^8)       3D c = 315 / (64 pi h^9)
//            smooth at the centre (zero slope), cheap (no sqrt), used for DENSITY. Its gradient vanishes at
//            r = 0, which is why it is bad for pressure (particles that get close feel almost no push).
//   spiky    W = c (h - r)^3                           2D c = 10 / (pi h^5)      3D c = 15 / (pi h^6)
//            has a cone-shaped tip, so its gradient has full strength at r -> 0: used for the PRESSURE force,
//            where particles must repel harder the closer they get. Gradient magnitude 3 c (h - r)^2.
//   viscosity Laplacian   L(r) = c (h - r)            2D c = 40 / (pi h^5)      3D c = 45 / (pi h^6)
//            positive and falling to 0 at r = h, so a viscosity force built from it only ever smooths velocity.
//            Only the Laplacian is ever evaluated, so the kernel W it belongs to is not needed. Its constant is
//            fixed by the consistency condition of a discrete Laplacian, integral L(r) r^2 / (2 D) dV = 1 (the
//            Laplacian of the field |x|^2 / (2 D) is 1): this is what makes a viscosity of nu m^2/s diffuse
//            velocity at exactly nu, independent of the particle spacing. 3D gives Muller's 45 / (pi h^6); 2D
//            gives 40 / (pi h^5) (D = 2). tests/sph.test.ts checks the condition by quadrature.
//   cohesion  phi(q) = 16 q^2 (1 - q)^2, q = r / h    a dimensionless attractive pair profile for surface
//            tension, 0 at q = 0 and q = 1, peak 1 at q = 1/2 (a particle spacing, when h = 2 s). No
//            normalisation; the solver scales it.
//
// 2D (h support radius, r2 = r * r so no sqrt is needed where none is)
//   sph_poly6_2d(r2, h) -> f32                    W_poly6; 0 for r2 >= h^2
//   sph_poly6_grad_2d(d, h) -> vec2f              gradient w.r.t. x_i of W_poly6(|d|); 0 at |d| >= h
//   sph_spiky_2d(r, h) -> f32                     W_spiky
//   sph_spiky_grad_2d(d, h) -> vec2f              gradient of W_spiky w.r.t. x_i; 0 at d = 0 and |d| >= h
//   sph_viscosity_laplacian_2d(r, h) -> f32       Laplacian of the viscosity kernel
// 3D
//   sph_poly6_3d(r2, h) -> f32                    sph_poly6_grad_3d(d, h) -> vec3f
//   sph_spiky_3d(r, h) -> f32                     sph_spiky_grad_3d(d, h) -> vec3f
//   sph_viscosity_laplacian_3d(r, h) -> f32
// Both
//   sph_cohesion(r, h) -> f32                     phi(r / h), in [0, 1]
//   sph_coincident_direction(i, j) -> vec2f       a unit vector for two particles at the same position: the same
//                                                 for the pair (i, j) and its negation for (j, i), so pair forces
//                                                 stay equal and opposite (momentum is conserved), and different
//                                                 for different pairs, so a stack of particles spreads out.
//
// Cost: a few multiplies each; sph_spiky_grad_* and the viscosity ones are one sqrt-free evaluation if the caller
// already has r. 3D kernels use the same arithmetic.

#include "math.wgsl"
#include "hash.wgsl"

fn sph_poly6_2d(r2: f32, h: f32) -> f32 {
  let h2 = h * h;
  let t = max(h2 - r2, 0.0);
  let k = 4.0 / (PI * h2 * h2 * h2 * h2);
  return k * t * t * t;
}

fn sph_poly6_grad_2d(d: vec2f, h: f32) -> vec2f {
  let h2 = h * h;
  let t = h2 - dot(d, d);
  if (t <= 0.0) {
    return vec2f(0.0);
  }
  return d * (-24.0 / (PI * h2 * h2 * h2 * h2) * t * t);
}

fn sph_spiky_2d(r: f32, h: f32) -> f32 {
  let t = max(h - r, 0.0);
  let h2 = h * h;
  return 10.0 / (PI * h2 * h2 * h) * t * t * t;
}

fn sph_spiky_grad_2d(d: vec2f, h: f32) -> vec2f {
  let r = length(d);
  if (r <= 0.0 || r >= h) {
    return vec2f(0.0);
  }
  let t = h - r;
  let h2 = h * h;
  return d * (-30.0 / (PI * h2 * h2 * h) * t * t / r);
}

fn sph_viscosity_laplacian_2d(r: f32, h: f32) -> f32 {
  let h2 = h * h;
  return 40.0 / (PI * h2 * h2 * h) * max(h - r, 0.0);
}

fn sph_poly6_3d(r2: f32, h: f32) -> f32 {
  let h2 = h * h;
  let t = max(h2 - r2, 0.0);
  let k = 315.0 / (64.0 * PI * h2 * h2 * h2 * h2 * h);
  return k * t * t * t;
}

fn sph_poly6_grad_3d(d: vec3f, h: f32) -> vec3f {
  let h2 = h * h;
  let t = h2 - dot(d, d);
  if (t <= 0.0) {
    return vec3f(0.0);
  }
  return d * (-945.0 / (32.0 * PI * h2 * h2 * h2 * h2 * h) * t * t);
}

fn sph_spiky_3d(r: f32, h: f32) -> f32 {
  let t = max(h - r, 0.0);
  let h3 = h * h * h;
  return 15.0 / (PI * h3 * h3) * t * t * t;
}

fn sph_spiky_grad_3d(d: vec3f, h: f32) -> vec3f {
  let r = length(d);
  if (r <= 0.0 || r >= h) {
    return vec3f(0.0);
  }
  let t = h - r;
  let h3 = h * h * h;
  return d * (-45.0 / (PI * h3 * h3) * t * t / r);
}

fn sph_viscosity_laplacian_3d(r: f32, h: f32) -> f32 {
  let h3 = h * h * h;
  return 45.0 / (PI * h3 * h3) * max(h - r, 0.0);
}

fn sph_cohesion(r: f32, h: f32) -> f32 {
  let q = clamp(r / max(h, 1e-9), 0.0, 1.0);
  let t = q * (1.0 - q);
  return 16.0 * t * t;
}

fn sph_coincident_direction(i: u32, j: u32) -> vec2f {
  let a = min(i, j);
  let b = max(i, j);
  let angle = to_unorm_1d(pcg_1d((a * 73856093u) ^ (b * 19349663u))) * TAU;
  let sign = select(-1.0, 1.0, i < j);
  return vec2f(cos(angle), sin(angle)) * sign;
}
