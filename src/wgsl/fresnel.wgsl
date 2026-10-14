// fresnel.wgsl -- Fresnel reflectance of a clear dielectric (water, glass): how much light a surface
// reflects as a function of the angle between the view and the surface normal.
//
// Conventions
//  * `cos_theta` is the cosine of the angle of incidence: dot(normal, view) with both vectors unit
//    length and `view` pointing from the surface TO the eye. 1 = looking straight down at the
//    surface, 0 = grazing. Every function clamps it to [0, 1], so a wave normal that tilts away from
//    the eye (dot < 0) is treated as grazing and reflects fully rather than producing garbage.
//  * Results are reflectance (the reflected fraction of incoming radiance), in [0, 1]. Refraction
//    gets the remainder: transmitted = 1 - fresnel_*(..). Light is treated as unpolarised.
//  * n1 is the index of refraction (IOR) on the side the light comes from, n2 the far side. Air is
//    1.0, water 1.333 (FRESNEL_IOR_*). Light travelling from air into water is n1 = 1, n2 = 1.333.
//
// Constants
//   FRESNEL_IOR_AIR (1.0)   FRESNEL_IOR_WATER (1.333)
//   FRESNEL_F0_WATER        reflectance of air -> water at normal incidence, (n1-n2)^2/(n1+n2)^2 = 0.0204
//
// Reflectance at normal incidence
//   fresnel_f0(n1, n2) -> f32      ((n1 - n2) / (n1 + n2))^2. Air to water: 0.0204; air to glass 1.5: 0.04
//
// Schlick's approximation: F0 + (1 - F0) (1 - cos)^5. Two ALU-cheap multiplies instead of two square
// roots and two divisions; the standard choice for real-time water.
//   fresnel_schlick(cos_theta, f0) -> f32       equals f0 at cos 1 and exactly 1 at cos 0
//   fresnel_schlick_vec(cos_theta, f0: vec3f) -> vec3f   per-channel F0, for tinted or metallic reflectors
//   fresnel_water(cos_theta) -> f32             fresnel_schlick with FRESNEL_F0_WATER (air -> water)
//   fresnel_schlick_roughness(cos_theta, f0, roughness) -> f32
//                                  Lagarde's roughness-aware form for reflections taken from a blurred
//                                  environment: the grazing limit is lowered to 1 - roughness (never
//                                  below f0), so a rough surface does not turn into a perfect mirror
//                                  at the edge. roughness in [0, 1]; roughness 0 equals fresnel_schlick
//   fresnel_schlick_dielectric(cos_theta, n1, n2) -> f32
//                                  Schlick that is also right when light goes from the denser medium
//                                  to the thinner one (n1 > n2, e.g. looking up from under water):
//                                  it feeds Schlick the cosine of the TRANSMITTED angle and returns 1
//                                  beyond the critical angle (total internal reflection)
//
// Exact form: the unpolarised average of the Fresnel equations for s and p polarisation. Use it as the
// reference to check the approximations, or where a few extra ALU ops do not matter.
//   fresnel_exact(cos_theta, n1, n2) -> f32     1 beyond the critical angle when n1 > n2
//
// Accuracy (checked in tests/optics-surface.test.ts): for air -> water Schlick is within 0.01 of
// fresnel_exact up to 70 degrees from the normal and within 0.06 everywhere. It reads a little low
// around 50 to 65 degrees and high above 70, worst at 84 degrees (0.58 against 0.53). Both agree at
// 0 and 90 degrees. Not visible on a wavy surface; if you need the exact grazing shoulder, use
// fresnel_exact.
// Cost: Schlick ~8 ALU; exact ~25 ALU with one sqrt and two divisions.
//
// Reference: C. Schlick, "An Inexpensive BRDF Model for Physically-based Rendering" (1994); the
// roughness-aware form is the one S. Lagarde describes in "Image-based lighting approaches and
// parallax-corrected cubemap" (2012).

const FRESNEL_IOR_AIR: f32 = 1.0;
const FRESNEL_IOR_WATER: f32 = 1.333;
const FRESNEL_F0_WATER: f32 = 0.020373;

fn _fresnel_pow5(x: f32) -> f32 {
  let x2 = x * x;
  return x2 * x2 * x;
}

fn fresnel_f0(n1: f32, n2: f32) -> f32 {
  let r = (n1 - n2) / (n1 + n2);
  return r * r;
}

fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
  let m = 1.0 - clamp(cos_theta, 0.0, 1.0);
  return f0 + (1.0 - f0) * _fresnel_pow5(m);
}

fn fresnel_schlick_vec(cos_theta: f32, f0: vec3f) -> vec3f {
  let m = 1.0 - clamp(cos_theta, 0.0, 1.0);
  return f0 + (vec3f(1.0) - f0) * _fresnel_pow5(m);
}

fn fresnel_water(cos_theta: f32) -> f32 {
  return fresnel_schlick(cos_theta, FRESNEL_F0_WATER);
}

fn fresnel_schlick_roughness(cos_theta: f32, f0: f32, roughness: f32) -> f32 {
  let m = 1.0 - clamp(cos_theta, 0.0, 1.0);
  let grazing = max(1.0 - clamp(roughness, 0.0, 1.0), f0);
  return f0 + (grazing - f0) * _fresnel_pow5(m);
}

fn fresnel_exact(cos_theta: f32, n1: f32, n2: f32) -> f32 {
  let c = clamp(cos_theta, 0.0, 1.0);
  let eta = n1 / n2;
  let sin2_t = eta * eta * (1.0 - c * c);
  if (sin2_t >= 1.0) {
    return 1.0;
  }
  let cos_t = sqrt(1.0 - sin2_t);
  let rs = (n1 * c - n2 * cos_t) / (n1 * c + n2 * cos_t);
  let rp = (n1 * cos_t - n2 * c) / (n1 * cos_t + n2 * c);
  return 0.5 * (rs * rs + rp * rp);
}

fn fresnel_schlick_dielectric(cos_theta: f32, n1: f32, n2: f32) -> f32 {
  let c = clamp(cos_theta, 0.0, 1.0);
  let f0 = fresnel_f0(n1, n2);
  if (n1 <= n2) {
    return fresnel_schlick(c, f0);
  }
  let eta = n1 / n2;
  let sin2_t = eta * eta * (1.0 - c * c);
  if (sin2_t >= 1.0) {
    return 1.0;
  }
  return fresnel_schlick(sqrt(1.0 - sin2_t), f0);
}
