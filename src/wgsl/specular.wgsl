// specular.wgsl -- sun glint on water: a tight, energy-conserving specular lobe, and stochastic
// sparkle (glitter) for the micro-facets the wave normals do not resolve.
//
// Conventions
//  * World space, all direction arguments are UNIT vectors pointing AWAY from the surface:
//    `n` surface normal, `v` toward the eye, `l` toward the sun (or other light).
//  * Results are dimensionless factors. Multiply by the light's radiance to get HDR radiance, e.g.
//        sky_sun_color(sun) * SKY_SUN_RADIANCE * spec_sun(n, v, l, roughness, SKY_SUN_ANGULAR_RADIUS)
//    The product never exceeds F x (sun disc radiance): the reflection of the sun cannot be brighter
//    than the sun itself times the Fresnel reflectance, however narrow the lobe is made.
//
// Lobe: GGX (Trowbridge-Reitz) normal distribution with the Smith visibility term and Fresnel from
// fresnel.wgsl (air -> water). Why GGX and not Blinn-Phong: GGX has a long tail, which gives the sun
// reflection on water its soft glow around a hard core; a Phong lobe falls off like a Gaussian and
// looks like a plastic dot. It also normalises the same way for every roughness, so one slider
// moves from mirror-calm to choppy without re-tuning brightness. Cost: no pow, one normalize, one
// cross, a handful of multiplies and one divide (~35 ALU).
//   roughness     perceptual, [0, 1]; alpha = roughness^2. Water at noon is 0.02 to 0.1, chop 0.1 to 0.3.
//                 0 is allowed: the sun's own size keeps the lobe finite.
//   sun_radius    angular radius of the light in radians (SKY_SUN_RADIANCE's disc: 0.0093). The lobe is
//                 widened to alpha_eff^2 = alpha^2 + (sun_radius / 2)^2, the standard way to give an area
//                 light a finite highlight without a reference mirror lookup; the peak stays below 1.
//
//   spec_ggx_d(n_dot_h, alpha) -> f32       GGX normal distribution; integrates to 1 over the hemisphere of
//                                           half vectors weighted by n_dot_h. Peak 1 / (pi alpha^2)
//   spec_ggx(n, v, l, alpha) -> f32         BRDF x cos(theta_l) with Fresnel: D x V x F x n_dot_l. Units 1/sr.
//                                           Its integral over l in the upper hemisphere is the directional
//                                           albedo and stays <= 1 (energy conserving). 0 when l or v is
//                                           below the surface
//   spec_sun(n, v, l, roughness, sun_radius) -> f32
//                                           fraction of the sun disc's radiance reflected toward the eye,
//                                           in [0, 1]. Equals the Fresnel reflectance F at the exact mirror
//                                           direction of a flat surface with roughness 0 and falls off with
//                                           the GGX lobe around it
//
// Sparkle: discrete glints that twinkle as the waves move. The water plane is cut into square cells
// in WORLD space (so glints stay put under camera motion). A cell either holds a glint (with
// probability `density`) or not. A glint is a small round patch inside its cell with its own micro-
// facet normal, tilted away from the wave normal by a random offset (up to 0.3 in slope) that slowly
// circles with time, so the glint lights up when the half vector sweeps past the facet. Brightness is a
// Gaussian of the slope distance between the half vector and the facet normal.
// Anti-aliasing: the cell size is raised with the pixel footprint (cells are never smaller than
// ~3 pixels) by blending two power-of-two cell levels, so glints stay about pixel-sized in the
// distance, never shimmer below the sampling rate, and the mean brightness does not depend on the
// level (area fraction and facet statistics are scale-free). Cost: two cell evaluations, each two
// pcg_4d hashes and ~30 ALU: roughly 150 ALU total inside the glitter path.
//   spec_sparkle(p, n, v, l, time, footprint, cell, density, sharpness) -> f32
//        p          world xz position of the shaded point
//        time       seconds; glints twinkle with a period of roughly 3 to 20 s each
//        footprint  pixel size on the water in metres (waterplane_footprint); 0 disables the
//                   anti-aliasing blend
//        cell       finest cell size in metres, e.g. 0.05 (smaller = finer glitter, more of it)
//        density    [0, 1], the fraction of cells that hold a glint
//        sharpness  [0.25, 8], 1 = nominal. Higher makes each glint's angular window narrower
//                   (fewer, hotter, more twinkly glints); lower makes a softer, steadier sheen
//        Returns [0, 1]: multiply by the sun's radiance, the Fresnel term and an intensity. Exactly 0
//        for density 0 and wherever the half vector is more than 0.3 + 4 sigma (slope) from the wave
//        normal, i.e. outside the glitter path, which also makes those pixels nearly free.
//        Deterministic for equal inputs
//
// References: GGX: B. Walter et al., "Microfacet Models for Refraction through Rough Surfaces"
// (2007). Visibility: E. Heitz, "Understanding the Masking-Shadowing Function" (2014) with the cheap
// approximation from S. Hammon, "PBR Diffuse Lighting for GGX+Smith Microsurfaces" (GDC 2017).
// Glitter by hashed world-space cells is a common shader-demo technique; spec_sparkle is a cheap
// shortcut, not a glint BRDF (those evaluate a distribution of facet normals per footprint).

#include "fresnel.wgsl"
#include "hash.wgsl"
#include "math.wgsl"

const _SPEC_SPARKLE_SPREAD: f32 = 0.3;
const _SPEC_SPARKLE_SIGMA: f32 = 0.05;

fn spec_ggx_d(n_dot_h: f32, alpha: f32) -> f32 {
  let a2 = alpha * alpha;
  let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}

// Hammon's approximation of the height-correlated Smith visibility G / (4 n.l n.v).
fn _spec_visibility(n_dot_l: f32, n_dot_v: f32, alpha: f32) -> f32 {
  return 0.5 / max(mix(2.0 * n_dot_l * n_dot_v, n_dot_l + n_dot_v, alpha), 1e-5);
}

// GGX distribution from the sine of the angle (cross product) rather than the cosine: stays
// accurate for very narrow lobes where 1 - cos^2 would cancel in f32.
fn _spec_ggx_d_sin(sin2_h: f32, n_dot_h: f32, alpha: f32) -> f32 {
  let a2 = alpha * alpha;
  let d = sin2_h + a2 * n_dot_h * n_dot_h;
  return a2 / (PI * d * d);
}

fn spec_ggx(n: vec3f, v: vec3f, l: vec3f, alpha: f32) -> f32 {
  let n_dot_l = dot(n, l);
  let n_dot_v = dot(n, v);
  if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
    return 0.0;
  }
  let h = normalize(v + l);
  let n_dot_h = saturate(dot(n, h));
  let c = cross(n, h);
  let a = max(alpha, 1e-3);
  let d = _spec_ggx_d_sin(dot(c, c), n_dot_h, a);
  let f = fresnel_water(saturate(dot(v, h)));
  return d * _spec_visibility(n_dot_l, n_dot_v, a) * f * n_dot_l;
}

fn spec_sun(n: vec3f, v: vec3f, l: vec3f, roughness: f32, sun_radius: f32) -> f32 {
  let n_dot_l = dot(n, l);
  let n_dot_v = dot(n, v);
  if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
    return 0.0;
  }
  let alpha = saturate(roughness) * saturate(roughness);
  let rho = 0.5 * sun_radius;
  let alpha_w = sqrt(alpha * alpha + rho * rho);
  let h = normalize(v + l);
  let v_dot_h = saturate(dot(v, h));
  // spec_ggx is per steradian of incoming light. The sun covers pi r^2 of them; a disc of radius
  // r / 2 in half-vector space maps to 4 (v.h) times that area in light-direction space, which
  // cancels the 4 in the BRDF's denominator and leaves v.h. At a perfect mirror alignment
  // (v.h == n.v) the product is exactly the Fresnel reflectance F.
  let solid_angle = PI * sun_radius * sun_radius * v_dot_h;
  return min(spec_ggx(n, v, l, alpha_w) * solid_angle, 1.0);
}

// Sparkle for one cell level. `slope` = half-vector slope minus the wave normal's slope.
fn _spec_sparkle_level(p: vec2f, cell: f32, level_seed: u32, slope: vec2f, time: f32, density: f32, sigma: f32) -> f32 {
  let q = p / cell;
  let id = floor(q);
  let f = q - id;
  let ids = vec2i(id);
  let key = vec4u(bitcast<u32>(ids.x), bitcast<u32>(ids.y), level_seed, 0x9e3779b9u);
  let r = to_unorm_4d(pcg_4d(key));
  if (r.x >= density) {
    return 0.0;
  }
  let r2 = to_unorm_4d(pcg_4d(key ^ vec4u(0x85ebca6bu, 0xc2b2ae35u, 0x27d4eb2fu, 1u)));
  let centre = vec2f(0.5) + (r.yz - 0.5) * 0.16;
  let mask = 1.0 - smoothstep(0.22, 0.42, length(f - centre));
  let radius = _SPEC_SPARKLE_SPREAD * sqrt(r2.x);
  let speed = (0.3 + 1.2 * r2.y) * (select(-1.0, 1.0, r2.z > 0.5));
  let angle = TAU * r.w + speed * time;
  let offset = radius * vec2f(cos(angle), sin(angle));
  let d = slope - offset;
  return mask * exp(-dot(d, d) / (sigma * sigma));
}

fn spec_sparkle(
  p: vec2f, n: vec3f, v: vec3f, l: vec3f, time: f32, footprint: f32,
  cell: f32, density: f32, sharpness: f32,
) -> f32 {
  let h = normalize(v + l);
  if (h.y <= 0.05 || n.y <= 0.05 || density <= 0.0) {
    return 0.0;
  }
  let slope = h.xz / h.y - n.xz / n.y;
  let sigma = _SPEC_SPARKLE_SIGMA / clamp(sharpness, 0.25, 8.0);
  // No facet can reach the half vector from this far (spread + 4 sigma, where the Gaussian is
  // 1e-7): skip the hashing. Most of the picture is outside the glitter path, so this is the big
  // saving.
  let reach = _SPEC_SPARKLE_SPREAD + 4.0 * sigma;
  if (dot(slope, slope) > reach * reach) {
    return 0.0;
  }
  let base = max(cell, 1e-3);
  // Cells at least ~3 pixels wide: pick the level from the footprint and blend the two nearest.
  let lod = clamp(log2(max(3.0 * footprint / base, 1.0)), 0.0, 10.0);
  let l0 = floor(lod);
  let blend = lod - l0;
  let a = _spec_sparkle_level(p, base * exp2(l0), u32(l0), slope, time, density, sigma);
  var s = a * (1.0 - blend);
  if (blend > 0.0) {
    s += blend * _spec_sparkle_level(p, base * exp2(l0 + 1.0), u32(l0) + 1u, slope, time, density, sigma);
  }
  return s;
}
