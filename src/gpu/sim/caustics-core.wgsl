// caustics-core.wgsl -- the shared optics of the heightfield caustics solver (src/gpu/sim/caustics.ts):
// what a ray of sunlight does at the water surface and whether a ball is in its way. Included (as an
// effect-local chunk) by caustics-mesh.wgsl and caustics-jacobian.wgsl, after the generated
// `caustics_params.wgsl` (struct CausticsParams). Bindings, group 0:
//   0  P         uniform CausticsParams
//   1  hf_tex    rgba16float, filterable   the heightfield solver's output (height, velocity, slope x, slope y)
//   2  hf_samp   filtering sampler, clamp-to-edge
//
// FRAME. The frame of the heightfield (src/wgsl/heightfield.wgsl): x to the right and y DOWN in units
// of the IMAGE HEIGHT (so a point at uv has x = uv.x * aspect, y = uv.y), z UP out of the water, zero at
// the mean surface. The floor is the plane z = -depth. `P.light` is the unit vector TOWARD the light.
// Displacements and landing points are returned in uv (x already divided by the aspect).
//
// The surface at a point is the single filtered texel `s` of hf_tex: s.x is the height in cells (times
// P.cell_size = image heights), s.zw the slope. The normal is normalize(-strength * slope, 1).

#include "caustics_params.wgsl"
#include "fresnel.wgsl"

@group(0) @binding(0) var<uniform> P: CausticsParams;
@group(0) @binding(1) var hf_tex: texture_2d<f32>;
@group(0) @binding(2) var hf_samp: sampler;

// A surface tilted so far that the sun grazes it (cosine of the incidence below this) transmits
// nothing: the light is reflected instead. Such texels contribute no energy.
const CAUSTICS_MIN_COS: f32 = 0.05;
// Most one texel's radiometric factor may reach.
const CAUSTICS_MAX_WEIGHT: f32 = 4.0;

fn caustics_surface(uv: vec2f) -> vec4f {
  return textureSampleLevel(hf_tex, hf_samp, uv, 0.0);
}

struct CausticsTransmit {
  // Unit direction of the refracted ray, z < 0.
  dir: vec3f,
  // Power per unit of horizontal surface area, relative to a flat surface (1 = the same as flat).
  weight: f32,
}

// Refracts the sun through the surface texel `s` at index of refraction `ior`.
//
// `weight` is the radiometry of a tilted patch: a patch of horizontal area dA with normal n catches
// l.n / n.z times the light a flat one does (it faces the sun more or less), and passes the fraction
// 1 - F(l.n) of it (Fresnel), both relative to the flat surface so that a flat surface weighs exactly
// 1. It is 1 when radiometry is off. A texel the sun cannot enter weighs 0 and refracts as if flat,
// so that a mesh built from such texels stays connected.
fn caustics_transmit(s: vec4f, ior: f32) -> CausticsTransmit {
  let l = P.light;
  var n = normalize(vec3f(-s.zw * P.strength, 1.0));
  var cos_i = dot(l, n);
  var weight = 1.0;
  if (cos_i <= CAUSTICS_MIN_COS) {
    n = vec3f(0.0, 0.0, 1.0);
    cos_i = l.z;
    weight = 0.0;
  }
  var r = refract(-l, n, 1.0 / ior);
  if (r.z > -0.05) {
    r = refract(-l, vec3f(0.0, 0.0, 1.0), 1.0 / ior);
    weight = 0.0;
  }
  if (weight > 0.0 && P.radiometry > 0.5) {
    let f0 = fresnel_f0(1.0, ior);
    let through = (1.0 - fresnel_schlick(cos_i, f0)) / (1.0 - fresnel_schlick(l.z, f0));
    weight = clamp(cos_i / (n.z * l.z) * through, 0.0, CAUSTICS_MAX_WEIGHT);
  }
  return CausticsTransmit(normalize(r), weight);
}

// How far, in uv, a ray refracted by texel `s` lands from the point where it entered, on a floor
// `P.depth` below the mean surface. The surface height is ignored (it is a cell or two).
fn caustics_displacement(s: vec4f, ior: f32) -> vec2f {
  let t = caustics_transmit(s, ior);
  return t.dir.xy / (-t.dir.z) * P.depth / vec2f(P.aspect, 1.0);
}

// Fraction of the light that reaches the floor along the ray `origin + dir * t`, t in [0, t_floor],
// after the occluding sphere P.occluder (centre xyz, radius w; w <= 0 disables it) has had its say.
// Two segments matter: the refracted ray under the surface, and the ray from the surface point up
// toward the sun (a ball floating above the water shades it too). Distance to the sphere is turned
// into a soft edge P.soft wide (a smoothstep centred on the sphere's silhouette), which is also what
// keeps a mesh with one value per vertex from showing its triangles in the shadow's rim.
fn caustics_lit(origin: vec3f, dir: vec3f, t_floor: f32) -> f32 {
  let radius = P.occluder.w;
  if (radius <= 0.0) {
    return 1.0;
  }
  let c = P.occluder.xyz;
  let rel = c - origin;
  let below = length(rel - dir * clamp(dot(rel, dir), 0.0, t_floor));
  let above = length(rel - P.light * max(dot(rel, P.light), 0.0));
  let d = min(below, above);
  let half_soft = max(0.5 * P.soft, 1e-5);
  return smoothstep(radius - half_soft, radius + half_soft, d);
}
