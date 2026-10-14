// Bakes the static part of the scene once per resize: what the bed looks like in dry light
// (rgb) and how deep the water over it is, normalised to 0..1 (a). The per-frame pass only reads it.
#include "fullscreen.wgsl"
#include "noise.wgsl"
#include "sdf.wgsl"
#include "backdrop.wgsl"
#include "bed_params.wgsl"

@group(0) @binding(0) var<uniform> bed: BedParams;

const LIGHT: vec3f = vec3f(-0.5, -0.6, 0.65);

struct Obj {
  height: f32,  // 0..1 profile of the object, 0 outside
  cover: f32,   // soft mask of its footprint
  shade: f32,   // simple diffuse shading of its top
}

fn dome_object(q: vec2f, centre: vec2f, radius: f32) -> Obj {
  let d = (q - centre) / radius;
  let rr = length(d);
  let h = sqrt(max(1.0 - rr * rr, 0.0));
  let n = normalize(vec3f(d, max(h, 0.25)));
  return Obj(h, 1.0 - smoothstep(0.9, 1.0, rr), 0.45 + 0.7 * max(dot(n, normalize(LIGHT)), 0.0));
}

fn box_object(q: vec2f, centre: vec2f, half_size: vec2f, angle: f32) -> Obj {
  let p = rotate_2d(q - centre, -angle);
  let sd = sd_box_2d(p, half_size);
  let top = 1.0 - smoothstep(-0.003, 0.003, sd);
  let rim = smoothstep(-0.02, -0.004, sd);
  return Obj(top, top, 0.85 - 0.25 * rim);
}

// Normalised water depth of the bare bed: a beach, a shelf, then a drop-off to the deep end.
fn basin_depth(uv: vec2f, aspect: f32) -> f32 {
  let q = vec2f(uv.x * aspect, uv.y);
  let u = uv.x + 0.035 * gnoise_2d(q * 2.4) + 0.012 * gnoise_2d(q * 9.0) + 0.02 * sin(uv.y * 5.0 + 1.0);
  var d = 0.38 * smoothstep(0.07, 0.6, u) + 0.62 * smoothstep(0.64, 0.7, u);
  d += 0.03 * gnoise_2d(q * 6.0) * smoothstep(0.0, 0.25, d);
  return max(d, 0.0);
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let uv = in.uv;
  let aspect = bed.aspect;
  let q = vec2f(uv.x * aspect, uv.y);
  var colour = backdrop_sample(bed.kind, uv, aspect, bed.aa, bed.density);
  var depth = basin_depth(uv, aspect);

  // Submerged objects stand on the bed and reach up toward the surface by a fraction of the local depth.
  let a = dome_object(q, vec2f(0.36 * aspect, 0.3), 0.075);
  let b = dome_object(q, vec2f(0.5 * aspect, 0.74), 0.1);
  let c = dome_object(q, vec2f(0.8 * aspect, 0.34), 0.11);
  let e = dome_object(q, vec2f(0.88 * aspect, 0.78), 0.07);
  let chest = box_object(q, vec2f(0.68 * aspect, 0.58), vec2f(0.07, 0.045), 0.4);
  depth *= (1.0 - 0.9 * a.height) * (1.0 - 0.8 * b.height) * (1.0 - 0.65 * c.height);
  depth *= (1.0 - 0.7 * e.height) * (1.0 - 0.55 * chest.height);

  let stone = srgb_to_linear(vec3f(0.5, 0.48, 0.45));
  let wood = srgb_to_linear(vec3f(0.62, 0.42, 0.26));
  colour = mix(colour, stone * a.shade, a.cover * 0.9);
  colour = mix(colour, stone * b.shade * 1.05, b.cover * 0.9);
  colour = mix(colour, stone * c.shade * 0.95, c.cover * 0.9);
  colour = mix(colour, stone * e.shade, e.cover * 0.9);
  colour = mix(colour, wood * chest.shade, chest.cover * 0.95);
  return vec4f(colour, depth);
}
