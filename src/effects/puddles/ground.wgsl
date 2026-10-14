// Bakes the dry ground once into a seamlessly tiling texture, so the per-frame pass reads two texels
// instead of evaluating noise and Voronoi per pixel.
//   target 0 (rgba16float): rgb = dry albedo (linear), a = height 0..1 (0 the lowest spot of the tile)
//   target 1 (rg16float):   xy  = tangent-space normal of the surface (u along world x, v along world z)
// The tile is TILE metres wide; heights span HEIGHT metres. Three surfaces: asphalt, cobbles, flagstones.
// Every pattern is built from the `_tile` functions, so the texture repeats without a seam.
#include "fullscreen.wgsl"
#include "math.wgsl"
#include "noise.wgsl"
#include "fbm.wgsl"
#include "voronoi.wgsl"
#include "normals.wgsl"
#include "ground_params.wgsl"

@group(0) @binding(0) var<uniform> G: GroundParams;

struct GroundOut {
  @location(0) albedo_height: vec4f,
  @location(1) normal: vec2f,
}

struct Ground {
  albedo: vec3f,
  height: f32,
}

// Low spots on a scale of a metre or two: where the puddles collect. 0..1, spread over the whole range.
fn macro_height(uv: vec2f) -> f32 {
  let n = fbm_2d_tile(uv * 3.0, vec2i(3), 3, 0.5);
  return saturate(0.5 + 1.5 * n);
}

fn ground_at(uv: vec2f) -> Ground {
  var out: Ground;
  let macro_h = macro_height(uv);
  let grain = 0.5 + 0.5 * fbm_2d_tile(uv * 96.0, vec2i(96), 3, 0.55);

  if (G.surface == 0) {
    // Asphalt: dark binder, lighter aggregate, hairline cracks, one worn lane line.
    let speck = 0.5 + 0.5 * gnoise_2d_tile(uv * 220.0, vec2i(220));
    let cracks = voronoi_2d_tile(uv * 5.0 + 0.3 * vec2f(grain), 0.9, vec2i(5));
    let crack = 1.0 - smoothstep(0.0, 0.05, cracks.f2 - cracks.f1);
    let line_x = abs(uv.x - 0.5) * G.tile_m;
    let wear = smoothstep(0.35, 0.65, 0.5 + 0.5 * fbm_2d_tile(uv * 24.0, vec2i(24), 3, 0.55));
    let line = (1.0 - smoothstep(0.055, 0.075, line_x)) * wear;
    var base = vec3f(0.07, 0.07, 0.075) * (0.55 + 0.9 * grain);
    base += vec3f(0.12, 0.115, 0.11) * smoothstep(0.78, 0.95, speck) * 0.6;
    base *= 1.0 - 0.7 * crack;
    base = mix(base, vec3f(0.55, 0.54, 0.5), line);
    out.albedo = base;
    out.height = saturate(0.9 * macro_h + 0.1 * (0.5 * grain + 0.5 * speck) - 0.12 * crack);
  } else if (G.surface == 1) {
    // Cobbles: domed stones about 12 cm across in dark grout.
    let v = voronoi_2d_tile(uv * 48.0, 0.55, vec2i(48));
    let e = v.f2 - v.f1;
    let dome = smoothstep(0.02, 0.5, e);
    let grout = 1.0 - smoothstep(0.03, 0.14, e);
    let tint = mix(vec3f(0.17, 0.155, 0.14), vec3f(0.3, 0.27, 0.24), v.id);
    let hue = mix(vec3f(1.0, 0.97, 0.92), vec3f(0.92, 0.97, 1.0), fract(v.id * 7.13));
    var base = tint * hue * (0.7 + 0.5 * grain);
    base = mix(base, vec3f(0.06, 0.055, 0.05), grout);
    out.albedo = base;
    let stone = 0.55 * dome + 0.35 * fract(v.id * 3.7) + 0.1 * grain;
    out.height = saturate(0.62 * macro_h + 0.38 * stone * (1.0 - 0.8 * grout));
  } else {
    // Flagstones: slabs about 75 cm across with narrow joints, each slightly tilted and tinted.
    let v = voronoi_2d_tile(uv * 8.0, 0.75, vec2i(8));
    let e = v.f2 - v.f1;
    let joint = 1.0 - smoothstep(0.02, 0.07, e);
    let bevel = smoothstep(0.02, 0.22, e);
    let tint = mix(vec3f(0.2, 0.2, 0.21), vec3f(0.33, 0.31, 0.28), fract(v.id * 5.31));
    let cool = mix(vec3f(0.95, 0.98, 1.04), vec3f(1.04, 1.0, 0.94), fract(v.id * 9.7));
    var base = tint * cool * (0.65 + 0.7 * grain);
    base = mix(base, vec3f(0.05, 0.05, 0.05), joint);
    out.albedo = base;
    let slab = 0.5 * fract(v.id * 11.3) + 0.3 * bevel + 0.2 * grain;
    out.height = saturate(0.6 * macro_h + 0.4 * slab * (1.0 - 0.7 * joint));
  }
  return out;
}

@fragment fn fs(in: FullscreenVarying) -> GroundOut {
  let uv = in.uv;
  let texel = 1.0 / G.size;
  let centre = ground_at(uv);
  let hx = ground_at(uv + vec2f(texel, 0.0)).height;
  let hy = ground_at(uv + vec2f(0.0, texel)).height;
  // Slope in metres of height per metre of ground.
  let metres_per_texel = G.tile_m * texel;
  let grad = vec2f(hx - centre.height, hy - centre.height) * G.height_m / metres_per_texel;
  let n = normal_from_gradient(grad, 1.0);
  return GroundOut(vec4f(centre.albedo, centre.height), n.xy);
}
