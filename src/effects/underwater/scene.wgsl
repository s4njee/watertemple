#include "fullscreen.wgsl"
#include "color.wgsl"
#include "math.wgsl"
#include "noise.wgsl"
#include "caustics.wgsl"
#include "shafts.wgsl"
#include "motes.wgsl"
#include "scene_params.wgsl"

@group(0) @binding(0) var<uniform> P: SceneParams;
@group(0) @binding(1) var caustic_tex: texture_2d<f32>;
@group(0) @binding(2) var caustic_samp: sampler;

const LIGHT: vec3f = vec3f(0.78, 0.95, 1.0);

// The whole picture is shifted by a slow, low-frequency displacement: the "looking through moving
// water" wobble. It is applied to the coordinates every layer uses, so shafts, seabed and motes move
// together and the scene never shears apart.
fn wobble(p: vec2f) -> vec2f {
  let t = P.time;
  return P.wobble * 0.012 * vec2f(
    sin(p.y * 6.0 + t * 0.9) + 0.5 * sin(p.y * 13.0 - t * 1.3 + p.x * 3.0),
    sin(p.x * 4.0 + t * 0.7) + 0.5 * sin(p.x * 9.0 + t * 1.1 + p.y * 2.0),
  );
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let raw = vec2f(in.uv.x * P.aspect, in.uv.y);
  let p = raw + wobble(raw);

  // Depth fog: bright, light-filled water at the top, the deep colour below.
  let fog = 1.0 - exp(-p.y * P.fog);
  let shallow = P.deep * 2.4 + vec3f(0.01, 0.05, 0.07);
  var colour = mix(shallow, P.deep, fog);
  // The sky through the surface, a soft lightening at the very top.
  colour += LIGHT * 0.07 * exp(-p.y * 4.0);

  // Light shafts, fanning from a point above the frame; a slant moves the source sideways.
  let source = vec2f(P.aspect * 0.5 - P.slant * (P.source_height + 0.5), -P.source_height);
  let shaft = shafts_light(p, source, P.time, P.shaft_frequency, P.shaft_contrast, P.shaft_reach);
  colour += LIGHT * shaft * P.shafts * 0.45;

  // The seabed, if any: sand lit by baked caustics, seen through the fog.
  if (P.seabed > 0.0) {
    let floor_y = 1.0 - 0.17 * P.seabed + 0.025 * sin(p.x * 3.1 + 1.0) + 0.015 * sin(p.x * 8.3);
    let mask = smoothstep(floor_y - 0.004, floor_y + 0.012, p.y);
    if (mask > 0.0) {
      // Foreshortened: the bed recedes upward, so the pattern is squashed vertically.
      let q = vec2f(p.x, (p.y - floor_y) * 0.55) * P.caustic_scale;
      let c = caustics_baked(caustic_tex, caustic_samp, q, P.time * 0.8, 1.2, CAUSTICS_BAKED_AVG);
      let sand = vec3f(0.62, 0.52, 0.36);
      let lit = sand * (0.1 + 0.4 * c) * (1.0 - 0.5 * fog);
      // It sits under water too: fog toward the deep colour with distance from the camera (the top
      // of the bed is far away).
      let haze = clamp(0.55 - (p.y - floor_y) * 2.5, 0.2, 0.85);
      colour = mix(colour, mix(lit, P.deep * 1.4, haze), mask);
    }
  }

  // Marine snow: a far layer (small, slow, dim) and a near one (larger, faster, brighter).
  let far = motes_layer(p, P.time, 0.07, P.particles, 0.09, vec2f(0.004, 0.012), 1.0);
  let near = motes_layer(p, P.time, 0.15, P.particles * 0.6, 0.1, vec2f(-0.008, 0.03), 2.0);
  colour += LIGHT * (far * 0.22 + near * 0.38) * (0.4 + 0.6 * exp(-p.y * 1.2));

  return vec4f(linear_to_srgb(clamp(colour, vec3f(0.0), vec3f(1.0))), 1.0);
}
