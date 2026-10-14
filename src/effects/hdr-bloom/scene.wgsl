// A dark rippling surface seen from above, lit so that its radiance spans several orders of
// magnitude. Output is LINEAR HDR (no tonemap, no sRGB): the post chain does that.
//
//   base water       ~0.005 .. 0.05     dark, so exposure and tonemapping have a toe to show
//   caustic lines    up to ~2.5         thin cyan creases from animated Voronoi cell borders
//   lantern streaks  peak 12 .. 30      vertical reflections of four lamps, warped by the ripples
//   sun glints       peak ~60           where a facet normal lines up with the half vector
#include "fullscreen.wgsl"
#include "fbm.wgsl"
#include "voronoi.wgsl"
#include "hash.wgsl"
#include "scene-params.wgsl"

@group(0) @binding(0) var<uniform> scene: SceneParams;

// Surface slope (d height / dx, d height / dy) of two ripple scales.
fn slope_at(p: vec2f, t: f32) -> vec2f {
  let big = fbm_3d_d(vec3f(p * 1.5 + vec2f(t * 0.05, 0.0), t * 0.18), 4, 2.0, 0.5);
  let fine = fbm_3d_d(vec3f(p * 9.0 + vec2f(0.0, t * 0.12), t * 0.55 + 7.0), 3, 2.0, 0.5);
  return big.yz * 0.07 + fine.yz * 0.05;
}

// A lamp's reflection: a soft blob stretched along the view's up axis, as on real water.
fn lantern(p: vec2f, centre: vec2f, size: vec2f, tint: vec3f, power: f32) -> vec3f {
  let d = (p - centre) / size;
  return tint * power * exp(-dot(d, d));
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let t = scene.time;
  let p = vec2f(in.uv.x * scene.aspect, in.uv.y);
  let slope = slope_at(p, t);

  // Darker towards the bottom of the frame, as if the water gets deeper.
  let depth = smoothstep(0.0, 1.0, in.uv.y);
  var colour = mix(vec3f(0.012, 0.045, 0.06), vec3f(0.002, 0.012, 0.022), depth);

  // Faint sky reflection: more of it where the surface tilts.
  colour += vec3f(0.02, 0.04, 0.07) * (0.4 + 6.0 * dot(slope, slope));

  // Caustic-like creases.
  let cell = voronoi_2d_anim(p * 5.0 + slope * 3.0, 0.9, t * 0.5);
  let crease = 1.0 - smoothstep(0.0, 0.05, cell.f2 - cell.f1);
  let mask = smoothstep(0.45, 0.95, 0.5 + 0.9 * fbm_2d(p * 1.2 + vec2f(3.0, t * 0.04), 3, 2.0, 0.5));
  colour += vec3f(0.25, 0.9, 1.0) * crease * crease * mask * 2.5 * scene.caustics;

  // Lamps hung above the water. Each reflection is displaced by the local slope, so it breaks up
  // into streaks that ripple. Radiance 12 to 30 is 12 to 30 times SDR white.
  let q = p + vec2f(slope.x * 0.35, slope.y * 0.12);
  let drift = vec2f(sin(t * 0.07), 0.0) * 0.05;
  var lamps = lantern(q, vec2f(0.30 * scene.aspect, 0.34) + drift, vec2f(0.006, 0.07), vec3f(1.0, 0.55, 0.22), 22.0);
  lamps += lantern(q, vec2f(0.62 * scene.aspect, 0.52) - drift, vec2f(0.005, 0.06), vec3f(1.0, 0.8, 0.5), 30.0);
  lamps += lantern(q, vec2f(0.82 * scene.aspect, 0.30) + drift * 0.5, vec2f(0.007, 0.09), vec3f(0.35, 0.7, 1.0), 12.0);
  lamps += lantern(q, vec2f(0.12 * scene.aspect, 0.70), vec2f(0.004, 0.04), vec3f(0.6, 1.0, 0.7), 16.0);
  colour += lamps * scene.lights;

  // Sun glints: a very tight lobe around the half vector between a high sun and the viewer.
  let n = normalize(vec3f(-slope.x, 1.0, -slope.y));
  let sun_dir = normalize(vec3f(0.22, 0.95, 0.16));
  let half_vec = normalize(sun_dir + vec3f(0.0, 1.0, 0.0));
  let lobe = pow(max(dot(n, half_vec), 0.0), 9000.0);
  colour += vec3f(1.0, 0.93, 0.8) * lobe * 60.0 * scene.sun;

  return vec4f(colour, 1.0);
}
