// Local sky model for the reflections demo: horizon-to-zenith gradient, sun, a few clouds.
// Used by the sky background pass, by the water shader as the fallback when a trace misses,
// and (through the planar pass) as the thing the sky strategy reflects.
// Functions: rf_sky(light, dir) -> vec3f   linear HDR colour looking along the unit vector `dir`.

#include "scene-light.wgsl"
#include "fbm.wgsl"

fn rf_sky(light: SceneLight, dir: vec3f) -> vec3f {
  let h = dir.y;
  var col = mix(light.horizon_color, light.sky_color * 1.15, pow(clamp(h, 0.0, 1.0), 0.5));
  col = mix(col, light.ground_color * 0.8, smoothstep(0.0, -0.3, h));

  // Clouds: a flat layer seen in perspective, thinning to nothing towards the horizon.
  if (h > 0.02) {
    let p = dir.xz / (h + 0.12) * 1.4;
    let n = fbm_2d(p + vec2f(3.1, 7.7), 4, 2.0, 0.5);
    let cover = smoothstep(0.02, 0.5, n * 1.3 + 0.12) * smoothstep(0.02, 0.25, h);
    col = mix(col, vec3f(1.0, 0.98, 0.95) * 1.25, cover * 0.85);
  }

  let s = max(dot(dir, light.sun_dir), 0.0);
  col += light.sun_color * (pow(s, 4000.0) * 60.0 + pow(s, 40.0) * 0.18 + pow(s, 5.0) * 0.04);
  return col;
}
