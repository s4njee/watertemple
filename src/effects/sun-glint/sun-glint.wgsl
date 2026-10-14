#include "fullscreen.wgsl"
#include "color.wgsl"
#include "sky.wgsl"
#include "fresnel.wgsl"
#include "specular.wgsl"
#include "waterplane.wgsl"
#include "params.wgsl"

@group(0) @binding(0) var<uniform> params: Params;

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let sun = waterplane_sun_dir(params.sun_elevation, params.sun_azimuth);
  let dir = waterplane_ray_dir(in.uv, params.aspect, params.pitch, params.fov);
  let origin = vec3f(params.drift * params.time, params.cam_height, 0.0);
  let t = waterplane_hit(origin, dir);
  let sun_radiance = sky_sun_color(sun) * SKY_SUN_RADIANCE * params.sun_gain;

  var colour = vec3f(0.0);
  if (t < 0.0) {
    colour = sky_radiance_ex(dir, sun, params.haze, params.sun_gain);
    if (params.view_mode != 0u) {
      colour = colour * 0.1;
    }
  } else {
    let footprint = waterplane_footprint(t, dir.y, params.pixel_angle);
    let p = origin + dir * t;
    let wave = waterplane_wave(p.xz, params.time, params.wave_strength, params.wave_scale, footprint);
    let normal = waterplane_normal(wave.yz);
    let view = -dir;

    let fresnel = fresnel_water(dot(normal, view));
    // The sun is added by the lobe and the sparkle below, so it is left out of the reflected sky.
    let reflected = sky_radiance_ex(waterplane_reflect(dir, normal), sun, params.haze, 0.0);
    let lit = sky_ambient(sun) * 0.8 + sky_sun_color(sun) * max(sun.y, 0.0) * 0.5;
    let body = params.body * lit;

    let lobe = spec_sun(normal, view, sun, params.roughness, SKY_SUN_ANGULAR_RADIUS);
    let half_vector = normalize(view + sun);
    let sparkle = spec_sparkle(
      p.xz, normal, view, sun, params.time, footprint,
      params.sparkle_cell, params.sparkle_density, params.sparkle_sharpness,
    ) * fresnel_water(saturate(dot(view, half_vector)));

    let base = mix(body, reflected, fresnel);
    let lobe_colour = sun_radiance * lobe;
    let sparkle_colour = sun_radiance * sparkle * params.sparkle_intensity;
    switch (params.view_mode) {
      case 1u: { colour = lobe_colour; }
      case 2u: { colour = sparkle_colour; }
      default: { colour = base + lobe_colour + sparkle_colour; }
    }
  }

  var out = linear_to_srgb(tonemap_aces(colour * params.exposure));
  out += sky_dither(in.position.xy) / 255.0;
  return vec4f(out, 1.0);
}
