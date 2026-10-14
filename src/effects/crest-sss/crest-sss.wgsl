#include "fullscreen.wgsl"
#include "color.wgsl"
#include "sky.wgsl"
#include "fresnel.wgsl"
#include "specular.wgsl"
#include "sss.wgsl"
#include "waterplane.wgsl"
#include "params.wgsl"

@group(0) @binding(0) var<uniform> params: Params;

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let sun = waterplane_sun_dir(params.sun_elevation, params.sun_azimuth);
  let dir = waterplane_ray_dir(in.uv, params.aspect, params.pitch, params.fov);
  let origin = vec3f(0.0, params.cam_height, 0.0);
  let t_plane = waterplane_hit(origin, dir);

  var colour = vec3f(0.0);
  if (t_plane < 0.0) {
    colour = sky_radiance_ex(dir, sun, params.haze, 1.0);
    if (params.view_mode != 0u) {
      colour = colour * 0.08;
    }
  } else {
    let footprint = waterplane_footprint(t_plane, dir.y, params.pixel_angle);
    // The waves are tall here: move the hit onto the displaced surface before reading them.
    let t = waterplane_hit_waves(origin, dir, t_plane, params.time, params.wave_strength, params.wave_scale, footprint);
    let p = origin + dir * t;
    let wave = waterplane_wave(p.xz, params.time, params.wave_strength, params.wave_scale, footprint);
    let normal = waterplane_normal(wave.yz);
    let view = -dir;

    let fresnel = fresnel_water(dot(normal, view));
    let reflected = sky_radiance_ex(waterplane_reflect(dir, normal), sun, params.haze, 0.0);
    let lit = sky_ambient(sun) * 0.8 + sky_sun_color(sun) * max(sun.y, 0.0) * 0.5;
    let body = params.body * lit;
    let sun_radiance = sky_sun_color(sun) * SKY_SUN_RADIANCE;
    let lobe = sun_radiance * spec_sun(normal, view, sun, 0.1, SKY_SUN_ANGULAR_RADIUS);

    let crest = waterplane_crest(wave.x, params.wave_strength, params.wave_scale);
    // Scattered light competes with the surface reflection, so it is weighted by (1 - F).
    let scatter = sss_crest_color(
      view, sun, normal, crest, params.scatter, sky_sun_color(sun) * params.sss_strength,
      params.sss_distortion, params.sss_power,
    ) * (1.0 - fresnel);

    let base = mix(body, reflected, fresnel) + lobe;
    switch (params.view_mode) {
      case 1u: { colour = scatter * f32(params.sss_on); }
      case 2u: { colour = vec3f(crest); }
      default: { colour = base + scatter * f32(params.sss_on); }
    }
  }

  var out = linear_to_srgb(tonemap_aces(colour * params.exposure));
  out += sky_dither(in.position.xy) / 255.0;
  return vec4f(out, 1.0);
}
