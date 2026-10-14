#include "fullscreen.wgsl"
#include "color.wgsl"
#include "sky.wgsl"
#include "fresnel.wgsl"
#include "waterplane.wgsl"
#include "params.wgsl"

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var sky_cube: texture_cube<f32>;
@group(0) @binding(2) var sky_sampler: sampler;

// The same sky, from either source, so the two can be compared pixel for pixel.
fn sky_lookup(dir: vec3f, sun: vec3f) -> vec3f {
  if (params.use_cube == 1u) {
    return textureSampleLevel(sky_cube, sky_sampler, dir, 0.0).rgb;
  }
  return sky_radiance_ex(dir, sun, params.haze, 1.0);
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let sun = waterplane_sun_dir(params.sun_elevation, params.sun_azimuth);
  let dir = waterplane_ray_dir(in.uv, params.aspect, params.pitch, params.fov);
  let origin = vec3f(0.0, params.cam_height, 0.0);
  let t_plane = waterplane_hit(origin, dir);

  var colour = vec3f(0.0);
  if (t_plane < 0.0) {
    // Sky: show it dimmed in the debug views so the water stays the subject.
    colour = sky_lookup(dir, sun);
    if (params.view_mode != 0u) {
      colour = colour * 0.15;
    }
  } else {
    let footprint = waterplane_footprint(t_plane, dir.y, params.pixel_angle);
    let p = origin + dir * t_plane;
    let wave = waterplane_wave(p.xz, params.time, params.wave_strength, params.wave_scale, footprint);
    let normal = waterplane_normal(wave.yz);
    let view = -dir;

    let fresnel = fresnel_schlick(dot(normal, view), params.f0);
    let reflected = sky_lookup(waterplane_reflect(dir, normal), sun);
    let lit = sky_ambient(sun) * 0.8 + sky_sun_color(sun) * max(sun.y, 0.0) * 0.5;
    let body = params.body * lit;

    switch (params.view_mode) {
      case 1u: { colour = vec3f(fresnel) * 2.0; }
      case 2u: { colour = reflected; }
      case 3u: { colour = body * 4.0; }
      default: { colour = mix(body, reflected, fresnel); }
    }
  }

  var out = linear_to_srgb(tonemap_aces(colour * params.exposure));
  out += sky_dither(in.position.xy) / 255.0;
  return vec4f(out, 1.0);
}
