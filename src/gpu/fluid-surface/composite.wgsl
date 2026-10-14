// Composite pass of the fluid surface, at the size of the output. It reads the scene the caller drew
// (colour and depth), the chain's depth and normals (raw and smoothed), and the blurred thickness,
// and writes scene-plus-liquid:
//   * the surface is rebuilt by interpolating the chain with weights that skip background texels, so
//     a half resolution chain still gives a clean, anti-aliased silhouette
//   * the scene behind the liquid is looked up through an offset taken from the normal and scaled by
//     the thickness (the offset is dropped when it would pull in something in FRONT of the liquid)
//   * Beer-Lambert absorption by the thickness, Fresnel-weighted sky reflection and a GGX sun
//     highlight sit on top; the scalar `w` of the particles can tint towards foam
//   * the liquid is hidden by scene geometry in front of it (depth compare against the scene depth)
// With debug views it shows one intermediate instead.

#include "fluid-frame.wgsl"
#include "fullscreen.wgsl"
#include "camera.wgsl"
#include "fluid-surface.wgsl"
#include "fresnel.wgsl"
#include "sky.wgsl"
#include "specular.wgsl"
#include "absorption.wgsl"
#include "refraction.wgsl"
#include "color.wgsl"

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> frame: FluidFrame;
@group(0) @binding(2) var scene_color: texture_2d<f32>;
@group(0) @binding(3) var scene_depth: texture_depth_2d;
@group(0) @binding(4) var samp: sampler;
@group(0) @binding(5) var raw_depth: texture_2d<f32>;
@group(0) @binding(6) var raw_normals: texture_2d<f32>;
@group(0) @binding(7) var smooth_depth: texture_2d<f32>;
@group(0) @binding(8) var smooth_normals: texture_2d<f32>;
@group(0) @binding(9) var thickness_image: texture_2d<f32>;

struct Surface {
  depth: f32,
  normal: vec3f,
  // Fraction of the four chain texels around the pixel that hold fluid (bilinear-weighted).
  coverage: f32,
}

fn fetch_surface(depth_image: texture_2d<f32>, normal_image: texture_2d<f32>, uv: vec2f) -> Surface {
  let size = vec2i(textureDimensions(depth_image));
  let position = uv * vec2f(size) - 0.5;
  let base = vec2i(floor(position));
  let f = position - floor(position);
  var depth_sum = 0.0;
  var normal_sum = vec3f(0.0);
  var weight_sum = 0.0;
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      let q = clamp(base + vec2i(i, j), vec2i(0), size - 1);
      let w = select(1.0 - f.x, f.x, i == 1) * select(1.0 - f.y, f.y, j == 1);
      let z = textureLoad(depth_image, q, 0).r;
      if (z > 0.0) {
        depth_sum += w * z;
        normal_sum += w * textureLoad(normal_image, q, 0).xyz;
        weight_sum += w;
      }
    }
  }
  var s: Surface;
  s.coverage = weight_sum;
  if (weight_sum <= 0.0) {
    s.depth = 0.0;
    s.normal = vec3f(0.0, 0.0, 1.0);
    return s;
  }
  s.depth = depth_sum / weight_sum;
  let n = normal_sum / weight_sum;
  s.normal = select(vec3f(0.0, 0.0, 1.0), normalize(n), dot(n, n) > 1e-8);
  return s;
}

fn finish(c: vec3f) -> vec4f {
  if (frame.output_srgb != 0u) {
    return vec4f(linear_to_srgb(tonemap_aces(max(c, vec3f(0.0)))), 1.0);
  }
  return vec4f(c, 1.0);
}

fn depth_colour(z: f32) -> vec3f {
  // Brightness falls with distance; thin dark lines mark equal depth every half radius, which makes
  // the lumps of a raw sphere cloud and the flatness of a smoothed surface obvious.
  let g = 0.95 * exp(-0.12 * z);
  let contour = abs(fract(z / (0.5 * frame.radius)) - 0.5) * 2.0;
  return vec3f(g * (1.0 - 0.55 * smoothstep(0.8, 1.0, contour)));
}

fn heat(v: f32) -> vec3f {
  let t = clamp(v, 0.0, 1.0);
  return mix(mix(vec3f(0.02, 0.04, 0.2), vec3f(0.1, 0.6, 0.7), smoothstep(0.0, 0.5, t)), vec3f(1.0, 0.92, 0.45), smoothstep(0.5, 1.0, t));
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let pixel = vec2i(in.position.xy);
  let uv = in.uv;
  let last = vec2i(frame.full_size) - 1;
  let scene = textureLoad(scene_color, pixel, 0).rgb;
  if (frame.live == 0u) {
    return finish(scene);
  }

  var use_raw = frame.surface == 1u;
  if (frame.surface == 2u) {
    use_raw = uv.x < frame.split;
  }
  var s: Surface;
  if (use_raw) {
    s = fetch_surface(raw_depth, raw_normals, uv);
  } else {
    s = fetch_surface(smooth_depth, smooth_normals, uv);
  }
  let thick_sample = textureSampleLevel(thickness_image, samp, uv, 0.0).xy;
  let thick = max(thick_sample.x, 0.0);
  let foam_mean = select(0.0, thick_sample.y / thick, thick > 1e-4);

  var divider = 0.0;
  if (frame.surface == 2u) {
    divider = 1.0 - smoothstep(0.5, 1.5, abs(in.position.x - frame.split * frame.full_size.x));
  }

  // Debug views replace the liquid with one intermediate on a dark ground.
  if (frame.debug != 0u) {
    var debug_colour = vec3f(0.035);
    let covered = clamp(s.coverage, 0.0, 1.0);
    if (frame.debug == 1u || frame.debug == 2u) {
      // The depth that was asked for, whichever half of a split it is.
      var d: Surface;
      if (frame.debug == 1u) {
        d = fetch_surface(raw_depth, raw_normals, uv);
      } else {
        d = fetch_surface(smooth_depth, smooth_normals, uv);
      }
      debug_colour = mix(debug_colour, depth_colour(d.depth), clamp(d.coverage, 0.0, 1.0));
    } else if (frame.debug == 3u) {
      debug_colour = mix(debug_colour, s.normal * 0.5 + 0.5, covered);
    } else if (frame.debug == 4u) {
      debug_colour = heat(1.0 - exp(-0.5 * thick));
    } else {
      debug_colour = vec3f(clamp(foam_mean * max(frame.foam_amount, 1.0), 0.0, 1.0));
    }
    return finish(mix(debug_colour, vec3f(1.0), divider));
  }

  if (s.coverage <= 0.0) {
    return finish(mix(scene, vec3f(1.0), divider));
  }

  // Hidden by the scene? Fade over a fraction of a particle so the cut with a floor is not a stair.
  let scene_z = camera_linear_depth(cam, textureLoad(scene_depth, pixel, 0));
  let slack = 0.35 * frame.radius;
  let hidden = smoothstep(scene_z - slack, scene_z + slack, s.depth);
  let alpha = clamp(s.coverage, 0.0, 1.0) * (1.0 - hidden);
  if (alpha <= 0.0) {
    return finish(mix(scene, vec3f(1.0), divider));
  }

  let n_view = s.normal;
  let p_view = fluid_surface_view_pos(cam, uv, s.depth);
  let v_view = normalize(-p_view);
  let n_world = normalize((cam.inv_view * vec4f(n_view, 0.0)).xyz);
  let v_world = normalize((cam.inv_view * vec4f(v_view, 0.0)).xyz);
  let sun = normalize(frame.sun_dir);

  // Refraction: the offset is in image-height units, so convert the thickness from world units.
  let aspect = frame.full_size.x / frame.full_size.y;
  let height_world = 2.0 * s.depth / cam.proj[1][1];
  let bend_thickness = min(thick, frame.max_refract) * frame.refraction / height_world;
  let n_screen = vec3f(n_view.x, -n_view.y, max(n_view.z, 0.1));
  let offset = refraction_to_uv(refraction_offset_cheap(n_screen, frame.ior, bend_thickness), aspect);
  var bed_uv = refraction_safe_uv(uv, offset, 0.03);
  let refracted_z = camera_linear_depth(cam, textureLoad(scene_depth, clamp(vec2i(bed_uv * frame.full_size), vec2i(0), last), 0));
  if (refracted_z < s.depth) {
    // Something stands in front of the liquid there: do not drag it into the liquid.
    bed_uv = uv;
  }
  let bed = textureSampleLevel(scene_color, samp, bed_uv, 0.0).rgb;

  var water = AbsorbWater(frame.sigma, frame.scatter, frame.turbidity);
  if (frame.water_type >= 0) {
    water = absorb_water(frame.water_type);
  }
  let body = absorb_composite(bed, water.sigma, water.turbidity, water.scatter, thick * frame.density);

  let f0 = fresnel_f0(1.0, frame.ior);
  let fresnel = fresnel_schlick(dot(n_world, v_world), f0) * frame.reflection;
  let sky = sky_radiance_ex(reflect(-v_world, n_world), sun, frame.haze, 0.0);
  let glint = frame.specular * sky_sun_color(sun) * SKY_SUN_RADIANCE
    * spec_sun(n_world, v_world, sun, frame.roughness, SKY_SUN_ANGULAR_RADIUS);
  var colour = body * (1.0 - fresnel) + sky * fresnel + glint;

  let foam = clamp(foam_mean * frame.foam_amount, 0.0, 1.0);
  let lit = sky_ambient(sun) + sky_sun_color(sun) * max(dot(n_world, sun), 0.0);
  colour = mix(colour, frame.foam_color * lit, foam);

  return finish(mix(mix(scene, colour, alpha), vec3f(1.0), divider));
}
