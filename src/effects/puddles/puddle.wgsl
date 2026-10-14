// The wet street, per frame. A perspective camera looks at an endless ground plane (the baked tile
// repeats); each pixel
//   - reads the baked albedo, height and normal (two anisotropic taps, before any branching),
//   - decides how deep the water is there (level minus height) and how damp the ground is,
//   - shades the ground: wet albedo, GGX sun, a dull sheen of the sky on damp stone,
//   - or, over water, mirrors the environment through the ripple normals, weighted by Fresnel, over the
//     ground seen through the water; the rings come from ripples.wgsl.
// Pixels above the horizon show the same environment, so the reflections match what is "there".
#include "fullscreen.wgsl"
#include "math.wgsl"
#include "hash.wgsl"
#include "color.wgsl"
#include "normals.wgsl"
#include "fresnel.wgsl"
#include "sky.wgsl"
#include "specular.wgsl"
#include "ripples.wgsl"
#include "wetness.wgsl"
#include "puddle_params.wgsl"

@group(0) @binding(0) var<uniform> P: PuddleParams;
@group(0) @binding(1) var ground_tex: texture_2d<f32>;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var ground_samp: sampler;

// --- The environment --------------------------------------------------------------------------------
// A dusk city: sky from sky.wgsl, two skylines with lit windows, and a few glowing lamps and signs,
// so a reflection has shapes and colours to show. Directions are world space, y up, +z straight ahead.

// A glowing point in the environment: `target_dir` is a precomputed unit direction.
fn lamp(dir: vec3f, target_dir: vec3f, size: f32, colour: vec3f) -> vec3f {
  let d = distance(dir, target_dir) / size;
  return colour * (exp(-d * d) + 0.06 * exp(-d * 0.5));
}

// Whether the direction hits a building of one skyline: a row of columns of random height.
fn skyline_hit(dir: vec3f, az01: f32, columns: f32, seed: i32, base: f32, tall: f32, gap: f32) -> bool {
  let h = ihash_2to1(vec2i(i32(floor(az01 * columns)), seed));
  return dir.y <= min(base + tall * h * h, gap);
}

fn env(dir_in: vec3f, sun: vec3f, haze: f32, sun_gain: f32, gap_gain: f32, lamp_gain: f32) -> vec3f {
  let dir = normalize(vec3f(dir_in.x, max(dir_in.y, 0.0), dir_in.z));
  // Rain clouds are darker than the clear-sky model, which only gets milkier with haze.
  var col = sky_radiance_ex(dir, sun, haze, sun_gain) * mix(1.0, 0.5, saturate((haze - 0.3) / 0.7));
  let horizon = sky_horizon_color(sun, haze);
  let az01 = atan2(dir.x, dir.z) / TAU + 0.5;
  // Keep the sun in a gap between the buildings: they get lower toward its azimuth.
  let sun_az01 = atan2(sun.x, sun.z) / TAU + 0.5;
  let daz = abs(fract(az01 - sun_az01 + 0.5) - 0.5);
  let gap = mix(1.0, mix(max(sun.y * 0.55, 0.01), 1.0, smoothstep(0.004, 0.05, daz)), smoothstep(0.4, 0.8, gap_gain));

  if (skyline_hit(dir, az01, 120.0, 11, 0.025, 0.16, gap)) {
    let body = horizon * 0.35 + vec3f(0.012, 0.014, 0.022);
    let cell = vec2f(az01 * 120.0 * 5.0, dir.y * 170.0);
    let wh = ihash_3to3(vec3i(vec2i(floor(cell)), 4));
    let lit = step(wh.x, 0.18) * step(0.15, fract(cell.x)) * step(0.2, fract(cell.y));
    col = body + vec3f(1.0, 0.7, 0.4) * lit * (0.4 + wh.y) * 0.45 * (0.3 + lamp_gain);
  }
  if (skyline_hit(dir, az01, 34.0, 23, 0.02, 0.2, gap)) {
    let body = horizon * 0.12 + vec3f(0.006, 0.007, 0.011);
    let cell = vec2f(az01 * 34.0 * 9.0, dir.y * 120.0);
    let wh = ihash_3to3(vec3i(vec2i(floor(cell)), 9));
    let lit = step(wh.x, 0.3) * step(0.18, fract(cell.x)) * step(0.22, fract(cell.y));
    let tint = mix(vec3f(1.0, 0.72, 0.4), vec3f(0.6, 0.8, 1.0), step(0.85, wh.z));
    col = body + tint * lit * (0.5 + wh.y) * 0.6 * (0.3 + lamp_gain);
  }
  // Lamps and signs: sodium, cyan, magenta, tungsten, red, green. Only the lamps near this direction
  // matter, so skip the lot for directions well above the roof line.
  var glow = vec3f(0.0);
  if (dir.y < 0.2) {
    glow += lamp(dir, vec3f(-0.5794, 0.0749, 0.8116), 0.02, vec3f(1.0, 0.5, 0.15) * 9.0);
    glow += lamp(dir, vec3f(-0.1970, 0.1296, 0.9718), 0.016, vec3f(0.2, 0.8, 1.0) * 8.0);
    glow += lamp(dir, vec3f(0.4429, 0.0699, 0.8939), 0.022, vec3f(1.0, 0.2, 0.7) * 8.0);
    glow += lamp(dir, vec3f(0.7786, 0.1098, 0.6179), 0.018, vec3f(1.0, 0.8, 0.5) * 9.0);
    glow += lamp(dir, vec3f(-0.8659, 0.0600, 0.4967), 0.02, vec3f(1.0, 0.1, 0.06) * 7.0);
    glow += lamp(dir, vec3f(0.9935, 0.0899, 0.0705), 0.02, vec3f(0.4, 1.0, 0.6) * 6.0);
  }
  col += glow * lamp_gain;
  return col;
}

// --- The pass ---------------------------------------------------------------------------------------

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let ndc = vec2f(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
  let dir = normalize(P.cam_fwd + P.cam_right * (ndc.x * P.tan_half * P.aspect) + P.cam_up * (ndc.y * P.tan_half));

  // Where the view ray meets the ground; clamped so pixels at the horizon keep sane derivatives.
  let t = min(P.cam_pos.y / max(-dir.y, 0.02), 90.0);
  let pw = P.cam_pos.xz + t * dir.xz;
  let uvw = pw / P.tile_m;
  // All texture reads happen here, in uniform control flow.
  let g0 = textureSample(ground_tex, ground_samp, uvw);
  let g1 = textureSample(normal_tex, ground_samp, uvw);

  let sun = P.sun;
  let rain = P.rain;
  let haze = 0.3 + 0.7 * rain;
  let sun_visible = 1.0 - 0.9 * rain;

  var col = vec3f(0.0);
  if (dir.y >= 0.0) {
    col = env(dir, sun, haze, sun_visible, sun_visible, P.lamps);
  } else {
    let footprint = t * P.pixel / max(-dir.y, 0.08);   // metres of ground per pixel, along the view
    let n_tex = normal_from_xy(g1.xy);
    let n_ground = normal_to_world_up(n_tex);
    let slope = length(g1.xy) / max(n_tex.z, 0.1);
    let h = g0.a * P.height_m;
    let level = P.level * P.height_m;
    let depth = wet_depth(h, level);
    let edge = clamp(slope * footprint, 0.0002, 0.012);
    let water = wet_water_mask(depth, edge);
    let wet = wet_damp(h, level, P.damp_reach, P.wetness);

    let view = -dir;
    let sun_colour = sky_sun_color(sun) * sun_visible;
    // The street lights fill the shadows with a warm glow.
    let ambient = sky_ambient(sun) * (0.6 + 0.4 * rain) + vec3f(0.07, 0.05, 0.03) * P.lamps;

    // Ground: wet albedo, sun and sky diffuse, glossier where wet.
    let albedo = wet_albedo(g0.rgb, wet);
    let rough = wet_roughness(P.roughness, wet);
    let ndl = max(dot(n_ground, sun), 0.0);
    var ground = albedo * (sun_colour * ndl * 2.4 + ambient * 1.6);
    let cos_g = max(dot(n_ground, view), 0.0);
    let glare = spec_sun(n_ground, view, sun, max(rough, 0.05), SKY_SUN_ANGULAR_RADIUS) * sun_colour * SKY_SUN_RADIANCE;
    ground += glare * smoothstep(0.2, 0.9, wet) * 0.35;
    let sheen_w = wet_reflectance(cos_g, wet, 0.0, rough);
    let sheen_dir = reflect(dir, n_ground);
    ground += sky_radiance_ex(vec3f(sheen_dir.x, max(sheen_dir.y, 0.0), sheen_dir.z), sun, haze, 0.0) * sheen_w;
    col = ground;

    if (water > 0.002) {
      // Rain ripples: two layers of rings at different sizes; the distant ones fade out.
      let cell = P.ripple_cell;
      let density = rain;
      let lod = ripple_lod_fade(footprint / cell);
      var slope_sum = vec2f(0.0);
      if (lod > 0.002 && rain > 0.01) {
        let gate = saturate(depth / 0.0025) * smoothstep(0.0, 0.05, rain);
        let a = ripple_rings(pw / cell, P.time, density, P.ripple_life, 0);
        slope_sum = a.yz * lod * gate;
        // The finer second layer only where its rings are still resolved.
        let lod_b = ripple_lod_fade(footprint / (cell * 0.62));
        if (lod_b > 0.002) {
          let b = ripple_rings(pw / (cell * 0.62) + vec2f(7.3, 2.9), P.time * 1.1, density * 0.8, P.ripple_life * 0.8, 1);
          slope_sum += b.yz * (0.6 * lod_b * gate);
        }
      }
      let n_water = normal_to_world_up(ripple_normal(vec3f(0.0, slope_sum), P.ripple_crest));
      let cos_w = max(dot(n_water, view), 0.0);
      // A little more reflective than clean water: puddles are dirty and the bed is dark.
      let fres = saturate(fresnel_water(cos_w) * 1.15 + 0.01);
      let refl_dir = reflect(dir, n_water);
      let reflection = env(refl_dir, sun, haze, 0.0, sun_visible, P.lamps);
      let bed_wet = wet_albedo(g0.rgb, 1.0) * (sun_colour * ndl * 2.4 + ambient * 1.6);
      let seen = wet_through_water(bed_wet, depth, vec3f(34.0, 24.0, 18.0));
      let glint = spec_sun(n_water, view, sun, 0.03, SKY_SUN_ANGULAR_RADIUS) * sun_colour * SKY_SUN_RADIANCE;
      let pool = seen * (1.0 - fres) + reflection * fres + glint;
      col = mix(ground, pool, water);
    }

    // Distance haze toward the horizon: by distance, and always fully at the horizon itself.
    let haze_amount = max(1.0 - exp(-t * (0.012 + 0.05 * rain)), 1.0 - smoothstep(0.0, 0.03, -dir.y));
    if (haze_amount > 0.01) {
      let far_colour = env(vec3f(dir.x, 0.0, dir.z), sun, haze, sun_visible, sun_visible, P.lamps);
      col = mix(col, far_colour, haze_amount);
    }
  }

  let mapped = tonemap_aces(col * P.exposure);
  return vec4f(linear_to_srgb(mapped), 1.0);
}
