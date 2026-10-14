// Six materials over the same retained fluid, with live head and spark geometry.
#include "fullscreen.wgsl"
#include "fluid.wgsl"
#include "color.wgsl"
#include "sdf.wgsl"
#include "display_params.wgsl"

@group(0) @binding(0) var<uniform> P: DisplayParams;
@group(0) @binding(1) var dye_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var fields_tex: texture_2d<f32>;

fn emitted(rgb: vec3f) -> vec3f {
  let light = max(rgb, vec3f(0.0)) * P.gain;
  let e = light - vec3f(min(light.r, min(light.g, light.b)) * 0.6);
  let peak = max(e.r, max(e.g, e.b));
  return e * (0.7 * (1.0 - exp(-peak / 0.7)) / max(peak, 1e-4));
}

fn spectral(t: f32) -> vec3f {
  return 0.48 + 0.46 * cos(6.28318 * (t + vec3f(0.0, 0.33, 0.67)));
}

// A procedural illuminated backdrop, sampled at displaced positions inside glass.
// Page text remains ordinary DOM and is never captured or distorted.
fn environment(p: vec2f) -> vec3f {
  let drift = P.time * 0.045;
  let window = pow(max(0.0, sin(p.x * 3.2 + p.y * 0.9 + drift)), 18.0);
  let horizon = exp(-pow((p.y - 0.42 - 0.09 * sin(p.x * 2.0)) * 8.0, 2.0));
  let line = pow(0.5 + 0.5 * cos(p.x * 45.0 + drift), 24.0);
  return P.ground + vec3f(0.012, 0.027, 0.035) * horizon
    + vec3f(0.015, 0.023, 0.03) * window + vec3f(0.004) * line;
}

fn glass(d: vec4f, p: vec2f, gradient: vec2f) -> vec3f {
  let depth = max(d.a, 0.0);
  let mask = 1.0 - exp(-depth * 1.8);
  let normal = normalize(vec3f(-gradient * 18.0, 1.0));
  let fresnel = pow(1.0 - normal.z, 2.0);
  let shift = normal.xy * 0.035 * mask;
  let refracted = environment(p + shift);
  let angle = dot(normal, normalize(vec3f(0.4 * sin(P.time * 0.1), -0.5, 1.0)));
  let film = 0.5 + 0.5 * sin(angle * 8.0 + depth * 0.9);
  let iridescence = mix(vec3f(0.015, 0.5, 0.4), vec3f(0.5, 0.025, 0.38), film);
  let light = normalize(vec3f(-0.45, -0.6, 0.8));
  let highlight = pow(max(dot(normal, normalize(light + vec3f(0.0, 0.0, 1.0))), 0.0), 120.0);
  let edge = clamp(length(gradient) * 3.0, 0.0, 1.0);
  let body = refracted * (0.7 + 0.3 * exp(-depth)) + iridescence * (0.08 + fresnel * 0.45);
  return mix(environment(p), body, mask) + mask * edge * (iridescence * 0.28 + vec3f(highlight * 0.65))
    + emitted(d.rgb) * mask * 0.06;
}

fn chrome(d: vec4f, gradient: vec2f) -> vec3f {
  let mask = 1.0 - exp(-max(d.a, 0.0) * 2.4);
  let normal = normalize(vec3f(-gradient * 24.0, 1.0));
  let reflected = reflect(vec3f(0.0, 0.0, -1.0), normal);
  let sweep = P.time * 0.075;
  let strip = exp(-pow((reflected.x + 0.28 * sin(sweep) - 0.2) * 15.0, 2.0));
  let softbox = exp(-pow((reflected.y + 0.38) * 4.0, 2.0));
  let edge = pow(1.0 - normal.z, 1.4);
  let metal = vec3f(0.012, 0.017, 0.027) + vec3f(0.68, 0.73, 0.8) * strip
    + vec3f(0.13, 0.16, 0.21) * softbox + vec3f(0.25, 0.29, 0.34) * edge;
  let prism = spectral(reflected.x * 0.7 + reflected.y * 0.3) * edge * 0.065;
  let alloy = max(d.rgb / max(d.a, 1e-4), vec3f(0.0));
  let hue = alloy / max(max(alloy.r, max(alloy.g, alloy.b)), 1e-4);
  return mix(P.ground, metal * mix(vec3f(1.0), hue, 0.55) + prism, mask);
}

fn head_lights(p: vec2f) -> vec3f {
  var result = vec3f(0.0);
  for (var i = 0u; i < P.head_count; i++) {
    let h = P.heads[i];
    let motion = P.head_motion[i];
    let colour = P.head_colours[i];
    let offset = p - h.xy;
    let velocity = motion.xy;
    let axis = select(vec2f(1.0, 0.0), velocity / max(length(velocity), 1e-4), length(velocity) > 1e-4);
    let along = dot(offset, axis);
    let across = dot(offset, vec2f(-axis.y, axis.x));
    let radius = max(P.pixel * 0.85, h.z * select(0.24, 2.1, P.style == 1u));
    var shape = exp(-pow(across / radius, 2.0) - pow(along / (radius * 1.7), 2.0));
    if (P.style == 1u) {
      // A soft ink blot with uneven edges and overlapping lobes, separate from the fine stroke.
      let angle = atan2(across, along);
      let phase = motion.z * 12.0 + P.time * 0.45;
      let edge = 1.0 + 0.13 * sin(angle * 3.0 + phase) + 0.09 * sin(angle * 5.0 - phase * 0.7);
      shape = exp(-pow(across / (radius * edge), 2.0) - pow(along / (radius * edge * 1.15), 2.0)) * 0.75;
      let lobe = vec2f(along - radius * 0.65, across + radius * 0.35);
      shape += exp(-dot(lobe, lobe) / (radius * radius * 0.5)) * 0.4;
    }
    let halo = exp(-dot(offset, offset) / (radius * radius * select(18.0, 3.0, P.style == 1u)));
    var tint = colour.rgb;
    var intensity = 0.8;
    if (P.style == 3u) {
      intensity = 0.15 + h.w * 1.5;
    }
    if (P.style == 5u) {
      tint = mix(tint, vec3f(1.0, 0.55, 0.1), 0.5);
      intensity = 0.6;
    }
    result += tint * colour.a * intensity * (shape + halo * 0.09);
  }
  if (P.style == 1u) {
    result -= vec3f(min(result.r, min(result.g, result.b)) * 0.75);
  }
  return result;
}

fn spark_lights(p: vec2f) -> vec3f {
  var result = vec3f(0.0);
  for (var i = 0u; i < P.spark_count; i++) {
    let spark = P.sparks[i];
    let radius = max(P.pixel * 0.7, spark.z);
    let offset = p - spark.xy;
    let distance = dot(offset, offset) / (radius * radius);
    let heat = spark.w * spark.w;
    let tint = mix(vec3f(0.6, 0.025, 0.003), vec3f(1.0, 0.65, 0.12), spark.w);
    result += tint * heat * (exp(-distance) + 0.055 * exp(-distance / 20.0));
  }
  return result;
}

fn eclipse(p: vec2f, d: vec4f) -> vec3f {
  var result = P.ground + emitted(d.rgb) * 0.3;
  for (var i = 0u; i < P.rect_count; i++) {
    let box = P.rects[i];
    let offset = p - box.xy;
    let distance = sd_rounded_box_2d(offset, box.zw, P.radii[i].x);
    let angle = atan2(offset.y, offset.x);
    let arc = pow(0.5 + 0.5 * sin(angle * 2.0 + P.time * 0.35 + f32(i) * 1.7), 5.0);
    let escape = pow(max(0.0, sin(P.time * 0.55 + f32(i) * 2.1)), 4.0);
    let shell = P.orbit_shell + arc * escape * 0.025;
    let width = max(P.pixel * 1.3, 0.0015);
    let rim = exp(-pow((distance - shell) / width, 2.0));
    let haze = exp(-pow((distance - shell) / (width * 5.0), 2.0));
    var tint = vec3f(0.12, 0.32, 0.8);
    if (P.head_count > 0u) {
      tint = P.head_colours[i % P.head_count].rgb;
    }
    result += tint * (rim * (0.12 + 0.58 * arc) + haze * 0.035);
  }
  return result;
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let uv = in.uv;
  let p = uv * vec2f(P.aspect, 1.0);
  let d = textureSampleLevel(dye_tex, samp, uv, 0.0);
  var colour = P.ground;
  if (P.view != 0u) {
    colour += fluid_flow_color(fluid_velocity(fluid_sample_fields(fields_tex, samp, uv)), 6.0);
  } else if (P.style == 1u) {
    // A dim diffuse afterimage beneath a narrower, denser stroke.
    let taper = smoothstep(0.015, 0.3, d.a);
    colour += emitted(d.rgb) * (0.025 + 0.075 * taper) + head_lights(p);
  } else if (P.style == 2u) {
    let gradient = fluid_dye_gradient(dye_tex, samp, uv, P.dye_texel) * P.dye_scale;
    colour = glass(d, p, gradient);
  } else if (P.style == 3u) {
    colour += emitted(d.rgb) * 0.23 + head_lights(p);
  } else if (P.style == 4u) {
    let gradient = fluid_dye_gradient(dye_tex, samp, uv, P.dye_texel) * P.dye_scale;
    colour = chrome(d, gradient);
  } else if (P.style == 5u) {
    let heat = 1.0 - exp(-max(d.a, 0.0) * P.gain * 1.5);
    let cooling = mix(vec3f(0.5, 0.015, 0.001), max(d.rgb / max(d.a, 1e-4), vec3f(0.0)), heat);
    let filament = pow(0.5 + 0.5 * sin(p.x * 650.0 + d.a * 6.0), 2.0);
    colour += emitted(cooling * d.a) * (0.25 + 0.5 * filament) + head_lights(p) + spark_lights(p);
  } else if (P.style == 6u) {
    colour = eclipse(p, d);
  }

  if (P.rect_count > 0u) {
    var distance = 1e9;
    for (var i = 0u; i < P.rect_count; i++) {
      let box = P.rects[i];
      distance = min(distance, sd_rounded_box_2d(p - box.xy, box.zw, P.radii[i].x));
    }
    colour = mix(colour, P.ground, clamp(0.5 - distance / P.pixel, 0.0, 1.0));
  }
  let radius = length((uv - 0.5) * vec2f(P.aspect, 1.0));
  colour *= 1.0 - P.vignette * smoothstep(0.35, 1.1, radius);
  // Compress combined tips and halos together, retaining hue at overlaps.
  let peak = max(colour.r, max(colour.g, colour.b));
  colour *= min(1.0, 0.88 / max(peak, 1e-4));
  return vec4f(linear_to_srgb(max(colour, vec3f(0.0)) * P.light), 1.0);
}
