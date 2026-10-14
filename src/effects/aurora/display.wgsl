// The orbit solver's retained dye becomes luminous folds and a soft upward curtain.
#include "fullscreen.wgsl"
#include "fluid.wgsl"
#include "color.wgsl"
#include "sdf.wgsl"
#include "display_params.wgsl"

@group(0) @binding(0) var<uniform> P: DisplayParams;
@group(0) @binding(1) var dye_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var fields_tex: texture_2d<f32>;

fn emission(rgb: vec3f) -> vec3f {
  let light = max(rgb, vec3f(0.0)) * P.gain;
  let e = light - vec3f(min(light.r, min(light.g, light.b)) * 0.5);
  let peak = max(e.r, max(e.g, e.b));
  return e * (0.65 * (1.0 - exp(-peak / 0.65)) / max(peak, 1e-4));
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let uv = in.uv;
  let d = textureSampleLevel(dye_tex, samp, uv, 0.0);
  let p = uv * vec2f(P.aspect, 1.0);
  var colour = P.ground;

  if (P.view == 0u) {
    // Fold phase follows the retained density, so the pleats bend with the flowing ribbons.
    let phase = p.x * 380.0 + sin(p.y * 15.0 + P.time * 0.13) * 4.0 + d.a * 6.0;
    let pleat = pow(0.5 + 0.5 * sin(phase), 3.0);
    let core = emission(d.rgb) * (0.46 + 0.54 * pleat);
    // A fine bright rim where the ribbon turns from dense to transparent.
    let gradient = fluid_dye_gradient(dye_tex, samp, uv, P.dye_texel) * P.dye_scale;
    let edge = clamp(length(gradient) * 2.0, 0.0, 0.5);
    var veil = vec3f(0.0);
    for (var i = 1; i <= 5; i++) {
      let height = f32(i) * 0.012;
      let sway = sin(p.y * 12.0 + P.time * 0.16 + f32(i) * 0.3) * height * 0.08;
      let source_uv = clamp(uv + vec2f(sway / P.aspect, height), vec2f(0.0), vec2f(1.0));
      let source = textureSampleLevel(dye_tex, samp, source_uv, 0.0);
      veil += emission(source.rgb) * (0.085 * exp(-f32(i) * 0.32));
    }
    colour += core * (1.0 + edge) + veil * (0.65 + 0.35 * pleat);
  } else {
    let velocity = fluid_velocity(fluid_sample_fields(fields_tex, samp, uv));
    colour += fluid_flow_color(velocity, 6.0);
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
  return vec4f(linear_to_srgb(clamp(colour, vec3f(0.0), vec3f(1.0)) * P.light), 1.0);
}
