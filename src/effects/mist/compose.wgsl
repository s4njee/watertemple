// Full-resolution pass: reconstruct the three densities, weigh them into one, turn that into a
// capped opacity and write premultiplied tinted mist. No noise here, only four texture taps.
#include "fullscreen.wgsl"
#include "wash.wgsl"
#include "mist.wgsl"
#include "sky.wgsl"
#include "compose_params.wgsl"

@group(0) @binding(0) var<uniform> mist: ComposeParams;
@group(0) @binding(1) var layers_tex: texture_2d<f32>;
@group(0) @binding(2) var layers_sampler: sampler;

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let d = max(wash_sample_bspline(layers_tex, layers_sampler, in.uv).xyz, vec3f(0.0));
  // Near mist weighs a little less than far mist so the veil stays even rather than lumpy.
  let weights = vec3f(0.38, 0.34, 0.28);
  let density = dot(d, weights) * mist.density;
  let alpha = mist_opacity(density, mist.max_opacity);

  // Far mist is a touch dimmer and cooler than near mist: depth without a second tint.
  let shade = dot(d * vec3f(0.84, 0.92, 1.0), weights) / max(dot(d, weights), 1e-4);
  let tint = mist.tint * clamp(shade, 0.0, 1.0);

  // Soft veils band badly in 8 bits (alpha steps of 1/255 are visible across a smooth gradient).
  let dither = sky_dither(in.position.xy) / 255.0 * step(0.0005, alpha);
  let a = clamp(alpha + dither, 0.0, mist.max_opacity);
  return vec4f(tint * a, a);
}
