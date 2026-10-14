#include "fullscreen.wgsl"
#include "waveline.wgsl"
#include "sky.wgsl"
#include "divider_params.wgsl"

@group(0) @binding(0) var<uniform> params: DividerParams;

const MAX_LAYERS: i32 = 5;
// Highest wave crest allowed this close to the open edge, so a tall wave never clips flat.
const EDGE_MARGIN: f32 = 2.0;

// Layers are drawn back (t = 0) to front (t = 1). Colours are sRGB-encoded and composited in sRGB,
// as the browser composites the rest of the page. The result is premultiplied RGBA.

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  // CSS pixels; y runs down from the OPEN (transparent) edge, which `flip` moves to the bottom.
  let x = in.uv.x * params.size.x;
  let y = select(in.uv.y, 1.0 - in.uv.y, params.flip != 0u) * params.size.y;
  let height = params.size.y;
  let layers = clamp(params.layers, 1, MAX_LAYERS);
  let last = f32(layers - 1);

  var acc = vec4f(0.0);
  var exact = 0.0;
  for (var i = 0; i < MAX_LAYERS; i++) {
    if (i >= layers) { break; }
    let t = select(f32(i) / last, 1.0, layers < 2);
    let front = i == layers - 1;
    let seed = f32(i) * 1.7 + 0.3;
    let dir = select(1.0, -1.0, (i & 1) == 1);

    // Back layers: longer, lazier, slower. The front layer moves fastest and with the page.
    let wavelength = params.wavelength * mix(1.55, 1.0, t) * (1.0 + 0.09 * f32(i % 3));
    let phase = params.phase * dir * mix(0.45, 1.0, t) + seed * 2.1;
    let x_i = x - params.shift * mix(0.12, 1.0, t);

    // Baselines run from the front layer's mean up to `spread` of the height above it.
    let front_mean = 0.58 * height;
    let mean_i = front_mean - params.spread * 0.55 * height * (1.0 - t)
      + params.lift * (1.0 - t) * height;
    var amp = params.amplitude * mix(1.0, 0.72, t) * params.boost;
    amp = min(amp, max(mean_i - EDGE_MARGIN, 0.0) / 1.6);

    let c = waveline_curve(x_i, phase, wavelength, amp, params.roughness, seed);
    let d = waveline_distance(y, mean_i + c.x, c.y);
    let cover = waveline_coverage(d, params.px);

    var color = mix(params.color_back, params.color_front, t);
    var alpha = cover;
    if (front) {
      // The front layer is one flat colour so it can match a page section exactly.
      exact = select(0.0, 1.0, cover >= 1.0);
    } else {
      // A soft highlight along the crest and a little darkening with depth, for body.
      let rim = waveline_band(d, -0.7, 0.7, params.px) * params.foam * 0.4;
      color = mix(color, params.color_foam, rim);
      color *= mix(1.0, 0.9, smoothstep(0.0, 0.5 * height, -d));
      alpha = cover * params.back_alpha;
    }
    acc = vec4f(color * alpha, alpha) + acc * (1.0 - alpha);

    if (front) {
      // Foam rides the front crest: a crisp line and lace that reaches a little way over the water
      // behind it as well as into the body, so it shows on a pale front colour too.
      let reach_out = 0.5 * params.foam_width;
      let depth = -d + reach_out;
      if (depth > -params.px && depth < params.foam_width + reach_out + 2.0) {
        let cell = 4.5;
        let p = vec2f(x_i + params.phase * wavelength / TAU, y) / cell;
        let lace = waveline_foam(max(depth, 0.0), p, params.time, params.foam_width + reach_out, params.foam, params.px / cell);
        let rim = waveline_band(d, -0.6, 1.0, params.px) * params.foam;
        let foam = clamp(max(lace * 0.85, rim), 0.0, 1.0);
        acc = vec4f(params.color_foam * foam, foam) + acc * (1.0 - foam);
        exact = select(exact, 0.0, foam > 0.0);
      }
    }
  }

  // 8-bit steps show in the back layers' gentle gradients; the flat front body stays exact.
  let dither = sky_dither(in.position.xy) / 255.0 * (1.0 - exact);
  let rgb = clamp(acc.rgb + vec3f(dither) * acc.a, vec3f(0.0), vec3f(acc.a));
  return vec4f(rgb, acc.a);
}
