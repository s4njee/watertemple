#include "fullscreen.wgsl"
#include "color.wgsl"
#include "caustics.wgsl"
#include "caustics_params.wgsl"

@group(0) @binding(0) var<uniform> P: CausticsParams;
@group(0) @binding(1) var baked_tex: texture_2d<f32>;
@group(0) @binding(2) var baked_samp: sampler;

const METHOD_BAKED: u32 = 2u;
const VIEW_OVERLAY: u32 = 0u;
const VIEW_OPAQUE: u32 = 1u;
const VIEW_SEAM: u32 = 2u;
// Pattern cells per tile, shared by the procedural methods and the bake (see index.ts).
const CELLS: i32 = 12;

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let pos = vec2f(in.uv.x * P.aspect, in.uv.y);
  // `scale` is cells across the image height; the tile is CELLS cells wide.
  var uv = pos * (P.scale / f32(CELLS));
  if (P.view == VIEW_SEAM) {
    uv += vec2f(0.5);
  }

  var c: vec3f;
  if (P.method == METHOD_BAKED) {
    c = caustics_baked_rgb(baked_tex, baked_samp, uv, P.time, P.lod, P.combine, P.split);
  } else {
    c = caustics_rgb(P.method, P.layers, uv, CELLS, P.time, P.sharp, P.split);
  }
  // Exposure: open-ended intensity to [0, 1) per channel.
  let e = vec3f(
    caustics_expose(c.r, P.intensity),
    caustics_expose(c.g, P.intensity),
    caustics_expose(c.b, P.intensity),
  );
  let light = P.tint * e;

  if (P.view == VIEW_OVERLAY) {
    // Premultiplied light over whatever the page has: alpha is the strongest channel, so a
    // bright filament covers the page a little and a faint one only tints it.
    let a = clamp(max(light.r, max(light.g, light.b)) * P.opacity, 0.0, 1.0);
    let straight = light / max(max(light.r, max(light.g, light.b)), 1e-4);
    return vec4f(linear_to_srgb(straight) * a, a);
  }

  var colour = P.deep + light * 0.9;
  if (P.view == VIEW_SEAM) {
    // The real tile seams, drawn where uv is a whole number; a pattern that tiles shows no break
    // in the filaments across them.
    let d = abs(fract(uv + vec2f(0.5)) - 0.5);
    let px = length(fwidth(uv));
    let line = 1.0 - smoothstep(0.0, 1.5 * px, min(d.x, d.y));
    colour = mix(colour, vec3f(1.0, 0.25, 0.2), 0.8 * line);
  }
  return vec4f(linear_to_srgb(clamp(colour, vec3f(0.0), vec3f(1.0))), 1.0);
}
