#include "fullscreen.wgsl"
#include "color.wgsl"
#include "life_params.wgsl"
@group(0) @binding(0) var<uniform> P: LifeParams;
@group(0) @binding(1) var cells: texture_2d<u32>;
fn at(p: vec2i) -> u32 { let n = vec2i(P.grid); return textureLoad(cells, ((p % n)+n)%n, 0).r; }
// Match ink orbit: each two-second leg crossfades linear RGB with zero slope at the primaries.
fn cycling_rgb(time: f32) -> vec3f {
  let phase = (time / 2.0) % 3.0;
  let channel = u32(floor(phase));
  let blend = smoothstep(0.0,1.0,fract(phase));
  var colour = vec3f(0);
  colour[channel] = 1.0-blend;
  colour[(channel+1u)%3u] = blend;
  return colour;
}
@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let q = in.uv * vec2f(P.grid); let p = vec2i(q); let value = at(p);
  let colour = cycling_rgb(P.time);
  let local = abs(fract(q)-0.5);
  let aa = min(0.16,max(fwidth(q.x),fwidth(q.y)) * 0.5);
  let cell = 1.0-smoothstep(0.34-aa,0.43+aa,max(local.x,local.y));
  let live = f32(value & 1u);
  let echo = f32((value>>16u)&255u)/9.0;
  var intensity = live * cell * 1.06 + echo * cell * (1.0-live) * 0.09;
  // Distance-weighted inner bloom and a broad aura, with no white added to the light.
  for (var y = -2; y <= 2; y++) { for (var x = -2; x <= 2; x++) {
    let offset = vec2i(x,y); let neighbour = at(p+offset);
    let distance = q - (vec2f(p+offset) + 0.5);
    let d2 = dot(distance,distance);
    let glow = exp(-d2 / 0.58) * 0.28 + exp(-d2 / 3.4) * 0.045;
    intensity += f32(neighbour & 1u) * glow;
  }}
  var emission = colour * intensity;
  let peak = max(emission.r,max(emission.g,emission.b));
  emission *= min(1.0,0.94/max(peak,1e-5));
  let grid = smoothstep(0.46,0.5,max(local.x,local.y));
  let ground = vec3f(0.001518,0.002125,0.005605) + vec3f(0.0008,0.0015,0.0025)*grid;
  let edge = min(min(in.uv.x,1.0-in.uv.x),min(in.uv.y,1.0-in.uv.y));
  return vec4f(linear_to_srgb(ground + emission * P.light * smoothstep(0.0,0.035,edge)),1);
}
