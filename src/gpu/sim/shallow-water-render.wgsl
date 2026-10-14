// The render-pass variant of the shallow-water solver: fullscreen fragment passes over grid-sized
// ping-pong render targets. The state is rgba32float (renderable everywhere, not blendable, which the
// solver never needs); the resolve pass writes three rgba16float targets at once.
#include "fullscreen.wgsl"
#include "sw_core.wgsl"

@fragment fn fs_upload(in: FullscreenVarying) -> @location(0) vec4f {
  return sw_upload(vec2i(in.position.xy));
}

@fragment fn fs_step(in: FullscreenVarying) -> @location(0) vec4f {
  return sw_step(vec2i(in.position.xy));
}

struct ResolveOutput {
  @location(0) water: vec4f,
  @location(1) ground: vec4f,
  @location(2) history: vec4f,
}

@fragment fn fs_resolve(in: FullscreenVarying) -> ResolveOutput {
  let r = sw_resolve(vec2i(in.position.xy));
  return ResolveOutput(r.water, r.ground, r.history);
}
