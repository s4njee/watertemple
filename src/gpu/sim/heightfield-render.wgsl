// The render-pass variant of the heightfield solver: three fullscreen fragment passes over the
// grid-sized ping-pong render targets. Nothing here needs compute or storage textures, so the same
// code ports to WebGL2 (texelFetch + a half-float colour attachment).
#include "fullscreen.wgsl"
#include "sim_core.wgsl"

@fragment fn fs_splat(in: FullscreenVarying) -> @location(0) vec4f {
  return sim_splat(vec2i(in.position.xy));
}

@fragment fn fs_step(in: FullscreenVarying) -> @location(0) vec4f {
  return sim_step(vec2i(in.position.xy));
}

@fragment fn fs_resolve(in: FullscreenVarying) -> @location(0) vec4f {
  return sim_resolve(vec2i(in.position.xy));
}
