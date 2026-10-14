// The compute-pass variant of the heightfield solver: the same three kernels as the render variant,
// one thread per cell, reading the previous state as a texture and writing a write-only storage
// texture (no read-write storage textures, so it runs everywhere compute does).
#include "sim_core.wgsl"

@group(0) @binding(3) var state_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16)
fn cs_splat(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.size)) {
    return;
  }
  textureStore(state_out, vec2i(id.xy), sim_splat(vec2i(id.xy)));
}

@compute @workgroup_size(16, 16)
fn cs_step(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.size)) {
    return;
  }
  textureStore(state_out, vec2i(id.xy), sim_step(vec2i(id.xy)));
}

@compute @workgroup_size(16, 16)
fn cs_resolve(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.size)) {
    return;
  }
  textureStore(state_out, vec2i(id.xy), sim_resolve(vec2i(id.xy)));
}
