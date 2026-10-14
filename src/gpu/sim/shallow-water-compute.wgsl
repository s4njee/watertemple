// The compute-pass variant of the shallow-water solver: the same kernels as the render variant, one
// thread per cell, reading the previous state as a texture and writing write-only storage textures
// (no read-write storage textures, so it runs everywhere compute does). The resolve kernel writes
// three storage textures, which is the most a default-limits device allows in a stage but one.
#include "sw_core.wgsl"

@group(0) @binding(3) var out_a: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var out_water: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var out_ground: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var out_history: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16)
fn cs_upload(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.size)) {
    return;
  }
  textureStore(out_a, vec2i(id.xy), sw_upload(vec2i(id.xy)));
}

@compute @workgroup_size(16, 16)
fn cs_step(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.size)) {
    return;
  }
  textureStore(out_a, vec2i(id.xy), sw_step(vec2i(id.xy)));
}

@compute @workgroup_size(16, 16)
fn cs_resolve(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.size)) {
    return;
  }
  let r = sw_resolve(vec2i(id.xy));
  textureStore(out_water, vec2i(id.xy), r.water);
  textureStore(out_ground, vec2i(id.xy), r.ground);
  textureStore(out_history, vec2i(id.xy), r.history);
}
