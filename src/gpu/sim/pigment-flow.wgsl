// pigment-flow.wgsl -- the paper holds the water back: damps the solver's velocity where the paper is
// dry (a custom pass for `fluid.hooks.forces`, see src/gpu/sim/pigment.ts). Water in the fibres of dry
// paper does not flow, so a face loses speed at the rate `drag` (1 / s) times how dry the paper under
// it is: v *= exp(-drag * (1 - ws) * dt). The pressure projection that follows removes the
// divergence this makes.
//
// Bindings: 0 U (PigmentUniforms: grid = velocity cells, size = dye texels, dt, drag, wet_lo/hi),
// 1 vel_in (the staggered velocity, (W+1) x (H+1) rgba16float: u in .x on the left face of a cell,
// v in .y on its top face; the last row of u and column of v repeat the previous one), 2 wet_in
// (r32float, dye sized), 3 vel_out (rgba16float storage, same size as vel_in). Entry point cs_flow.

#include "pigment_uniforms.wgsl"

@group(0) @binding(0) var<uniform> U: PigmentUniforms;
@group(0) @binding(1) var vel_in: texture_2d<f32>;
@group(0) @binding(2) var wet_in: texture_2d<f32>;
@group(0) @binding(3) var vel_out: texture_storage_2d<rgba16float, write>;

// Shaped wetness under a point given in cells of the velocity grid.
fn wetness_at(p_cells: vec2f) -> f32 {
  let uv = p_cells / vec2f(U.grid);
  let t = clamp(vec2i(floor(uv * vec2f(U.size))), vec2i(0), vec2i(U.size) - vec2i(1));
  return smoothstep(U.wet_lo, U.wet_hi, textureLoad(wet_in, t, 0).x);
}

@compute @workgroup_size(16, 16)
fn cs_flow(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= U.grid + vec2u(1u))) {
    return;
  }
  let c = vec2i(id.xy);
  let g = vec2i(U.grid);
  let v = textureLoad(vel_in, c, 0).xy;
  // The face each component belongs to (the padding row / column repeats the last real face).
  let u_at = vec2f(f32(c.x), f32(min(c.y, g.y - 1)) + 0.5);
  let v_at = vec2f(f32(min(c.x, g.x - 1)) + 0.5, f32(c.y));
  let du = exp(-U.drag * (1.0 - wetness_at(u_at)) * U.dt);
  let dv = exp(-U.drag * (1.0 - wetness_at(v_at)) * U.dt);
  textureStore(vel_out, c, vec4f(v.x * du, v.y * dv, 0.0, 0.0));
}
