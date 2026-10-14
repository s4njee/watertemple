// volume-prep.wgsl -- the two small compute passes that make the ray march cheap.
//
//   cs_light      for every voxel of a coarse light volume, the transmittance of the ink between that voxel and the
//                 light (rgb, coloured: an ink casts a tinted shadow), and in alpha how open the voxel is to the sky
//                 straight above it. One march of `light_steps` taps per voxel, done once per frame instead of
//                 once per ray sample.
//   cs_occupancy  the largest ink amount in each 4 x 4 x 4 block of the ink grid (and a texel of margin), so the
//                 march can skip empty space with one tap of a tiny texture instead of a tap of the big one plus
//                 the lighting.
//
// Bindings: 0 `P` (ink.wgsl), 1 the ink (filterable 3D), 2 a linear sampler, 3 the output.

#include "volume.wgsl"
#include "ink.wgsl"

@group(0) @binding(1) var dens: texture_3d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var out_light: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var out_occ: texture_storage_3d<rgba16float, write>;

fn box_uvw(p: vec3f) -> vec3f {
  return volume_to_uvw(p, P.box_min, P.box_max);
}

// Optical depth (per channel) from p along the unit direction d to the wall of the box, in `n` mid-point taps.
fn optical_depth(p: vec3f, d: vec3f, n: u32, max_length: f32) -> vec3f {
  let span = volume_ray_box(p, d, P.box_min, P.box_max);
  let length = min(max(span.y, 0.0), max_length);
  if (length <= 0.0) {
    return vec3f(0.0);
  }
  let ds = length / f32(n);
  var tau = vec3f(0.0);
  for (var i = 0u; i < n; i++) {
    let q = p + d * ((f32(i) + 0.5) * ds);
    tau += ink_extinction(textureSampleLevel(dens, samp, box_uvw(q), 0.0)) * ds;
  }
  return tau;
}

@compute @workgroup_size(4, 4, 4)
fn cs_light(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= P.light_size)) {
    return;
  }
  let p = P.box_min + (vec3f(id) + 0.5) / vec3f(P.light_size) * (P.box_max - P.box_min);
  let toward_light = optical_depth(p, P.light_dir, P.light_steps, 1e9);
  // How much ink is straight above, over at most half a box height: ambient light from the sky.
  let above = optical_depth(p, vec3f(0.0, 1.0, 0.0), max(P.light_steps / 3u, 3u), 0.5 * (P.box_max.y - P.box_min.y));
  let open_sky = exp(-dot(above, vec3f(1.0 / 3.0)));
  textureStore(out_light, vec3i(id), vec4f(exp(-toward_light), open_sky));
}

@compute @workgroup_size(4, 4, 4)
fn cs_occupancy(@builtin(global_invocation_id) id: vec3u) {
  let size = vec3i(textureDimensions(out_occ));
  if (any(vec3i(id) >= size)) {
    return;
  }
  let fine = vec3i(textureDimensions(dens));
  let lo = vec3i(id) * 4 - vec3i(1);
  var most = 0.0;
  for (var k = 0; k < 6; k++) {
    for (var j = 0; j < 6; j++) {
      for (var i = 0; i < 6; i++) {
        let c = clamp(lo + vec3i(i, j, k), vec3i(0), fine - vec3i(1));
        most = max(most, textureLoad(dens, c, 0).a);
      }
    }
  }
  textureStore(out_occ, vec3i(id), vec4f(most, 0.0, 0.0, 0.0));
}
