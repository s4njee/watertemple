// shallow-water.wgsl -- read the output textures of the shallow-water solver
// (src/gpu/sim/shallow-water.ts) without knowing how they are laid out. Everything a shading pass
// needs: water depth, flow velocity, foam, the bed, the slope of the water surface and its normal.
//
// THE TEXTURES. The solver publishes two rgba16float textures, filterable, one texel per grid cell,
// covering the whole domain: uv (0, 0) is the top-left corner and (1, 1) the bottom-right. A 3D
// consumer maps uv.x to world x and uv.y to world z (the grid's "down" is +z).
//   water  = solver.view         x depth (m)   y velocity x (m/s)   z velocity y (m/s)   w foam [0, 1]
//   ground = solver.groundView   x bed height (m)   y surface slope x   z surface slope y   w ground
//                                wetness [0, 1]
// Velocity is the depth-averaged flow, zero where there is no water. The slopes are d(level)/dx and
// d(level)/dy of the water surface (bed + depth) where there is water, of the bed where there is
// none, per metre, so a slope of 0.1 is a 5.7 degree tilt: they are the same number whatever the grid
// resolution. At a shoreline the surface is taken to be flat up to the dry ground, so the water
// does not tilt up the beach. Foam is born at bores, on thin fast sheets and at advancing fronts, is
// carried with the flow and fades over solver.params.foamLife seconds. Ground wetness is 1 under
// water and dries out over solver.params.wetLife seconds, so the sand stays dark behind a retreating
// shore.
//
// FRAME. World axes: x right, y up, z toward the viewer (uv y). The free surface normal of a flat
// pool is (0, 1, 0).
//
// Decoding a water texel `w` (a vec4f from the water texture)
//   swe_depth(w) -> f32                      water depth in metres, >= 0
//   swe_velocity(w) -> vec2f                 (x, z) velocity in m/s
//   swe_velocity_world(w) -> vec3f           (vx, 0, vz) in world axes
//   swe_speed(w) -> f32                      |velocity| in m/s
//   swe_foam(w) -> f32                       foam amount [0, 1]
//   swe_froude(w, gravity) -> f32            speed / sqrt(g depth): above 1 the flow is supercritical
//                                            (a torrent, or a bore front); 0 on dry ground
//
// Decoding a ground texel `g`
//   swe_bed(g) -> f32                        bed height in metres
//   swe_surface_slope(g) -> vec2f            (d/dx, d/dz) of the visible surface
//   swe_ground_wetness(g) -> f32             [0, 1]
//
// Combining
//   swe_surface(w, g) -> f32                 water surface height: bed + depth
//   swe_wet_mask(depth, edge) -> f32         coverage [0, 1]: 0 at depth <= 0, 1 at depth >= edge, smooth
//                                            and monotonic between. Use it as the alpha of a water mesh so
//                                            the shoreline is soft and does not stair-step. edge <= 0
//                                            gives a hard step. Roughly one cell's rise of a beach (the
//                                            beach slope times the cell size) gives a crisp edge
//   swe_surface_normal(g, strength) -> vec3f unit normal of the water surface from the stored slopes;
//                                            strength 1 is physical, larger exaggerates
//   swe_terrain_normal(bed_w, bed_e, bed_n, bed_s, spacing) -> vec3f
//                                            unit normal of the bed from four bed heights one `spacing`
//                                            metres west, east, north (-z) and south (+z) of the point
//
// Fetching (the texture and sampler are parameters; this chunk declares no bindings)
//   swe_sample_water(tex, samp, uv) -> vec4f            one bilinear tap; needs a filtering sampler
//   swe_sample_ground(tex, samp, uv) -> vec4f
//   swe_sample_bed_normal(tex, samp, uv, texel, cell_size) -> vec3f
//                                            bed normal from four taps of the ground texture, `texel`
//                                            = solver.grid.texelSize (uv), `cell_size` = grid.cellSize (m)
//   swe_load_water(tex, cell) -> vec4f       the exact texel of a cell (clamped to the grid)
//
// Cost: a decode is free; the normal is a normalize (about 10 ALU); swe_sample_bed_normal is four taps.
//
// Why bilinear taps are fine here: depth, velocity and slope are all smooth fields on the grid. The
// one thing a bilinear tap cannot give is a sharp shoreline, which swe_wet_mask provides.

fn swe_depth(w: vec4f) -> f32 {
  return max(w.x, 0.0);
}

fn swe_velocity(w: vec4f) -> vec2f {
  return w.yz;
}

fn swe_velocity_world(w: vec4f) -> vec3f {
  return vec3f(w.y, 0.0, w.z);
}

fn swe_speed(w: vec4f) -> f32 {
  return length(w.yz);
}

fn swe_foam(w: vec4f) -> f32 {
  return clamp(w.w, 0.0, 1.0);
}

fn swe_froude(w: vec4f, gravity: f32) -> f32 {
  let h = max(w.x, 0.0);
  if (h <= 1e-4) {
    return 0.0;
  }
  return length(w.yz) / sqrt(max(gravity, 1e-3) * h);
}

fn swe_bed(g: vec4f) -> f32 {
  return g.x;
}

fn swe_surface_slope(g: vec4f) -> vec2f {
  return g.yz;
}

fn swe_ground_wetness(g: vec4f) -> f32 {
  return clamp(g.w, 0.0, 1.0);
}

fn swe_surface(w: vec4f, g: vec4f) -> f32 {
  return g.x + max(w.x, 0.0);
}

fn swe_wet_mask(depth: f32, edge: f32) -> f32 {
  if (edge <= 0.0) {
    return select(0.0, 1.0, depth > 0.0);
  }
  let t = clamp(depth / edge, 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

fn swe_surface_normal(g: vec4f, strength: f32) -> vec3f {
  return normalize(vec3f(-g.y * strength, 1.0, -g.z * strength));
}

fn swe_terrain_normal(bed_w: f32, bed_e: f32, bed_n: f32, bed_s: f32, spacing: f32) -> vec3f {
  let inv = 0.5 / max(spacing, 1e-6);
  return normalize(vec3f(-(bed_e - bed_w) * inv, 1.0, -(bed_s - bed_n) * inv));
}

fn swe_sample_water(tex: texture_2d<f32>, samp: sampler, uv: vec2f) -> vec4f {
  return textureSampleLevel(tex, samp, uv, 0.0);
}

fn swe_sample_ground(tex: texture_2d<f32>, samp: sampler, uv: vec2f) -> vec4f {
  return textureSampleLevel(tex, samp, uv, 0.0);
}

fn swe_sample_bed_normal(
  tex: texture_2d<f32>,
  samp: sampler,
  uv: vec2f,
  texel: vec2f,
  cell_size: f32,
) -> vec3f {
  let w = textureSampleLevel(tex, samp, uv - vec2f(texel.x, 0.0), 0.0).x;
  let e = textureSampleLevel(tex, samp, uv + vec2f(texel.x, 0.0), 0.0).x;
  let n = textureSampleLevel(tex, samp, uv - vec2f(0.0, texel.y), 0.0).x;
  let s = textureSampleLevel(tex, samp, uv + vec2f(0.0, texel.y), 0.0).x;
  return swe_terrain_normal(w, e, n, s, cell_size);
}

fn swe_load_water(tex: texture_2d<f32>, cell: vec2i) -> vec4f {
  let size = vec2i(textureDimensions(tex));
  return textureLoad(tex, clamp(cell, vec2i(0), size - vec2i(1)), 0);
}
