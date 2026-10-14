// Per-particle scalars for the 'colour by' views of pbf-2d: one f32 per particle in a buffer the billboard renderer can
// read as a vertex buffer (a ParticleSet carries only the foam amount in positions.w). Both entry points run over the
// particles in the solver's CURRENT order, which is the order of `velocities` and of `pbf.solve`.
//
//   speed          the length of the velocity, world units per second
//   density_error  rho / rho_0 - 1 of the last solver iteration: 0 at rest density, negative at a free surface or in a
//                  splash, positive where the fluid is compressed (the number the solver is trying to drive to zero)

struct Count {
  n: u32,
  unused0: u32,
  unused1: u32,
  unused2: u32,
}

@group(0) @binding(0) var<uniform> params: Count;
@group(0) @binding(1) var<storage, read> velocities: array<vec4f>;
@group(0) @binding(2) var<storage, read> solve: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> scalar: array<f32>;

@compute @workgroup_size(64)
fn speed(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.n) { return; }
  scalar[i] = length(velocities[i].xy);
}

@compute @workgroup_size(64)
fn density_error(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.n) { return; }
  scalar[i] = solve[i].y - 1.0;
}
