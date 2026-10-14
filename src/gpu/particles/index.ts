// The GPU particle framework (Epic 7). Import from here.
//
//   system.ts          ParticleSystem: buffers, spawning, emitting, reordering, ParticleSet view
//   compute.ts         createComputePass: pipeline and bind-group caching, dispatch sizing
//   integrator.ts      createIntegrator: gravity, damping, force field, box, respawn, fixed steps
//   billboards.ts      createBillboardRenderer: instanced sprites for a ParticleSet
//   cameras.ts         orthographicCamera for 2D systems
//   neighbor-grid.ts   createNeighborGrid: the uniform-grid neighbour search (with scan.ts)
//   neighbor-count.ts  reference consumers of the grid
//   emitter.ts         Emitter descriptions and their CPU sampling

export type { ParticleSet } from './types.ts';
export * from './billboards.ts';
export * from './cameras.ts';
export * from './compute.ts';
export * from './emitter.ts';
export * from './integrator.ts';
export * from './neighbor-count.ts';
export * from './neighbor-grid.ts';
export * from './scan.ts';
export * from './system.ts';
export * from './whitewater.ts';
export * from './container.ts';
