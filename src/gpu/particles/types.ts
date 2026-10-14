/**
 * What every particle simulation hands to whatever draws or post-processes it. Simulations
 * (SPH, PBF, MPM) produce one; renderers (billboards, the screen-space fluid surface) consume one.
 * Neither side needs to know the other.
 */
export interface ParticleSet {
  /**
   * `array<vec4f>`, one element per particle: xyz is the position in world units, w is a
   * per-particle scalar the simulation chooses and documents (speed, density, age, ...), which a
   * renderer may map to colour. Two-dimensional systems keep z at 0.
   *
   * Created with STORAGE | VERTEX | COPY_SRC | COPY_DST usage, so it can be bound as a read-only
   * storage buffer or as an instance-step vertex buffer and read back in tests.
   *
   * A simulation that ping-pongs its buffers returns a different buffer from frame to frame:
   * cache bind groups per buffer object, never just once.
   */
  readonly positions: GPUBuffer;
  /** `array<vec4f>`: xyz is the velocity in world units per second, w is free. Same usage flags. */
  readonly velocities?: GPUBuffer;
  /** Live particles, occupying indices 0 to count - 1. */
  readonly count: number;
  /** Size of the buffers, in particles. */
  readonly capacity: number;
  /** World-space radius, the same for every particle in the set. */
  readonly radius: number;
  /**
   * Volume of liquid one particle stands for at rest, in world units cubed. Optional: a renderer
   * that turns particle overlap into liquid thickness needs it; billboards do not.
   */
  readonly restVolume?: number;
  /** Bumps when the buffers are re-created (capacity change), so consumers drop cached bind groups. */
  readonly generation: number;
}
