import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'particles',
  title: 'Particle field and neighbour grid',
  epic: '7',
  tier: 'B',
  tags: ['particles', 'compute', 'simulation', 'interactive', '3d', 'spray'],
  summary:
    'A fountain of up to a million spray particles carried by a curl-noise flow, drawn as instanced billboards. The foundation of the particle fluids: a GPU particle system, a fixed-step integrator, and a uniform-grid neighbour search built by a counting sort and prefix sum every frame. Colour the cloud by its neighbour counts, or light up the neighbours of the particle under the pointer, to see the search at work. Move the pointer to aim the nozzle, drag to orbit.',
  scorecard: {
    gpuMs: [
      { device: 'Apple M1 Max · Dawn/Metal (workload slope)', ms: 1.38, resolution: '1920 × 1080' },
    ],
    bundleKb: 4.46,
    // Every pass but the draw is a compute dispatch (integrate, counting sort, scan, reorder, neighbour count).
    compute: true,
    // Not measured on a phone. A million particles do not belong there; the default count is the question.
    mobile: 'untested',
    features: { required: [], optional: ['timestamp-query'] },
    fallback:
      'Without WebGPU or compute, show a looping video of the fountain, or a WebGL2 version that moves a few tens of thousands of particles with transform feedback or a vertex-shader flow; the neighbour views have no fallback and are a research tool, not a feature.',
    useCases: [
      'Mist, spray, snow, embers and dust that need hundreds of thousands of GPU-simulated sprites: the cheapest honest particle effect, and the framework the fluid solvers build on',
      'A hero or landing-page set piece where the particles are the subject, on desktop-class GPUs',
      'The reference for a GPU neighbour search (cell ids, counting sort, prefix sum, reorder) that SPH, position-based fluids and collisions reuse',
    ],
  },
};
