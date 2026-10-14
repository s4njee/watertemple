import { noProfiler, type PassProfiler } from '../profiler.ts';
import { createComputePass, type ComputePass } from './compute.ts';
import countSource from './neighbor-count.wgsl';
import type { NeighborGrid, NeighborMode, PositionSource } from './neighbor-grid.ts';
import type { ParticleSystem } from './system.ts';

// Two reference consumers of the grid, written with the loop from src/wgsl/neighbors.wgsl. They exist to
// inspect the search (the particles demo colours particles by them) and to measure it: "count my neighbours"
// is the cheapest honest stand-in for the first pass of any SPH or PBF step. Binding layout, which a solver
// can copy: positions at 0, the result at 1, the grid's bindings from 2 (2 or 3 of them), the pick at 5.

export interface NeighborCounter {
  /**
   * Writes the number of neighbours within the grid's radius of each particle into `result` (one f32 per
   * particle, STORAGE usage). Call after `grid.build` on the same source: in reorder mode the positions
   * are in cell order and so is the result.
   */
  count(
    encoder: GPUCommandEncoder,
    grid: NeighborGrid,
    source: ParticleSystem | PositionSource,
    result: GPUBuffer,
  ): void;
  /**
   * Zeroes `result` (so it needs COPY_DST usage as well as STORAGE), then writes 1 for every neighbour of particle `pick[0]` and 2 for the particle itself.
   * `pick` is a buffer holding one u32 (STORAGE usage); an out-of-range index marks nothing.
   */
  mark(
    encoder: GPUCommandEncoder,
    grid: NeighborGrid,
    source: ParticleSystem | PositionSource,
    pick: GPUBuffer,
    result: GPUBuffer,
  ): void;
  destroy(): void;
}

export interface NeighborCounterOptions {
  device: GPUDevice;
  profiler?: PassProfiler;
  label?: string;
}

const PICK_BINDING = 5;

export function createNeighborCounter(options: NeighborCounterOptions): NeighborCounter {
  const { device, profiler = noProfiler, label = 'neighbours' } = options;
  const passes = new Map<string, ComputePass>();

  function passFor(grid: NeighborGrid, entry: 'count_neighbors' | 'mark_neighbors'): ComputePass {
    const mode: NeighborMode = grid.mode;
    const key = `${entry}:${mode}`;
    let pass = passes.get(key);
    if (!pass) {
      pass = createComputePass(device, {
        label: 'particles/neighbor-count.wgsl',
        code: countSource.code,
        chunks: grid.chunks({ group: 0, binding: 2 }),
        entryPoint: entry,
        workgroupSize: entry === 'mark_neighbors' ? 1 : 64,
        profiler,
        timestamp: entry === 'mark_neighbors' ? `${label} mark` : `${label} count`,
      });
      passes.set(key, pass);
    }
    return pass;
  }

  const positionsOf = (source: ParticleSystem | PositionSource): [GPUBuffer, number] => [
    source.positions,
    source.count,
  ];

  return {
    count(encoder, grid, source, result) {
      const [positions, count] = positionsOf(source);
      if (count <= 0) return;
      passFor(grid, 'count_neighbors').runCount(
        encoder,
        [
          { buffer: positions, size: count * 16 },
          { buffer: result, size: count * 4 },
          ...grid.resources(),
        ],
        count,
      );
    },
    mark(encoder, grid, source, pick, result) {
      const [positions, count] = positionsOf(source);
      if (count <= 0) return;
      encoder.clearBuffer(result, 0, count * 4);
      const resources: (GPUBuffer | GPUBufferBinding)[] = [
        { buffer: positions, size: count * 16 },
        { buffer: result, size: count * 4 },
        ...grid.resources(),
      ];
      // The pick lives at a fixed binding, after however many the grid took.
      const sparse: (GPUBuffer | GPUBufferBinding | undefined)[] = [...resources];
      while (sparse.length < PICK_BINDING) sparse.push(undefined);
      sparse[PICK_BINDING] = pick;
      passFor(grid, 'mark_neighbors').runGroups(encoder, sparse, 1);
    },
    destroy() {
      for (const pass of passes.values()) pass.destroy();
      passes.clear();
    },
  };
}
