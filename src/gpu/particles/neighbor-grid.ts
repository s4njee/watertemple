import { noProfiler, type PassProfiler } from '../profiler.ts';
import { createUniformBuffer, defineUniformStruct, type UniformBuffer } from '../uniforms.ts';
import { createComputePass } from './compute.ts';
import type { Vec3 } from './emitter.ts';
import cellsSource from './neighbor-grid.wgsl?raw';
import scatterSource from './neighbor-scatter.wgsl?raw';
import { createScan, MAX_SCAN_ELEMENTS } from './scan.ts';
import type { ParticleSystem } from './system.ts';

// The neighbour search (story 7.2): for every particle, the particles within a radius h, rebuilt every
// step on the GPU. A UNIFORM GRID in a BOUNDED box, built by a COUNTING SORT.
//
// ALGORITHM (one `build` call, 4 compute passes plus the scan's):
//   0. clear the cell counters (a buffer clear, no pass)
//   1. cells    each particle computes its cell and takes a ticket with atomicAdd on the cell's counter; the
//               ticket is its rank among the cell's particles. (Cell id and count fused: one pass, one atomic.)
//   2. scan     an exclusive prefix sum over the cells + 1 counters (scan.ts): counter c becomes the start
//               offset of cell c in the sorted order, and the last entry the particle count
//   3. scatter  particle i is written to slot start[cell] + rank: `sorted[slot] = i`. Plain stores, no atomics.
//   4. reorder  (optional) every per-particle array is gathered through `sorted` so that slot k IS particle k
//
// The ranks come from atomics, so the order of particles WITHIN a cell varies from run to run (the sort is not
// stable). Sums of floating-point terms over a cell therefore differ in the last bits between runs. That is
// invisible in a fluid; if you need bit-exact repeatability, sort by (cell, index) instead.
//
// BOUNDED RATHER THAN HASHED. The domain is a box `[min, max]`; positions outside it are clamped into the edge
// cells. A spatial hash would support an unbounded domain, at the price of collisions (distinct cells sharing a
// bucket make the neighbour loop visit strangers and, worse, break the "three x-adjacent cells are contiguous"
// layout that lets the loop read 3 or 9 ranges instead of 27 cells). Every solver this serves (SPH, PBF, MLS-MPM)
// runs in a box anyway: a bounded grid is exact, simpler, and the clamp costs nothing. Clamping is also
// correct for positions outside the box: two points within h of each other are never more than one cell apart
// in any axis, and clamping preserves that.
//
// SIZE AND LIMITS. cells = nx * ny * nz with cell side >= radius. A grid costs 4 bytes per cell (the counters,
// which become the start offsets) plus the tile sums of the scan (a 1/1024 fraction) and, per particle, 8
// bytes (cell and rank) + 4 bytes (the sorted index). 64^3 is 262 144 cells = 1 MiB; 128^3 is 2 097 152 cells
// = 8 MiB; the default `maxCells` of 4 194 304 (about 161^3) = 16 MiB. A bigger box or a smaller radius would
// exceed that, and the grid COARSENS instead of failing: the cell side grows until the cells fit (the search
// stays exact because a bigger cell is still >= radius; the loop just visits more candidates), and
// `layout.coarsened` says so. Absolute limits: the scan handles 67 M elements, a storage binding 128 MiB by
// default = 33.5 M cells; `maxCells` is clamped to the device's maxStorageBufferBindingSize / 4 - 1.
// Particles: 4.19 M at workgroup size 64 (one dispatch dimension), and the 16-byte position array of the
// system limits it to 8.4 M under default limits.
//
// PRECISION. The cell side is padded by 2^-12 over the radius so a pair exactly one radius apart can never be
// split by float rounding; positions are exact to about 1e-4 of a cell up to ~1000 cells from the origin.
//
// A CELL THAT HAS MORE PARTICLES THAN EXPECTED costs time, never correctness: there is no per-cell capacity
// to overflow. All particles in one cell make the counting pass serialise on one atomic (measured in the cost
// study) and make that cell's neighbour loops quadratic, but the result is still exact.
//
// BINDINGS A SOLVER USES. In reorder mode: the grid uniform (not a storage binding) and ONE storage buffer
// (cell_start). In sorted-index mode: TWO (cell_start, sorted). That leaves a solver 7 or 6 of the 8 storage
// buffers every shader stage is guaranteed.

export const NEIGHBOR_GRID = defineUniformStruct('NeighborGrid', {
  origin: 'vec3f',
  cell_size: 'f32',
  dims: 'vec3i',
  rows: 'u32',
  radius: 'f32',
  inv_cell_size: 'f32',
  cells: 'u32',
  dimension: 'u32',
});

/** Name under which `grid.chunks()` offers the generated bindings to `createShader`. */
export const NEIGHBOR_BINDINGS_CHUNK = 'neighbor_bindings.wgsl';

export type NeighborMode = 'sorted' | 'reorder';
export type BuildStage = 'cells' | 'scan' | 'scatter' | 'reorder';

export interface NeighborDomain {
  min: Vec3;
  max: Vec3;
  /** Search radius h. */
  radius: number;
  /** Cell side; at least `radius` (smaller values are raised). Default `radius`. */
  cellSize?: number;
}

export interface GridLayout {
  readonly origin: readonly [number, number, number];
  readonly cellSize: number;
  readonly dims: readonly [number, number, number];
  readonly cells: number;
  readonly radius: number;
  /** The cell side had to grow to fit `maxCells`. */
  readonly coarsened: boolean;
}

export interface PositionSource {
  /** `array<vec4f>`; xyz is the position. */
  positions: GPUBuffer;
  count: number;
}

export interface NeighborGridOptions {
  device: GPUDevice;
  dimension?: 2 | 3;
  /** `reorder` (default) moves the particle data into cell order; `sorted` only produces an index list. */
  mode?: NeighborMode;
  /** Particles the per-particle scratch is sized for at first; it grows on demand. Default 65536. */
  capacity?: number;
  /** Cell budget before the grid coarsens. Default 4 194 304. */
  maxCells?: number;
  profiler?: PassProfiler;
  label?: string;
}

export interface BuildOptions {
  /** Which attribute of a ParticleSystem to bin. Default 'position'. */
  attribute?: string;
  /** Overrides the particle count (at most the source's). */
  count?: number;
  /** Stop after this stage. For cost studies and debugging; the default runs everything. */
  stopAfter?: BuildStage;
}

export interface BindingRecipe {
  /** Bind group index. Default 0. */
  group?: number;
  /** First binding number the search takes; it uses 2 (reorder) or 3 (sorted). */
  binding: number;
}

export interface NeighborGrid {
  readonly dimension: 2 | 3;
  /** Switch freely between steps; regenerate any shader chunk made with `bindingsWgsl` afterwards. */
  mode: NeighborMode;
  /** Bumps when the cell or sorted buffers are re-created: drop bind groups that hold them. */
  readonly generation: number;
  readonly layout: GridLayout;
  readonly buffers: {
    /** NeighborGrid uniform. */
    readonly uniform: GPUBuffer;
    /** `cells + 1` u32: start offset of every cell in sorted order; the last entry is the particle count. */
    readonly cellStart: GPUBuffer;
    /** u32 per particle: sorted[k] is the particle (before any reorder) that belongs in slot k. */
    readonly sorted: GPUBuffer;
  };
  /** Storage buffers a solver's shader spends on the search: 1 (reorder) or 2 (sorted). */
  readonly storageBuffers: number;
  /** Sets the box and radius. Cheap to call every frame; reallocates only when the cell count outgrows the buffer. */
  configure(domain: NeighborDomain): GridLayout;
  /**
   * Sorts the particles into cells. With a ParticleSystem and `mode: 'reorder'`, every attribute of the system
   * is reordered too, so afterwards `system.positions` etc. are in cell order. Records the passes into `encoder`.
   * Call `configure` first. May be called more than once per submit.
   */
  build(
    encoder: GPUCommandEncoder,
    source: ParticleSystem | PositionSource,
    options?: BuildOptions,
  ): void;
  /** The WGSL that declares the bindings and `neighbor_row_span` / `neighbor_particle` (see neighbors.wgsl). */
  bindingsWgsl(recipe: BindingRecipe): string;
  /** `{ 'neighbor_bindings.wgsl': bindingsWgsl(recipe) }`, ready to spread into a ShaderSource's chunks. */
  chunks(recipe: BindingRecipe): Record<string, string>;
  /** The buffers for the recipe's bindings, in binding order: uniform, cell_start, then sorted if present. */
  resources(): GPUBuffer[];
  /** Total bytes of GPU memory the grid holds right now. */
  readonly memoryBytes: number;
  destroy(): void;
}

const CELL_PADDING = 1 + 2 ** -12;
const DEFAULT_MAX_CELLS = 1 << 22;

export function computeGridLayout(
  domain: NeighborDomain,
  dimension: 2 | 3,
  maxCells: number,
): GridLayout {
  const { min, max, radius } = domain;
  if (!(radius > 0) || !Number.isFinite(radius)) {
    throw new RangeError(`neighbour radius must be positive and finite, got ${radius}`);
  }
  const extent = [0, 1, 2].map((axis) => Math.max(max[axis]! - min[axis]!, 0));
  if (extent.some((e) => !Number.isFinite(e))) throw new RangeError('domain must be finite');
  let cellSize = Math.max(domain.cellSize ?? radius, radius) * CELL_PADDING;
  const dimsFor = (side: number): [number, number, number] => [
    Math.max(1, Math.ceil(extent[0]! / side)),
    Math.max(1, Math.ceil(extent[1]! / side)),
    dimension === 2 ? 1 : Math.max(1, Math.ceil(extent[2]! / side)),
  ];
  let dims = dimsFor(cellSize);
  let coarsened = false;
  while (dims[0] * dims[1] * dims[2] > maxCells) {
    const factor = (dims[0] * dims[1] * dims[2]) / maxCells;
    cellSize *= Math.max(1.001, Math.pow(factor, 1 / dimension) * 1.0001);
    dims = dimsFor(cellSize);
    coarsened = true;
  }
  return {
    origin: [min[0]!, min[1]!, dimension === 2 ? 0 : min[2]!],
    cellSize,
    dims,
    cells: dims[0] * dims[1] * dims[2],
    radius,
    coarsened,
  };
}

const isSystem = (source: ParticleSystem | PositionSource): source is ParticleSystem =>
  'permute' in source;

export function createNeighborGrid(options: NeighborGridOptions): NeighborGrid {
  const { device, dimension = 3, profiler = noProfiler, label = 'neighbor grid' } = options;
  let mode: NeighborMode = options.mode ?? 'reorder';
  const bindingLimit = device.limits.maxStorageBufferBindingSize;
  const maxCells = Math.max(
    1,
    Math.min(
      options.maxCells ?? DEFAULT_MAX_CELLS,
      Math.floor(bindingLimit / 4) - 1,
      MAX_SCAN_ELEMENTS - 1,
    ),
  );

  const scan = createScan({ device, maxElements: maxCells + 1, profiler, label: `${label} scan` });
  const cellsPass = createComputePass(device, {
    label: 'particles/neighbor-grid.wgsl',
    code: cellsSource,
    entryPoint: 'cells',
    profiler,
    timestamp: `${label} cells`,
  });
  const scatterPass = createComputePass(device, {
    label: 'particles/neighbor-scatter.wgsl',
    code: scatterSource,
    entryPoint: 'scatter',
    profiler,
    timestamp: `${label} scatter`,
  });
  const uniform: UniformBuffer<typeof NEIGHBOR_GRID.fields> = createUniformBuffer(
    device,
    NEIGHBOR_GRID,
    `${label} uniform`,
  );

  let layout: GridLayout | undefined;
  let layoutKey = '';
  let generation = 0;
  let particleCapacity = 0;
  let cellCapacity = 0;
  let cellStart: GPUBuffer | undefined;
  let cellRank: GPUBuffer | undefined;
  let sorted: GPUBuffer | undefined;
  let destroyed = false;

  const make = (name: string, bytes: number, extra = 0): GPUBuffer =>
    device.createBuffer({
      label: `${label} ${name}`,
      size: Math.max(16, bytes),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extra,
    });

  function ensureCells(cells: number): void {
    if (cellStart && cells + 1 <= cellCapacity) return;
    cellStart?.destroy();
    // Headroom, so a radius slider does not reallocate on every tick.
    cellCapacity = Math.min(maxCells + 1, Math.max(cells + 1, Math.ceil(cellCapacity * 1.5)));
    cellStart = make('cell start', cellCapacity * 4);
    generation++;
  }

  function ensureParticles(count: number): void {
    if (cellRank && count <= particleCapacity) return;
    cellRank?.destroy();
    sorted?.destroy();
    particleCapacity = Math.max(
      count,
      Math.ceil(particleCapacity * 1.5),
      options.capacity ?? 65536,
    );
    cellRank = make('cell and rank', particleCapacity * 8);
    sorted = make('sorted index', particleCapacity * 4);
    generation++;
  }

  // Allocated now, so `buffers` is valid before the first build.
  ensureParticles(1);
  ensureCells(1);

  const grid: NeighborGrid = {
    dimension,
    get mode() {
      return mode;
    },
    set mode(value: NeighborMode) {
      mode = value;
    },
    get generation() {
      return generation;
    },
    get layout() {
      if (!layout) throw new Error(`${label}: configure() has not been called`);
      return layout;
    },
    buffers: {
      get uniform() {
        return uniform.buffer;
      },
      get cellStart() {
        return cellStart!;
      },
      get sorted() {
        return sorted!;
      },
    },
    get storageBuffers() {
      return mode === 'sorted' ? 2 : 1;
    },
    get memoryBytes() {
      return (cellStart?.size ?? 0) + (cellRank?.size ?? 0) + (sorted?.size ?? 0);
    },

    configure(domain) {
      const next = computeGridLayout(domain, dimension, maxCells);
      const key = `${next.origin}|${next.cellSize}|${next.dims}|${next.radius}`;
      if (!layout || key !== layoutKey) {
        layout = next;
        layoutKey = key;
        ensureCells(next.cells);
        uniform.set({
          origin: [...next.origin],
          cell_size: next.cellSize,
          dims: [...next.dims],
          rows: dimension === 2 ? 3 : 9,
          radius: next.radius,
          inv_cell_size: 1 / next.cellSize,
          cells: next.cells,
          dimension,
        });
        uniform.upload();
      }
      return layout;
    },

    build(encoder, source, buildOptions = {}) {
      const current = grid.layout;
      const stopAfter = buildOptions.stopAfter ?? 'reorder';
      let positions: GPUBuffer;
      let count: number;
      if (isSystem(source)) {
        positions = source.attribute(buildOptions.attribute ?? 'position');
        if (source.components(buildOptions.attribute ?? 'position') !== 4) {
          throw new TypeError(`${label}: the binned attribute must be a vec4 array`);
        }
        count = source.count;
      } else {
        if (mode === 'reorder') {
          throw new TypeError(
            `${label}: reorder mode needs a ParticleSystem (to move every attribute); use mode 'sorted' for a bare buffer`,
          );
        }
        positions = source.positions;
        count = source.count;
      }
      count = Math.min(count, buildOptions.count ?? count);
      if (count <= 0) return;
      ensureParticles(count);
      const cellsPlusOne = current.cells + 1;

      encoder.clearBuffer(cellStart!, 0, cellsPlusOne * 4);
      cellsPass.runCount(
        encoder,
        [
          uniform.buffer,
          { buffer: positions, size: count * 16 },
          { buffer: cellStart!, size: cellsPlusOne * 4 },
          { buffer: cellRank!, size: count * 8 },
        ],
        count,
      );
      if (stopAfter === 'cells') return;

      scan.run(encoder, cellStart!, cellsPlusOne);
      if (stopAfter === 'scan') return;

      scatterPass.runCount(
        encoder,
        [
          { buffer: cellStart!, size: cellsPlusOne * 4 },
          { buffer: cellRank!, size: count * 8 },
          { buffer: sorted!, size: count * 4 },
        ],
        count,
      );
      if (stopAfter === 'scatter') return;

      if (mode === 'reorder' && isSystem(source)) source.permute(encoder, sorted!, count);
    },

    bindingsWgsl({ group = 0, binding }) {
      const sortedMode = mode === 'sorted';
      return `// Generated by NeighborGrid.bindingsWgsl: the grid's bindings and the two functions that hide them.
#include "neighbors.wgsl"
@group(${group}) @binding(${binding}) var<uniform> neighbor_grid: NeighborGrid;
@group(${group}) @binding(${binding + 1}) var<storage, read> neighbor_cell_start: array<u32>;
${sortedMode ? `@group(${group}) @binding(${binding + 2}) var<storage, read> neighbor_sorted: array<u32>;` : ''}

fn neighbor_row_span(c: vec3i, row: u32) -> vec2u {
  let r = neighbor_row_cells(neighbor_grid, c, row);
  if (r.x < 0) {
    return vec2u(0u, 0u);
  }
  return vec2u(neighbor_cell_start[u32(r.x)], neighbor_cell_start[u32(r.y) + 1u]);
}

fn neighbor_particle(k: u32) -> u32 {
  return ${sortedMode ? 'neighbor_sorted[k]' : 'k'};
}
`;
    },

    chunks(recipe) {
      return { [NEIGHBOR_BINDINGS_CHUNK]: grid.bindingsWgsl(recipe) };
    },

    resources() {
      const list = [uniform.buffer, cellStart!];
      if (mode === 'sorted') list.push(sorted!);
      return list;
    },

    destroy() {
      if (destroyed) return;
      destroyed = true;
      scan.destroy();
      cellsPass.destroy();
      scatterPass.destroy();
      uniform.destroy();
      cellStart?.destroy();
      cellRank?.destroy();
      sorted?.destroy();
    },
  };
  return grid;
}
