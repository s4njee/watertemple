import { describe, expect, it, vi } from 'vitest';
import { createComputePass, MAX_WORKGROUPS_PER_DIMENSION } from '../src/gpu/particles/compute.ts';
import {
  computeGridLayout,
  createNeighborGrid,
  NEIGHBOR_GRID,
  type NeighborGrid,
  type NeighborMode,
} from '../src/gpu/particles/neighbor-grid.ts';
import { createNeighborCounter } from '../src/gpu/particles/neighbor-count.ts';
import {
  createScan,
  MAX_SCAN_ELEMENTS,
  SCAN_TILE,
  scanLevels,
  scanLevelSizes,
} from '../src/gpu/particles/scan.ts';
import { createParticleSystem, type ParticleSystem } from '../src/gpu/particles/system.ts';
import { getChunk } from '../src/wgsl/index.ts';
import { expectValid, getTestDevice } from './gpu.ts';

// Story 7.2: the neighbour search. Everything here runs the real shaders on the GPU and compares with plain
// JavaScript: a brute-force O(n^2) neighbour reference and a CPU prefix sum.

vi.setConfig({ testTimeout: 60_000 });

// --- Helpers -----------------------------------------------------------------------------------

function makeRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

async function readBuffer(
  device: GPUDevice,
  buffer: GPUBuffer,
  bytes: number,
): Promise<ArrayBuffer> {
  const readback = device.createBuffer({
    size: Math.max(4, bytes),
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, readback, 0, Math.max(4, bytes));
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const data = readback.getMappedRange().slice(0, bytes);
  readback.unmap();
  readback.destroy();
  return data;
}

async function submit(device: GPUDevice, record: (encoder: GPUCommandEncoder) => void) {
  const encoder = device.createCommandEncoder();
  record(encoder);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

// --- The prefix sum -----------------------------------------------------------------------------

function cpuExclusiveScan(values: Uint32Array): Uint32Array {
  const out = new Uint32Array(values.length);
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    out[i] = sum;
    sum = (sum + values[i]!) >>> 0;
  }
  return out;
}

describe('prefix sum', () => {
  async function scanOnGpu(values: Uint32Array, extra = 0): Promise<Uint32Array> {
    const device = await getTestDevice();
    const scan = createScan({ device, maxElements: Math.max(values.length, 1) });
    const buffer = device.createBuffer({
      size: (values.length + extra) * 4 + 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const padded = new Uint32Array(values.length + extra + 1).fill(0xdeadbeef);
    padded.set(values);
    device.queue.writeBuffer(buffer, 0, padded);
    await expectValid(device, () => submit(device, (e) => scan.run(e, buffer, values.length)));
    const out = new Uint32Array(await readBuffer(device, buffer, padded.byteLength));
    scan.destroy();
    buffer.destroy();
    // Elements past the scanned range must be untouched.
    for (let i = values.length; i < out.length; i++) expect(out[i]).toBe(0xdeadbeef);
    return out.subarray(0, values.length);
  }

  const random = makeRandom(7);
  // 1 and 2 are degenerate; 255/256/1023/1024/1025 straddle a thread and a tile; the rest need several
  // tiles; 1 048 577 needs three levels (1025 tiles, then 2, then one).
  it.each([1, 2, 3, 255, 256, 257, 1023, 1024, 1025, 2048, 4097, 100_003, 1_048_577])(
    'matches a CPU exclusive scan for %i random elements',
    async (n) => {
      const values = new Uint32Array(n);
      for (let i = 0; i < n; i++) values[i] = Math.floor(random() * 1000);
      const out = await scanOnGpu(values);
      expect(out).toEqual(cpuExclusiveScan(values));
    },
  );

  it('is exact for all ones, and wraps like u32 for huge values', async () => {
    const ones = new Uint32Array(5000).fill(1);
    expect(Array.from(await scanOnGpu(ones))).toEqual(Array.from({ length: 5000 }, (_, i) => i));
    const huge = new Uint32Array(3000);
    for (let i = 0; i < huge.length; i++) huge[i] = Math.floor(random() * 2 ** 32) >>> 0;
    expect(await scanOnGpu(huge)).toEqual(cpuExclusiveScan(huge));
  });

  it('leaves the buffer alone for a count of 0 and rejects counts it cannot handle', async () => {
    const device = await getTestDevice();
    const scan = createScan({ device, maxElements: 100 });
    const buffer = device.createBuffer({ size: 400, usage: GPUBufferUsage.STORAGE });
    const encoder = device.createCommandEncoder();
    expect(() => scan.run(encoder, buffer, 0)).not.toThrow();
    expect(() => scan.run(encoder, buffer, 101)).toThrow(RangeError);
    expect(() => scan.run(encoder, buffer, 1.5)).toThrow(RangeError);
    expect(() => createScan({ device, maxElements: MAX_SCAN_ELEMENTS + 1 })).toThrow(RangeError);
    scan.destroy();
    buffer.destroy();
  });

  it('plans one level per factor of 1024', () => {
    expect(scanLevelSizes(1)).toEqual([1]);
    expect(scanLevelSizes(SCAN_TILE)).toEqual([1024]);
    expect(scanLevelSizes(SCAN_TILE + 1)).toEqual([1025, 2]);
    expect(scanLevels(64 ** 3 + 1)).toBe(2); // a 64^3 grid: 257 tiles
    expect(scanLevels(128 ** 3 + 1)).toBe(3);
    expect(scanLevels(1_048_577)).toBe(3);
  });

  it('refuses a dispatch that would need more workgroups than one dimension allows', async () => {
    const device = await getTestDevice();
    const pass = createComputePass(device, {
      label: 'test/limit.wgsl',
      code: '@group(0) @binding(0) var<storage, read_write> a: array<u32>; @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) { a[id.x] = 1u; }',
    });
    const encoder = device.createCommandEncoder();
    expect(() => pass.runCount(encoder, [], MAX_WORKGROUPS_PER_DIMENSION * 64 + 1)).toThrow(
      RangeError,
    );
    expect(() => pass.runCount(encoder, [], Number.NaN)).toThrow(RangeError);
  });
});

// --- The grid: layout ----------------------------------------------------------------------------

describe('grid layout', () => {
  it('sizes cells at least one radius wide and covers the box', () => {
    const layout = computeGridLayout({ min: [0, 0, 0], max: [1, 2, 3], radius: 0.1 }, 3, 1 << 22);
    expect(layout.cellSize).toBeGreaterThan(0.1);
    expect(layout.cellSize).toBeLessThan(0.1001);
    expect(layout.dims).toEqual([10, 20, 30]);
    expect(layout.cells).toBe(6000);
    expect(layout.coarsened).toBe(false);
    for (let axis = 0; axis < 3; axis++) {
      expect(layout.dims[axis]! * layout.cellSize).toBeGreaterThanOrEqual([1, 2, 3][axis]!);
    }
  });

  it('is flat in 2D, honours a larger cell size, and coarsens instead of exceeding maxCells', () => {
    expect(computeGridLayout({ min: [0, 0, 0], max: [1, 1, 5], radius: 0.1 }, 2, 1e6).dims[2]).toBe(
      1,
    );
    const wide = computeGridLayout(
      { min: [0, 0, 0], max: [1, 1, 1], radius: 0.1, cellSize: 0.25 },
      3,
      1e6,
    );
    expect(wide.dims).toEqual([4, 4, 4]);
    const huge = computeGridLayout({ min: [0, 0, 0], max: [1, 1, 1], radius: 0.001 }, 3, 1 << 20);
    expect(huge.coarsened).toBe(true);
    expect(huge.cells).toBeLessThanOrEqual(1 << 20);
    expect(huge.cellSize).toBeGreaterThanOrEqual(0.001);
    expect(() => computeGridLayout({ min: [0, 0, 0], max: [1, 1, 1], radius: 0 }, 3, 10)).toThrow(
      RangeError,
    );
  });

  it('keeps the uniform in step with struct NeighborGrid in neighbors.wgsl', () => {
    const source = getChunk('neighbors.wgsl') ?? '';
    const declared = /struct NeighborGrid \{([^}]*)\}/.exec(source)?.[1] ?? '';
    const normalise = (text: string) => text.replace(/\s+/g, ' ').replace(/,\s*$/, '').trim();
    const generated = /struct NeighborGrid \{([^}]*)\}/.exec(NEIGHBOR_GRID.wgsl)?.[1] ?? '';
    expect(normalise(declared).replace(/,$/, '')).toBe(normalise(generated).replace(/,$/, ''));
  });
});

// --- The grid: correctness against brute force ----------------------------------------------------

interface Scenario {
  name: string;
  dimension: 2 | 3;
  radius: number;
  min: [number, number, number];
  max: [number, number, number];
  /** vec4 rows (x y z w). */
  positions: (grid: NeighborGrid) => Float32Array;
}

const UNIT = {
  min: [0, 0, 0] as [number, number, number],
  max: [1, 1, 1] as [number, number, number],
};

function uniform(n: number, seed: number, lo: number, hi: number, dimension: 2 | 3): Float32Array {
  const random = makeRandom(seed);
  const rows = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    rows[i * 4] = lo + (hi - lo) * random();
    rows[i * 4 + 1] = lo + (hi - lo) * random();
    rows[i * 4 + 2] = dimension === 2 ? 0 : lo + (hi - lo) * random();
    rows[i * 4 + 3] = i;
  }
  return rows;
}

const scenarios: Scenario[] = [
  {
    name: 'uniform 3D',
    dimension: 3,
    radius: 0.1,
    ...UNIT,
    positions: () => uniform(3000, 1, 0, 1, 3),
  },
  {
    name: 'uniform 2D',
    dimension: 2,
    radius: 0.06,
    ...UNIT,
    positions: () => uniform(3000, 2, 0, 1, 2),
  },
  {
    name: 'everything in one cell (3D)',
    dimension: 3,
    radius: 0.1,
    ...UNIT,
    positions: (grid) => {
      const random = makeRandom(3);
      const cell = grid.layout.cellSize;
      const rows = new Float32Array(600 * 4);
      for (let i = 0; i < 600; i++) {
        // Strictly inside cell (4, 4, 4), and small enough that every pair is a neighbour.
        rows[i * 4] = cell * 4.2 + random() * cell * 0.05;
        rows[i * 4 + 1] = cell * 4.2 + random() * cell * 0.05;
        rows[i * 4 + 2] = cell * 4.2 + random() * cell * 0.05;
        rows[i * 4 + 3] = i;
      }
      return rows;
    },
  },
  {
    name: 'everything in one cell (2D)',
    dimension: 2,
    radius: 0.1,
    ...UNIT,
    positions: (grid) => {
      const random = makeRandom(4);
      const cell = grid.layout.cellSize;
      const rows = new Float32Array(500 * 4);
      for (let i = 0; i < 500; i++) {
        rows[i * 4] = cell * 3.3 + random() * cell * 0.3;
        rows[i * 4 + 1] = cell * 3.3 + random() * cell * 0.3;
        rows[i * 4 + 3] = i;
      }
      return rows;
    },
  },
  {
    name: 'on cell boundaries (lattice at half a cell)',
    dimension: 3,
    radius: 0.125,
    ...UNIT,
    positions: (grid) => {
      // Spacing of half a cell puts every other plane exactly on a cell boundary, and pairs two steps
      // apart sit a hair beyond the radius: the case the 2^-12 padding of the cell side exists for.
      const step = grid.layout.cellSize / 2;
      const rows: number[] = [];
      for (let z = 0; z < 6; z++)
        for (let y = 0; y < 6; y++)
          for (let x = 0; x < 6; x++) rows.push(x * step, y * step, z * step, rows.length / 4);
      return new Float32Array(rows);
    },
  },
  {
    name: 'on cell boundaries (lattice at exactly the radius)',
    dimension: 3,
    radius: 0.125,
    ...UNIT,
    positions: (grid) => {
      const step = grid.layout.cellSize;
      const rows: number[] = [];
      for (let z = 0; z < 6; z++)
        for (let y = 0; y < 6; y++)
          for (let x = 0; x < 6; x++) rows.push(x * step, y * step, z * step, rows.length / 4);
      return new Float32Array(rows);
    },
  },
  {
    name: 'at and beyond the edges of the domain',
    dimension: 3,
    radius: 0.1,
    ...UNIT,
    // Uniform in a box 40 percent larger than the domain, plus particles exactly on its faces.
    positions: () => {
      const rows = uniform(3000, 5, -0.2, 1.2, 3);
      const faces = [0, 1];
      for (let i = 0; i < 60; i++) {
        rows[i * 4] = faces[i % 2]!;
        rows[(i + 60) * 4 + 1] = faces[i % 2]!;
        rows[(i + 120) * 4 + 2] = faces[i % 2]!;
      }
      return rows;
    },
  },
  {
    name: 'clustered: a dense blob over a sparse background',
    dimension: 3,
    radius: 0.08,
    ...UNIT,
    positions: () => {
      const random = makeRandom(6);
      const rows = uniform(2700, 6, 0, 1, 3);
      for (let i = 0; i < 1200; i++) {
        // A Gaussian blob: 1200 particles inside two or three cells.
        const r = Math.sqrt(-2 * Math.log(random() + 1e-9)) * 0.03;
        const a = random() * Math.PI * 2;
        const b = random() * Math.PI;
        rows[i * 4] = 0.4 + r * Math.sin(b) * Math.cos(a);
        rows[i * 4 + 1] = 0.5 + r * Math.sin(b) * Math.sin(a);
        rows[i * 4 + 2] = 0.6 + r * Math.cos(b);
      }
      return rows;
    },
  },
  {
    name: 'a single particle, and an empty neighbourhood',
    dimension: 3,
    radius: 0.1,
    ...UNIT,
    positions: () => new Float32Array([0.5, 0.5, 0.5, 0, 0.9, 0.9, 0.9, 1]),
  },
];

/** A reference neighbour list per particle: pairs surely inside, and pairs possibly inside (the rounding band). */
function bruteForce(rows: Float32Array, n: number, radius: number, dimension: 2 | 3) {
  const inner: Set<number>[] = [];
  const outer: Set<number>[] = [];
  const lo = radius * radius * (1 - 1e-4);
  const hi = radius * radius * (1 + 1e-4);
  for (let i = 0; i < n; i++) {
    const a = new Set<number>();
    const b = new Set<number>();
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const dx = rows[j * 4]! - rows[i * 4]!;
      const dy = rows[j * 4 + 1]! - rows[i * 4 + 1]!;
      const dz = dimension === 2 ? 0 : rows[j * 4 + 2]! - rows[i * 4 + 2]!;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 <= lo) a.add(j);
      if (d2 <= hi) b.add(j);
    }
    inner.push(a);
    outer.push(b);
  }
  return { inner, outer };
}

const MAX_LIST = 1500;

/** The listing shader: for every slot, the count and the ids (an extra attribute that travels with the data) of its neighbours. */
function listingShader(): string {
  return `
#include "neighbor_bindings.wgsl"
@group(0) @binding(0) var<storage, read> positions: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> lists: array<u32>;
@group(0) @binding(5) var<storage, read> ids: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&ids)) { return; }
  let p = positions[i].xyz;
  let c = neighbor_cell_of(neighbor_grid, p);
  let r2 = neighbor_grid.radius * neighbor_grid.radius;
  var n = 0u;
  for (var row = 0u; row < neighbor_grid.rows; row++) {
    let span = neighbor_row_span(c, row);
    for (var k = span.x; k < span.y; k++) {
      let j = neighbor_particle(k);
      if (j != i && neighbor_dist2(neighbor_grid, positions[j].xyz - p) <= r2) {
        if (n < ${MAX_LIST}u) { lists[i * ${MAX_LIST + 1}u + 1u + n] = u32(ids[j]); }
        n++;
      }
    }
  }
  lists[i * ${MAX_LIST + 1}u] = n;
}`;
}

interface Rig {
  device: GPUDevice;
  system: ParticleSystem;
  grid: NeighborGrid;
  destroy(): void;
}

async function makeRig(
  dimension: 2 | 3,
  mode: NeighborMode,
  capacity: number,
  extra: { maxCells?: number } = {},
): Promise<Rig> {
  const device = await getTestDevice();
  const system = createParticleSystem({
    device,
    capacity,
    radius: 0.01,
    dimension,
    extras: { id: 1 },
  });
  const grid = createNeighborGrid({ device, dimension, mode, capacity: 64, ...extra });
  return {
    device,
    system,
    grid,
    destroy() {
      system.destroy();
      grid.destroy();
    },
  };
}

/** Writes the rows with ids 0..n-1 in order, then builds the grid. */
async function loadAndBuild(rig: Rig, rows: Float32Array, ids?: Float32Array): Promise<number> {
  const n = rows.length / 4;
  const idRows = ids ?? Float32Array.from({ length: n }, (_, i) => i);
  rig.system.set({ positions: rows, extras: { id: idRows } });
  await expectValid(rig.device, () => submit(rig.device, (e) => rig.grid.build(e, rig.system)));
  return n;
}

/** Everything a step should satisfy, checked against the CPU. `truth` holds the particles by id. */
async function verify(
  rig: Rig,
  truth: Float32Array,
  radius: number,
  label: string,
): Promise<{ lists: number[][]; ids: number[] }> {
  const { device, system, grid } = rig;
  const n = system.count;
  const ids = Array.from(await system.read('id'));
  const positions = await system.read('position');

  // 1. No particle lost or duplicated, and the data moved together: slot k's position is truth[ids[k]].
  expect(ids.length, label).toBe(n);
  expect(
    [...ids].sort((a, b) => a - b),
    `${label}: ids are a permutation`,
  ).toEqual(Array.from({ length: n }, (_, i) => i));
  if (grid.mode === 'reorder') {
    for (let k = 0; k < n; k++) {
      const id = ids[k]!;
      if (
        positions[k * 4] !== truth[id * 4] ||
        positions[k * 4 + 1] !== truth[id * 4 + 1] ||
        positions[k * 4 + 2] !== truth[id * 4 + 2] ||
        positions[k * 4 + 3] !== truth[id * 4 + 3]
      ) {
        throw new Error(`${label}: slot ${k} (id ${id}) does not hold that particle's position`);
      }
    }
  } else {
    // Sorted-index mode leaves the arrays alone.
    expect(ids, `${label}: untouched`).toEqual(Array.from({ length: n }, (_, i) => i));
  }

  // 2. The sorted index is a permutation, and cell_start is consistent with it.
  const sorted = new Uint32Array(await readBuffer(device, grid.buffers.sorted, n * 4));
  expect(
    [...sorted].sort((a, b) => a - b),
    `${label}: sorted is a permutation`,
  ).toEqual(Array.from({ length: n }, (_, i) => i));
  const cells = grid.layout.cells;
  const starts = new Uint32Array(await readBuffer(device, grid.buffers.cellStart, (cells + 1) * 4));
  expect(starts[0], label).toBe(0);
  expect(starts[cells], `${label}: the last start is the particle count`).toBe(n);
  for (let c = 0; c < cells; c++) {
    if (starts[c]! > starts[c + 1]!) throw new Error(`${label}: cell_start decreases at cell ${c}`);
  }
  // Every particle sits in the cell range its position says (to within a cell for the ones within rounding of a boundary).
  const { origin, cellSize, dims } = grid.layout;
  // In reorder mode the data is already in slot order; in sorted mode slot k holds particle sorted[k].
  const cellOfSlot = (slot: number) => {
    const particle = grid.mode === 'reorder' ? slot : sorted[slot]!;
    const out: number[] = [];
    for (let axis = 0; axis < 3; axis++) {
      const f =
        axis === 2 && grid.dimension === 2
          ? 0
          : (positions[particle * 4 + axis]! - origin[axis]!) / cellSize;
      const c = Math.min(Math.max(Math.floor(f), 0), dims[axis]! - 1);
      out.push(c);
    }
    return out;
  };
  let c = 0;
  for (let slot = 0; slot < n; slot++) {
    while (starts[c + 1]! <= slot) c++;
    const [cx, cy, cz] = cellOfSlot(slot);
    const gx = c % dims[0];
    const gy = Math.floor(c / dims[0]) % dims[1];
    const gz = Math.floor(c / (dims[0] * dims[1]));
    if (Math.abs(gx - cx!) > 1 || Math.abs(gy - cy!) > 1 || Math.abs(gz - cz!) > 1) {
      throw new Error(
        `${label}: slot ${slot} is in cell ${gx},${gy},${gz} but its position says ${cx},${cy},${cz}`,
      );
    }
    // And exactly, unless the position is within rounding of a boundary.
    const particle = grid.mode === 'reorder' ? slot : sorted[slot]!;
    const near = [0, 1, 2].some((axis) => {
      if (axis === 2 && grid.dimension === 2) return false;
      const f = (positions[particle * 4 + axis]! - origin[axis]!) / cellSize;
      const frac = f - Math.floor(f);
      return frac < 1e-3 || frac > 1 - 1e-3;
    });
    if (!near && (gx !== cx || gy !== cy || gz !== cz)) {
      throw new Error(
        `${label}: slot ${slot} is in cell ${gx},${gy},${gz} but its position says ${cx},${cy},${cz}`,
      );
    }
  }

  // 3. The neighbour sets, from the documented loop, equal brute force (inside the rounding band).
  const lists = await listNeighbors(rig, n);
  const { inner, outer } = bruteForce(truth, n, radius, grid.dimension);
  const problems: string[] = [];
  const out: number[][] = [];
  for (let slot = 0; slot < n; slot++) {
    const id = ids[slot]!;
    const count = lists[slot * (MAX_LIST + 1)]!;
    const found = new Set<number>();
    for (let m = 0; m < Math.min(count, MAX_LIST); m++)
      found.add(lists[slot * (MAX_LIST + 1) + 1 + m]!);
    out.push([...found]);
    if (found.size !== Math.min(count, MAX_LIST))
      problems.push(`id ${id}: a neighbour was listed twice`);
    if (count < inner[id]!.size || count > outer[id]!.size) {
      problems.push(
        `id ${id}: counted ${count}, expected ${inner[id]!.size} to ${outer[id]!.size}`,
      );
    }
    for (const j of inner[id]!)
      if (!found.has(j)) problems.push(`id ${id}: missing neighbour ${j}`);
    for (const j of found) if (!outer[id]!.has(j)) problems.push(`id ${id}: extra neighbour ${j}`);
    if (problems.length > 6) break;
  }
  expect(problems, `${label}: neighbour sets`).toEqual([]);
  return { lists: out, ids };
}

async function listNeighbors(rig: Rig, n: number): Promise<Uint32Array> {
  const { device, system, grid } = rig;
  const pass = createComputePass(device, {
    label: `test/listing-${grid.mode}.wgsl`,
    code: listingShader(),
    chunks: grid.chunks({ group: 0, binding: 2 }),
  });
  const lists = device.createBuffer({
    size: n * (MAX_LIST + 1) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const resources: (GPUBuffer | GPUBufferBinding | undefined)[] = [
    { buffer: system.positions, size: n * 16 },
    lists,
    ...grid.resources(),
  ];
  while (resources.length < 5) resources.push(undefined);
  resources[5] = { buffer: system.attribute('id'), size: n * 4 };
  await expectValid(device, () => submit(device, (e) => pass.runCount(e, resources, n)));
  const data = new Uint32Array(await readBuffer(device, lists, n * (MAX_LIST + 1) * 4));
  lists.destroy();
  pass.destroy();
  return data;
}

describe.each<NeighborMode>(['reorder', 'sorted'])('neighbour search, %s mode', (mode) => {
  it.each(scenarios)('finds exactly the brute-force neighbours: $name', async (scenario) => {
    const rig = await makeRig(scenario.dimension, mode, 8192);
    rig.grid.configure({ min: scenario.min, max: scenario.max, radius: scenario.radius });
    const rows = scenario.positions(rig.grid);
    const truth = rows.slice();
    await loadAndBuild(rig, rows);
    await verify(rig, truth, scenario.radius, scenario.name);
    rig.destroy();
  });

  it('stays correct while the particles move and the count changes between steps', async () => {
    const rig = await makeRig(3, mode, 8192);
    const radius = 0.09;
    rig.grid.configure({ ...UNIT, radius });
    const random = makeRandom(11);
    let n = 2000;
    let truth = uniform(n, 21, 0, 1, 3);
    await loadAndBuild(rig, truth.slice());
    await verify(rig, truth, radius, 'step 0');

    const plan = [2000, 2000, 700, 700, 2600, 1, 2600, 0 + 1500];
    for (let step = 1; step < plan.length; step++) {
      const next = plan[step]!;
      // Move everything a little, then grow or shrink by adding or dropping ids at the end of the id range.
      const moved = new Float32Array(next * 4);
      for (let id = 0; id < next; id++) {
        const old = id < n;
        for (let axis = 0; axis < 3; axis++) {
          const base = old ? truth[id * 4 + axis]! : random();
          moved[id * 4 + axis] = Math.min(1, Math.max(0, base + (random() - 0.5) * 0.04));
        }
        moved[id * 4 + 3] = id;
      }
      truth = moved;
      n = next;
      // Reorder mode keeps particles in cell order, so each slot has to be rewritten in its own id's place.
      const ids = Array.from(await rig.system.read('id'));
      const slotIds: number[] = ids.filter((id) => id < n);
      const present = new Set(slotIds);
      for (let id = 0; id < n; id++) if (!present.has(id)) slotIds.push(id);
      const rows = new Float32Array(n * 4);
      slotIds.forEach((id, slot) => rows.set(truth.subarray(id * 4, id * 4 + 4), slot * 4));
      await loadAndBuild(rig, rows, Float32Array.from(slotIds));
      expect(rig.system.count).toBe(n);
      await verify(rig, truth, radius, `step ${step} (${n} particles)`);
    }
    rig.destroy();
  });

  it('follows a change of radius and domain', async () => {
    const rig = await makeRig(3, mode, 4096);
    const truth = uniform(1500, 31, 0, 2, 3);
    rig.grid.configure({ min: [0, 0, 0], max: [2, 2, 2], radius: 0.2 });
    await loadAndBuild(rig, truth.slice());
    await verify(rig, truth, 0.2, 'radius 0.2');
    // Smaller radius: more cells than the buffer was made for, so it grows.
    const before = rig.grid.generation;
    rig.grid.configure({ min: [0, 0, 0], max: [2, 2, 2], radius: 0.05 });
    expect(rig.grid.generation).toBeGreaterThan(before);
    await submit(rig.device, (e) => rig.grid.build(e, rig.system));
    // The particles are in whatever order the first build left them; the truth by id is unchanged.
    await verify(rig, truth, 0.05, 'radius 0.05');
    rig.destroy();
  });

  it('is exact when the grid coarsens to fit a small cell budget', async () => {
    const rig = await makeRig(3, mode, 4096, { maxCells: 64 });
    const truth = uniform(1200, 41, 0, 1, 3);
    const layout = rig.grid.configure({ ...UNIT, radius: 0.05 });
    expect(layout.coarsened).toBe(true);
    expect(layout.cells).toBeLessThanOrEqual(64);
    await loadAndBuild(rig, truth.slice());
    await verify(rig, truth, 0.05, 'coarsened');
    rig.destroy();
  });

  it('does not lose particles with NaN, infinite or enormous positions', async () => {
    const rig = await makeRig(3, mode, 256);
    rig.grid.configure({ ...UNIT, radius: 0.1 });
    const rows = uniform(200, 51, 0, 1, 3);
    rows.set([Number.NaN, 0.5, 0.5, 0], 0);
    rows.set([Number.POSITIVE_INFINITY, 0.5, 0.5, 0], 4);
    rows.set([0.5, Number.NEGATIVE_INFINITY, 0.5, 0], 8);
    rows.set([1e30, -1e30, 3e38, 0], 12);
    await loadAndBuild(rig, rows);
    const ids = Array.from(await rig.system.read('id'));
    expect([...ids].sort((a, b) => a - b)).toEqual(Array.from({ length: 200 }, (_, i) => i));
    const sorted = new Uint32Array(await readBuffer(rig.device, rig.grid.buffers.sorted, 200 * 4));
    expect([...sorted].sort((a, b) => a - b)).toEqual(Array.from({ length: 200 }, (_, i) => i));
    const starts = new Uint32Array(
      await readBuffer(rig.device, rig.grid.buffers.cellStart, (rig.grid.layout.cells + 1) * 4),
    );
    expect(starts[rig.grid.layout.cells]).toBe(200);
    rig.destroy();
  });
});

// --- The documented loop -------------------------------------------------------------------------

describe('the loop in neighbors.wgsl', () => {
  /** The worked example, pulled out of the header comment and uncommented. */
  function documentedExample(): string {
    const source = getChunk('neighbors.wgsl') ?? '';
    const lines = source.split('\n');
    const begin = lines.findIndex((line) => line.includes('---- example begin ----'));
    const end = lines.findIndex((line) => line.includes('---- example end ----'));
    expect(begin).toBeGreaterThan(0);
    expect(end).toBeGreaterThan(begin);
    return lines
      .slice(begin + 1, end)
      .map((line) => line.replace(/^\/\/ ?/, ''))
      .join('\n');
  }

  it.each<NeighborMode>(['reorder', 'sorted'])(
    'counts neighbours as documented (%s)',
    async (mode) => {
      const rig = await makeRig(3, mode, 4096);
      const radius = 0.1;
      rig.grid.configure({ ...UNIT, radius });
      const rows = uniform(2500, 61, 0, 1, 3);
      await loadAndBuild(rig, rows.slice());
      const { device, system, grid } = rig;
      const n = system.count;

      const pass = createComputePass(device, {
        label: `test/documented-${mode}.wgsl`,
        code: documentedExample(),
        chunks: grid.chunks({ group: 0, binding: 2 }),
      });
      const result = device.createBuffer({
        size: n * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      await expectValid(device, () =>
        submit(device, (e) =>
          pass.runCount(
            e,
            [{ buffer: system.positions, size: n * 16 }, result, ...grid.resources()],
            n,
          ),
        ),
      );
      const counts = new Float32Array(await readBuffer(device, result, n * 4));
      const ids = Array.from(await system.read('id'));
      const { inner, outer } = bruteForce(rows, n, radius, 3);
      for (let slot = 0; slot < n; slot++) {
        const id = ids[slot]!;
        expect(counts[slot]!, `id ${id}`).toBeGreaterThanOrEqual(inner[id]!.size);
        expect(counts[slot]!, `id ${id}`).toBeLessThanOrEqual(outer[id]!.size);
      }
      result.destroy();
      pass.destroy();
      rig.destroy();
    },
  );

  it('agrees with the reference counter, and the marker lights exactly the neighbours of the pick', async () => {
    for (const mode of ['reorder', 'sorted'] as const) {
      const rig = await makeRig(3, mode, 4096);
      const radius = 0.12;
      rig.grid.configure({ ...UNIT, radius });
      const rows = uniform(2000, 71, 0, 1, 3);
      await loadAndBuild(rig, rows.slice());
      const { device, system, grid } = rig;
      const n = system.count;
      const counter = createNeighborCounter({ device });
      const counts = device.createBuffer({
        size: n * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      const marks = device.createBuffer({
        size: n * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      const pick = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const picked = 123;
      device.queue.writeBuffer(pick, 0, new Uint32Array([picked]));
      await expectValid(device, () =>
        submit(device, (e) => {
          counter.count(e, grid, system, counts);
          counter.mark(e, grid, system, pick, marks);
        }),
      );
      const found = new Float32Array(await readBuffer(device, counts, n * 4));
      const lit = new Float32Array(await readBuffer(device, marks, n * 4));
      const ids = Array.from(await system.read('id'));
      const positions = await system.read('position');
      const { inner, outer } = bruteForce(rows, n, radius, 3);
      for (let slot = 0; slot < n; slot++) {
        const id = ids[slot]!;
        expect(found[slot]!).toBeGreaterThanOrEqual(inner[id]!.size);
        expect(found[slot]!).toBeLessThanOrEqual(outer[id]!.size);
      }
      // The pick is a slot index; its neighbours are the lit slots.
      expect(lit[picked]).toBe(2);
      const pickedId = ids[picked]!;
      for (let slot = 0; slot < n; slot++) {
        if (slot === picked) continue;
        const id = ids[slot]!;
        if (inner[pickedId]!.has(id)) expect(lit[slot], `slot ${slot}`).toBe(1);
        else if (!outer[pickedId]!.has(id)) expect(lit[slot], `slot ${slot}`).toBe(0);
      }
      expect(positions.length).toBe(n * 4);
      for (const b of [counts, marks, pick]) b.destroy();
      counter.destroy();
      rig.destroy();
    }
  });
});

describe('binding recipe', () => {
  it('spends 1 storage buffer in reorder mode and 2 in sorted-index mode, plus the uniform', async () => {
    const device = await getTestDevice();
    const grid = createNeighborGrid({ device, mode: 'reorder' });
    grid.configure({ ...UNIT, radius: 0.1 });
    expect(grid.storageBuffers).toBe(1);
    expect(grid.resources()).toHaveLength(2);
    expect(grid.bindingsWgsl({ group: 1, binding: 4 })).toContain(
      '@group(1) @binding(5) var<storage, read> neighbor_cell_start',
    );
    expect(grid.bindingsWgsl({ binding: 0 })).not.toContain('neighbor_sorted: array');
    grid.mode = 'sorted';
    expect(grid.storageBuffers).toBe(2);
    expect(grid.resources()).toHaveLength(3);
    expect(grid.bindingsWgsl({ binding: 0 })).toContain(
      '@binding(2) var<storage, read> neighbor_sorted',
    );
    grid.destroy();
  });

  it('refuses a reorder build from a bare buffer, and an unconfigured grid', async () => {
    const device = await getTestDevice();
    const grid = createNeighborGrid({ device, mode: 'reorder' });
    const positions = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE });
    const encoder = device.createCommandEncoder();
    expect(() => grid.build(encoder, { positions, count: 4 })).toThrow(/configure/);
    grid.configure({ ...UNIT, radius: 0.1 });
    expect(() => grid.build(encoder, { positions, count: 4 })).toThrow(/ParticleSystem/);
    grid.mode = 'sorted';
    expect(() => grid.build(encoder, { positions, count: 4 })).not.toThrow();
    grid.destroy();
    positions.destroy();
  });

  it('builds into a bare position buffer in sorted-index mode', async () => {
    const device = await getTestDevice();
    const grid = createNeighborGrid({ device, mode: 'sorted' });
    grid.configure({ ...UNIT, radius: 0.25 });
    const rows = uniform(400, 81, 0, 1, 3);
    const positions = device.createBuffer({
      size: rows.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(positions, 0, rows);
    await expectValid(device, () =>
      submit(device, (e) => grid.build(e, { positions, count: 400 })),
    );
    const sorted = new Uint32Array(await readBuffer(device, grid.buffers.sorted, 400 * 4));
    expect([...sorted].sort((a, b) => a - b)).toEqual(Array.from({ length: 400 }, (_, i) => i));
    grid.destroy();
    positions.destroy();
  });
});

// --- Cost study ------------------------------------------------------------------------------------
//
// The numbers in src/effects/particles/notes.md. Skipped unless PARTICLE_STUDY names a file to write the report to:
//
//     PARTICLE_STUDY=/tmp/particle-study.txt npx vitest run tests/neighbors.test.ts -t study
//
// Method (docs/effect-authoring.md, "Reading the HUD honestly"): every figure is the SLOPE of the GPU span between a small
// and a large number of repetitions of the workload recorded into ONE command buffer, so the ~1 ms floor of a single pass
// and the submit latency drop out. The span is measured from the begin timestamp of the first pass of the first
// repetition to the end timestamp of the last pass of the last (two timestamps in all, however many passes ran). The
// repetition count is chosen from a pilot run so the large run lasts roughly 3 ms. Small and large runs alternate 40 times;
// the figure is the slope between the quietest of each (a busy GPU only ever adds time), with the slope between the median
// runs in brackets. See `slope` for why not long runs and medians.

const studyFile = (globalThis as { process?: { env: Record<string, string | undefined> } }).process
  ?.env.PARTICLE_STUDY;

describe.skipIf(!studyFile)('study', () => {
  vi.setConfig({ testTimeout: 3_600_000 });
  const lines: string[] = [];
  const log = (text = '') => lines.push(text);

  async function finish() {
    // A variable specifier keeps the type checker (which has no Node types here) out of it.
    const specifier = 'node:fs';
    const { writeFileSync } = (await import(/* @vite-ignore */ specifier)) as {
      writeFileSync(path: string, text: string): void;
    };
    writeFileSync(studyFile!, lines.join('\n') + '\n');
  }

  /** A profiler that stamps only the very first pass's start and the very last pass's end. */
  async function makeTimer() {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter?.features.has('timestamp-query'))
      throw new Error('timestamp-query is not available');
    const device = await adapter.requestDevice({ requiredFeatures: ['timestamp-query'] });
    void device.lost.then((info) => {
      throw new Error(`the study's device was lost: ${info.message}`);
    });
    // Slots 0 and 1 are the two readings; 2 and 3 take the unwanted end of the first pass and begin of the last (Dawn on
    // Metal misbehaves, and runs out of memory, when a pass is given only one of its two timestamps).
    const querySet = device.createQuerySet({ type: 'timestamp', count: 4 });
    const resolve = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const read = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    let seen = 0;
    let expected = 0;
    const profiler = {
      pass() {
        const index = seen++;
        if (expected === 0) return undefined;
        if (expected === 1)
          return { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };
        if (index === 0) return { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 2 };
        if (index === expected - 1)
          return { querySet, beginningOfPassWriteIndex: 3, endOfPassWriteIndex: 1 };
        return undefined;
      },
    };
    /** How many timed passes `record` makes. */
    function countPasses(record: (encoder: GPUCommandEncoder) => void): number {
      seen = 0;
      expected = 0;
      // Finished and dropped, never submitted: an encoder left open is not safe to garbage collect.
      const encoder = device.createCommandEncoder();
      record(encoder);
      encoder.finish();
      return seen;
    }
    /** GPU milliseconds from the first pass starting to the last ending. */
    async function time(
      record: (encoder: GPUCommandEncoder) => void,
      passes: number,
    ): Promise<number> {
      seen = 0;
      expected = passes;
      device.pushErrorScope('validation');
      const encoder = device.createCommandEncoder();
      record(encoder);
      expected = 0;
      encoder.resolveQuerySet(querySet, 0, 4, resolve, 0);
      encoder.copyBufferToBuffer(resolve, 0, read, 0, 32);
      device.queue.submit([encoder.finish()]);
      const problem = await device.popErrorScope();
      if (problem)
        throw new Error(`validation error while timing ${passes} passes: ${problem.message}`);
      await read.mapAsync(GPUMapMode.READ);
      const stamps = new BigUint64Array(read.getMappedRange().slice(0, 32));
      read.unmap();
      return Number(stamps[1]! - stamps[0]!) / 1e6;
    }
    return { device, profiler, time, countPasses };
  }

  /** The quietest sample and the median sample, as ms per repetition. */
  interface Stat {
    best: number;
    median: number;
  }
  const fmt = (s: Stat, digits = 3) =>
    `${s.best.toFixed(digits)} (med ${s.median.toFixed(digits)})`;

  type Timer = Awaited<ReturnType<typeof makeTimer>>;

  /**
   * The cost of one repetition of `work`. The GPU is shared, and a busy moment only ever adds time (a pass that gets
   * pre-empted by another process can read 13 ms for 0.1 ms of work, the length of a scheduler slice), so the figure
   * is the slope between the QUIETEST small run and the QUIETEST large run of many alternating ones, as in
   * the fluid cost study; `median` is the same slope from the median runs, which is what a busy machine shows.
   */
  async function slope(
    timer: Timer,
    work: (encoder: GPUCommandEncoder) => void,
    rounds = 40,
  ): Promise<Stat> {
    const per = timer.countPasses(work);
    const repeat = (n: number) => (e: GPUCommandEncoder) => {
      for (let i = 0; i < n; i++) work(e);
    };
    // A pilot (its quietest of 5 runs) picks the repetition count: about 3 ms in the large run, at most 120 repetitions.
    let pilot = Infinity;
    for (let i = 0; i < 5; i++) pilot = Math.min(pilot, (await timer.time(repeat(4), per * 4)) / 4);
    const hi = Math.min(120, Math.max(8, Math.round(3 / Math.max(pilot, 0.005))));
    const lo = Math.max(1, Math.round(hi / 4));
    const small: number[] = [];
    const large: number[] = [];
    for (let round = 0; round < rounds; round++) {
      small.push(await timer.time(repeat(lo), per * lo));
      large.push(await timer.time(repeat(hi), per * hi));
    }
    small.sort((x, y) => x - y);
    large.sort((x, y) => x - y);
    const middle = Math.floor(rounds / 2);
    return {
      best: (large[0]! - small[0]!) / (hi - lo),
      median: (large[middle]! - small[middle]!) / (hi - lo),
    };
  }

  /** A fluid-like fill: uniform random in the lower 40 percent of the unit box, with a target neighbour count. */
  function poolScene(n: number, dimension: 2 | 3, seed: number) {
    const fill = 0.4;
    const target = dimension === 3 ? 32 : 16;
    const radius =
      dimension === 3
        ? Math.cbrt((target * fill) / (n * (4 / 3) * Math.PI))
        : Math.sqrt((target * fill) / (n * Math.PI));
    const random = makeRandom(seed);
    const rows = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      rows[i * 4] = random();
      rows[i * 4 + 1] = random() * fill;
      rows[i * 4 + 2] = dimension === 2 ? 0 : random();
    }
    return { rows, radius };
  }

  const SIZES = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000];

  it('cost against particle count, 2D and 3D, sorted index versus reorder', async () => {
    const timer = await makeTimer();
    const { device } = timer;
    log('NEIGHBOUR GRID COST. Apple M1 Max, Metal through Dawn (Node), timestamp queries.');
    log(
      'Milliseconds per call, quietest of 40 alternating runs (the median-run figure in brackets). Particles in a fluid-like pool (lower 40 percent of the unit box), in random order in memory,',
    );
    log(
      'with a radius giving about 32 neighbours in 3D and 16 in 2D. Each particle is a position and a velocity (32 bytes) that move together.',
    );
    log();

    const main: string[][] = [];
    const stages: string[][] = [];
    for (const dimension of [3, 2] as const) {
      for (const n of SIZES) {
        const { rows: positions, radius } = poolScene(n, dimension, 100 + n);
        const system = createParticleSystem({
          device,
          capacity: n,
          radius: 0.001,
          dimension,
          profiler: timer.profiler,
        });
        system.set({ positions });
        const grid = createNeighborGrid({
          device,
          dimension,
          mode: 'sorted',
          capacity: n,
          profiler: timer.profiler,
          label: 'grid',
        });
        const layout = grid.configure({ min: [0, 0, 0], max: [1, 1, 1], radius });
        const counter = createNeighborCounter({ device, profiler: timer.profiler });
        const counts = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE });

        // Sorted-index mode: the particle arrays stay in their random order.
        await submit(device, (e) => grid.build(e, system));
        const cumulativeRandom: Stat[] = [];
        for (const stopAfter of ['cells', 'scan', 'scatter'] as const) {
          cumulativeRandom.push(await slope(timer, (e) => grid.build(e, system, { stopAfter })));
        }
        const buildSorted = cumulativeRandom[2]!;
        const countSorted = await slope(timer, (e) => counter.count(e, grid, system, counts));

        // Reorder mode: sort the data once so it is in steady state (each step starts from last step's order).
        grid.mode = 'reorder';
        await submit(device, (e) => grid.build(e, system));
        await submit(device, (e) => grid.build(e, system));
        const cumulativeOrdered: Stat[] = [];
        for (const stopAfter of ['cells', 'scan', 'scatter', 'reorder'] as const) {
          cumulativeOrdered.push(await slope(timer, (e) => grid.build(e, system, { stopAfter })));
        }
        const buildReorder = cumulativeOrdered[3]!;
        const countReorder = await slope(timer, (e) => counter.count(e, grid, system, counts));

        main.push([
          `${dimension}D`,
          String(n),
          String(layout.cells),
          fmt(buildSorted),
          fmt(countSorted),
          (buildSorted.best + countSorted.best).toFixed(3),
          fmt(buildReorder),
          fmt(countReorder),
          (buildReorder.best + countReorder.best).toFixed(3),
        ]);
        const delta = (list: Stat[], i: number) =>
          (list[i]!.best - (i > 0 ? list[i - 1]!.best : 0)).toFixed(3);
        stages.push([
          `${dimension}D`,
          String(n),
          delta(cumulativeRandom, 0),
          delta(cumulativeRandom, 1),
          delta(cumulativeRandom, 2),
          delta(cumulativeOrdered, 0),
          delta(cumulativeOrdered, 1),
          delta(cumulativeOrdered, 2),
          delta(cumulativeOrdered, 3),
        ]);
        counts.destroy();
        counter.destroy();
        grid.destroy();
        system.destroy();
        await finish();
      }
    }
    const table = (header: string[], body: string[][]) => {
      log(header.join(' | '));
      for (const row of body) log(row.join(' | '));
      log();
    };
    log('A. Build, reference count pass, and their sum per frame, sorted index vs reorder (ms)');
    table(
      [
        'dim',
        'particles',
        'cells',
        'build sorted',
        'count sorted',
        'sum',
        'build reorder',
        'count reorder',
        'sum',
      ],
      main,
    );
    log(
      'B. Where the build goes: the cost each stage adds (ms, quietest-run slopes of the cumulative build that stops after the stage).',
    );
    log(
      '   cells = clear + cell id + atomic count; scan = prefix sum over the cells; scatter = write the sorted index; gather = move the 32 bytes of every particle.',
    );
    log(
      '   Left: particles in random order (what sorted-index mode always sees). Right: reorder mode in steady state (the data is already nearly in cell order).',
    );
    table(
      [
        'dim',
        'particles',
        'cells',
        '+ scan',
        '+ scatter',
        'cells',
        '+ scan',
        '+ scatter',
        '+ gather',
      ],
      stages,
    );
    await finish();
  });

  it('the count pass against the neighbour count, and the index indirection on coherent data', async () => {
    const timer = await makeTimer();
    const { device } = timer;
    log(
      'E. The reference count pass against how many neighbours there are (3D, pool, ms). h sets the neighbour count: about 32 is a typical SPH',
    );
    log(
      '   (h = 2 particle spacings), 60 or more is position-based fluids with a wide kernel. "random" and "reordered" are the two modes; "coherent index" is',
    );
    log(
      '   sorted-index mode on data that was reordered earlier (an index list over memory that is already in cell order).',
    );
    log(
      [
        'particles',
        'neighbours',
        'count, random order',
        'count, reordered',
        'count, coherent index',
        'cost per neighbour candidate (reordered, ns)',
      ].join(' | '),
    );
    for (const n of [250_000, 1_000_000]) {
      for (const target of [8, 16, 32, 64, 128]) {
        const fill = 0.4;
        const radius = Math.cbrt((target * fill) / (n * (4 / 3) * Math.PI));
        const random = makeRandom(900 + n + target);
        const rows = new Float32Array(n * 4);
        for (let i = 0; i < n; i++) {
          rows[i * 4] = random();
          rows[i * 4 + 1] = random() * fill;
          rows[i * 4 + 2] = random();
        }
        const system = createParticleSystem({
          device,
          capacity: n,
          radius: 0.001,
          profiler: timer.profiler,
        });
        system.set({ positions: rows });
        const grid = createNeighborGrid({
          device,
          mode: 'sorted',
          capacity: n,
          profiler: timer.profiler,
          label: 'grid',
        });
        grid.configure({ min: [0, 0, 0], max: [1, 1, 1], radius });
        const counter = createNeighborCounter({ device, profiler: timer.profiler });
        const counts = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE });
        await submit(device, (e) => grid.build(e, system));
        const random_ = await slope(timer, (e) => counter.count(e, grid, system, counts), 25);
        grid.mode = 'reorder';
        await submit(device, (e) => grid.build(e, system));
        await submit(device, (e) => grid.build(e, system));
        const reordered = await slope(timer, (e) => counter.count(e, grid, system, counts), 25);
        grid.mode = 'sorted';
        await submit(device, (e) => grid.build(e, system));
        const coherent = await slope(timer, (e) => counter.count(e, grid, system, counts), 25);
        // Candidates visited per particle: 27 cells' worth at density n * fill, about 6.4 x the neighbours in 3D.
        const candidates = (27 * target) / ((4 / 3) * Math.PI);
        const ns = (reordered.best * 1e6) / (n * candidates);
        log(
          `${n} | ${target} | ${fmt(random_)} | ${fmt(reordered)} | ${fmt(coherent)} | ${ns.toFixed(3)}`,
        );
        counts.destroy();
        counter.destroy();
        grid.destroy();
        system.destroy();
        await finish();
      }
    }
    log();
    await finish();
  });

  it('atomics under contention, and the scan against the cell count', async () => {
    const timer = await makeTimer();
    const { device } = timer;
    const n = 250_000;
    log(
      'C. Counting pass under contention: 250 000 particles in 3D, full sorted-index build (ms), by how many cells they share',
    );
    log(['particles share', 'build (ms)', 'fullest cell'].join(' | '));
    const random = makeRandom(5);
    for (const [name, cellsAcross] of [
      ['all in 1 cell', 1],
      ['64 cells (4x4x4)', 4],
      ['4096 cells (16x16x16)', 16],
      ['the pool (about 85 000 cells)', 0],
    ] as const) {
      const system = createParticleSystem({
        device,
        capacity: n,
        radius: 0.001,
        profiler: timer.profiler,
      });
      const rows = new Float32Array(n * 4);
      const grid = createNeighborGrid({
        device,
        mode: 'sorted',
        capacity: n,
        profiler: timer.profiler,
        label: 'grid',
      });
      const radius = cellsAcross === 0 ? poolScene(n, 3, 1).radius : 0.05;
      const layout = grid.configure({ min: [0, 0, 0], max: [1, 1, 1], radius });
      for (let i = 0; i < n; i++) {
        const span = cellsAcross === 0 ? 1 : cellsAcross * layout.cellSize;
        const height = cellsAcross === 0 ? 0.4 : span;
        rows[i * 4] = random() * span * 0.999;
        rows[i * 4 + 1] = random() * height * 0.999;
        rows[i * 4 + 2] = random() * span * 0.999;
      }
      system.set({ positions: rows });
      await submit(device, (e) => grid.build(e, system));
      const build = await slope(timer, (e) => grid.build(e, system));
      const starts = new Uint32Array(
        await readBuffer(device, grid.buffers.cellStart, (layout.cells + 1) * 4),
      );
      let biggest = 0;
      for (let c = 0; c < layout.cells; c++)
        biggest = Math.max(biggest, starts[c + 1]! - starts[c]!);
      log(`${name} | ${fmt(build)} | ${biggest}`);
      grid.destroy();
      system.destroy();
    }
    log();

    log(
      'D. The prefix sum alone, in place over u32 (ms per scan), by element count and the grid it would serve',
    );
    log(['elements', 'dispatches', 'ms', 'grid'].join(' | '));
    const scan = createScan({
      device,
      maxElements: 4_200_000,
      profiler: timer.profiler,
      label: 'scan',
    });
    for (const [elements, grid] of [
      [16 ** 3 + 1, '16^3'],
      [32 ** 3 + 1, '32^3'],
      [64 ** 3 + 1, '64^3'],
      [128 ** 3 + 1, '128^3'],
      [160 ** 3 + 1, '160^3 (about the default cap)'],
    ] as const) {
      const buffer = device.createBuffer({ size: elements * 4, usage: GPUBufferUsage.STORAGE });
      const s = await slope(timer, (e) => scan.run(e, buffer, elements));
      log(`${elements} | ${scanLevels(elements) * 2 - 1} | ${fmt(s)} | ${grid}`);
      buffer.destroy();
    }
    scan.destroy();
    log();
    await finish();
  });
});
