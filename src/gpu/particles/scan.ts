import { noProfiler, type PassProfiler } from '../profiler.ts';
import { createComputePass } from './compute.ts';
import scanSource from './scan.wgsl?raw';

// A parallel exclusive prefix sum over u32 in a storage buffer, in place, exact for any length up to
// MAX_SCAN_ELEMENTS. It is a "reduce, scan the sums, add back" scan (Harris, Sengupta and Owens, GPU Gems 3
// ch. 39, in the tiled form of Merrill and Grimshaw):
//
//   level 0   scan_blocks over the data: each 1024-element tile is scanned in workgroup memory
//             (256 invocations x 4 elements, a Hillis-Steele scan of the 256 totals) and its total goes
//             to sums[0]
//   level 1   the tile totals are themselves an array of n / 1024 elements: scan them the same way,
//   ...       and so on until a level fits in one tile (n <= 1024)
//   back up   add_offsets adds each tile's scanned total to its elements, from the top level down
//
// So n = 1 024 takes 1 dispatch, 262 144 (a 64^3 grid) 3, 2 097 152 (128^3) 5, and each level is a
// factor of 1024 smaller than the one below, so the cost is dominated by the one pass over the data. The
// scan is exclusive: out[i] = sum of in[0 .. i-1]; to get the total as well, scan one element more than
// you need (the grid does: cell_start has cells + 1 entries).
//
// Why not a single-pass decoupled look-back scan: it relies on forward-progress guarantees between
// workgroups that WebGPU does not give. This one only relies on the barriers between dispatches.

/** Elements one workgroup scans: 256 invocations x 4. */
export const SCAN_TILE = 1024;
/** Longest array: the number of tiles must fit one dispatch (65535). */
export const MAX_SCAN_ELEMENTS = SCAN_TILE * 65535;

export interface Scan {
  readonly maxElements: number;
  /**
   * Replaces the first `count` u32 of `data` with their exclusive prefix sums. The buffer needs
   * STORAGE usage and at least `count * 4` bytes. Records `scanLevels(count) * 2 - 1` compute passes.
   */
  run(encoder: GPUCommandEncoder, data: GPUBuffer, count: number): void;
  destroy(): void;
}

/** Element counts of each level for an array of `count` elements: [count, tiles, tiles of tiles, ...]. */
export function scanLevelSizes(count: number): number[] {
  const sizes = [count];
  while (sizes[sizes.length - 1]! > SCAN_TILE) {
    sizes.push(Math.ceil(sizes[sizes.length - 1]! / SCAN_TILE));
  }
  return sizes;
}

/** Number of levels (dispatches of scan_blocks) for `count` elements. */
export function scanLevels(count: number): number {
  return scanLevelSizes(count).length;
}

export interface ScanOptions {
  device: GPUDevice;
  /** Longest array it will be asked to scan; sizes the scratch buffers. */
  maxElements: number;
  profiler?: PassProfiler;
  label?: string;
}

export function createScan(options: ScanOptions): Scan {
  const { device, maxElements, profiler = noProfiler, label = 'particles scan' } = options;
  if (!(maxElements >= 1) || maxElements > MAX_SCAN_ELEMENTS) {
    throw new RangeError(`scan length must be 1..${MAX_SCAN_ELEMENTS}, got ${maxElements}`);
  }
  const scanBlocks = createComputePass(device, {
    label: 'particles/scan.wgsl',
    code: scanSource,
    entryPoint: 'scan_blocks',
    workgroupSize: 256,
    profiler,
    timestamp: `${label}`,
  });
  const addOffsets = createComputePass(device, {
    label: 'particles/scan.wgsl',
    code: scanSource,
    entryPoint: 'add_offsets',
    workgroupSize: 256,
    profiler,
    timestamp: `${label}`,
  });

  // One scratch array per level above 0, sized for the longest input. Level k's `sums` is level k + 1's data.
  const maxSizes = scanLevelSizes(maxElements);
  const scratch = maxSizes.slice(1).map((size, index) =>
    device.createBuffer({
      label: `${label} tile sums ${index}`,
      size: size * 4,
      usage: GPUBufferUsage.STORAGE,
    }),
  );
  // The top level has nowhere to put its total; it writes here.
  const sink = device.createBuffer({
    label: `${label} sink`,
    size: 16,
    usage: GPUBufferUsage.STORAGE,
  });

  function run(encoder: GPUCommandEncoder, data: GPUBuffer, count: number): void {
    if (!Number.isInteger(count) || count < 0 || count > maxElements) {
      throw new RangeError(`scan count must be an integer 0..${maxElements}, got ${count}`);
    }
    if (count === 0) return;
    if (data.size < count * 4)
      throw new RangeError(`${label}: buffer holds fewer than ${count} u32`);
    const sizes = scanLevelSizes(count);
    const buffers = [data, ...scratch];
    const dataOf = (level: number): GPUBufferBinding => ({
      buffer: buffers[level]!,
      size: sizes[level]! * 4,
    });
    const sumsOf = (level: number): GPUBufferBinding =>
      level + 1 < sizes.length ? dataOf(level + 1) : { buffer: sink, size: 16 };

    for (let level = 0; level < sizes.length; level++) {
      scanBlocks.runGroups(
        encoder,
        [dataOf(level), sumsOf(level)],
        Math.ceil(sizes[level]! / SCAN_TILE),
      );
    }
    for (let level = sizes.length - 2; level >= 0; level--) {
      addOffsets.runGroups(
        encoder,
        [dataOf(level), sumsOf(level)],
        Math.ceil(sizes[level]! / SCAN_TILE),
      );
    }
  }

  return {
    maxElements,
    run,
    destroy() {
      for (const buffer of scratch) buffer.destroy();
      sink.destroy();
      scanBlocks.destroy();
      addOffsets.destroy();
    },
  };
}
