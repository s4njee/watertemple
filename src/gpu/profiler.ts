export interface PassTiming {
  label: string;
  /** GPU time the pass took, in milliseconds. */
  ms: number;
}

/** The part of the profiler an effect sees. */
export interface PassProfiler {
  /**
   * Timestamp writes for one render or compute pass in the current frame:
   * `encoder.beginRenderPass({ ..., timestampWrites: profiler.pass('blur') })`.
   * Undefined when timing is unavailable or switched off, which a pass descriptor accepts.
   */
  pass(label: string): GPURenderPassTimestampWrites | undefined;
}

export interface GpuProfiler extends PassProfiler {
  /** False when the device lacks `timestamp-query`; `pass` then always returns undefined. */
  readonly supported: boolean;
  /** Timing costs a little, so it only runs while something is displaying it. */
  enabled: boolean;
  /**
   * Per-pass timings of the most recent frame whose results have come back from the GPU, one entry
   * per label (repeats within a frame are summed). A new array is published for each frame that
   * arrives, so a changed identity means a new sample. Empty while disabled.
   */
  readonly latest: readonly PassTiming[];
  /**
   * GPU time of that same frame: from its first timed pass starting to its last one ending, in
   * milliseconds. This is the frame's cost. Do not add up `latest` instead: on tile-based GPUs
   * (Apple, most phones) the intervals of consecutive passes overlap, so their sum can be several
   * times the real total. Undefined until a frame has come back.
   */
  readonly latestFrameMs: number | undefined;
  beginFrame(): void;
  /** Call after the frame's passes are encoded and before `encoder.finish()`. */
  endFrame(encoder: GPUCommandEncoder): void;
  /** Call after the frame's commands are submitted. */
  afterSubmit(): void;
  destroy(): void;
}

/** For tests and anywhere else timing is not wanted. */
export const noProfiler: PassProfiler = { pass: () => undefined };

/** Passes timed per frame; later calls to `pass` get undefined. */
export const MAX_TIMED_PASSES = 32;
/** Frames of results that may be in flight at once, i.e. readback buffers. */
export const READBACK_POOL_SIZE = 3;
/** A single pass longer than this is a counter reset or a stall, not a measurement. */
const MAX_PLAUSIBLE_NS = 1_000_000_000n;

/**
 * Turns resolved timestamps (begin/end pairs in nanoseconds) into per-label milliseconds. Pairs
 * that run backwards, are unwritten (zero), or span an absurd time are dropped, and labels used
 * more than once are summed.
 */
export function timingsFromTimestamps(
  stamps: ArrayLike<bigint>,
  labels: readonly string[],
): PassTiming[] {
  const totals = new Map<string, bigint>();
  labels.forEach((label, index) => {
    const begin = stamps[index * 2];
    const end = stamps[index * 2 + 1];
    if (begin === undefined || end === undefined || begin === 0n || end < begin) return;
    const duration = end - begin;
    if (duration > MAX_PLAUSIBLE_NS) return;
    totals.set(label, (totals.get(label) ?? 0n) + duration);
  });
  return [...totals].map(([label, ns]) => ({ label, ms: Number(ns) / 1e6 }));
}

/**
 * Time from the earliest begin to the latest end among the valid pairs, in milliseconds, or
 * undefined when no pair is valid. Pairs are judged as in `timingsFromTimestamps`.
 */
export function frameSpanFromTimestamps(
  stamps: ArrayLike<bigint>,
  passCount: number,
): number | undefined {
  let first: bigint | undefined;
  let last: bigint | undefined;
  for (let index = 0; index < passCount; index++) {
    const begin = stamps[index * 2];
    const end = stamps[index * 2 + 1];
    if (begin === undefined || end === undefined || begin === 0n || end < begin) continue;
    if (end - begin > MAX_PLAUSIBLE_NS) continue;
    if (first === undefined || begin < first) first = begin;
    if (last === undefined || end > last) last = end;
  }
  if (first === undefined || last === undefined) return undefined;
  return Number(last - first) / 1e6;
}

interface Readback {
  buffer: GPUBuffer;
  /** Free to be used by the next frame. */
  free: boolean;
  /** One entry per pair of timestamps written into this buffer. */
  labels: string[];
}

interface GpuResources {
  querySet: GPUQuerySet;
  resolve: GPUBuffer;
  readbacks: Readback[];
}

const BYTES_PER_PASS = 16;

export function createGpuProfiler(device: GPUDevice): GpuProfiler {
  const supported = device.features.has('timestamp-query');
  let enabled = false;
  let destroyed = false;
  let latest: readonly PassTiming[] = [];
  let latestFrameMs: number | undefined;
  // Created on first use, so a profiler nobody enables never touches the GPU.
  let resources: GpuResources | undefined;
  // State of the frame being encoded.
  let slot: Readback | undefined;
  let labels: string[] = [];
  // Set by endFrame once the copy is encoded; afterSubmit then starts the map.
  let encoded: Readback | undefined;

  function ensureResources(): GpuResources {
    resources ??= {
      querySet: device.createQuerySet({
        label: 'profiler timestamps',
        type: 'timestamp',
        count: MAX_TIMED_PASSES * 2,
      }),
      resolve: device.createBuffer({
        label: 'profiler resolve',
        size: MAX_TIMED_PASSES * BYTES_PER_PASS,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      }),
      readbacks: Array.from({ length: READBACK_POOL_SIZE }, (_, index) => ({
        buffer: device.createBuffer({
          label: `profiler readback ${index}`,
          size: MAX_TIMED_PASSES * BYTES_PER_PASS,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        }),
        free: true,
        labels: [],
      })),
    };
    return resources;
  }

  function abandonFrame() {
    // A frame that was encoded but never submitted never starts a map, so its buffer is free again.
    if (slot) slot.free = true;
    if (encoded) encoded.free = true;
    slot = undefined;
    encoded = undefined;
    labels = [];
  }

  function read(readback: Readback) {
    const { buffer } = readback;
    buffer.mapAsync(GPUMapMode.READ).then(
      () => {
        if (destroyed) return;
        const bytes = readback.labels.length * BYTES_PER_PASS;
        const stamps = new BigUint64Array(buffer.getMappedRange(0, bytes).slice(0));
        buffer.unmap();
        readback.free = true;
        // Results that land after the HUD closed would show up stale if it reopened.
        if (!enabled) return;
        const timings = timingsFromTimestamps(stamps, readback.labels);
        if (timings.length > 0) {
          latest = timings;
          latestFrameMs = frameSpanFromTimestamps(stamps, readback.labels.length);
        }
      },
      () => {
        // The buffer was destroyed under the map, or the device was lost: nothing to report.
        readback.free = true;
      },
    );
  }

  return {
    supported,
    get enabled() {
      return enabled;
    },
    set enabled(value) {
      const next = value && supported && !destroyed;
      if (next === enabled) return;
      enabled = next;
      if (!enabled) {
        latest = [];
        latestFrameMs = undefined;
        abandonFrame();
      }
    },
    get latest() {
      return enabled ? latest : [];
    },
    get latestFrameMs() {
      return enabled ? latestFrameMs : undefined;
    },
    pass(label) {
      if (!slot || labels.length >= MAX_TIMED_PASSES) return undefined;
      const index = labels.length * 2;
      labels.push(label);
      return {
        querySet: ensureResources().querySet,
        beginningOfPassWriteIndex: index,
        endOfPassWriteIndex: index + 1,
      };
    },
    beginFrame() {
      abandonFrame();
      if (!enabled) return;
      // Every buffer still waiting on the GPU or a map: skip this frame's timing rather than
      // queue without bound.
      slot = ensureResources().readbacks.find((readback) => readback.free);
      if (slot) slot.free = false;
    },
    endFrame(encoder) {
      if (!slot || !resources) return;
      if (labels.length === 0) {
        abandonFrame();
        return;
      }
      const bytes = labels.length * BYTES_PER_PASS;
      encoder.resolveQuerySet(resources.querySet, 0, labels.length * 2, resources.resolve, 0);
      encoder.copyBufferToBuffer(resources.resolve, 0, slot.buffer, 0, bytes);
      slot.labels = labels;
      encoded = slot;
      slot = undefined;
      labels = [];
    },
    afterSubmit() {
      const readback = encoded;
      encoded = undefined;
      abandonFrame();
      if (readback) read(readback);
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      enabled = false;
      latest = [];
      latestFrameMs = undefined;
      slot = undefined;
      encoded = undefined;
      resources?.querySet.destroy();
      resources?.resolve.destroy();
      for (const readback of resources?.readbacks ?? []) readback.buffer.destroy();
      resources = undefined;
    },
  };
}
