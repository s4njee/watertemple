import { describe, it, vi } from 'vitest';
import { createPbf, type PbfParams } from '../src/gpu/sim/pbf.ts';
import { getTestDevice } from '../tests/gpu.ts';
import { createNeighborCounter, createNeighborGrid, createParticleSystem } from '../src/gpu/particles/index.ts';
import { loadEffect } from '../src/effects/registry.ts';
import { defaultParams } from '../src/effects/types.ts';

vi.setConfig({ testTimeout: 7_200_000 });

const env = (globalThis as { process?: { env: Record<string, string | undefined> } }).process!.env;
const studyFile = env.PBF_STUDY;

describe.skipIf(!studyFile)('study', () => {
  const lines: string[] = [];
  const json: Record<string, unknown> = {};
  const log = (text = '') => { lines.push(text); console.log(text); };

  async function finish() {
    const { writeFileSync } = (await import(/* @vite-ignore */ 'node:fs')) as { writeFileSync(p: string, t: string): void };
    writeFileSync(studyFile!, JSON.stringify(json, null, 2));
  }
  async function timedDevice() {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter?.features.has('timestamp-query')) throw new Error('timestamp-query is not available');
    return adapter.requestDevice({ requiredFeatures: ['timestamp-query'] });
  }
  const median = (a: number[]) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]!; };
  const f = (x: number, d = 2) => x.toFixed(d);

  // A GPU span recorder: first begin to last end of the timed passes of a recording.
  function recorder(device: GPUDevice) {
    const querySet = device.createQuerySet({ type: 'timestamp', count: 512 });
    const resolve = device.createBuffer({ size: 512 * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: 512 * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const state = { mode: 'off' as 'off' | 'span' | 'all', expected: 0, seen: 0, labels: [] as string[] };
    const profiler = {
      pass(label: string) {
        const index = state.seen++;
        if (state.mode === 'all') {
          state.labels.push(label);
          if (index >= 250) return undefined;
          return { querySet, beginningOfPassWriteIndex: index * 2, endOfPassWriteIndex: index * 2 + 1 };
        }
        if (state.mode === 'span') {
          if (state.expected === 1) return { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };
          if (index === 0) return { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 2 };
          if (index === state.expected - 1) return { querySet, beginningOfPassWriteIndex: 3, endOfPassWriteIndex: 1 };
        }
        return undefined;
      },
    };
    const run = async (record: (e: GPUCommandEncoder) => void, slots: number) => {
      const encoder = device.createCommandEncoder();
      record(encoder);
      encoder.resolveQuerySet(querySet, 0, slots, resolve, 0);
      encoder.copyBufferToBuffer(resolve, 0, read, 0, slots * 8);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const stamps = new BigUint64Array(read.getMappedRange().slice(0, slots * 8));
      read.unmap();
      return stamps;
    };
    /** Quietest and median span (ms) of `unit()` recorded `lo` and `hi` times into one command buffer. */
    async function slope(unit: (e: GPUCommandEncoder) => void, passesPerUnit: number, lo: number, hi: number, rounds: number) {
      const time = async (n: number) => {
        state.mode = 'span'; state.seen = 0; state.expected = passesPerUnit * n;
        const stamps = await run((e) => { for (let i = 0; i < n; i++) unit(e); }, 4);
        return Number(stamps[1]! - stamps[0]!) / 1e6;
      };
      const small: number[] = [], large: number[] = [];
      for (let r = 0; r < rounds; r++) { small.push(await time(lo)); large.push(await time(hi)); }
      small.sort((a, b) => a - b); large.sort((a, b) => a - b);
      const mid = Math.floor(rounds / 2);
      const q = (a: number[], p: number) => a[Math.floor(p * (a.length - 1))]!;
      return {
        best: (large[0]! - small[0]!) / (hi - lo),
        median: (large[mid]! - small[mid]!) / (hi - lo),
        p25: (q(large, 0.25) - q(small, 0.25)) / (hi - lo),
        p75: (q(large, 0.75) - q(small, 0.75)) / (hi - lo),
      };
    }
    async function countPasses(unit: (e: GPUCommandEncoder) => void) {
      state.mode = 'all'; state.seen = 0; state.labels = [];
      await run(unit, 2);
      return state.seen;
    }
    async function perLabel(unit: (e: GPUCommandEncoder) => void, rounds: number) {
      const best: Record<string, number> = {};
      for (let r = 0; r < rounds; r++) {
        state.mode = 'all'; state.seen = 0; state.labels = [];
        const stamps = await run(unit, Math.min(500, 2 * 250));
        const sums = new Map<string, number>();
        state.labels.forEach((l, i) => sums.set(l, (sums.get(l) ?? 0) + Number(stamps[i * 2 + 1]! - stamps[i * 2]!) / 1e6));
        for (const [l, v] of sums) best[l] = Math.min(best[l] ?? 1e9, v);
      }
      return best;
    }
    return { profiler, state, slope, countPasses, perLabel, run, destroy() { querySet.destroy(); resolve.destroy(); read.destroy(); } };
  }

  async function psSnapshot(): Promise<string> {
    try {
      const { execSync } = (await import(/* @vite-ignore */ 'node:child_process')) as { execSync(c: string): { toString(): string } };
      return execSync("ps -Ao pcpu,comm -r | head -8 | sed 's/ \\+/ /g'").toString();
    } catch { return 'unavailable'; }
  }


  // A fixed yardstick, so that a reading can be judged against how busy the machine was: the framework's own
  // neighbour-count pass over 250 000 particles in a pool (documented in src/effects/particles/notes.md as 0.39 ms on a
  // quiet M1 Max). A reading well above that means somebody else was using the GPU.
  async function calibrate(device: GPUDevice, rec: ReturnType<typeof recorder>): Promise<number> {
    const n = 250000;
    const system = createParticleSystem({ device, capacity: n, radius: 0.01, dimension: 3 });
    const side = Math.ceil(Math.cbrt(n));
    system.fill(n, (i, p) => {
      p.position[0] = (i % side) * 0.0125 + 0.0003 * ((i * 7) % 5);
      p.position[1] = (Math.floor(i / side) % side) * 0.0125 * 0.4;
      p.position[2] = Math.floor(i / (side * side)) * 0.0125;
    });
    const grid = createNeighborGrid({ device, dimension: 3, mode: 'reorder', capacity: n, profiler: rec.profiler });
    const counter = createNeighborCounter({ device, profiler: rec.profiler });
    const result = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    grid.configure({ min: [0, 0, 0], max: [side * 0.0125, side * 0.0125 * 0.4 + 0.0125, side * 0.0125], radius: 0.0275 });
    const unit = (e: GPUCommandEncoder) => { grid.build(e, system); counter.count(e, grid, system, result); };
    for (let i = 0; i < 4; i++) await rec.run(unit, 2);
    const best = await rec.perLabel(unit, 14);
    grid.destroy(); counter.destroy(); system.destroy(); result.destroy();
    return best['neighbours count'] ?? 0;
  }

  // --- A: the solver alone, stationary pools ---------------------------------------------------------
  it('A. solver cost', async () => {
    const device = await timedDevice();
    const rec = recorder(device);
    const yardstickBefore = await calibrate(device, rec);
    json.machine = { gpu: 'Apple M1 Max (Metal through Dawn, Node)', note: env.PBF_STUDY_NOTE ?? '', ps: await psSnapshot(), yardstickBeforeMs: yardstickBefore };
    log(`Yardstick (framework neighbour-count pass, 250k particles, 3D): ${f(yardstickBefore, 3)} ms; 0.39 ms when the GPU is quiet.`);
    const solverRows: unknown[] = [];
    json.solver = solverRows;
    for (const dim of [2, 3] as const) {
      const counts = dim === 2 ? [5000, 10000, 25000, 50000, 100000, 250000] : [10000, 30000, 60000, 120000, 250000, 500000];
      log(`## Solver, ${dim}D: ms per sub-step (stationary pool filling its tank, gravity off, reorder mode)`);
      log('| particles | iterations | span/sub-step | per-iteration | fixed part | predict | grid | density it | correct it | viscosity |');
      log('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |');
      for (const n of counts) {
        const spacing = dim === 3 ? 0.02 : 0.01;
        const side = Math.ceil(dim === 3 ? Math.cbrt(n) : Math.sqrt(n));
        const nx = side, nz = dim === 3 ? side : 1;
        const ny = Math.ceil(n / (nx * nz));
        const pbf = createPbf({ device, dimension: dim, capacity: nx * ny * nz + 16, spacing, profiler: rec.profiler, timedPasses: 100000,
          bounds: { min: [0, 0, dim === 3 ? 0 : -1], max: [nx * spacing, ny * spacing, dim === 3 ? nz * spacing : 1] }, params: { gravity: [0, 0, 0], iterations: 2 } });
        pbf.reset({ blocks: [{ min: [0, 0, 0], max: [nx * spacing, ny * spacing, nz * spacing], jitter: 0.1 }] });
        // warm: pipelines and a few steps so the data is in cell order
        for (let i = 0; i < 6; i++) await rec.run((e) => pbf.run(e), 2);
        const per: Record<number, { best: number; median: number; p25: number; p75: number }> = {};
        for (const it of [1, 2, 4, 8]) {
          pbf.params.iterations = it;
          const passes = await rec.countPasses((e) => pbf.run(e));
          per[it] = await rec.slope((e) => pbf.run(e), passes, 1, 5, 18);
        }
        const perIter = (per[8]!.best - per[2]!.best) / 6;
        const fixed = per[2]!.best - 2 * perIter;
        pbf.params.iterations = 2;
        const labels = await rec.perLabel((e) => pbf.run(e), 20);
        const g = (k: string) => labels[k] ?? 0;
        const grid = g('pbf grid cells') + g('pbf grid scan') + g('pbf grid scatter') + g('pbf reorder');
        log(`| ${pbf.count} | 2 | ${f(per[2]!.best)} (med ${f(per[2]!.median)}) | ${f(perIter)} | ${f(fixed)} | ${f(g('pbf predict'), 3)} | ${f(grid, 3)} | ${f(g('pbf density') / 2, 3)} | ${f(g('pbf correct') / 2, 3)} | ${f(g('pbf viscosity'), 3)} |`);
        solverRows.push({ dimension: dim, particles: pbf.count, spans: per, perIteration: perIter, fixed, labels });
        pbf.destroy();
        await finish();
      }
      log();
    }
    // iterations curve at one count per dimension
    const itRows: unknown[] = [];
    json.iterations = itRows;
    for (const [dim, n] of [[2, 100000], [3, 100000]] as const) {
      log(`## ${dim}D, ${n} particles: ms per sub-step against iterations`);
      const spacing = dim === 3 ? 0.02 : 0.01;
      const side = Math.ceil(dim === 3 ? Math.cbrt(n) : Math.sqrt(n));
      const nx = side, nz = dim === 3 ? side : 1, ny = Math.ceil(n / (nx * nz));
      const pbf = createPbf({ device, dimension: dim, capacity: nx * ny * nz + 16, spacing, profiler: rec.profiler, timedPasses: 100000,
        bounds: { min: [0, 0, dim === 3 ? 0 : -1], max: [nx * spacing, ny * spacing, dim === 3 ? nz * spacing : 1] }, params: { gravity: [0, 0, 0] } });
      pbf.reset({ blocks: [{ min: [0, 0, 0], max: [nx * spacing, ny * spacing, nz * spacing], jitter: 0.1 }] });
      for (let i = 0; i < 6; i++) await rec.run((e) => pbf.run(e), 2);
      for (const it of [1, 2, 3, 4, 6, 8, 12]) {
        pbf.params.iterations = it;
        const passes = await rec.countPasses((e) => pbf.run(e));
        const r = await rec.slope((e) => pbf.run(e), passes, 1, 5, 18);
        log(`| ${it} | ${f(r.best)} (median ${f(r.median)}, p25-p75 ${f(r.p25)}-${f(r.p75)}) |`);
        itRows.push({ dimension: dim, particles: pbf.count, iterations: it, ...r });
      }
      for (const [name, over] of [['no viscosity', { viscosity: 0 }], ['vorticity 0.1', { vorticity: 0.1 }]] as const) {
        Object.assign(pbf.params, { iterations: 2, viscosity: 0.01, vorticity: 0 }, over);
        const passes = await rec.countPasses((e) => pbf.run(e));
        const r = await rec.slope((e) => pbf.run(e), passes, 1, 5, 18);
        log(`| 2, ${name} | ${f(r.best)} |`);
        itRows.push({ dimension: dim, particles: pbf.count, iterations: 2, variant: name, ...r });
      }
      pbf.destroy();
      log();
      await finish();
    }
    rec.destroy();
  });

  // --- C: how much compression, how calm: iterations against sub-steps ------------------------------------------
  it('C. quality against iterations and sub-steps', async () => {
    const device = await getTestDevice();
    const quality: unknown[] = [];
    json.quality = quality;
    const frames = async (pbf: ReturnType<typeof createPbf>, n: number, delta: number) => {
      for (let i = 0; i < n; i++) {
        const e = device.createCommandEncoder();
        pbf.update(e, delta);
        device.queue.submit([e.finish()]);
        if (i % 10 === 9) await device.queue.onSubmittedWorkDone();
      }
      await device.queue.onSubmittedWorkDone();
    };
    const cases = [
      { dim: 2 as const, spacing: 0.02, width: 0.4, height: 0.6, label: '2D column 0.4 x 0.6 m (30 rows of s = 0.02)' },
      { dim: 3 as const, spacing: 0.025, width: 0.3, height: 0.5, label: '3D column 0.3 x 0.3 x 0.5 m (20 rows of s = 0.025)' },
    ];
    for (const c of cases) {
      log(`## ${c.label}: compression of the settled column (volume height, 100 % = incompressible) and rms speed after 3 s, default parameters`);
      const subs = [1, 2, 3, 4, 6];
      log(['iterations \\ sub-steps per 1/60 s', ...subs.map((s) => `${s} (dt ${(1000 / 60 / s).toFixed(1)} ms)`)].join(' | '));
      for (const it of [1, 2, 3, 4, 6, 8]) {
        const row: string[] = [];
        for (const sub of subs) {
          const pbf = createPbf({ device, dimension: c.dim, capacity: 20000, spacing: c.spacing,
            bounds: { min: [0, 0, c.dim === 3 ? 0 : -1], max: [c.width, 2, c.dim === 3 ? c.width : 1] }, params: { iterations: it, substeps: sub } });
          pbf.reset({ blocks: [{ min: [0, 0, 0], max: [c.width, c.height, c.width], jitter: 0.05 }] });
          await frames(pbf, 180, 1 / 60);
          const pos = await pbf.system.read('position');
          const vel = await pbf.system.read('velocity');
          const n = pbf.count;
          let ys = 0, ke = 0, top = 0;
          for (let i = 0; i < n; i++) { ys += pos[i * 4 + 1]!; top = Math.max(top, pos[i * 4 + 1]!); ke += vel[i * 4]! ** 2 + vel[i * 4 + 1]! ** 2 + vel[i * 4 + 2]! ** 2; }
          const height = (2 * ys) / n;
          const compression = 1 - height / c.height;
          const exploded = top > 1.5 * c.height;
          row.push(exploded ? 'unstable' : `${(compression * 100).toFixed(1)} % (${Math.sqrt(ke / n).toFixed(2)})`);
          quality.push({ dimension: c.dim, iterations: it, substeps: sub, compression, rms: Math.sqrt(ke / n), unstable: exploded });
          pbf.destroy();
        }
        log(`${it} | ${row.join(' | ')}`);
      }
      log();
      await finish();
    }
    // Time step: a bigger fixed step with the same sub-step length.
    log('## 3D column, 1/30 s steps with 6 sub-steps (dt 5.6 ms) against 1/60 s with 3 (dt 5.6 ms): the same fluid?');
    for (const [timeStep, sub] of [[1 / 60, 3], [1 / 30, 6], [1 / 20, 9 > 8 ? 8 : 9]] as const) {
      const pbf = createPbf({ device, dimension: 3, capacity: 20000, spacing: 0.025,
        bounds: { min: [0, 0, 0], max: [0.3, 2, 0.3] }, params: { iterations: 2, substeps: sub, timeStep } });
      pbf.reset({ blocks: [{ min: [0, 0, 0], max: [0.3, 0.5, 0.3], jitter: 0.05 }] });
      await frames(pbf, 180, 1 / 60);
      const pos = await pbf.system.read('position');
      const n = pbf.count;
      let ys = 0;
      for (let i = 0; i < n; i++) ys += pos[i * 4 + 1]!;
      log(`| ${timeStep.toFixed(4)} s x ${sub} sub-steps | compression ${(100 * (1 - (2 * ys) / n / 0.5)).toFixed(1)} % |`);
      pbf.destroy();
    }
    await finish();
  });

  // --- B: the 3D effect, simulation against rendering --------------------------------------------------
  it('B. pbf-3d simulation versus rendering', async () => {
    const device = await timedDevice();
    const rec = recorder(device);
    const effect = await loadEffect('pbf-3d');
    const W = 1920, H = 1080;
    const target = device.createTexture({ size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    const rows: unknown[] = [];
    json.effect = rows;
    async function measure(overrides: Record<string, unknown>, warmFrames: number) {
      const values = { ...defaultParams(effect.params), ...overrides } as Record<string, unknown>;
      const canvas = Object.assign(new EventTarget(), { width: W, height: H, style: {}, getBoundingClientRect: () => ({ left: 0, top: 0, width: W, height: H }), setPointerCapture() {}, releasePointerCapture() {} }) as unknown as HTMLCanvasElement;
      const instance = await effect.create({ device, format: 'rgba8unorm', canvas, params: values as never, profiler: rec.profiler, hdr: false });
      instance.resize(W, H, 1);
      let frame = 0;
      const unit = (e: GPUCommandEncoder) => { instance.frame({ time: frame / 60, delta: 1 / 60, frame }, e, target.createView()); frame++; };
      // Warm up with real time passing, so the pool has collapsed and the splashes are on the way down.
      rec.state.mode = 'off';
      for (let i = 0; i < warmFrames; i++) { const e = device.createCommandEncoder(); unit(e); device.queue.submit([e.finish()]); if (i % 10 === 9) await device.queue.onSubmittedWorkDone(); }
      await device.queue.onSubmittedWorkDone();
      // Freeze the state: from here on the solver does nothing when speed = 0, which gives the rendering alone.
      const passes = await rec.countPasses(unit);
      const full = await rec.slope(unit, passes, 1, 4, 20);
      const labels = await rec.perLabel(unit, 12);
      values.speed = 0;
      const idlePasses = await rec.countPasses(unit);
      const idle = await rec.slope(unit, idlePasses, 1, 4, 20);
      const idleLabels = await rec.perLabel(unit, 12);
      instance.destroy();
      return { full, idle, labels, idleLabels };
    }
    log('## pbf-3d at 1920x1080: ms per frame (default settings: 2 iterations x 3 sub-steps, surface at half resolution, 3 filter iterations)');
    log('| particles | frame | rendering only (solver idle) | solver (difference) | solver median | frame median | fluid depth | fluid smooth | fluid thickness | fluid blur | fluid composite |');
    log('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |');
    for (const n of [10000, 30000, 60000, 120000, 250000, 500000]) {
      const r = await measure({ particles: n, autoRotate: false, ambient: false }, 150);
      const l = r.idleLabels;
      const solver = r.full.best - r.idle.best;
      log(`| ${n} | ${f(r.full.best)} | ${f(r.idle.best)} | ${f(solver)} | ${f(r.full.median - r.idle.median)} | ${f(r.full.median)} | ${f(l['fluid depth'] ?? 0)} | ${f(l['fluid smooth'] ?? 0)} | ${f(l['fluid thickness'] ?? 0)} | ${f(l['fluid blur'] ?? 0)} | ${f(l['fluid composite'] ?? 0)} |`);
      rows.push({ particles: n, frame: r.full, idle: r.idle, solver, labels: r.labels, idleLabels: r.idleLabels });
      await finish();
    }
    log();
    log('## pbf-3d, 60000 particles: iterations and sub-steps (frame ms and solver share)');
    for (const [it, sub] of [[1, 3], [2, 2], [2, 3], [2, 4], [3, 3], [4, 2], [4, 3], [6, 3]] as const) {
      const r = await measure({ particles: 60000, iterations: it, substeps: sub, autoRotate: false, ambient: false }, 150);
      log(`| ${it} x ${sub} | ${f(r.full.best)} | solver ${f(r.full.best - r.idle.best)} |`);
      rows.push({ particles: 60000, iterations: it, substeps: sub, frame: r.full, idle: r.idle });
      await finish();
    }
    log();
    log('## pbf-3d, 60000 particles: renderer quality (frame ms, solver idle)');
    for (const [name, over] of [
      ['default (scale 1/2, 3 iterations, thickness every particle)', {}],
      ['scale 1/4, 1 iteration, thickness every 2nd', { renderScale: '1/4', filterIterations: 1, thicknessStride: 2, filterRadius: 1.5 }],
      ['scale 1', { renderScale: '1' }],
      ['spheres view (billboards)', { view: 'spheres' }],
      ['tier B cut preset (15000 particles, 2 x 2, scale 1/4)', { ...((effect.presets?.['tier B cut'] ?? {}) as Record<string, unknown>), particles: 15000 }],
    ] as const) {
      const r = await measure({ particles: 60000, autoRotate: false, ambient: false, ...over }, 150);
      log(`| ${name} | idle ${f(r.idle.best)} | full ${f(r.full.best)} |`);
      rows.push({ particles: 60000, quality: name, frame: r.full, idle: r.idle });
      await finish();
    }
    rec.destroy();
    target.destroy();
  });
});
