import { describe, it, vi } from 'vitest';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createPbf, type Pbf, type PbfParams } from '../src/gpu/sim/pbf.ts';
import { getTestDevice } from '../tests/gpu.ts';

vi.setConfig({ testTimeout: 900_000 });
const env = (globalThis as { process?: { env: Record<string, string | undefined> } }).process!.env;

const OUT = join(tmpdir(), env.OUT ?? 'pbf2d-scratch-out.txt');
const lines: string[] = [];
const spec = 'node:fs';
async function flush() {
  const { writeFileSync } = (await import(/* @vite-ignore */ spec)) as {
    writeFileSync(p: string, t: string): void;
  };
  writeFileSync(OUT, lines.join('\n') + '\n');
}
const log = (t: string) => {
  lines.push(t);
};

let simTime = 0;
const RAMP = Number(env.RAMP ?? 0);
async function frames(device: GPUDevice, pbf: Pbf, n: number, delta = 1 / 60) {
  for (let i = 0; i < n; i++) {
    simTime += delta;
    if (RAMP > 0) {
      const q = Math.min(1, simTime / RAMP);
      pbf.params.gravity = [0, -9.81 * q * q * (3 - 2 * q), 0];
    }
    const e = device.createCommandEncoder();
    pbf.update(e, delta);
    device.queue.submit([e.finish()]);
    if (i % 8 === 7) await device.queue.onSubmittedWorkDone();
  }
  await device.queue.onSubmittedWorkDone();
}

describe('dam', () => {
  it('dam break stats', async () => {
    const device = await getTestDevice();
    const s = Number(env.S ?? 0.008);
    const extra = JSON.parse(env.P ?? '{}') as Partial<PbfParams>;
    const secs = Number(env.SECS ?? 8);
    const delta = Number(env.DELTA ?? 1 / 60);
    const pbf = createPbf({
      device,
      dimension: 2,
      capacity: Math.ceil(0.9 / s / s) + 500,
      spacing: s,
      bounds: { min: [0, 0, -1], max: [2, 1.2, 1] },
      params: extra,
    });
    pbf.reset({
      blocks: [{ min: [0, 0, 0], max: [Number(env.BW ?? 0.55), Number(env.BH ?? 0.85), 0] }],
    });
    const n = pbf.count;
    const area = n * s * s;
    log(`s=${s} n=${n} area=${area.toFixed(4)} flat height=${(area / 2).toFixed(4)}`);
    const steps = Math.round(0.25 / delta);
    for (let k = 0; k < secs * 4; k++) {
      await frames(device, pbf, steps, delta);
      const pos = await pbf.system.read('position');
      const vel = await pbf.system.read('velocity');
      const st = await pbf.readState();
      let top = 0,
        high = 0,
        ke = 0,
        dsum = 0,
        dmax = 0,
        ysum = 0;
      for (let i = 0; i < n; i++) {
        const y = pos[i * 4 + 1]!;
        top = Math.max(top, y);
        ysum += y;
        if (y > 0.6) high++;
        ke += vel[i * 4]! ** 2 + vel[i * 4 + 1]! ** 2;
        dsum += st.density[i]!;
        dmax = Math.max(dmax, st.density[i]!);
      }
      log(
        `t=${((k + 1) * 0.25).toFixed(2)} top=${top.toFixed(3)} n>0.6=${high} meanY=${(ysum / n).toFixed(3)} rms=${Math.sqrt(ke / n).toFixed(3)} rho mean=${(dsum / n).toFixed(3)} max=${dmax.toFixed(3)}`,
      );
    }
    pbf.destroy();
    await flush();
  });
});

// dam break scan: for a list of (spacing, iterations, substeps) cases print the worst p99 density and the end state.
describe('scan', () => {
  it('scan', async () => {
    const device = await getTestDevice();
    const cases = JSON.parse(env.CASES ?? '[]') as [number, number, number][];
    const bw = Number(env.BW ?? 0.6),
      bh = Number(env.BH ?? 0.5);
    const secs = Number(env.SECS ?? 4);
    for (const [s, iterations, substeps] of cases) {
      const pbf = createPbf({
        device,
        dimension: 2,
        capacity: Math.ceil(0.9 / s / s) + 500,
        spacing: s,
        bounds: { min: [0, 0, -1], max: [2, 1.2, 1] },
        params: { iterations, substeps, ...(JSON.parse(env.P ?? '{}') as Partial<PbfParams>) },
      });
      pbf.reset({ blocks: [{ min: [0, 0, 0], max: [bw, bh, 0] }] });
      const n = pbf.count;
      let worstP99 = 0,
        worstMax = 0,
        topMax = 0;
      let last = '';
      for (let k = 0; k < secs * 4; k++) {
        await frames(device, pbf, 15);
        const pos = await pbf.system.read('position');
        const vel = await pbf.system.read('velocity');
        const st = await pbf.readState();
        const sorted = Float32Array.from(st.density).sort();
        const p99 = sorted[Math.floor(n * 0.99)]!;
        let ke = 0,
          top = 0,
          ymean = 0;
        for (let i = 0; i < n; i++) {
          ke += vel[i * 4]! ** 2 + vel[i * 4 + 1]! ** 2;
          top = Math.max(top, pos[i * 4 + 1]!);
          ymean += pos[i * 4 + 1]!;
        }
        if (k < 6) {
          worstP99 = Math.max(worstP99, p99);
          worstMax = Math.max(worstMax, sorted[n - 1]!);
          topMax = Math.max(topMax, top);
        }
        let mean = 0;
        for (let i = 0; i < n; i++) mean += st.density[i]!;
        last = `rho mean=${(mean / n).toFixed(3)} p99=${p99.toFixed(3)} rms=${Math.sqrt(ke / n).toFixed(3)} meanH=${((2 * ymean) / n / ((n * s * s) / 2)).toFixed(3)} (rel to flat)`;
      }
      log(
        `s=${s} it=${iterations} sub=${substeps} n=${n}: first1.5s worst p99=${worstP99.toFixed(2)} max=${worstMax.toFixed(2)} top=${topMax.toFixed(2)} | end ${last}`,
      );
      pbf.destroy();
    }
    await flush();
  });
});

describe('jet', () => {
  it('jet width', async () => {
    const device = await getTestDevice();
    const s = Number(env.S ?? 0.008);
    const extra = JSON.parse(env.P ?? '{}') as Partial<PbfParams>;
    const pbf = createPbf({
      device,
      dimension: 2,
      capacity: 30000,
      spacing: s,
      bounds: { min: [0, 0, -1], max: [2, 1.2, 1] },
      params: { iterations: 3, substeps: 3, ...extra },
    });
    pbf.reset({
      emitter: {
        position: [0.45, 1.15, 0],
        velocity: [0, Number(env.VY ?? -1.3), 0],
        radius: Number(env.R ?? 0.035),
      },
      blocks: [],
    });
    for (let k = 0; k < 6; k++) {
      await frames(device, pbf, 15);
      const pos = await pbf.system.read('position');
      const n = pbf.count;
      const bins = [0.9, 0.7, 0.5, 0.3, 0.1];
      const row: string[] = [];
      for (const y of bins) {
        let lo = 9,
          hi = -9,
          c = 0;
        for (let i = 0; i < n; i++) {
          const py = pos[i * 4 + 1]!;
          if (Math.abs(py - y) < 0.1) {
            lo = Math.min(lo, pos[i * 4]!);
            hi = Math.max(hi, pos[i * 4]!);
            c++;
          }
        }
        row.push(c ? `y${y}: w=${(hi - lo).toFixed(3)} n=${c}` : `y${y}: -`);
      }
      log(`t=${(k + 1) * 0.25} n=${n} ${row.join(' | ')}`);
    }
    await flush();
  });
});

describe('wave', () => {
  it('wave maker', async () => {
    const device = await getTestDevice();
    const s = Number(env.S ?? 0.008);
    const A = Number(env.A ?? 0.18),
      F = Number(env.F ?? 0.75),
      D = Number(env.D ?? 0.42);
    const extra = JSON.parse(env.P ?? '{}') as Partial<PbfParams>;
    const secs = Number(env.SECS ?? 10);
    const pbf = createPbf({
      device,
      dimension: 2,
      capacity: Math.ceil((1.1 * 2 * D) / s / s) + 500,
      spacing: s,
      bounds: { min: [0, 0, -1], max: [2, 1.2, 1] },
      params: {
        iterations: Number(env.IT ?? 3),
        substeps: Number(env.SUB ?? 3),
        maxWallSpeed: 2,
        artificialPressure: 0.02,
        ...extra,
      },
    });
    pbf.reset({ blocks: [{ min: [0, 0, 0], max: [2, D, 0] }] });
    const n = pbf.count;
    log(`s=${s} n=${n} A=${A} F=${F} D=${D}`);
    let t = 0;
    for (let k = 0; k < secs * 4; k++) {
      for (let i = 0; i < 15; i++) {
        t += 1 / 60;
        const ramp = Math.min(1, t / 2);
        const wall = A * ramp * 0.5 * (1 - Math.cos(2 * Math.PI * F * t));
        pbf.setBounds([wall, 0, -1], [2, 1.2, 1]);
        await frames(device, pbf, 1);
      }
      const pos = await pbf.system.read('position');
      const vel = await pbf.system.read('velocity');
      const st = await pbf.readState();
      let top = 0,
        high = 0,
        ke = 0,
        d = 0,
        p = 0,
        minx = 9;
      for (let i = 0; i < n; i++) {
        top = Math.max(top, pos[i * 4 + 1]!);
        if (pos[i * 4 + 1]! > D + 0.15) high++;
        ke += vel[i * 4]! ** 2 + vel[i * 4 + 1]! ** 2;
        d += st.density[i]!;
        minx = Math.min(minx, pos[i * 4]!);
      }
      log(
        `t=${t.toFixed(2)} wall=${pbf.bounds.min[0]!.toFixed(3)} top=${top.toFixed(3)} high=${high} rms=${Math.sqrt(ke / n).toFixed(3)} rho=${(d / n).toFixed(3)} minx=${minx.toFixed(3)}`,
      );
    }
    await flush();
  });
});

describe('rest', () => {
  it('rest column', async () => {
    const device = await getTestDevice();
    const rowsList = JSON.parse(env.ROWS ?? '[30]') as number[];
    const cases = JSON.parse(env.CASES ?? '[[3,3]]') as [number, number][];
    const s = Number(env.S ?? 0.008);
    const W = Number(env.W ?? 0.4);
    const extra = JSON.parse(env.P ?? '{}') as Partial<PbfParams>;
    for (const rows of rowsList)
      for (const [iterations, substeps] of cases) {
        const pbf = createPbf({
          device,
          dimension: 2,
          capacity: Math.ceil(W / s) * (rows + 4) + 100,
          spacing: s,
          bounds: { min: [0, 0, -1], max: [W, 1.5, 1] },
          params: { iterations, substeps, ...extra },
        });
        pbf.reset({ blocks: [{ min: [0, 0, 0], max: [W, rows * s, 0] }] });
        const n = pbf.count;
        const out: string[] = [];
        let top0 = 0;
        for (let k = 0; k < 6; k++) {
          await frames(device, pbf, k === 0 ? 15 : 30);
          const pos = await pbf.system.read('position');
          const vel = await pbf.system.read('velocity');
          const st = await pbf.readState();
          let top = 0,
            ke = 0,
            d = 0,
            dmax = 0;
          for (let i = 0; i < n; i++) {
            top = Math.max(top, pos[i * 4 + 1]!);
            ke += vel[i * 4]! ** 2 + vel[i * 4 + 1]! ** 2;
            d += st.density[i]!;
            dmax = Math.max(dmax, st.density[i]!);
          }
          if (k === 0) top0 = top;
          out.push(`${top.toFixed(3)}/${Math.sqrt(ke / n).toFixed(2)}/${(d / n).toFixed(2)}`);
        }
        log(
          `rows=${rows} it=${iterations} sub=${substeps} s=${s} (top/rms/rho: t=0.25 ... 2.75) ${out.join('  ')}   column=${(rows * s).toFixed(3)} 1st=${top0.toFixed(3)}`,
        );
        pbf.destroy();
      }
    await flush();
  });
});

// ---- Cost study: sim and draw timed separately by the repeat-N slope -------------------------------------------------
import { createBillboardRenderer, orthographicCamera } from '../src/gpu/particles/index.ts';

async function timedDevice() {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter?.features.has('timestamp-query')) throw new Error('no timestamp-query');
  return adapter.requestDevice({ requiredFeatures: ['timestamp-query'] });
}

function makeTimer(device: GPUDevice) {
  const querySet = device.createQuerySet({ type: 'timestamp', count: 8 });
  const resolve = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const read = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let mode: 'off' | 'count' | 'span' = 'off';
  let expected = 0;
  let seen = 0;
  const profiler = {
    pass() {
      const index = seen++;
      if (mode !== 'span') return undefined;
      if (expected === 1) return { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };
      if (index === 0) return { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 2 };
      if (index === expected - 1)
        return { querySet, beginningOfPassWriteIndex: 3, endOfPassWriteIndex: 1 };
      return undefined;
    },
  };
  async function time(record: (e: GPUCommandEncoder) => void, reps: number, perRep: number) {
    mode = 'span';
    seen = 0;
    expected = perRep * reps;
    const e = device.createCommandEncoder();
    for (let i = 0; i < reps; i++) record(e);
    e.resolveQuerySet(querySet, 0, 4, resolve, 0);
    e.copyBufferToBuffer(resolve, 0, read, 0, 32);
    device.queue.submit([e.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const stamps = new BigUint64Array(read.getMappedRange().slice(0, 32));
    read.unmap();
    return Number(stamps[1]! - stamps[0]!) / 1e6;
  }
  async function slope(record: (e: GPUCommandEncoder) => void, lo = 1, hi = 4, rounds = 14) {
    mode = 'count';
    seen = 0;
    const e = device.createCommandEncoder();
    record(e);
    device.queue.submit([e.finish()]);
    const perRep = seen;
    for (let i = 0; i < 3; i++) await time(record, 1, perRep);
    const small: number[] = [],
      large: number[] = [];
    for (let r = 0; r < rounds; r++) {
      small.push(await time(record, lo, perRep));
      large.push(await time(record, hi, perRep));
    }
    small.sort((a, b) => a - b);
    large.sort((a, b) => a - b);
    const mid = Math.floor(rounds / 2);
    return {
      best: (large[0]! - small[0]!) / (hi - lo),
      median: (large[mid]! - small[mid]!) / (hi - lo),
      perRep,
    };
  }
  return { profiler, slope };
}

const studyOut = env.STUDY;
describe.skipIf(!studyOut)('study', () => {
  vi.setConfig({ testTimeout: 3_600_000 });
  const f = (x: number) => x.toFixed(2);

  async function pool(
    device: GPUDevice,
    profiler: unknown,
    n: number,
    its: number,
    subs: number,
    extra: Partial<PbfParams> = {},
  ) {
    const s = Math.sqrt(0.8 / n);
    const pbf = createPbf({
      device,
      dimension: 2,
      capacity: Math.ceil(n * 1.1),
      spacing: s,
      bounds: { min: [0, 0, -1], max: [2, 1.2, 1] },
      timedPasses: 1_000_000,
      profiler: profiler as never,
      params: { iterations: its, substeps: subs, gravity: [0, 0, 0], ...extra },
    });
    pbf.reset({ blocks: [{ min: [0, 0, 0], max: [2, 0.4, 0] }] });
    return pbf;
  }

  it('sim cost', async () => {
    const device = await timedDevice();
    const t = makeTimer(device);
    log(
      'PBF 2D SIM COST: ms per 1/60 s step (one update), quietest-run slope [median-run slope]. Pool 2 x 0.4 m at rest lattice, gravity 0 (cost does not depend on it), 24 neighbours per particle.',
    );
    log(
      'N      s(mm)  rows | it x sub: ' +
        [
          [2, 3],
          [4, 3],
          [4, 4],
          [6, 6],
          [8, 6],
          [1, 1],
          [2, 1],
          [4, 1],
          [8, 1],
        ]
          .map(([a, b]) => `${a}x${b}`)
          .join('   '),
    );
    const counts = JSON.parse(env.NS ?? '[5000,10000,25000,50000,100000,250000]') as number[];
    for (const n of counts) {
      const row: string[] = [];
      for (const [it, sub] of [
        [2, 3],
        [4, 3],
        [4, 4],
        [6, 6],
        [8, 6],
        [1, 1],
        [2, 1],
        [4, 1],
        [8, 1],
      ] as const) {
        const pbf = await pool(device, t.profiler, n, it, sub);
        const r = await t.slope(
          (e) => {
            pbf.update(e, 1 / 60);
          },
          1,
          n > 100000 ? 2 : 4,
          Number(env.ROUNDS ?? 12),
        );
        row.push(`${f(r.best)} [${f(r.median)}]`);
        pbf.destroy();
      }
      const s = Math.sqrt(0.8 / n);
      log(`${n} ${(s * 1000).toFixed(1)} ${Math.round(0.4 / s)} | ${row.join('   ')}`);
      await flush2();
    }
  });

  it('sim stages', async () => {
    const device = await timedDevice();
    const t = makeTimer(device);
    log(
      'STAGES of one sub-step at N, ms (quietest slope). predict | grid | one iteration (density + correct) | viscosity',
    );
    for (const n of JSON.parse(env.NS2 ?? '[10000,50000,250000]') as number[]) {
      const pbf = await pool(device, t.profiler, n, 1, 1);
      const out: string[] = [];
      for (const stages of [
        ['predict'],
        ['predict', 'grid'],
        ['predict', 'grid', 'density', 'correct'],
        ['predict', 'grid', 'density', 'correct', 'viscosity'],
      ] as const) {
        const r = await t.slope(
          (e) => {
            pbf.run(e, { stages: stages as never, iterations: 1 });
          },
          1,
          4,
          Number(env.ROUNDS ?? 12),
        );
        out.push(`${stages.join('+')}: ${f(r.best)}`);
      }
      log(`N=${n}: ${out.join(' | ')}`);
      pbf.destroy();
      await flush2();
    }
  });

  it('draw cost', async () => {
    const device = await timedDevice();
    const t = makeTimer(device);
    const W = Number(env.DW ?? 1920),
      H = Number(env.DH ?? 1080);
    const target = device.createTexture({
      size: [W, H],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    log(
      `DRAW COST at ${W}x${H}: ms per frame, billboards over a 2 x 0.4 m pool in a 2 x 1.2 tank (quietest slope [median])`,
    );
    for (const n of JSON.parse(env.NS3 ?? '[5000,10000,25000,50000,100000,250000]') as number[]) {
      const pbf = await pool(device, undefined, n, 1, 1);
      const row: string[] = [];
      for (const mode of ['opaque', 'alpha', 'additive'] as const) {
        const renderer = createBillboardRenderer({
          device,
          format: 'rgba8unorm',
          profiler: t.profiler,
          style: { mode, softness: 0.5, shading: 0.5, intensity: 0.92, minPixels: 0.7 },
        });
        renderer.resize(W, H);
        const cam = orthographicCamera({ center: [1, 0.6], halfHeight: 0.6 * 1.08, aspect: W / H });
        const r = await t.slope(
          (e) => {
            renderer.draw(e, pbf.particles, cam, target.createView(), {
              width: W,
              height: H,
              clear: null,
            });
          },
          1,
          6,
          14,
        );
        row.push(`${mode} ${f(r.best)} [${f(r.median)}]`);
        renderer.destroy();
      }
      log(`N=${n} (s=${(Math.sqrt(0.8 / n) * 1000).toFixed(1)} mm): ${row.join(' | ')}`);
      pbf.destroy();
      await flush2();
    }
  });
});

async function flush2() {
  const { appendFileSync, writeFileSync } = (await import(/* @vite-ignore */ spec)) as {
    appendFileSync(p: string, t: string): void;
    writeFileSync(p: string, t: string): void;
  };
  void appendFileSync;
  writeFileSync(OUT, lines.join('\n') + '\n');
}

// ---- Outside view of the solver for the notes: how incompressible, how fast it settles, 1/30 s steps ------------------
describe('compare', () => {
  it('pbf numbers', async () => {
    const device = await getTestDevice();
    const spacings = JSON.parse(env.SPACINGS ?? '[0.012,0.008,0.006,0.004]') as number[];
    const stat = async (pbf: Pbf, n: number) => {
      const pos = await pbf.system.read('position');
      const vel = await pbf.system.read('velocity');
      const st = await pbf.readState();
      let ke = 0,
        ym = 0,
        ymax = 0;
      for (let i = 0; i < n; i++) {
        ke += vel[i * 4]! ** 2 + vel[i * 4 + 1]! ** 2;
        ym += pos[i * 4 + 1]!;
        ymax = Math.max(ymax, pos[i * 4 + 1]!);
      }
      const sorted = Float32Array.from(st.density).sort();
      let mean = 0;
      for (let i = 0; i < n; i++) mean += st.density[i]!;
      return {
        rms: Math.sqrt(ke / n),
        ymean: ym / n,
        ymax,
        rho: mean / n,
        p99: sorted[Math.floor(n * 0.99)]!,
        p01: sorted[Math.floor(n * 0.01)]!,
      };
    };
    const autoIts = (s: number) => Math.max(4, 2 * Math.round(0.013125 / s));
    const autoSub = (s: number) => Math.min(8, Math.max(2, Math.round(0.025 / s)));
    for (const s of spacings) {
      const its = autoIts(s),
        subs = autoSub(s);
      // (a) a pool at rest: 2 x 0.42, narrower here (0.6 wide) to keep the run short
      {
        const rows = Math.round(0.42 / s);
        const W = 0.6;
        const pbf = createPbf({
          device,
          dimension: 2,
          capacity: Math.ceil(W / s) * (rows + 4),
          spacing: s,
          bounds: { min: [0, 0, -1], max: [W, 1.2, 1] },
          params: { iterations: its, substeps: subs, artificialPressure: 0.02 },
        });
        pbf.reset({ blocks: [{ min: [0, 0, 0], max: [W, rows * s, 0] }] });
        const n = pbf.count;
        await frames(device, pbf, 180);
        const r = await stat(pbf, n);
        // an incompressible column's mean height: rows*s/2 (centres), mean y of the lattice = rows * s / 2
        log(
          `POOL s=${s * 1000}mm its=${its} sub=${subs} rows=${rows} n=${n}: mean y / incompressible = ${(r.ymean / ((rows * s) / 2)).toFixed(4)}  top=${r.ymax.toFixed(3)} (lattice top ${(rows * s - s / 2).toFixed(3)})  rho mean=${r.rho.toFixed(3)} p99=${r.p99.toFixed(3)} p01=${r.p01.toFixed(3)} rms=${r.rms.toFixed(4)}`,
        );
        pbf.destroy();
      }
      // (b) dam break: peak compression, time to settle
      {
        const W = 0.6,
          Hh = 0.55;
        const pbf = createPbf({
          device,
          dimension: 2,
          capacity: Math.ceil(0.4 / s / s),
          spacing: s,
          bounds: { min: [0, 0, -1], max: [2, 1.2, 1] },
          params: { iterations: its, substeps: subs, artificialPressure: 0.02 },
        });
        pbf.reset({ blocks: [{ min: [0, 0, 0], max: [W, Hh, 0] }] });
        const n = pbf.count;
        let worstP99 = 0,
          topMax = 0,
          t005: number | null = null,
          t01: number | null = null,
          t02: number | null = null;
        const rmsAt: string[] = [];
        for (let k = 1; k <= 120; k++) {
          // 30 s in 0.25 s samples
          await frames(device, pbf, 15);
          const r = await stat(pbf, n);
          if (k <= 8) {
            worstP99 = Math.max(worstP99, r.p99);
            topMax = Math.max(topMax, r.ymax);
          }
          if (k > 8) {
            if (t02 === null && r.rms < 0.2) t02 = k * 0.25;
            if (t01 === null && r.rms < 0.1) t01 = k * 0.25;
            if (t005 === null && r.rms < 0.05) t005 = k * 0.25;
          }
          if (k === 8 || k === 20 || k === 40 || k === 80 || k === 120)
            rmsAt.push(`t=${k * 0.25}:${r.rms.toFixed(3)}`);
        }
        const fin = await stat(pbf, n);
        log(
          `DAM s=${s * 1000}mm its=${its} sub=${subs} n=${n}: first 2 s worst p99 rho=${worstP99.toFixed(2)} topmax=${topMax.toFixed(2)}; t(rms<0.2)=${t02} t(<0.1)=${t01} t(<0.05)=${t005} s; rms ${rmsAt.join(' ')}; end mean rho=${fin.rho.toFixed(3)} mean y=${fin.ymean.toFixed(4)} (flat ${((n * s * s) / 2 / 2).toFixed(4)})`,
        );
        pbf.destroy();
      }
      // (c) 1/30 s steps: same sub-steps per step (dt per sub-step doubles), and double the sub-steps (same dt per sub-step)
      for (const [label, ts, sb] of [
        ['1/30 s, same sub-steps per step', 1 / 30, subs],
        ['1/30 s, doubled sub-steps', 1 / 30, Math.min(8, subs * 2)],
      ] as const) {
        const rows = Math.round(0.42 / s),
          W = 0.6;
        const pbf = createPbf({
          device,
          dimension: 2,
          capacity: Math.ceil(W / s) * (rows + 4),
          spacing: s,
          bounds: { min: [0, 0, -1], max: [W, 1.2, 1] },
          params: {
            iterations: its,
            substeps: sb,
            timeStep: ts,
            artificialPressure: 0.02,
            maxSteps: 4,
          },
        });
        pbf.reset({ blocks: [{ min: [0, 0, 0], max: [W, rows * s, 0] }] });
        const n = pbf.count;
        await frames(device, pbf, 90, 1 / 30);
        const r = await stat(pbf, n);
        log(
          `POOL at ${label}: s=${s * 1000}mm its=${its} sub=${sb}: mean y/incompressible=${(r.ymean / ((rows * s) / 2)).toFixed(4)} top=${r.ymax.toFixed(3)} rho mean=${r.rho.toFixed(3)} p99=${r.p99.toFixed(3)} rms=${r.rms.toFixed(4)}`,
        );
        pbf.destroy();
      }
      await flush2();
    }
  });
});

describe.skipIf(!studyOut)('study2', () => {
  vi.setConfig({ testTimeout: 3_600_000 });
  const f = (x: number) => x.toFixed(2);

  it('effect configs', async () => {
    const device = await timedDevice();
    const t = makeTimer(device);
    log(
      "SIM ONLY at the effect's auto settings: ms per 1/60 s step, pool 2 x 0.42 m (the wave-maker scene), quietest slope [median]",
    );
    const cfgs = JSON.parse(
      env.CFGS ?? '[[0.012,4,2],[0.008,4,3],[0.006,4,4],[0.004,6,6],[0.003,8,8],[0.002,14,8]]',
    ) as [number, number, number][];
    for (const [s, it, sub] of cfgs) {
      const pbf = createPbf({
        device,
        dimension: 2,
        capacity: Math.ceil((0.84 / s / s) * 1.2),
        spacing: s,
        bounds: { min: [0, 0, -1], max: [2, 1.2, 1] },
        timedPasses: 1_000_000,
        profiler: t.profiler as never,
        params: { iterations: it, substeps: sub, gravity: [0, 0, 0], artificialPressure: 0.02 },
      });
      pbf.reset({ blocks: [{ min: [0, 0, 0], max: [2, 0.42, 0] }] });
      const r = await t.slope(
        (e) => {
          pbf.update(e, 1 / 60);
        },
        1,
        pbf.count > 100000 ? 2 : 4,
        Number(env.ROUNDS ?? 30),
      );
      log(
        `s=${s * 1000} mm N=${pbf.count} ${it}x${sub}: ${f(r.best)} [${f(r.median)}] ms, ${r.perRep} passes`,
      );
      pbf.destroy();
      await flush2();
    }
  });

  it('effect frames', async () => {
    const device = await timedDevice();
    const t = makeTimer(device);
    const { loadEffect } = await import('../src/effects/registry.ts');
    const { defaultParams } = await import('../src/effects/types.ts');
    const effect = await loadEffect('pbf-2d');
    const W = 1920,
      H = 1080;
    log(
      `WHOLE EFFECT FRAME at ${W}x${H} (sim + backdrop + draw), scene wave-maker, view water, quietest slope [median]`,
    );
    for (const resolution of ['12 mm', '8 mm', '6 mm', '4 mm', '3 mm', '2 mm']) {
      for (const view of ['water', 'neighbours']) {
        const canvas = Object.assign(new EventTarget(), {
          width: W,
          height: H,
          style: {},
          getBoundingClientRect: () => ({ left: 0, top: 0, width: W, height: H }),
          setPointerCapture() {},
          releasePointerCapture() {},
        }) as unknown as HTMLCanvasElement;
        const target = device.createTexture({
          size: [W, H],
          format: 'rgba8unorm',
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        const params = {
          ...defaultParams(effect.params),
          scene: 'wave-maker',
          resolution,
          view,
          waveAmplitude: 0,
        } as never;
        const instance = await effect.create({
          device,
          format: 'rgba8unorm',
          canvas,
          params,
          profiler: t.profiler as never,
          hdr: false,
        });
        instance.resize(W, H, 1);
        let frame = 0;
        const rec = (e: GPUCommandEncoder) => {
          instance.frame({ time: frame / 60, delta: 1 / 60, frame }, e, target.createView());
          frame++;
        };
        for (let i = 0; i < 90; i++) {
          const e = device.createCommandEncoder();
          rec(e);
          device.queue.submit([e.finish()]);
        }
        await device.queue.onSubmittedWorkDone();
        const r = await t.slope(rec, 1, 4, Number(env.ROUNDS ?? 30));
        log(
          `${resolution} view=${view}: ${f(r.best)} [${f(r.median)}] ms/frame (${r.perRep} timed passes)`,
        );
        instance.destroy();
        target.destroy();
        await flush2();
      }
    }
  });
});
