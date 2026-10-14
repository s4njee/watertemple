import { describe, it, vi } from 'vitest';
import { createMpm, type MpmParams } from '../src/gpu/sim/mpm.ts';
import { getTestDevice } from '../tests/gpu.ts';

vi.setConfig({ testTimeout: 600_000 });

async function stats(s: ReturnType<typeof createMpm>) {
  const pos = await s.system.read('position');
  const vel = await s.system.read('velocity');
  const n = s.count;
  const ys: number[] = [];
  let ke = 0, maxv = 0, nan = 0;
  for (let i = 0; i < n; i++) {
    const y = pos[i*4+1]!;
    if (!Number.isFinite(y) || !Number.isFinite(vel[i*4]!)) nan++;
    ys.push(y);
    const sp = Math.hypot(vel[i*4]!, vel[i*4+1]!, vel[i*4+2]!);
    ke += 0.5*sp*sp; maxv = Math.max(maxv, sp);
  }
  ys.sort((a,b)=>a-b);
  return { top: ys[Math.floor(n*0.99)]!, median: ys[Math.floor(n/2)]!, low: ys[Math.floor(n*0.01)]!, ke: ke/n, maxv, nan };
}

describe('scratch', () => {
  it('rest', async () => {
    const device = await getTestDevice();
    const params: Partial<MpmParams> = JSON.parse(process.env.MPM_P ?? '{}');
    const s = createMpm({ device, grid: 32, cellSize: 0.05, capacity: 40000, params });
    s.reset({ bounds: { min: [0.4, 0.1, 0.4], max: [1.2, Number(process.env.TOP ?? 1.5), 1.2] }, blocks: [{ shape: 'box', min: [0.4, 0.1, 0.4], max: [1.2, 0.5, 1.2], jitter: Number(process.env.JIT ?? 0.15) }] });
    console.log('count', s.count, 'dt', s.effective.dt, 'ppc', s.params.particlesPerCell);
    let t = 0;
    for (let frame = 0; frame <= Number(process.env.FR ?? 180); frame++) {
      if (frame % Number(process.env.EV ?? 20) === 0) console.log(frame, JSON.stringify(await stats(s)));
      const e = device.createCommandEncoder();
      s.update(e, 1 / 60);
      device.queue.submit([e.finish()]);
    }
    const pos = await s.system.read('position');
    const a0 = await s.system.read('affine0');
    {
      // interior true density vs estimator: count per 3x3x3-cell block in the interior
      const h = 0.05; const counts = new Map<string, number>(); const rsum = new Map<string, number>();
      for (let i = 0; i < s.count; i++) {
        const x = pos[i*4]!, y = pos[i*4+1]!, z = pos[i*4+2]!;
        const key = `${Math.floor(x/h)},${Math.floor(y/h)},${Math.floor(z/h)}`;
        counts.set(key, (counts.get(key) ?? 0) + 1); rsum.set(key, (rsum.get(key) ?? 0) + a0[i*4+3]!);
      }
      let n = 0, c = 0, r = 0; const dens: number[] = [];
      for (const [key, cnt] of counts) {
        const [ix, iy, iz] = key.split(',').map(Number) as [number, number, number];
        if (ix < 10 || ix > 21 || iz < 10 || iz > 21 || iy < 4 || iy > 7) continue;
        n++; c += cnt; r += rsum.get(key)! / cnt; dens.push(cnt);
      }
      dens.sort((a,b)=>a-b);
      console.log(`interior cells ${n}: mean count/cell ${(c/n).toFixed(2)} (rest 8), mean r_est ${(r/n).toFixed(3)}, count p5 ${dens[Math.floor(n*0.05)]} p95 ${dens[Math.floor(n*0.95)]}`);
    }
    const bins = 20; const lo = 0.1, hi = 0.5;
    const cnt = new Array(bins).fill(0), rs = new Array(bins).fill(0);
    for (let i = 0; i < s.count; i++) {
      const b = Math.floor((pos[i*4+1]! - lo) / (hi - lo) * bins);
      if (b >= 0 && b < bins) { cnt[b]++; rs[b] += a0[i*4+3]!; }
    }
    const ppcBin = (0.8*0.8*((hi-lo)/bins)) / (0.05**3); // cells per bin
    console.log(cnt.map((c, b) => `${(lo + (b+0.5)*(hi-lo)/bins).toFixed(3)}: n/cell ${(c/ppcBin).toFixed(2)} r ${(rs[b]/Math.max(c,1)).toFixed(2)}`).join('\n'));
    s.destroy();
  });
});

describe('scratch2', () => {
  it('initial density estimate', async () => {
    const device = await getTestDevice();
    const s = createMpm({ device, grid: 32, cellSize: 0.05, capacity: 40000 });
    s.reset({ bounds: { min: [0.4, 0.1, 0.4], max: [1.2, Number(process.env.TOP ?? 1.5), 1.2] }, blocks: [{ shape: 'box', min: [0.4, 0.1, 0.4], max: [1.2, 0.5, 1.2], jitter: Number(process.env.JIT ?? 0.15) }] });
    const e = device.createCommandEncoder();
    s.step(e, { passes: ['clear', 'p2g1', 'p2g2'] });
    device.queue.submit([e.finish()]);
    const pos = await s.system.read('position');
    const a0 = await s.system.read('affine0');
    // r by depth below top (0.5) in cells
    const by: number[][] = Array.from({ length: 9 }, () => []);
    for (let i = 0; i < s.count; i++) {
      const x = pos[i*4]!, y = pos[i*4+1]!, z = pos[i*4+2]!;
      if (x < 0.6 || x > 1.0 || z < 0.6 || z > 1.0) continue;
      const d = Math.floor((0.5 - y) / 0.05);
      if (d >= 0 && d < 9) by[d]!.push(a0[i*4+3]!);
    }
    console.log(by.map((v, d) => `depth ${d}: n ${v.length} r ${(v.reduce((a,b)=>a+b,0)/Math.max(v.length,1)).toFixed(3)} min ${Math.min(...v).toFixed(3)} max ${Math.max(...v).toFixed(3)}`).join('\n'));
    s.destroy();
  });
});

describe('scratch3', () => {
  it('dam', async () => {
    const device = await getTestDevice();
    const params: Partial<MpmParams> = JSON.parse(process.env.MPM_P ?? '{}');
    const s = createMpm({ device, grid: [48, 32, 32], cellSize: 0.05, capacity: 120000, params });
    s.reset({ blocks: [{ shape: 'box', min: [0.1, 0.1, 0.1], max: [0.7, 1.1, 1.4] }] });
    console.log('count', s.count, 'dt', s.effective.dt, 'bounds', JSON.stringify(s.bounds));
    for (let frame = 0; frame <= Number(process.env.FR ?? 240); frame++) {
      if (frame % Number(process.env.EV ?? 30) === 0) {
        const st = await stats(s);
        const pos = await s.system.read('position');
        let maxx = 0; for (let i = 0; i < s.count; i++) maxx = Math.max(maxx, pos[i*4]!);
        console.log(frame, `ke ${st.ke.toFixed(3)} maxv ${st.maxv.toFixed(2)} top ${st.top.toFixed(3)} front ${maxx.toFixed(2)} nan ${st.nan}`);
      }
      const e = device.createCommandEncoder();
      s.update(e, 1 / 60);
      device.queue.submit([e.finish()]);
    }
    s.destroy();
  });
});

describe('scratch4', () => {
  it('perf', async () => {
    const device = await getTestDevice();
    const counts = (process.env.COUNTS ?? '100000').split(',').map(Number);
    const grids = (process.env.GRIDS ?? '64').split(',').map(Number);
    const wg = Number(process.env.WG ?? 64) as 64;
    for (const g of grids) for (const n of counts) {
      const s = createMpm({ device, grid: g, cellSize: 4 / g, capacity: n, workgroupSize: wg, gridWorkgroupSize: wg, params: JSON.parse(process.env.MPM_P ?? '{}') });
      // fill a block at 8 ppc up to n particles
      const side = Math.cbrt(n / 8) * s.cellSize; // cube side
      s.reset({ blocks: [{ shape: 'box', min: [0.3, 0.3, 0.3], max: [0.3 + side * 1.0, 0.3 + side * 1.0, 0.3 + side * 1.0] }] });
      const order = process.env.ORDER ?? 'scan';
      if (order !== 'scan') {
        const pos = await s.system.read('position');
        const idx = Array.from({ length: s.count }, (_, i) => i);
        const h = s.cellSize;
        const morton = (x: number, y: number, z: number) => { let m = 0; for (let b = 0; b < 8; b++) m |= (((x >> b) & 1) << (3*b)) | (((y >> b) & 1) << (3*b+1)) | (((z >> b) & 1) << (3*b+2)); return m; };
        const key = (i: number) => { const cx = Math.floor(pos[i*4]!/h), cy = Math.floor(pos[i*4+1]!/h), cz = Math.floor(pos[i*4+2]!/h); return order === 'morton' ? morton(cx, cy, cz) : (order === 'cell' ? (cz * 1000 + cy) * 1000 + cx : Math.random()); };
        const keys = idx.map(key); idx.sort((a, b) => keys[a]! - keys[b]!);
        const out = new Float32Array(s.count * 4); idx.forEach((src, dst) => out.set(pos.subarray(src*4, src*4+4), dst*4));
        s.system.set({ positions: out });
      }
      const run = async (passes: any, N: number) => {
        const e = device.createCommandEncoder();
        s.step(e, { count: N, passes });
        const t0 = performance.now();
        device.queue.submit([e.finish()]);
        await device.queue.onSubmittedWorkDone();
        return performance.now() - t0;
      };
      for (let i = 0; i < 3; i++) await run(undefined, 4);
      const slope = async (passes: any) => {
        const a: number[] = [], b: number[] = [];
        for (let i = 0; i < 9; i++) { a.push(await run(passes, 8)); b.push(await run(passes, 40)); }
        a.sort((x, y) => x - y); b.sort((x, y) => x - y);
        return (b[0]! - a[0]!) / 32;
      };
      const out: string[] = [];
      for (const p of ['clear', 'sort', 'p2g1', 'p2g2', 'grid', 'g2p'] as const) out.push(`${p} ${(await slope([p])).toFixed(3)}`);
      if (process.env.QUICK) { console.log(`grid ${g} n ${s.count} ALL ${(await slope(undefined)).toFixed(3)} ${process.env.MPM_P ?? ''}`); s.destroy(); continue; }
      out.push(`ALL ${(await slope(undefined)).toFixed(3)}`);
      console.log(`grid ${g} n ${s.count} wg ${wg}: ${out.join('  ')}`);
      s.destroy();
    }
  });
});
