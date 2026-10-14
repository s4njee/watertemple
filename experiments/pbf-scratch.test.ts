import { describe, it, vi } from 'vitest';
import { createPbf, restState, poly6, type Pbf, type PbfParams } from '../src/gpu/sim/pbf.ts';
import { expectValid, getTestDevice } from '../tests/gpu.ts';

vi.setConfig({ testTimeout: 600_000 });

async function frames(device: GPUDevice, pbf: Pbf, n: number, delta = 1 / 60) {
  for (let i = 0; i < n; i++) {
    const e = device.createCommandEncoder();
    pbf.update(e, delta);
    device.queue.submit([e.finish()]);
    if (i % 8 === 7) await device.queue.onSubmittedWorkDone();
  }
  await device.queue.onSubmittedWorkDone();
}

function density(dim: 2 | 3, pos: Float32Array, n: number, s: number, h: number, n0: number, lo: number[], hi: number[]) {
  // mean rho/n0 of particles farther than h from walls & the highest 1.2h: interior
  const rho = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) {
      const dx = pos[i*4]!-pos[j*4]!, dy = pos[i*4+1]!-pos[j*4+1]!, dz = dim===3?pos[i*4+2]!-pos[j*4+2]!:0;
      const r = Math.hypot(dx,dy,dz);
      if (r < h) sum += poly6(dim, h, r);
    }
    rho[i] = sum / n0;
  }
  return rho;
}

describe('columns', () => {
  it('compression vs iterations', async () => {
    const device = await getTestDevice();
    const dim = Number(process.env.PBF_DIM ?? 2) as 2 | 3;
    const s = Number(process.env.PBF_S ?? 0.02);
    const W = Number(process.env.PBF_W ?? 0.4);
    const cases: [number, number, number][] = JSON.parse(process.env.PBF_CASES ?? '[[4,2,1]]');
    const extra = JSON.parse(process.env.PBF_P ?? '{}') as Partial<PbfParams>;
    const H0 = Number(process.env.PBF_H ?? 1.0);
    for (const [iterations, substeps, relaxation] of cases) {
      const pbf = createPbf({ device, dimension: dim, capacity: 40000, spacing: s,
        bounds: { min: [0,0,dim===3?0:-1], max: [W,2,dim===3?W:1] }, params: { iterations, substeps, relaxation: relaxation ?? 0.2, ...extra } });
      pbf.reset({ blocks: [{ min: [0,0,0], max: [W, H0, W], jitter: 0.05 }] });
      const n = pbf.count;
      const out: string[] = [];
      for (let k = 0; k < 4; k++) {
        await frames(device, pbf, 45);
        const pos = await pbf.system.read('position');
        const vel = await pbf.system.read('velocity');
        let ysum = 0, ke = 0, ytop = 0;
        for (let i = 0; i < n; i++) { ysum += pos[i*4+1]!; ytop=Math.max(ytop,pos[i*4+1]!); ke += vel[i*4]!**2+vel[i*4+1]!**2; }
        out.push(`t${((k+1)*0.75).toFixed(2)}: H=${(2*ysum/n/H0).toFixed(3)} top=${ytop.toFixed(3)} v=${Math.sqrt(ke/n).toFixed(3)}`);
      }
      {
        const pos = await pbf.system.read('position');
        const vel = await pbf.system.read('velocity');
        const bins = new Array(10).fill(0), cnt = new Array(10).fill(0), bvx = new Array(10).fill(0), bvy=new Array(10).fill(0);
        for (let i = 0; i < n; i++) { const b = Math.min(9, Math.floor(pos[i*4+1]!/H0*10)); bins[b]+= vel[i*4]!**2+vel[i*4+1]!**2; cnt[b]++; bvx[b]+=vel[i*4]!; bvy[b]+=vel[i*4+1]!; }
        console.log('rms by height bin: ' + bins.map((v,i)=>Math.sqrt(v/Math.max(1,cnt[i])).toFixed(2)).join(' ') + ' | mean vy ' + bvy.map((v,i)=>(v/Math.max(1,cnt[i])).toFixed(2)).join(' '));
      }
      console.log(`it=${iterations} sub=${substeps} eps=${relaxation} n=${n} ${out.join(' | ')}`);
      pbf.destroy();
    }
  });
});

describe('early', () => {
  it('early dynamics', async () => {
    const device = await getTestDevice();
    const s = 0.02;
    const [iterations, substeps] = JSON.parse(process.env.PBF_CASES ?? '[[4,1]]')[0] as number[];
    const extra = JSON.parse(process.env.PBF_P ?? '{}') as Partial<PbfParams>;
    const pbf = createPbf({ device, dimension: 2, capacity: 4000, spacing: s,
      bounds: { min: [0,0,-1], max: [0.4,2,1] }, params: { iterations, substeps, ...extra } });
    pbf.reset({ blocks: [{ min: [0,0,0], max: [0.4, 1, 0], jitter: 0.05 }] });
    const n = pbf.count;
    for (let k = 0; k < 24; k++) {
      await frames(device, pbf, 3);
      const pos = await pbf.system.read('position');
      const vel = await pbf.system.read('velocity');
      let ysum = 0, ke = 0, ytop = 0, vy = 0, ybot=1e9;
      for (let i = 0; i < n; i++) { ysum += pos[i*4+1]!; ytop=Math.max(ytop,pos[i*4+1]!); ybot=Math.min(ybot,pos[i*4+1]!); ke += vel[i*4]!**2+vel[i*4+1]!**2; vy+=vel[i*4+1]!; }
      console.log(`f${(k+1)*3}: H=${(2*ysum/n).toFixed(3)} top=${ytop.toFixed(3)} bot=${ybot.toFixed(3)} rms v=${Math.sqrt(ke/n).toFixed(3)} mean vy=${(vy/n).toFixed(3)}`);
    }
    pbf.destroy();
  });
});

describe('compressed', () => {
  it('error vs iterations', async () => {
    const device = await getTestDevice();
    const dim = Number(process.env.PBF_DIM ?? 2) as 2 | 3;
    const s = 0.02;
    const extra = JSON.parse(process.env.PBF_P ?? '{}') as Partial<PbfParams>;
    const its = JSON.parse(process.env.PBF_ITS ?? '[1,2,4,8,16,32]') as number[];
    const row: string[] = [];
    for (const iterations of its) {
      const pbf = createPbf({ device, dimension: dim, capacity: 20000, spacing: s,
        bounds: { min: [-5,-5,-5], max: [5,5,5] }, params: { iterations, substeps: 1, gravity: [0,0,0], ...extra } });
      const c = pbf.constants;
      // compressed lattice: spacing 0.8 s, 14 x 14 (x 14)
      const m = dim === 2 ? 20 : 12; const sp = 0.8 * s;
      const pos: number[] = []; 
      for (let k = 0; k < (dim===3?m:1); k++) for (let j = 0; j < m; j++) for (let i = 0; i < m; i++) pos.push(i*sp, j*sp, k*sp, 0);
      pbf.system.set({ positions: new Float32Array(pos) });
      const n = pbf.count;
      const out: string[] = [];
      const measure = async () => {
        const p = await pbf.system.read('position');
        const lo = (m*sp)/2;
        let sum=0, cnt=0, ke=0;
        for (let i=0;i<n;i++){
          const x=p[i*4]!, y=p[i*4+1]!, z=p[i*4+2]!;
          // interior only
          const mid = (m-1)*sp/2;
          if (Math.hypot(x-mid,y-mid, dim===3? z-mid:0) > 0.2*m*sp) continue;
          let rho=0;
          for (let j=0;j<n;j++){ const r=Math.hypot(x-p[j*4]!,y-p[j*4+1]!,dim===3?z-p[j*4+2]!:0); if (r<c.h) rho+=poly6(dim,c.h,r);} 
          sum+=rho/c.n0; cnt++;
        }
        return sum/cnt;
      };
      const before = await measure();
      const e = device.createCommandEncoder();
      pbf.run(e, { dt: 1/120 });
      device.queue.submit([e.finish()]);
      const after = await measure();
      const vel = await pbf.system.read('velocity');
      let ke = 0;
      for (let i=0;i<n;i++) ke += vel[i*4]!**2+vel[i*4+1]!**2+vel[i*4+2]!**2;
      row.push(`${iterations}:${after.toFixed(3)}/v${Math.sqrt(ke/n).toFixed(2)}`);
      pbf.destroy();
    }
    console.log(`dim=${dim} ${JSON.stringify(extra)} ${row.join('  ')}`);
  });
});

describe('grid', () => {
  it('param grid', async () => {
    const device = await getTestDevice();
    const dim = Number(process.env.PBF_DIM ?? 2) as 2 | 3;
    const s = Number(process.env.PBF_S ?? 0.02);
    const W = Number(process.env.PBF_W ?? 0.4);
    const H0 = Number(process.env.PBF_H ?? 1.0);
    const grid = JSON.parse(process.env.PBF_GRID ?? '{}') as Record<string, number[]>;
    const its = JSON.parse(process.env.PBF_CASES ?? '[[2,4]]') as number[][];
    const keys = Object.keys(grid);
    const combos: Record<string, number>[] = [{}];
    for (const k of keys) { const next: Record<string, number>[] = []; for (const c of combos) for (const v of grid[k]!) next.push({ ...c, [k]: v }); combos.splice(0, combos.length, ...next); }
    for (const [iterations, substeps] of its) {
      for (const combo of combos) {
        const pbf = createPbf({ device, dimension: dim, capacity: 40000, spacing: s,
          bounds: { min: [0,0,dim===3?0:-1], max: [W,2,dim===3?W:1] }, params: { iterations, substeps, ...combo } , kernelScale: process.env.PBF_K ? Number(process.env.PBF_K) : undefined });
        pbf.reset({ blocks: [{ min: [0,0,0], max: [W, H0, W], jitter: 0.05 }] });
        const n = pbf.count;
        await frames(device, pbf, 120);
        const pos = await pbf.system.read('position');
        const vel = await pbf.system.read('velocity');
        let ysum=0, ytop=0, ke=0, vmax=0;
        for (let i=0;i<n;i++){ ysum+=pos[i*4+1]!; ytop=Math.max(ytop,pos[i*4+1]!); const v2=vel[i*4]!**2+vel[i*4+1]!**2+vel[i*4+2]!**2; ke+=v2; vmax=Math.max(vmax,Math.sqrt(v2)); }
        console.log(`G it=${iterations} sub=${substeps} ${JSON.stringify(combo)} H=${(2*ysum/n/H0).toFixed(3)} top=${ytop.toFixed(3)} v=${Math.sqrt(ke/n).toFixed(3)} vmax=${vmax.toFixed(2)}`);
        pbf.destroy();
      }
    }
  });
});

describe('scratch', () => {
  it('2d column settles', async () => {
    const device = await getTestDevice();
    const s = 0.02;
    const pbf = createPbf({ device, dimension: 2, capacity: 8000, spacing: s,
      bounds: { min: [0,0,-1], max: [1,1,1] }, params: JSON.parse(process.env.PBF_P ?? '{}') as Partial<PbfParams> });
    const c = pbf.constants;
    console.log('consts', JSON.stringify(c));
    pbf.reset({ blocks: [{ min: [0,0,0], max: [0.4, 0.6, 0], jitter: 0.05 }] });
    const n = pbf.count;
    console.log('n', n, 'incompressible height', n*s*s/0.4);
    for (let k = 0; k < 6; k++) {
      await frames(device, pbf, 40);
      const pos = await pbf.system.read('position');
      const vel = await pbf.system.read('velocity');
      let ymax = 0, ke = 0, ysum=0, vmax=0, xmax=0;
      for (let i = 0; i < n; i++) { ymax = Math.max(ymax, pos[i*4+1]!); ysum += pos[i*4+1]!; const v=Math.hypot(vel[i*4]!,vel[i*4+1]!); ke += v*v; vmax=Math.max(vmax,v); xmax=Math.max(xmax,pos[i*4]!);}
      const rho = density(2, pos, n, s, c.h, c.n0, [0,0],[1,1]);
      // interior
      let m=0, cnt=0, mx=0;
      for (let i=0;i<n;i++){ const y=pos[i*4+1]!, x=pos[i*4]!; if (x>c.h && x<1-c.h && y>c.h && y<ymax-1.5*c.h) { m+=rho[i]!; cnt++; mx=Math.max(mx,rho[i]!);} }
      console.log(`t=${((k+1)*40/60).toFixed(2)} ymax=${ymax.toFixed(3)} mean y=${(ysum/n).toFixed(3)} xmax=${xmax.toFixed(3)} rmsv=${Math.sqrt(ke/n).toFixed(3)} vmax=${vmax.toFixed(3)} interior rho mean=${(m/cnt).toFixed(3)} max=${mx.toFixed(3)}`);
    }
    pbf.destroy();
  });
});

describe('cost quick', () => {
  it('wall clock', async () => {
    const device = await getTestDevice();
    const dim = Number(process.env.PBF_DIM ?? 3) as 2 | 3;
    const counts = JSON.parse(process.env.PBF_N ?? '[30000,100000,250000]') as number[];
    for (const n of counts) {
      const s = dim === 3 ? 0.02 : 0.01;
      const side = Math.ceil(dim === 3 ? Math.cbrt(n) : Math.sqrt(n));
      const pbf = createPbf({ device, dimension: dim, capacity: n + 10, spacing: s,
        bounds: { min: [0,0,dim===3?0:-1], max: [side*s*1.5, side*s*1.2, dim===3?side*s*1.5:1] }, params: JSON.parse(process.env.PBF_P ?? '{}') });
      pbf.reset({ blocks: [{ min: [0,0,0], max: [side*s*1.5, (n/(side*(dim===3?side*1.5:1)))*s/1.5 , side*s*1.5] , jitter: 0.1 }] });
      const run = async (k: number) => { const e = device.createCommandEncoder(); for (let i=0;i<k;i++) pbf.run(e); device.queue.submit([e.finish()]); const t=performance.now(); await device.queue.onSubmittedWorkDone(); return t; };
      // warm
      for (let i=0;i<3;i++) await run(2);
      const timeIt = async (k: number) => { const t0 = performance.now(); await run(k); return performance.now() - t0; };
      const small: number[] = [], large: number[] = [];
      for (let r=0;r<6;r++){ small.push(await timeIt(2)); large.push(await timeIt(10)); }
      small.sort((a,b)=>a-b); large.sort((a,b)=>a-b);
      const per = (large[0]! - small[0]!)/8;
      console.log(`C dim=${dim} n=${pbf.count} per substep ${per.toFixed(2)} ms (it=${pbf.params.iterations})`);
      pbf.destroy();
    }
  });
});

describe('cost stages', () => {
  it('stage slopes', async () => {
    const device = await getTestDevice();
    const dim = Number(process.env.PBF_DIM ?? 3) as 2 | 3;
    const n = Number(process.env.PBF_N1 ?? 100000);
    const s = dim === 3 ? 0.02 : 0.01;
    const side = Math.ceil(dim === 3 ? Math.cbrt(n) : Math.sqrt(n));
    const pbf = createPbf({ device, dimension: dim, capacity: n + 10, spacing: s,
      bounds: { min: [0,0,dim===3?0:-1], max: [side*s*1.5, side*s*1.2, dim===3?side*s*1.5:1] }, params: { gravity: [0,0,0], ...JSON.parse(process.env.PBF_P ?? '{}') } });
    pbf.reset({ blocks: [{ min: [0,0,0], max: [side*s*1.5, (n/(side*(dim===3?side*1.5:1)))*s/1.5 , side*s*1.5] , jitter: 0.1 }] });
    await frames(device, pbf, 20);
    const stageList: [string[], number][] = [[['predict'],1],[['predict','grid'],1],[['predict','grid','density','correct'],1],[['predict','grid','density','correct'],2],[['predict','grid','density','correct'],4],[['predict','grid','density','correct','viscosity'],2],[['predict','grid','density','correct','viscosity','vorticity'],2]];
    for (const [stages, its] of stageList) {
      const run = async (k: number) => { const e = device.createCommandEncoder(); for (let i=0;i<k;i++) pbf.run(e, { stages: stages as never, iterations: its }); device.queue.submit([e.finish()]); await device.queue.onSubmittedWorkDone(); };
      for (let i=0;i<3;i++) await run(2);
      const timeIt = async (k: number) => { const t0 = performance.now(); await run(k); return performance.now() - t0; };
      const small: number[] = [], large: number[] = [];
      for (let r=0;r<8;r++){ small.push(await timeIt(2)); large.push(await timeIt(20)); }
      small.sort((a,b)=>a-b); large.sort((a,b)=>a-b);
      console.log(`S dim=${dim} n=${pbf.count} it=${its} ${stages.join('+')} ${((large[0]! - small[0]!)/18).toFixed(3)} ms`);
    }
    pbf.destroy();
  });
});

describe('cost labels', () => {
  it('per pass', async () => {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter!.requestDevice({ requiredFeatures: ['timestamp-query'] });
    const dim = Number(process.env.PBF_DIM ?? 3) as 2 | 3;
    const n = Number(process.env.PBF_N1 ?? 100000);
    const s = dim === 3 ? 0.02 : 0.01;
    const side = Math.ceil(dim === 3 ? Math.cbrt(n) : Math.sqrt(n));
    const querySet = device.createQuerySet({ type: 'timestamp', count: 256 });
    const resolve = device.createBuffer({ size: 256*8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: 256*8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    let labels: string[] = [];
    const profiler = { pass(label: string) { const i = labels.length; labels.push(label); if (i >= 127) return undefined; return { querySet, beginningOfPassWriteIndex: i*2, endOfPassWriteIndex: i*2+1 }; } };
    const pbf = createPbf({ device, dimension: dim, capacity: n + 10, spacing: s, profiler, timedPasses: 1000,
      bounds: { min: [0,0,dim===3?0:-1], max: [side*s*1.5, side*s*1.2, dim===3?side*s*1.5:1] }, params: { gravity: [0,0,0], ...JSON.parse(process.env.PBF_P ?? '{}') } });
    pbf.reset({ blocks: [{ min: [0,0,0], max: [side*s*1.5, (n/(side*(dim===3?side*1.5:1)))*s/1.5 , side*s*1.5] , jitter: 0.1 }] });
    const best: Record<string, number> = {};
    for (let r = 0; r < 30; r++) {
      labels = [];
      const e = device.createCommandEncoder();
      pbf.run(e);
      e.resolveQuerySet(querySet, 0, Math.min(256, labels.length*2), resolve, 0);
      e.copyBufferToBuffer(resolve, 0, read, 0, 256*8);
      device.queue.submit([e.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const st = new BigUint64Array(read.getMappedRange().slice(0));
      read.unmap();
      if (r < 5) continue;
      const sums = new Map<string, number>();
      labels.forEach((l, i) => sums.set(l, (sums.get(l) ?? 0) + Number(st[i*2+1]! - st[i*2]!)/1e6));
      for (const [l, v] of sums) best[l] = Math.min(best[l] ?? 1e9, v);
    }
    console.log(`L n=${pbf.count} ` + Object.entries(best).map(([l, v]) => `${l}=${v.toFixed(3)}`).join(' | '));
    pbf.destroy();
  });
});

describe('dam3d', () => {
  it('3d dam break', async () => {
    const device = await getTestDevice();
    const s = 0.03;
    const pbf = createPbf({ device, dimension: 3, capacity: 40000, spacing: s,
      bounds: { min: [0,0,0], max: [1.5,0.9,0.6] }, params: JSON.parse(process.env.PBF_P ?? '{}') });
    pbf.reset({ blocks: [{ min: [0,0,0], max: [0.45, 0.6, 0.6], jitter: 0.05 }] });
    const n = pbf.count;
    console.log('D n', n, 'rest height if spread', (n*s**3/(1.5*0.6)).toFixed(3));
    for (let k = 0; k < 16; k++) {
      await frames(device, pbf, 30);
      const pos = await pbf.system.read('position');
      const vel = await pbf.system.read('velocity');
      let xs=0, ys=0, ke=0, ymax=0, xmax=0, esc=0, foam=0, vmax=0;
      for (let i=0;i<n;i++){ const x=pos[i*4]!,y=pos[i*4+1]!,z=pos[i*4+2]!; xs+=x; ys+=y; ymax=Math.max(ymax,y); xmax=Math.max(xmax,x); if (x<0||x>1.5||y<0||y>0.9||z<0||z>0.6) esc++; foam+=pos[i*4+3]!; const v2=vel[i*4]!**2+vel[i*4+1]!**2+vel[i*4+2]!**2; ke+=v2; vmax=Math.max(vmax,Math.sqrt(v2)); }
      console.log(`D t=${((k+1)*0.5).toFixed(1)} mean x=${(xs/n).toFixed(3)} mean y=${(ys/n).toFixed(3)} ymax=${ymax.toFixed(3)} xmax=${xmax.toFixed(3)} rmsv=${Math.sqrt(ke/n).toFixed(3)} vmax=${vmax.toFixed(2)} esc=${esc} foam=${(foam/n).toFixed(3)}`);
    }
    pbf.destroy();
  });
});

describe('wave3d', () => {
  it('wave maker', async () => {
    const device = await getTestDevice();
    const s = 0.03;
    const dim = Number(process.env.PBF_DIM ?? 3) as 2|3;
    const pbf = createPbf({ device, dimension: dim, capacity: 40000, spacing: s,
      bounds: { min: [0,0,0], max: [1.5,0.9,0.45] }, params: JSON.parse(process.env.PBF_P ?? '{}') });
    pbf.reset({ blocks: [{ min: [0.0,0,0], max: [1.5, 0.3, 0.45], jitter: 0.05 }] });
    const n = pbf.count;
    let t = 0;
    const amp = 0.12, f = Number(process.env.PBF_F ?? 0.7);
    for (let k = 0; k < 20; k++) {
      for (let i = 0; i < 30; i++) {
        t += 1/60;
        const x0 = amp * (1 - Math.cos(2*Math.PI*f*t)) * 0.5 * 2;
        pbf.setBounds([x0,0,0],[1.5,0.9,0.45]);
        const e = device.createCommandEncoder(); pbf.update(e, 1/60); device.queue.submit([e.finish()]);
      }
      await device.queue.onSubmittedWorkDone();
      const pos = await pbf.system.read('position');
      let esc=0, hFar=0, cFar=0, hNear=0, cNear=0, ymax=0; const b = pbf.bounds;
      for (let i=0;i<n;i++){ const x=pos[i*4]!,y=pos[i*4+1]!,z=pos[i*4+2]!; if (x<b.min[0]!-1e-4||x>b.max[0]!+1e-4||y<-1e-4||y>0.9||(dim===3&&(z<0||z>0.45))) esc++; ymax=Math.max(ymax,y); if (x>1.2){ hFar+=y; cFar++; } if (x<0.4+b.min[0]!) { hNear+=y; cNear++; } }
      console.log(`W t=${t.toFixed(1)} wall=${b.min[0]!.toFixed(3)} esc=${esc} far mean y*2=${(2*hFar/Math.max(1,cFar)).toFixed(3)} near=${(2*hNear/Math.max(1,cNear)).toFixed(3)} ymax=${ymax.toFixed(2)}`);
    }
    pbf.destroy();
  });
});

describe('free', () => {
  it('free block', async () => {
    const device = await getTestDevice();
    for (const dim of [2, 3] as const) for (const [k, rest] of [[0.06, 0], [0.1, 0], [0.2, 0], [0.4, 0]] as const) {
      const s = 0.03;
      const pbf = createPbf({ device, dimension: dim, capacity: 5000, spacing: s, bounds: { min: [-5,-5,-5], max: [5,5,5] }, params: { gravity: [0,0,0], artificialPressure: k, artificialPressureRest: rest } });
      pbf.reset({ blocks: [{ min: [0,0,0], max: [0.3,0.3,0.3], jitter: 0.05 }] });
      const n = pbf.count;
      await frames(device, pbf, 90);
      const pos = await pbf.system.read('position'); const vel = await pbf.system.read('velocity');
      let r=0, ke=0; for (let i=0;i<n;i++){ r=Math.max(r,Math.hypot(pos[i*4]!-0.15,pos[i*4+1]!-0.15, dim===3?pos[i*4+2]!-0.15:0)); ke+=vel[i*4]!**2+vel[i*4+1]!**2+vel[i*4+2]!**2; }
      console.log(`F dim=${dim} k=${k} rest=${rest} n=${n} max radius ${r.toFixed(3)} (start ${(dim===3?0.26:0.21).toFixed(2)}) rms v ${Math.sqrt(ke/n).toFixed(3)}`);
      pbf.destroy();
    }
  });
});
