import { expect, it } from 'vitest';
import { createMyceliumSimulation } from '../src/effects/neon-mycelium/simulation.ts';
import { halfToFloat } from '../src/gpu/sim/fluid.ts';
import { boxDistance } from '../src/effects/ink-orbit/heads.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';

async function readAgents(device: GPUDevice, source: GPUBuffer) {
  const copy = device.createBuffer({
    size: source.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, copy, 0, source.size);
  device.queue.submit([encoder.finish()]);
  await copy.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(copy.getMappedRange().slice(0));
  copy.unmap();
  copy.destroy();
  return data;
}
async function concentration(device: GPUDevice, texture: GPUTexture) {
  const bytes = await readTexture(device, texture, 8);
  const half = new Uint16Array(bytes.buffer);
  return Float32Array.from({ length: half.length / 4 }, (_, index) =>
    halfToFloat(half[index * 4 + 3]!),
  );
}

it('Physarum grows outside page walls, follows moved walls and keeps reorganizing instead of forming a viewport border', async () => {
  const device = await getTestDevice();
  await expectValid(device, async () => {
    const width = 128,
      height = 96,
      count = 2048,
      aspect = width / height;
    const sim = createMyceliumSimulation(device, 'rgba8unorm', width, height, count);
    try {
      const box = { cx: aspect * 0.5, cy: 0.5, hx: 0.2, hy: 0.12, radius: 0.025, spin: 0 };
      sim.configure({
        aspect,
        rect_count: 1,
        boxes: [box.cx, box.cy, box.hx, box.hy],
        extras: [box.radius, 1, 0, 0],
        deposit: 0.055,
        diffusion: 0.3,
        decay: 1.8,
        pull: 0.15,
      });
      const start = device.createCommandEncoder();
      sim.updateWalls(start);
      sim.initialize(start);
      sim.step(start, 150);
      device.queue.submit([start.finish()]);
      const before = await concentration(device, sim.trail);
      expect(Math.max(...before)).toBeGreaterThan(0.1);
      const agents = await readAgents(device, sim.agents);
      for (let index = 0; index < count; index++) {
        expect(
          boxDistance(agents[index * 4]! * aspect, agents[index * 4 + 1]!, box).d,
        ).toBeGreaterThan(-0.005);
      }
      let edge = 0,
        total = 0;
      for (let y = 0; y < height; y++)
        for (let x = 0; x < width; x++) {
          const value = before[y * width + x]!;
          total += value;
          if (x < 6 || x >= width - 6 || y < 6 || y >= height - 6) edge += value;
          if (boxDistance(((x + 0.5) / width) * aspect, (y + 0.5) / height, box).d < 0)
            expect(value).toBe(0);
        }
      expect(edge / total).toBeLessThan(0.35);
      const continued = device.createCommandEncoder();
      sim.step(continued, 120);
      device.queue.submit([continued.finish()]);
      const after = await concentration(device, sim.trail);
      let changed = 0;
      for (let i = 0; i < after.length; i++) changed += Math.abs(after[i]! - before[i]!);
      expect(changed / total).toBeGreaterThan(0.1);
      box.cx = aspect * 0.3;
      box.cy = 0.25;
      sim.configure({ boxes: [box.cx, box.cy, box.hx, box.hy] });
      const moved = device.createCommandEncoder();
      sim.updateWalls(moved);
      sim.step(moved, 3);
      device.queue.submit([moved.finish()]);
      const relocated = await concentration(device, sim.trail);
      for (let y = 0; y < height; y++)
        for (let x = 0; x < width; x++) {
          if (boxDistance(((x + 0.5) / width) * aspect, (y + 0.5) / height, box).d < 0)
            expect(relocated[y * width + x]).toBe(0);
        }
    } finally {
      sim.destroy();
    }
  });
}, 30000);
