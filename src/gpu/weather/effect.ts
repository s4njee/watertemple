import { defineEffect, type EffectMeta, type ParamSchema } from '../../effects/types.ts';
import { createFullscreenPipeline, renderFullscreen } from '../fullscreen.ts';
import { createUniformBuffer } from '../uniforms.ts';
import { createShader } from '../shader.ts';
import { trackPointers } from '../pointer.ts';
import { generateMipmaps, mipLevelCount } from '../mipmap.ts';
import { createWeatherMemory } from './memory.ts';
import { WeatherParams } from './params.ts';
import {
  seededRandom,
  newRain,
  newBubble,
  splashBurst,
  advanceWeather,
  packWeather,
  MAX_WEATHER_DROPS,
  type WeatherDrop,
} from './dynamics.ts';
import background from './backdrop.wgsl';
import scene from './scene.wgsl';
import display from './display.wgsl';
import particles from './particles.wgsl';
export type WeatherMode = 0 | 1 | 2 | 3 | 4 | 5;
const number = (value: number, min: number, max: number, label: string, step = 0.01) => ({
  type: 'number' as const,
  default: value,
  min,
  max,
  step,
  label,
});
export function weatherSchema(mode: WeatherMode): ParamSchema {
  const schema: ParamSchema = {
    strength: number(1, 0, 2, 'Effect strength'),
    speed: number(1, 0, 3, 'Animation speed'),
    tint: { type: 'color', default: [0.35, 0.7, 0.8], label: 'Water / mist tint' },
  };
  if (mode === 0) {
    schema.count = number(320, 16, 1000, 'Rain streaks', 1);
    schema.wind = number(0.3, -2, 2, 'Wind');
  }
  if (mode === 1) schema.density = number(0.65, 0, 1, 'Rain density');
  if (mode === 2) {
    schema.count = number(48, 4, 140, 'Bubbles', 1);
    schema.style = { type: 'select', default: 'underwater', options: ['underwater', 'soap'] };
    schema.film = number(420, 100, 1000, 'Film thickness (soap, nm)', 1);
  }
  if (mode === 3) {
    schema.count = number(48, 8, 128, 'Droplets per burst', 1);
    schema.gravity = number(0.55, 0, 2, 'Gravity');
    schema.automatic = { type: 'boolean', default: true, label: 'Ambient bursts' };
  }
  if (mode === 4) {
    schema.density = number(0.86, 0, 1, 'Condensation');
    schema.refog = number(0.12, 0, 1, 'Refog rate / second');
    schema.brush = number(0.065, 0.015, 0.2, 'Wipe radius');
  }
  if (mode === 5) {
    schema.growth = number(0.5, 0.05, 2, 'Crystal growth rate');
    schema.brush = number(0.065, 0.015, 0.2, 'Wipe radius');
    schema.cycle = {
      type: 'select',
      default: 'freeze and thaw',
      options: ['freeze and thaw', 'freeze', 'melt'],
    };
  }
  if (mode !== 0 && mode !== 4)
    schema.refraction = number(mode === 2 ? 1.8 : 2, 0, 5, 'Refraction');
  return schema;
}
const PRESETS: Record<WeatherMode, Record<string, Record<string, string | number | boolean>>> = {
  0: {
    drizzle: { count: 120, wind: 0.15, speed: 0.7, strength: 0.7 },
    storm: { count: 850, wind: 1.2, speed: 1.6, strength: 1.25 },
  },
  1: {
    'quiet rain': { density: 0.3, speed: 0.65, strength: 0.7 },
    'silver rings': { density: 1, speed: 1.35, strength: 1.5 },
  },
  2: {
    'underwater air': { style: 'underwater', count: 48, refraction: 1.8 },
    'soap film': { style: 'soap', count: 28, film: 520, refraction: 2.4 },
  },
  3: {
    'tap feedback': { automatic: false, count: 32, gravity: 0.7 },
    'fountain bursts': { automatic: true, count: 96, gravity: 0.4, strength: 1.35 },
  },
  4: {
    'morning window': { density: 0.72, refog: 0.055, brush: 0.08 },
    'steamy glass': { density: 0.98, refog: 0.22, brush: 0.05 },
  },
  5: {
    'winter cycle': { cycle: 'freeze and thaw', growth: 0.5, refraction: 2 },
    'slow frost': { cycle: 'freeze', growth: 0.2, refraction: 3 },
    'thaw the pane': { cycle: 'melt', growth: 0.5, refraction: 2.5 },
  },
};
/** Shared ownership only: each mode has distinct particles, memory mechanics and optics. */
export function weatherEffect(meta: EffectMeta, mode: WeatherMode) {
  return defineEffect({
    meta,
    params: weatherSchema(mode),
    presets: PRESETS[mode],
    async create({ device, format, canvas, params, profiler }) {
      const uniforms = createUniformBuffer(device, WeatherParams, `${meta.id} params`);
      const pointer = trackPointers(canvas, { capture: mode >= 4, maxPointers: 4 });
      const reduced =
        typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
      const sampler = device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear',
        mipmapFilter: 'linear',
      });
      const chunks = {
        'weather_params.wgsl': WeatherParams.wgsl,
        'weather_scene.wgsl': scene.code,
        ...scene.chunks,
      };
      const backgroundPipeline = createFullscreenPipeline(device, {
        format: 'rgba16float',
        fragment: { label: 'weather/backdrop.wgsl', code: background.code, chunks },
      });
      const backgroundGroup = device.createBindGroup({
        layout: backgroundPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: uniforms.buffer } }],
      });
      const pipeline = createFullscreenPipeline(device, {
        format,
        fragment: { label: 'weather/display.wgsl', code: display.code, chunks },
      });
      const module = createShader(device, {
        label: 'weather/particles.wgsl',
        code: particles.code,
        chunks,
      });
      const dropPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module, entryPoint: 'vs' },
        fragment: {
          module,
          entryPoint: 'fs',
          targets: [
            {
              format,
              blend: {
                color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
                alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
              },
            },
          ],
        },
      });
      const buffer = device.createBuffer({
        label: 'weather instanced particles',
        size: MAX_WEATHER_DROPS * 48,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const dummy = device.createTexture({
        size: [1, 1],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      device.queue.writeTexture({ texture: dummy }, new Uint8Array([0, 0, 0, 255]), {}, [1, 1]);
      let backdrop: GPUTexture | undefined,
        memory: ReturnType<typeof createWeatherMemory> | undefined;
      let displayGroups: GPUBindGroup[] = [],
        dropGroup: GPUBindGroup | undefined;
      let width = 1,
        height = 1,
        aspect = 1,
        phase = 0,
        cycleTime = 0,
        burstElapsed = 0,
        backdropKey = '',
        generation = 0,
        initialized = false;
      const random = seededRandom(407 + mode * 31),
        drops: WeatherDrop[] = [];
      const strokes = new Float32Array(16 * 4);
      function allocate() {
        backdrop?.destroy();
        memory?.destroy();
        aspect = width / height;
        const scale = Math.min(0.5, 768 / Math.max(width, height));
        const w = Math.max(8, Math.round(width * scale)),
          h = Math.max(8, Math.round(height * scale));
        backdrop = device.createTexture({
          label: 'weather refractable scene',
          size: [w, h],
          format: 'rgba16float',
          mipLevelCount: mipLevelCount(w, h),
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
        });
        memory =
          mode >= 4
            ? createWeatherMemory(
                device,
                Math.max(32, Math.min(1024, Math.round(256 * aspect))),
                Math.max(32, Math.min(256, Math.round(1024 / aspect))),
              )
            : undefined;
        displayGroups = (memory?.views ?? [dummy.createView()]).map((view) =>
          device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: uniforms.buffer } },
              { binding: 1, resource: backdrop!.createView() },
              { binding: 2, resource: view },
              { binding: 3, resource: sampler },
            ],
          }),
        );
        dropGroup = device.createBindGroup({
          layout: dropPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: uniforms.buffer } },
            { binding: 1, resource: { buffer } },
            { binding: 2, resource: backdrop.createView() },
            { binding: 3, resource: sampler },
          ],
        });
        backdropKey = '';
        generation++;
        initialized = false;
        cycleTime = 0;
        drops.length = 0;
      }
      return {
        get generation() {
          return generation;
        },
        resize(w, h) {
          width = Math.max(1, w);
          height = Math.max(1, h);
          allocate();
        },
        frame({ delta, time }, encoder, target) {
          if (!backdrop || !dropGroup) return;
          const value = (name: string, fallback: number) =>
            typeof params[name] === 'number' ? Number(params[name]) : fallback;
          const dt = reduced ? 0 : Math.min(0.1, Math.max(0, delta)) * value('speed', 1);
          phase += dt;
          cycleTime += dt;
          const wind = value('wind', 0),
            count = Math.round(value('count', 48)),
            strength = value('strength', 1);
          if (!initialized) {
            if (mode === 0)
              for (let i = 0; i < count; i++) drops.push(newRain(random, aspect, wind, true));
            if (mode === 2)
              for (let i = 0; i < count; i++) drops.push(newBubble(random, aspect, true));
            if (mode === 3 && params.automatic)
              splashBurst(drops, aspect * 0.5, 0.56, random, count, strength);
            initialized = true;
          }
          const inputs = pointer.poll(time * 1000);
          let segmentCount = 0;
          strokes.fill(0);
          for (const input of inputs) {
            if (mode === 3 && input.justPressed && !input.cancelled)
              splashBurst(drops, input.x * aspect, input.y, random, count, strength);
            if (mode >= 4 && input.down)
              for (let i = 0; i < Math.max(1, input.path.length - 1) && segmentCount < 16; i++) {
                const a = input.path[i]!,
                  b = input.path[i + 1] ?? a;
                strokes.set([a.x, a.y, b.x, b.y], segmentCount++ * 4);
              }
          }
          if (dt > 0) {
            advanceWeather(drops, dt, aspect, wind, value('gravity', 0.7), random);
            if (mode === 0 || mode === 2) {
              const kind = mode === 0 ? 0 : 2;
              let present = drops.filter((p) => p.kind === kind).length;
              // Live count changes take effect without leaking old particles or resetting the whole scene.
              for (let i = drops.length - 1; i >= 0 && present > count; i--)
                if (drops[i]!.kind === kind) {
                  drops.splice(i, 1);
                  present--;
                }
              while (present < count && drops.length < MAX_WEATHER_DROPS) {
                drops.push(mode === 0 ? newRain(random, aspect, wind) : newBubble(random, aspect));
                present++;
              }
            }
            if (mode === 3 && params.automatic) {
              burstElapsed += dt;
              if (burstElapsed >= 1.5) {
                burstElapsed %= 1.5;
                splashBurst(
                  drops,
                  aspect * (0.2 + random() * 0.6),
                  0.4 + random() * 0.3,
                  random,
                  count,
                  strength,
                );
              }
            }
          }
          // Cycle includes a hold before thaw. Explicit melt starts from a developed crystal pane.
          const thaw =
            params.cycle === 'melt' || (params.cycle === 'freeze and thaw' && cycleTime % 22 >= 13);
          const frostClock =
            reduced || params.cycle === 'melt'
              ? 12
              : params.cycle === 'freeze and thaw'
                ? cycleTime % 22
                : cycleTime;
          const backgroundKind =
            mode === 2 ? (params.style === 'soap' ? 2 : 1) : mode === 1 || mode === 3 ? 3 : 0;
          const values = {
            resolution: [width, height] as [number, number],
            aspect,
            clock: phase,
            tint: params.tint as [number, number, number],
            strength,
            wind,
            refraction: value('refraction', 2),
            density: value('density', 0.8),
            speed: value('speed', 1),
            film: value('film', 420),
            background: backgroundKind,
            mode,
            stroke_count: segmentCount,
            brush: value('brush', 0.065),
            refog: value('refog', 0.12),
            growth: value('growth', 0.5),
            heat: thaw ? 1 : 0,
            delta: dt,
            cycle: frostClock,
            strokes,
          };
          uniforms.set(values);
          uniforms.upload();
          const key = `${backgroundKind}:${generation}`;
          if (key !== backdropKey) {
            backdropKey = key;
            renderFullscreen(
              encoder,
              backdrop.createView({ baseMipLevel: 0, mipLevelCount: 1 }),
              backgroundPipeline,
              [backgroundGroup],
              { timestampWrites: profiler.pass('weather backdrop (on change)') },
            );
            generateMipmaps(device, backdrop, encoder);
          }
          if (memory) memory.update(encoder, values, profiler.pass('weather glass memory'));
          renderFullscreen(encoder, target, pipeline, [displayGroups[memory?.index ?? 0]!], {
            timestampWrites: profiler.pass('weather optics'),
          });
          if (drops.length > 0) {
            const data = packWeather([...drops].sort((a, b) => a.depth - b.depth));
            device.queue.writeBuffer(buffer, 0, data as Float32Array<ArrayBuffer>);
            const pass = encoder.beginRenderPass({
              label: 'instanced weather',
              colorAttachments: [{ view: target, loadOp: 'load', storeOp: 'store' }],
              timestampWrites: profiler.pass('instanced weather'),
            });
            pass.setPipeline(dropPipeline);
            pass.setBindGroup(0, dropGroup);
            pass.draw(6, drops.length);
            pass.end();
          }
        },
        destroy() {
          pointer.destroy();
          uniforms.destroy();
          buffer.destroy();
          backdrop?.destroy();
          memory?.destroy();
          dummy.destroy();
        },
      };
    },
  });
}
