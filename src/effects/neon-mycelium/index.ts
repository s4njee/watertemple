import { trackDomRects } from '../../gpu/dom-rects.ts';
import { trackPointer } from '../../gpu/pointer.ts';
import { defineEffect } from '../types.ts';
import { createMyceliumSimulation, MAX_ROOTS } from './simulation.ts';
import { meta } from './meta.ts';

const PALETTES = {
  electric: [0.015, 0.65, 1, 0, 0.75, 0.015, 0.45, 0, 0.25, 0.025, 1, 0],
  verdant: [0.025, 0.8, 0.26, 0, 0.015, 0.48, 0.9, 0, 0.35, 0.9, 0.06, 0],
  ember: [1, 0.25, 0.015, 0, 0.8, 0.015, 0.06, 0, 1, 0.65, 0.08, 0],
} as const;

export const effect = defineEffect({
  meta,
  hdr: true,
  params: {
    agents: {
      type: 'number',
      default: 8192,
      min: 1024,
      max: 65536,
      step: 1024,
      label: 'Physarum agents',
    },
    resolution: {
      type: 'select',
      default: '512',
      options: ['256', '384', '512', '640'],
      label: 'Grid cells on the short side',
    },
    growth: { type: 'number', default: 1, min: 0, max: 3, step: 0.05, label: 'Growth rate' },
    sensorDistance: {
      type: 'number',
      default: 18,
      min: 3,
      max: 40,
      step: 1,
      label: 'Trail sensor distance (cells)',
    },
    sensorAngle: {
      type: 'number',
      default: 0.55,
      min: 0.1,
      max: 1.5,
      step: 0.05,
      label: 'Trail sensor angle',
    },
    turn: {
      type: 'number',
      default: 0.22,
      min: 0.02,
      max: 1,
      step: 0.01,
      label: 'Turning per simulation step',
    },
    speed: {
      type: 'number',
      default: 1.3,
      min: 0.1,
      max: 4,
      step: 0.05,
      label: 'Travel (cells per step)',
    },
    diffusion: {
      type: 'number',
      default: 0.3,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Trail diffusion',
    },
    decay: {
      type: 'number',
      default: 1.8,
      min: 0.2,
      max: 6,
      step: 0.05,
      label: 'Trail decay (per simulation second)',
    },
    deposit: {
      type: 'number',
      default: 0.055,
      min: 0.01,
      max: 2,
      step: 0.01,
      label: 'Trail reinforcement',
    },
    pull: {
      type: 'number',
      default: 0.15,
      min: 0,
      max: 3,
      step: 0.05,
      label: 'Nutrient attraction near the page',
    },
    shell: {
      type: 'number',
      default: 0.03,
      min: 0.005,
      max: 0.15,
      step: 0.001,
      label: 'Nutrient band outside outlines',
    },
    palette: { type: 'select', default: 'electric', options: ['electric', 'verdant', 'ember'] },
    exposure: {
      type: 'number',
      default: 2.0,
      min: 0.3,
      max: 3,
      step: 0.05,
      label: 'Neon intensity',
    },
    paperLight: {
      type: 'number',
      default: 1.2,
      min: 1,
      max: 5,
      step: 0.05,
      label: 'HDR brightness',
    },
  },
  presets: {
    'neon mycelium': {},
    'fine roots': { sensorDistance: 8, turn: 0.3, diffusion: 0.35 },
    'broad networks': { sensorDistance: 22, turn: 0.15, diffusion: 0.7 },
  },
  async create({ device, format, canvas, params, profiler, hdr = false }) {
    const dom = trackDomRects(canvas, { epsilon: 0.0002 });
    const pointer = trackPointer(canvas);
    const reduced =
      typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
    let width = canvas.width || 1,
      height = canvas.height || 1;
    let simulation: ReturnType<typeof createMyceliumSimulation> | undefined;
    let sizeKey = '',
      aspect = width / height,
      initialized = false,
      geometryDirty = true,
      accumulator = 0;
    const boxes = new Float32Array(MAX_ROOTS * 4),
      extras = new Float32Array(MAX_ROOTS * 4);
    return {
      resize(w, h) {
        width = Math.max(1, w);
        height = Math.max(1, h);
        dom.invalidate();
        geometryDirty = true;
      },
      frame({ delta }, encoder, target) {
        const nextAspect = width / height;
        const requested = Math.min(65536, Math.max(1024, Math.round(params.agents)));
        const short = Number(params.resolution);
        // Avoid repeatedly resetting a growing network while resizing by a few pixels.
        if (
          !simulation ||
          Math.abs(Math.log(nextAspect / aspect)) > 0.06 ||
          sizeKey !== `${short}:${requested}`
        ) {
          simulation?.destroy();
          aspect = nextAspect;
          const w = Math.round(short * Math.max(1, aspect)),
            h = Math.round(short * Math.max(1, 1 / aspect));
          simulation = createMyceliumSimulation(device, format, w, h, requested);
          sizeKey = `${short}:${requested}`;
          initialized = false;
          geometryDirty = true;
          accumulator = 0;
        }
        const measured = dom.update(delta);
        const roots = measured.obstacles.slice(0, MAX_ROOTS);
        boxes.fill(0);
        extras.fill(0);
        roots.forEach((obstacle, index) => {
          const r = obstacle.exact ?? obstacle.rect;
          boxes.set(
            [
              (r.x + r.width / 2) * nextAspect,
              r.y + r.height / 2,
              (r.width * nextAspect) / 2,
              r.height / 2,
            ],
            index * 4,
          );
          extras[index * 4] = r.radius ?? 0;
          extras[index * 4 + 1] = obstacle.element?.getAttribute('data-wt-root') === 'off' ? 0 : 1;
        });
        simulation.configure({
          aspect: nextAspect,
          pixel: 1 / height,
          rect_count: roots.length,
          boxes,
          extras,
          sensor_distance: params.sensorDistance,
          sensor_angle: params.sensorAngle,
          speed: params.speed,
          turn: params.turn,
          diffusion: params.diffusion,
          decay: params.decay,
          deposit: params.deposit,
          pull: params.pull,
          shell: params.shell,
          exposure: params.exposure,
          light: hdr ? params.paperLight : 1,
          colours: PALETTES[params.palette as keyof typeof PALETTES] ?? PALETTES.electric,
          pointer: [pointer.state.x, pointer.state.y, pointer.state.down && !reduced ? 1 : 0, 0],
        });
        if (measured.changed || geometryDirty) {
          simulation.updateWalls(encoder);
          geometryDirty = false;
        }
        if (!initialized) {
          simulation.initialize(encoder);
          // A young rooted network is visible immediately, including the reduced-motion still.
          simulation.step(encoder, 48);
          initialized = true;
        } else if (!reduced) {
          accumulator += Math.min(0.05, Math.max(0, delta)) * 60 * params.growth * 3;
          const iterations = Math.min(9, Math.floor(accumulator));
          accumulator -= iterations;
          if (iterations > 0) simulation.step(encoder, iterations);
        }
        simulation.draw(encoder, target, profiler.pass('mycelium shade'));
      },
      destroy() {
        dom.destroy();
        pointer.destroy();
        simulation?.destroy();
      },
    };
  },
});
