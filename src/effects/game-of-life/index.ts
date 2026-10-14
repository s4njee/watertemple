import { defineEffect } from '../types.ts';
import { meta } from './meta.ts';
import { createLifeSimulation } from './simulation.ts';
import { makeSeed } from './seeds.ts';
export const effect = defineEffect({
  meta,
  hdr: true,
  params: {
    paperLight: {
      type: 'number',
      default: 2,
      min: 1,
      max: 5,
      step: 0.05,
      label: 'HDR neon brightness',
    },
    speed: {
      type: 'number',
      default: 12,
      min: 1,
      max: 30,
      step: 1,
      label: 'Generations per second',
    },
    cellSize: {
      type: 'number',
      default: 7,
      min: 4,
      max: 16,
      step: 1,
      label: 'Cell size (CSS pixels)',
    },
    pattern: { type: 'select', default: 'soup', options: ['soup', 'garden', 'oscillators'] },
    seed: { type: 'number', default: 1, min: 1, max: 1000000, step: 1 },
    running: { type: 'boolean', default: true },
    reseedInterval: {
      type: 'number',
      default: 18,
      min: 0,
      max: 60,
      step: 1,
      label: 'Fresh seed interval (seconds; 0 disables)',
    },
    singleStep: { type: 'number', default: 0, min: 0, max: 1000000, step: 1 },
    plantRequest: { type: 'number', default: 0, min: 0, max: 1000000, step: 1 },
    plantX: { type: 'number', default: 0.5, min: 0, max: 1 },
    plantY: { type: 'number', default: 0.5, min: 0, max: 1 },
    generation: { type: 'number', default: 0, min: 0, max: 100000000, step: 1 },
  },
  async create({ device, format, params, profiler, hdr = false }) {
    let width = 1,
      height = 1,
      dpr = 1,
      key = '',
      accumulator = 0,
      colourTime = 0,
      seedElapsed = 0,
      seedEvent = 0;
    let simulation: ReturnType<typeof createLifeSimulation> | undefined;
    let stepRequest = params.singleStep,
      plantRequest = params.plantRequest;
    return {
      resize(w, h, d) {
        width = w;
        height = h;
        dpr = d;
      },
      frame({ delta }, encoder, target) {
        const w = Math.max(16, Math.round(width / dpr / params.cellSize)),
          h = Math.max(16, Math.round(height / dpr / params.cellSize));
        const nextKey = `${w}:${h}:${params.pattern}:${params.seed}`;
        let fresh = false;
        if (!simulation || key !== nextKey) {
          simulation?.destroy();
          simulation = createLifeSimulation(device, format, w, h);
          simulation.seed(makeSeed(w, h, params.pattern, params.seed));
          key = nextKey;
          accumulator = 0;
          params.generation = 0;
          seedElapsed = 0;
          seedEvent = 0;
          fresh = true;
        }
        if (plantRequest !== params.plantRequest) {
          simulation.plant(encoder, params.plantX, params.plantY);
          plantRequest = params.plantRequest;
        }
        let steps = 0;
        if (params.running && !fresh) {
          accumulator += Math.min(0.1, Math.max(0, delta)) * params.speed;
          steps = Math.min(3, Math.floor(accumulator));
          accumulator -= steps;
        } else if (!params.running) {
          accumulator = 0;
        }
        if (stepRequest !== params.singleStep) {
          steps += Math.max(0, params.singleStep - stepRequest);
          stepRequest = params.singleStep;
        }
        if (steps) {
          simulation.step(encoder, steps);
          params.generation += steps;
        }
        if (params.running) {
          const elapsed = Math.min(0.1, Math.max(0, delta));
          colourTime += elapsed;
          seedElapsed = fresh || params.reseedInterval <= 0 ? 0 : seedElapsed + elapsed;
          if (params.reseedInterval > 0 && seedElapsed >= params.reseedInterval) {
            seedElapsed %= params.reseedInterval;
            simulation.sprinkle(encoder, params.seed + ++seedEvent * 104729);
          }
        }
        simulation.setTime(colourTime, hdr ? params.paperLight : 1);
        simulation.draw(encoder, target, profiler.pass('Life display'));
      },
      destroy() {
        simulation?.destroy();
      },
    };
  },
});
