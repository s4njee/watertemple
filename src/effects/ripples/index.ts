import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { trackPointers } from '../../gpu/pointer.ts';
import {
  createHeightfield,
  type BoundaryMode,
  type Heightfield,
  type HeightfieldKind,
} from '../../gpu/sim/heightfield.ts';
import { createStirrer, type StirMode } from '../../gpu/sim/stir.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import { meta } from './meta.ts';
import sceneShader from './scene.wgsl';
import waterShader from './water.wgsl';

const SceneParams = defineUniformStruct('SceneParams', {
  aspect: 'f32',
  aa: 'f32',
  density: 'f32',
  kind: 'i32',
});

const WaterParams = defineUniformStruct('WaterParams', {
  tint: 'vec3f',
  aspect: 'f32',
  strength: 'f32',
  ior: 'f32',
  thickness: 'f32',
  dispersion: 'f32',
  fade: 'f32',
  shine: 'f32',
  relief: 'f32',
  height_gain: 'f32',
  view: 'u32',
});

const SOLVERS = ['render', 'compute'] as const;
const RESOLUTIONS = ['128', '192', '256', '384', '512', '768', '1024'] as const;
const BOUNDARIES = ['absorb', 'reflect'] as const;
const BACKDROPS = ['pool tiles', 'test card', 'sand'] as const;
const POINTER_MODES = ['hover and press', 'press only', 'hover only', 'off'] as const;
const TOUCH_MODES = ['scrolls the page', 'stirs the water'] as const;
const VIEWS = ['water', 'split', 'height', 'normals', 'velocity'] as const;

const BACKDROP_FORMAT: GPUTextureFormat = 'rgba8unorm-srgb';
/** Raindrops hit harder than a gentle touch: they have to read from across a page. */
const RAIN_GAIN = 2.2;

/** mulberry32: a small seeded generator, so the rain of a given preset repeats. */
function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const effect = defineEffect({
  meta,
  params: {
    solver: {
      type: 'select',
      default: 'render',
      options: SOLVERS,
      label: 'Solver (re-creates the surface)',
    },
    resolution: {
      type: 'select',
      default: '256',
      options: RESOLUTIONS,
      label: 'Grid cells on the long side',
    },
    waveSpeed: {
      type: 'number',
      default: 0.45,
      min: 0.1,
      max: 1.5,
      step: 0.01,
      label: 'Wave speed (canvas heights / s)',
    },
    damping: { type: 'number', default: 0.7, min: 0, max: 4, step: 0.01, label: 'Damping (1 / s)' },
    boundary: { type: 'select', default: 'absorb', options: BOUNDARIES, label: 'Edges' },
    dropSize: {
      type: 'number',
      default: 0.032,
      min: 0.008,
      max: 0.12,
      step: 0.001,
      label: 'Drop size (canvas heights)',
    },
    dropStrength: {
      type: 'number',
      default: 1,
      min: 0,
      max: 4,
      step: 0.01,
      label: 'Drop strength',
    },
    rainRate: {
      type: 'number',
      default: 2.5,
      min: 0,
      max: 40,
      step: 0.1,
      label: 'Rain (drops / s)',
    },
    swell: { type: 'number', default: 0.35, min: 0, max: 1, step: 0.01, label: 'Idle swell' },
    calm: {
      type: 'number',
      default: 0.6,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Quiet down while touched',
    },
    pointer: { type: 'select', default: 'hover and press', options: POINTER_MODES },
    touch: {
      type: 'select',
      default: 'scrolls the page',
      options: TOUCH_MODES,
      label: 'Touch drags',
    },
    strength: {
      type: 'number',
      default: 2,
      min: 0.2,
      max: 8,
      step: 0.01,
      label: 'Refraction strength (slope gain)',
    },
    depth: {
      type: 'number',
      default: 0.35,
      min: 0.05,
      max: 1.2,
      step: 0.01,
      label: 'Water depth (canvas heights)',
    },
    ior: { type: 'number', default: 1.333, min: 1, max: 2, step: 0.001, label: 'IOR' },
    dispersion: { type: 'number', default: 0.012, min: 0, max: 0.1, step: 0.001 },
    shine: { type: 'number', default: 0.45, min: 0, max: 1, step: 0.01 },
    relief: { type: 'number', default: 0.3, min: 0, max: 1, step: 0.01, label: 'Relief shading' },
    tint: { type: 'color', default: [0.8, 0.95, 1] },
    backdrop: { type: 'select', default: 'pool tiles', options: BACKDROPS },
    view: { type: 'select', default: 'water', options: VIEWS, label: 'View' },
    stepsPerFrame: {
      type: 'number',
      default: 0,
      min: 0,
      max: 200,
      step: 1,
      label: 'Steps per frame (0 = by clock; for benchmarks)',
    },
  },
  presets: {
    'still pond': {
      waveSpeed: 0.35,
      damping: 0.5,
      boundary: 'absorb',
      dropSize: 0.04,
      dropStrength: 1,
      rainRate: 0.6,
      swell: 0.3,
      calm: 0.5,
      strength: 2.4,
      depth: 0.4,
      shine: 0.45,
      relief: 0.3,
    },
    rain: {
      waveSpeed: 0.5,
      damping: 1,
      boundary: 'absorb',
      dropSize: 0.022,
      dropStrength: 0.9,
      rainRate: 16,
      swell: 0.15,
      calm: 0.7,
      strength: 2.2,
      depth: 0.35,
      shine: 0.55,
      relief: 0.3,
    },
    stirred: {
      waveSpeed: 0.75,
      damping: 0.3,
      boundary: 'absorb',
      dropSize: 0.05,
      dropStrength: 2.2,
      rainRate: 0.8,
      swell: 0.5,
      calm: 0.2,
      strength: 2,
      depth: 0.5,
      shine: 0.5,
      relief: 0.35,
    },
    'tank (reflecting)': {
      waveSpeed: 0.6,
      damping: 0.12,
      boundary: 'reflect',
      dropSize: 0.035,
      dropStrength: 1.4,
      rainRate: 0.3,
      swell: 0,
      calm: 0,
      strength: 1.8,
      depth: 0.4,
      shine: 0.5,
      relief: 0.3,
    },
  },

  async create({ device, format, canvas, params, profiler }) {
    const sceneUniforms = createUniformBuffer(device, SceneParams, 'ripples scene params');
    const waterUniforms = createUniformBuffer(device, WaterParams, 'ripples water params');
    const pointers = trackPointers(canvas);
    const stirrer = createStirrer();
    const random = createRandom(0x5eed);

    const scenePipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'ripples/scene.wgsl',
        code: sceneShader.code,
        chunks: { ...sceneShader.chunks, 'scene_params.wgsl': SceneParams.wgsl },
      },
      format: BACKDROP_FORMAT,
    });
    const waterPipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'ripples/water.wgsl',
        code: waterShader.code,
        chunks: { ...waterShader.chunks, 'water_params.wgsl': WaterParams.wgsl },
      },
      format,
    });
    const sceneGroup = device.createBindGroup({
      label: 'ripples scene',
      layout: scenePipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: sceneUniforms.buffer } }],
    });
    const sampler = device.createSampler({
      label: 'ripples sampler',
      magFilter: 'linear',
      minFilter: 'linear',
    });

    let width = Math.max(1, canvas.width || 1);
    let height = Math.max(1, canvas.height || 1);
    let backdrop: GPUTexture | undefined;
    let backdropView: GPUTextureView | undefined;
    let waterGroup: GPUBindGroup | undefined;
    let waterGroupGeneration = -1;
    let drawnKey = '';

    // The solver. Switching `solver` throws it away and builds the other implementation: the
    // surface starts flat again, which is the honest cost of switching.
    let solver: Heightfield = makeSolver();
    function makeSolver(): Heightfield {
      return createHeightfield({
        device,
        kind: params.solver as HeightfieldKind,
        resolution: Number(params.resolution),
        width,
        height,
        profiler,
        label: 'ripples',
      });
    }

    function allocateBackdrop() {
      backdrop?.destroy();
      backdrop = device.createTexture({
        label: 'ripples backdrop',
        size: [width, height],
        format: BACKDROP_FORMAT,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      backdropView = backdrop.createView();
      waterGroup = undefined;
      drawnKey = '';
    }

    let rainDebt = 0;
    let interaction = 0;
    let swellPhase = random() * 100;
    let swellPoint: [number, number] | undefined;

    function ambient(delta: number) {
      // Rain: a Poisson process, at a rate that eases off while someone is touching the water.
      const quiet = 1 - params.calm * interaction;
      rainDebt += delta * params.rainRate * quiet;
      let drops = 0;
      while (rainDebt >= 1 && drops < 12) {
        rainDebt -= 1;
        drops++;
        const size = params.dropSize * (0.7 + 0.6 * random());
        solver.addDrop(
          0.02 + random() * 0.96,
          0.02 + random() * 0.96,
          size,
          -params.dropStrength * RAIN_GAIN * (0.7 + 0.7 * random()),
        );
      }
      rainDebt = Math.min(rainDebt, 1);

      // Idle swell: a broad, slow disc wandering over the surface, so the water is never dead.
      // The wake is proportional to how far it moved this frame, so it is the same at any frame rate.
      swellPhase += delta;
      const next: [number, number] = [
        0.5 + 0.44 * Math.sin(swellPhase * 0.19 + 1.3) * Math.cos(swellPhase * 0.07),
        0.5 + 0.4 * Math.sin(swellPhase * 0.27),
      ];
      if (swellPoint && params.swell > 0) {
        solver.displace(swellPoint, next, 0.16, params.swell * quiet * 0.9);
      }
      swellPoint = next;
    }

    return {
      resize(w, h) {
        width = Math.max(1, Math.round(w));
        height = Math.max(1, Math.round(h));
        allocateBackdrop();
        solver.resize(width, height);
      },

      frame({ delta }, encoder, target) {
        if (!backdrop || !backdropView) return;
        const aspect = width / height;

        // Parameters that need the solver rebuilt or re-gridded.
        if (solver.kind !== params.solver) {
          solver.destroy();
          solver = makeSolver();
        }
        solver.setResolution(Number(params.resolution));
        const wave = solver.params;
        wave.waveSpeed = params.waveSpeed;
        wave.damping = params.damping;
        wave.boundary = params.boundary as BoundaryMode;

        const key = `${params.backdrop}:${width}x${height}`;
        if (key !== drawnKey) {
          drawnKey = key;
          sceneUniforms.set({
            aspect,
            aa: 1 / height,
            density: 1,
            kind: Math.max(0, BACKDROPS.indexOf(params.backdrop as (typeof BACKDROPS)[number])),
          });
          sceneUniforms.upload();
          renderFullscreen(encoder, backdropView, scenePipeline, [sceneGroup], {
            timestampWrites: profiler.pass('backdrop (on change)'),
          });
        }

        // Input: touch drags scroll the page unless the visitor asked for the water to take them.
        pointers.setCapture(params.touch === 'stirs the water');
        const stir = stirrer.stir(solver, pointers.poll(), aspect, {
          mode: params.pointer as StirMode,
          radius: params.dropSize,
          strength: params.dropStrength,
        });
        const blend = 1 - Math.exp(-delta * (stir.active ? 6 : 0.8));
        interaction += ((stir.active ? 1 : 0) - interaction) * blend;

        ambient(delta);

        if (params.stepsPerFrame > 0) solver.advance(encoder, params.stepsPerFrame);
        else solver.update(encoder, delta);

        if (!waterGroup || waterGroupGeneration !== solver.generation) {
          waterGroupGeneration = solver.generation;
          waterGroup = device.createBindGroup({
            label: 'ripples water',
            layout: waterPipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: waterUniforms.buffer } },
              { binding: 1, resource: backdropView },
              { binding: 2, resource: sampler },
              { binding: 3, resource: solver.view },
              { binding: 4, resource: sampler },
            ],
          });
        }
        waterUniforms.set({
          tint: params.tint,
          aspect,
          strength: params.strength,
          ior: params.ior,
          thickness: params.depth,
          dispersion: params.dispersion,
          fade: 0.05,
          shine: params.shine,
          relief: params.relief,
          height_gain: 1.5,
          view: Math.max(0, VIEWS.indexOf(params.view as (typeof VIEWS)[number])),
        });
        waterUniforms.upload();
        renderFullscreen(encoder, target, waterPipeline, [waterGroup], {
          timestampWrites: profiler.pass('shade'),
        });
      },

      destroy() {
        pointers.destroy();
        solver.destroy();
        backdrop?.destroy();
        sceneUniforms.destroy();
        waterUniforms.destroy();
      },
    };
  },
});
