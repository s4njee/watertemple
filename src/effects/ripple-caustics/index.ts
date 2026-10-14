import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { trackPointers } from '../../gpu/pointer.ts';
import {
  createHeightfieldCaustics,
  lightDirection,
  type CausticsMethod,
} from '../../gpu/sim/caustics.ts';
import {
  createHeightfield,
  type BoundaryMode,
  type Heightfield,
} from '../../gpu/sim/heightfield.ts';
import { createStirrer, type StirMode } from '../../gpu/sim/stir.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import { meta } from './meta.ts';
import sceneShader from './scene.wgsl';
import shadeShader from './shade.wgsl';

const SceneParams = defineUniformStruct('SceneParams', {
  aspect: 'f32',
  aa: 'f32',
  density: 'f32',
  kind: 'i32',
});

const ShadeParams = defineUniformStruct('ShadeParams', {
  sigma: 'vec3f',
  aspect: 'f32',
  scatter: 'vec3f',
  strength: 'f32',
  light: 'vec3f',
  ior: 'f32',
  depth: 'f32',
  fade: 'f32',
  shine: 'f32',
  intensity: 'f32',
  ambient: 'f32',
  sun: 'f32',
  turbidity: 'f32',
  white: 'f32',
  view: 'u32',
  split: 'f32',
  time: 'f32',
  fake_scale: 'f32',
  fake_cells: 'i32',
  fake_sharp: 'f32',
});

const METHODS = ['mesh', 'jacobian'] as const;
const CAUSTIC_RESOLUTIONS = ['auto', '256', '384', '512', '768', '1024'] as const;
const MESH_DENSITIES = ['0.5', '1', '2'] as const;
const GRID_RESOLUTIONS = ['128', '192', '256', '384', '512'] as const;
const BOUNDARIES = ['absorb', 'reflect'] as const;
const BACKDROPS = ['pool tiles', 'sand'] as const;
const POINTER_MODES = ['hover and press', 'press only', 'hover only', 'off'] as const;
const TOUCH_MODES = ['scrolls the page', 'stirs the water'] as const;
const VIEWS = ['lit floor', 'caustics only', 'compare lit', 'compare caustics only'] as const;

const BACKDROP_FORMAT: GPUTextureFormat = 'rgba8unorm-srgb';
/** Raindrops hit harder than a gentle touch: they have to read from across a page. */
const RAIN_GAIN = 2.2;
const DEG = Math.PI / 180;

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
    method: {
      type: 'select',
      default: 'jacobian',
      options: METHODS,
      label: 'Caustics method (mesh: projected light mesh, jacobian: per-pixel)',
    },
    causticsResolution: {
      type: 'select',
      default: 'auto',
      options: CAUSTIC_RESOLUTIONS,
      label: 'Caustics texels on the long side (auto: half the canvas)',
    },
    meshDensity: {
      type: 'select',
      default: '1',
      options: MESH_DENSITIES,
      label: 'Mesh vertices per grid cell (mesh only)',
    },
    resolution: {
      type: 'select',
      default: '256',
      options: GRID_RESOLUTIONS,
      label: 'Surface grid cells on the long side',
    },
    intensity: {
      type: 'number',
      default: 1.4,
      min: 0,
      max: 3,
      step: 0.01,
      label: 'Caustic intensity (contrast; 1 = as computed)',
    },
    sunElevation: {
      type: 'number',
      default: 72,
      min: 10,
      max: 90,
      step: 0.5,
      label: 'Sun elevation (degrees; 90 = overhead)',
    },
    sunAzimuth: {
      type: 'number',
      default: 35,
      min: 0,
      max: 360,
      step: 1,
      label: 'Sun direction on the canvas (degrees)',
    },
    depth: {
      type: 'number',
      default: 0.35,
      min: 0.05,
      max: 1.2,
      step: 0.01,
      label: 'Water depth (canvas heights)',
    },
    strength: {
      type: 'number',
      default: 2.2,
      min: 0.2,
      max: 6,
      step: 0.01,
      label: 'Wave steepness (slope gain)',
    },
    colorSplit: {
      type: 'number',
      default: 0.012,
      min: 0,
      max: 0.08,
      step: 0.001,
      label: 'Chromatic split of the caustics (0 = one grey pass)',
    },
    smooth: {
      type: 'number',
      default: 0.8,
      min: 0,
      max: 4,
      step: 0.1,
      label: 'Caustic blur (sigma, texels)',
    },
    waveSpeed: {
      type: 'number',
      default: 0.45,
      min: 0.1,
      max: 1.5,
      step: 0.01,
      label: 'Wave speed (canvas heights / s)',
    },
    damping: { type: 'number', default: 0.6, min: 0, max: 4, step: 0.01, label: 'Damping (1 / s)' },
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
      default: 3,
      min: 0,
      max: 40,
      step: 0.1,
      label: 'Rain (drops / s)',
    },
    swell: { type: 'number', default: 0.3, min: 0, max: 1, step: 0.01, label: 'Idle swell' },
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
    ior: { type: 'number', default: 1.333, min: 1.05, max: 2, step: 0.001, label: 'IOR' },
    tint: {
      type: 'color',
      default: [0.55, 0.88, 0.95],
      label: 'Water colour after 1 canvas height',
    },
    shine: { type: 'number', default: 0.35, min: 0, max: 1, step: 0.01 },
    backdrop: { type: 'select', default: 'pool tiles', options: BACKDROPS },
    view: { type: 'select', default: 'lit floor', options: VIEWS, label: 'View' },
    repeat: {
      type: 'number',
      default: 1,
      min: 1,
      max: 24,
      step: 1,
      label: 'Run the caustics pass N times per frame (benchmark: read the slope)',
    },
  },
  presets: {
    'midday pool': {
      method: 'jacobian',
      sunElevation: 84,
      depth: 0.35,
      strength: 2.2,
      intensity: 1.4,
      colorSplit: 0.012,
      rainRate: 3,
      dropSize: 0.032,
      damping: 0.6,
      swell: 0.3,
      view: 'lit floor',
    },
    'low sun': {
      method: 'mesh',
      sunElevation: 28,
      sunAzimuth: 20,
      depth: 0.55,
      strength: 1.6,
      intensity: 1.2,
      colorSplit: 0.02,
      rainRate: 4,
      dropSize: 0.04,
      damping: 0.5,
      swell: 0.4,
      view: 'lit floor',
    },
    stirred: {
      method: 'mesh',
      sunElevation: 68,
      depth: 0.4,
      strength: 2.6,
      intensity: 1,
      colorSplit: 0.01,
      waveSpeed: 0.8,
      damping: 0.25,
      dropSize: 0.05,
      dropStrength: 2.2,
      rainRate: 6,
      swell: 0.5,
      calm: 0.1,
      view: 'lit floor',
    },
    'real vs fake': {
      method: 'jacobian',
      sunElevation: 80,
      depth: 0.35,
      strength: 2.2,
      rainRate: 5,
      intensity: 1.4,
      view: 'compare lit',
    },
    'caustics only': {
      sunElevation: 78,
      view: 'caustics only',
      intensity: 1,
      colorSplit: 0,
      rainRate: 5,
    },
  },

  async create({ device, format, canvas, params, profiler }) {
    const sceneUniforms = createUniformBuffer(device, SceneParams, 'ripple-caustics scene params');
    const shadeUniforms = createUniformBuffer(device, ShadeParams, 'ripple-caustics shade params');
    const pointers = trackPointers(canvas);
    const stirrer = createStirrer();
    const random = createRandom(0xca57);

    const scenePipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'ripple-caustics/scene.wgsl',
        code: sceneShader.code,
        chunks: { ...sceneShader.chunks, 'scene_params.wgsl': SceneParams.wgsl },
      },
      format: BACKDROP_FORMAT,
    });
    const shadePipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'ripple-caustics/shade.wgsl',
        code: shadeShader.code,
        chunks: { ...shadeShader.chunks, 'shade_params.wgsl': ShadeParams.wgsl },
      },
      format,
    });
    const sceneGroup = device.createBindGroup({
      label: 'ripple-caustics scene',
      layout: scenePipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: sceneUniforms.buffer } }],
    });
    const sampler = device.createSampler({
      label: 'ripple-caustics sampler',
      magFilter: 'linear',
      minFilter: 'linear',
    });

    let width = Math.max(1, canvas.width || 1);
    let height = Math.max(1, canvas.height || 1);
    let backdrop: GPUTexture | undefined;
    let backdropView: GPUTextureView | undefined;
    let shadeGroup: GPUBindGroup | undefined;
    let boundKey = '';
    let drawnKey = '';

    const solver: Heightfield = createHeightfield({
      device,
      kind: 'render',
      resolution: Number(params.resolution),
      width,
      height,
      profiler,
      label: 'ripple-caustics',
    });
    const caustics = createHeightfieldCaustics({
      device,
      heightfield: solver,
      resolution: 512,
      method: params.method as CausticsMethod,
      profiler,
      label: 'caustics',
    });

    function allocateBackdrop() {
      backdrop?.destroy();
      backdrop = device.createTexture({
        label: 'ripple-caustics floor',
        size: [width, height],
        format: BACKDROP_FORMAT,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      backdropView = backdrop.createView();
      shadeGroup = undefined;
      drawnKey = '';
    }

    const causticsLongSide = () =>
      params.causticsResolution === 'auto'
        ? Math.min(1024, Math.max(256, Math.round(Math.max(width, height) / 2)))
        : Number(params.causticsResolution);

    let rainDebt = 0;
    let interaction = 0;
    let swellPhase = random() * 100;
    let swellPoint: [number, number] | undefined;
    let time = 0;

    function ambient(delta: number) {
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
        caustics.resize(width, height);
      },

      frame({ delta }, encoder, target) {
        if (!backdrop || !backdropView) return;
        const aspect = width / height;
        time += delta;

        solver.setResolution(Number(params.resolution));
        const wave = solver.params;
        wave.waveSpeed = params.waveSpeed;
        wave.damping = params.damping;
        wave.boundary = params.boundary as BoundaryMode;

        const backdropKey = `${params.backdrop}:${width}x${height}`;
        if (backdropKey !== drawnKey) {
          drawnKey = backdropKey;
          sceneUniforms.set({
            aspect,
            aa: 1 / height,
            density: 1,
            kind: params.backdrop === 'sand' ? 2 : 0,
          });
          sceneUniforms.upload();
          renderFullscreen(encoder, backdropView, scenePipeline, [sceneGroup], {
            timestampWrites: profiler.pass('floor (on change)'),
          });
        }

        pointers.setCapture(params.touch === 'stirs the water');
        const stir = stirrer.stir(solver, pointers.poll(), aspect, {
          mode: params.pointer as StirMode,
          radius: params.dropSize,
          strength: params.dropStrength,
        });
        const blend = 1 - Math.exp(-delta * (stir.active ? 6 : 0.8));
        interaction += ((stir.active ? 1 : 0) - interaction) * blend;

        ambient(delta);
        solver.update(encoder, delta);

        // The caustics, from the surface the solver just produced.
        caustics.method = params.method as CausticsMethod;
        caustics.setResolution(causticsLongSide());
        caustics.setMeshScale(Number(params.meshDensity));
        const lightDir = lightDirection(params.sunElevation * DEG, params.sunAzimuth * DEG);
        const repeats = Math.max(1, Math.round(params.repeat));
        for (let i = 0; i < repeats; i++) {
          caustics.update(encoder, {
            lightDir,
            depth: params.depth,
            ior: params.ior,
            strength: params.strength,
            dispersion: params.colorSplit,
            smooth: params.smooth,
          });
        }

        const key = `${solver.generation}:${caustics.generation}:${width}x${height}`;
        if (!shadeGroup || key !== boundKey) {
          boundKey = key;
          shadeGroup = device.createBindGroup({
            label: 'ripple-caustics shade',
            layout: shadePipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: shadeUniforms.buffer } },
              { binding: 1, resource: backdropView },
              { binding: 2, resource: sampler },
              { binding: 3, resource: solver.view },
              { binding: 4, resource: caustics.view },
              { binding: 5, resource: sampler },
            ],
          });
        }

        // Water colour: `tint` is what white turns into after one canvas height of water.
        const sigma = params.tint.map((c) => -Math.log(Math.min(Math.max(c, 1e-3), 1))) as [
          number,
          number,
          number,
        ];
        shadeUniforms.set({
          sigma,
          aspect,
          scatter: [0.05, 0.22, 0.3],
          strength: params.strength,
          light: lightDir,
          ior: params.ior,
          depth: params.depth,
          fade: 0.05,
          shine: params.shine,
          intensity: params.intensity,
          ambient: 0.4,
          sun: 0.85,
          turbidity: 0.4,
          white: 3,
          view: Math.max(0, VIEWS.indexOf(params.view as (typeof VIEWS)[number])),
          split: 0.5,
          time: (time * 0.6) % (Math.PI * 2),
          fake_scale: 1.8,
          fake_cells: 8,
          fake_sharp: 0.78,
        });
        shadeUniforms.upload();
        renderFullscreen(encoder, target, shadePipeline, [shadeGroup], {
          timestampWrites: profiler.pass('shade'),
        });
      },

      destroy() {
        pointers.destroy();
        caustics.destroy();
        solver.destroy();
        backdrop?.destroy();
        sceneUniforms.destroy();
        shadeUniforms.destroy();
      },
    };
  },
});
