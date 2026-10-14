import { edgeWake, trackDomRects, type DomRectsOptions } from '../../gpu/dom-rects.ts';
import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { trackPointers } from '../../gpu/pointer.ts';
import {
  createHeightfield,
  type BoundaryMode,
  type Heightfield,
} from '../../gpu/sim/heightfield.ts';
import { createStirrer, type StirMode } from '../../gpu/sim/stir.ts';
import { createUniformBuffer, defineUniformStruct, uniformArray } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import { createDemoElements, type DemoElements } from './demo.ts';
import { meta } from './meta.ts';
import sceneShader from './scene.wgsl';
import waterShader from './water.wgsl';

/** Page elements the shading pass cuts out exactly; further ones still act on the water. */
const MAX_RECTS = 24;

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
  edge: 'f32',
  pixel: 'f32',
  view: 'u32',
  rect_count: 'u32',
  boxes: uniformArray('vec4f', MAX_RECTS),
  radii: uniformArray('vec4f', MAX_RECTS),
});

const RESOLUTIONS = ['192', '256', '384', '512'] as const;
const BOUNDARIES = ['absorb', 'reflect'] as const;
const BACKDROPS = ['pool tiles', 'test card', 'sand'] as const;
const POINTER_MODES = ['hover and press', 'press only', 'hover only', 'off'] as const;
const TOUCH_MODES = ['scrolls the page', 'stirs the water'] as const;
const DEMOS = ['auto', 'always', 'never'] as const;
const VIEWS = ['water', 'mask', 'height', 'normals'] as const;

const BACKDROP_FORMAT: GPUTextureFormat = 'rgba8unorm-srgb';
const RAIN_GAIN = 2.2;
/** Impulse per image height a wall advances, at wake = 1 (see edgeWake). */
const WAKE_GAIN = 10;
/** Most wake drops one frame may queue, so the pointer and the rain keep their share of the 64. */
const MAX_WAKE_DROPS = 24;

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
    obstacles: {
      type: 'boolean',
      default: true,
      label: 'Page elements are solid (off = open water)',
    },
    demo: {
      type: 'select',
      default: 'auto',
      options: DEMOS,
      label: 'Demo cards (auto: only where the page has no obstacles)',
    },
    outset: {
      type: 'number',
      default: 2,
      min: -12,
      max: 32,
      step: 1,
      label: 'Wall margin around each element (px)',
    },
    edge: { type: 'number', default: 0.7, min: 0, max: 1, step: 0.01, label: 'Contact edge' },
    wake: {
      type: 'number',
      default: 1,
      min: 0,
      max: 3,
      step: 0.01,
      label: 'Wake of a moving element',
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
    damping: { type: 'number', default: 0.5, min: 0, max: 4, step: 0.01, label: 'Damping (1 / s)' },
    boundary: { type: 'select', default: 'absorb', options: BOUNDARIES, label: 'Canvas edges' },
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
      default: 1.6,
      min: 0,
      max: 40,
      step: 0.1,
      label: 'Ambient rain (drops / s)',
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
      default: 2.2,
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
    shine: { type: 'number', default: 0.45, min: 0, max: 1, step: 0.01 },
    relief: { type: 'number', default: 0.3, min: 0, max: 1, step: 0.01, label: 'Relief shading' },
    tint: { type: 'color', default: [0.8, 0.95, 1] },
    backdrop: { type: 'select', default: 'pool tiles', options: BACKDROPS },
    view: {
      type: 'select',
      default: 'water',
      options: VIEWS,
      label: 'View (mask: solver mask in red, exact outlines in green)',
    },
  },
  presets: {
    'calm pool': {
      obstacles: true,
      outset: 2,
      edge: 0.7,
      wake: 1,
      waveSpeed: 0.4,
      damping: 0.5,
      boundary: 'absorb',
      dropSize: 0.032,
      dropStrength: 1,
      rainRate: 1.6,
      strength: 2.2,
      depth: 0.35,
      shine: 0.45,
      relief: 0.3,
    },
    'tank (reflecting)': {
      obstacles: true,
      outset: 4,
      edge: 0.8,
      wake: 1.4,
      waveSpeed: 0.65,
      damping: 0.12,
      boundary: 'reflect',
      dropSize: 0.035,
      dropStrength: 1.4,
      rainRate: 0.5,
      strength: 1.8,
      depth: 0.4,
      shine: 0.5,
      relief: 0.3,
    },
    'rain on the page': {
      obstacles: true,
      outset: 0,
      edge: 0.5,
      wake: 0.8,
      waveSpeed: 0.55,
      damping: 1,
      boundary: 'absorb',
      dropSize: 0.02,
      dropStrength: 0.9,
      rainRate: 18,
      strength: 2.2,
      depth: 0.35,
      shine: 0.55,
      relief: 0.3,
    },
    'open water': {
      obstacles: false,
      edge: 0,
      wake: 0,
      waveSpeed: 0.45,
      damping: 0.6,
      boundary: 'absorb',
      dropSize: 0.032,
      dropStrength: 1,
      rainRate: 2.5,
      strength: 2.2,
      depth: 0.35,
      shine: 0.45,
      relief: 0.3,
    },
  },

  async create({ device, format, canvas, params, profiler }) {
    const sceneUniforms = createUniformBuffer(device, SceneParams, 'ripple-dom scene params');
    const waterUniforms = createUniformBuffer(device, WaterParams, 'ripple-dom water params');
    const pointers = trackPointers(canvas);
    const stirrer = createStirrer();
    const random = createRandom(0xd0f);
    // Mutable: the outset and the change threshold are read by the tracker on every update.
    const domOptions: DomRectsOptions = { outsetPx: params.outset, epsilon: 1e-3 };
    const dom = trackDomRects(canvas, domOptions);
    let demo: DemoElements | undefined;

    const scenePipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'ripple-dom/scene.wgsl',
        code: sceneShader.code,
        chunks: { ...sceneShader.chunks, 'scene_params.wgsl': SceneParams.wgsl },
      },
      format: BACKDROP_FORMAT,
    });
    const waterPipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'ripple-dom/water.wgsl',
        code: waterShader.code,
        chunks: { ...waterShader.chunks, 'water_params.wgsl': WaterParams.wgsl },
      },
      format,
    });
    const sceneGroup = device.createBindGroup({
      label: 'ripple-dom scene',
      layout: scenePipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: sceneUniforms.buffer } }],
    });
    const sampler = device.createSampler({
      label: 'ripple-dom sampler',
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
    let maskApplied = false;

    const solver: Heightfield = createHeightfield({
      device,
      kind: 'render',
      resolution: Number(params.resolution),
      width,
      height,
      profiler,
      label: 'ripple-dom',
    });

    function allocateBackdrop() {
      backdrop?.destroy();
      backdrop = device.createTexture({
        label: 'ripple-dom backdrop',
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
      const quiet = 1 - 0.6 * interaction;
      rainDebt += delta * params.rainRate * quiet;
      let drops = 0;
      while (rainDebt >= 1 && drops < 8) {
        rainDebt -= 1;
        drops++;
        solver.addDrop(
          0.02 + random() * 0.96,
          0.02 + random() * 0.96,
          params.dropSize * (0.7 + 0.6 * random()),
          -params.dropStrength * RAIN_GAIN * (0.7 + 0.7 * random()),
        );
      }
      rainDebt = Math.min(rainDebt, 1);

      // A broad, slow swell wandering over the surface, so the water is never dead.
      swellPhase += delta;
      const next: [number, number] = [
        0.5 + 0.44 * Math.sin(swellPhase * 0.19 + 1.3) * Math.cos(swellPhase * 0.07),
        0.5 + 0.4 * Math.sin(swellPhase * 0.27),
      ];
      if (swellPoint && params.rainRate > 0) solver.displace(swellPoint, next, 0.16, 0.3 * quiet);
      swellPoint = next;
    }

    /** Fills the shader's rectangle list from this frame's measured elements. */
    const boxes = new Float32Array(MAX_RECTS * 4);
    const radii = new Float32Array(MAX_RECTS * 4);
    let rectCount = 0;

    return {
      resize(w, h) {
        width = Math.max(1, Math.round(w));
        height = Math.max(1, Math.round(h));
        allocateBackdrop();
        solver.resize(width, height);
        dom.invalidate();
        maskApplied = false;
      },

      frame({ delta }, encoder, target) {
        if (!backdrop || !backdropView) return;
        const aspect = width / height;

        // 1. Read the page first, before anything is written. Skipped when obstacles are off.
        domOptions.outsetPx = params.outset;
        domOptions.epsilon = 0.25 / Math.max(1, solver.grid.width);
        const measured = params.obstacles ? dom.update(delta) : undefined;

        // 2. Demonstration cards where the page has none of its own. `found` counts the demo cards
        // too when they sit under the search root, so they are taken off before judging the page.
        const demoCounted = demo && dom.root?.contains(demo.container) ? demo.count : 0;
        const own = measured ? measured.found - demoCounted : 0;
        const wantDemo =
          params.obstacles && (params.demo === 'always' || (params.demo === 'auto' && own <= 0));
        if (wantDemo && !demo) {
          demo = createDemoElements(canvas);
          if (demo) dom.invalidate();
        } else if (!wantDemo && demo) {
          demo.destroy();
          demo = undefined;
          dom.invalidate();
        }

        // 3. Parameters.
        solver.setResolution(Number(params.resolution));
        const wave = solver.params;
        wave.waveSpeed = params.waveSpeed;
        wave.damping = params.damping;
        wave.boundary = params.boundary as BoundaryMode;

        // 4. The page into the solver, only when it changed.
        rectCount = 0;
        if (measured) {
          if (measured.changed || !maskApplied) {
            solver.obstacles.setRects(measured.rects);
            maskApplied = true;
          }
          for (const o of measured.obstacles) {
            if (rectCount < MAX_RECTS) {
              // The wall may sit a margin outside the element; the water is cut out at the element itself.
              const r = o.exact ?? o.rect;
              boxes.set(
                [
                  (r.x + r.width / 2) * aspect,
                  r.y + r.height / 2,
                  (r.width * aspect) / 2,
                  r.height / 2,
                ],
                rectCount * 4,
              );
              radii[rectCount * 4] = r.radius ?? 0;
              rectCount++;
            }
          }
          // A moving element pushes the water: raised ahead of its advancing edges, dented behind.
          if (params.wake > 0) {
            let queued = 0;
            for (const o of measured.obstacles) {
              if (!o.motion) continue;
              for (const drop of edgeWake(o.box, o.motion, {
                aspect,
                dt: delta,
                radius: params.dropSize,
                gain: WAKE_GAIN * params.wake,
              })) {
                if (queued++ >= MAX_WAKE_DROPS) break;
                solver.addDrop(drop.x, drop.y, drop.radius, drop.strength);
              }
            }
          }
        } else if (maskApplied) {
          solver.obstacles.clear();
          maskApplied = false;
        }

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

        // 5. Input and rain.
        pointers.setCapture(params.touch === 'stirs the water');
        const stir = stirrer.stir(solver, pointers.poll(), aspect, {
          mode: params.pointer as StirMode,
          radius: params.dropSize,
          strength: params.dropStrength,
        });
        interaction += ((stir.active ? 1 : 0) - interaction) * (1 - Math.exp(-delta * 4));
        ambient(delta);

        solver.update(encoder, delta);

        if (!waterGroup || waterGroupGeneration !== solver.generation) {
          waterGroupGeneration = solver.generation;
          waterGroup = device.createBindGroup({
            label: 'ripple-dom water',
            layout: waterPipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: waterUniforms.buffer } },
              { binding: 1, resource: backdropView },
              { binding: 2, resource: sampler },
              { binding: 3, resource: solver.view },
              { binding: 4, resource: sampler },
              { binding: 5, resource: solver.obstacles.view },
            ],
          });
        }
        waterUniforms.set({
          tint: params.tint,
          aspect,
          strength: params.strength,
          ior: 1.333,
          thickness: params.depth,
          dispersion: 0.01,
          fade: 0.05,
          shine: params.shine,
          relief: params.relief,
          height_gain: 1.5,
          edge: params.edge,
          pixel: 1 / height,
          view: Math.max(0, VIEWS.indexOf(params.view as (typeof VIEWS)[number])),
          rect_count: rectCount,
          boxes,
          radii,
        });
        waterUniforms.upload();
        renderFullscreen(encoder, target, waterPipeline, [waterGroup], {
          timestampWrites: profiler.pass('shade'),
        });
      },

      destroy() {
        demo?.destroy();
        demo = undefined;
        dom.destroy();
        pointers.destroy();
        solver.destroy();
        backdrop?.destroy();
        sceneUniforms.destroy();
        waterUniforms.destroy();
      },
    };
  },
});
