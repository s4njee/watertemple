import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { trackPointers } from '../../gpu/pointer.ts';
import {
  createFluid,
  type Fluid,
  type FluidBoundary,
  type FluidPressureSolver,
  type FluidScheme,
} from '../../gpu/sim/fluid.ts';
import {
  createFluidStirrer,
  type FluidStirMode,
  type FluidStroke,
} from '../../gpu/sim/fluid-stir.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import displayShader from './display.wgsl';
import { meta } from './meta.ts';

const DisplayParams = defineUniformStruct('DisplayParams', {
  ground: 'vec3f',
  aspect: 'f32',
  grid: 'vec2f',
  dye_texel: 'vec2f',
  gain: 'f32',
  sheen: 'f32',
  dye_scale: 'f32',
  debug_gain: 'f32',
  vignette: 'f32',
  mode: 'u32',
  view: 'u32',
});

const RESOLUTIONS = ['64', '96', '128', '192', '256'] as const;
const DYE_RESOLUTIONS = ['256', '512', '768', '1024'] as const;
const SCHEMES = ['semi-lagrangian', 'maccormack', 'bfecc'] as const;
const SOLVERS = ['jacobi', 'sor', 'multigrid'] as const;
const BOUNDARIES = ['wall', 'open', 'wrap'] as const;
const PALETTES = ['aurora', 'ember', 'ocean', 'candy', 'sumi', 'rainbow'] as const;
const GROUNDS = ['dark', 'light'] as const;
const POINTER_MODES = ['hover and press', 'press only', 'hover only', 'off'] as const;
const TOUCH_MODES = ['scrolls the page', 'paints'] as const;
const VIEWS = ['dye', 'velocity', 'pressure', 'divergence', 'vorticity'] as const;

type Vec3 = readonly [number, number, number];

// Colours are designed in sRGB with I. Quilez's cosine palettes (colour.wgsl's palette_cosine, here on
// the CPU because the strokes are coloured when they are injected) and converted to linear.
interface Palette {
  a: Vec3;
  b: Vec3;
  c: Vec3;
  d: Vec3;
}

const PALETTE_TABLE: Record<(typeof PALETTES)[number], Palette> = {
  aurora: { a: [0.42, 0.55, 0.62], b: [0.4, 0.38, 0.38], c: [1, 0.8, 0.6], d: [0.55, 0.15, 0.7] },
  ember: { a: [0.62, 0.35, 0.18], b: [0.4, 0.3, 0.2], c: [0.8, 0.9, 0.7], d: [0.0, 0.1, 0.2] },
  ocean: { a: [0.15, 0.5, 0.65], b: [0.2, 0.35, 0.3], c: [0.9, 0.7, 0.5], d: [0.5, 0.3, 0.2] },
  candy: { a: [0.62, 0.45, 0.6], b: [0.38, 0.45, 0.38], c: [1, 1, 1], d: [0.0, 0.33, 0.67] },
  sumi: { a: [0.14, 0.14, 0.24], b: [0.12, 0.1, 0.16], c: [0.6, 0.5, 0.4], d: [0.1, 0.3, 0.5] },
  rainbow: { a: [0.5, 0.5, 0.5], b: [0.5, 0.5, 0.5], c: [1, 1, 1], d: [0.0, 0.33, 0.67] },
};

function srgbToLinear(x: number): number {
  return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
}

/** A colour of the palette, linear RGB, t cycling with period 1. */
export function paletteColour(name: string, t: number): [number, number, number] {
  const p = PALETTE_TABLE[name as (typeof PALETTES)[number]] ?? PALETTE_TABLE.aurora;
  const out: [number, number, number] = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    const srgb = p.a[i]! + p.b[i]! * Math.cos(Math.PI * 2 * (p.c[i]! * t + p.d[i]!));
    out[i] = srgbToLinear(Math.min(1, Math.max(0, srgb)));
  }
  return out;
}

/** The per-channel optical density of a colour: -ln(colour), as fluid.wgsl's fluid_absorbance. */
function absorbance(c: readonly number[]): [number, number, number] {
  return [0, 1, 2].map((i) => -Math.log(Math.min(1, Math.max(0.02, c[i]!)))) as [
    number,
    number,
    number,
  ];
}

/** mulberry32: a small seeded generator, so the idle strokes of a given preset repeat. */
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
    resolution: {
      type: 'select',
      default: '128',
      options: RESOLUTIONS,
      label: 'Velocity cells (short side; re-creates the fluid)',
    },
    dyeResolution: {
      type: 'select',
      default: '512',
      options: DYE_RESOLUTIONS,
      label: 'Dye texels (short side; re-creates the fluid)',
    },
    pressureSolver: {
      type: 'select',
      default: 'jacobi',
      options: SOLVERS,
      label: 'Pressure solver',
    },
    pressureIterations: {
      type: 'number',
      default: 30,
      min: 2,
      max: 120,
      step: 1,
      label: 'Pressure iterations',
    },
    velocityScheme: {
      type: 'select',
      default: 'maccormack',
      options: SCHEMES,
      label: 'Velocity advection',
    },
    dyeScheme: { type: 'select', default: 'maccormack', options: SCHEMES, label: 'Dye advection' },
    vorticity: {
      type: 'number',
      default: 8,
      min: 0,
      max: 60,
      step: 0.5,
      label: 'Vorticity confinement (keep below ~14 x dissipation)',
    },
    velocityDissipation: {
      type: 'number',
      default: 0.6,
      min: 0,
      max: 4,
      step: 0.01,
      label: 'Velocity dissipation (1 / s)',
    },
    dyeDissipation: {
      type: 'number',
      default: 0.45,
      min: 0,
      max: 3,
      step: 0.01,
      label: 'Dye dissipation (1 / s)',
    },
    buoyancy: {
      type: 'number',
      default: 0,
      min: -2,
      max: 2,
      step: 0.01,
      label: 'Buoyancy (dye rises when positive)',
    },
    boundary: { type: 'select', default: 'wall', options: BOUNDARIES, label: 'Edges' },
    radius: {
      type: 'number',
      default: 0.04,
      min: 0.01,
      max: 0.15,
      step: 0.001,
      label: 'Splat radius (canvas heights)',
    },
    force: {
      type: 'number',
      default: 0.6,
      min: 0,
      max: 4,
      step: 0.01,
      label: 'Push (fraction of pointer velocity)',
    },
    dye: { type: 'number', default: 0.9, min: 0, max: 3, step: 0.01, label: 'Dye amount' },
    palette: { type: 'select', default: 'aurora', options: PALETTES },
    ground: {
      type: 'select',
      default: 'dark',
      options: GROUNDS,
      label: 'Ground (dark glows, light is ink)',
    },
    gain: { type: 'number', default: 1.4, min: 0.2, max: 4, step: 0.01, label: 'Dye strength' },
    sheen: { type: 'number', default: 0.4, min: 0, max: 1, step: 0.01, label: 'Liquid sheen' },
    ambient: { type: 'number', default: 0.6, min: 0, max: 1, step: 0.01, label: 'Idle strokes' },
    calm: {
      type: 'number',
      default: 0.8,
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
    view: { type: 'select', default: 'dye', options: VIEWS, label: 'View' },
    debugGain: {
      type: 'number',
      default: 1,
      min: 0.1,
      max: 10,
      step: 0.1,
      label: 'Debug view gain',
    },
    stepsPerFrame: {
      type: 'number',
      default: 0,
      min: 0,
      max: 64,
      step: 1,
      label: 'Steps per frame (0 = by clock; for benchmarks)',
    },
  },
  presets: {
    aurora: {},
    ink: {
      ground: 'light',
      palette: 'sumi',
      gain: 1.6,
      sheen: 0.25,
      dye: 1.1,
      radius: 0.035,
      force: 0.6,
      vorticity: 12,
      velocityDissipation: 1,
      dyeDissipation: 0.12,
      buoyancy: 0,
      ambient: 0.45,
    },
    smoke: {
      ground: 'dark',
      palette: 'ocean',
      gain: 0.9,
      sheen: 0.1,
      dye: 0.7,
      radius: 0.04,
      force: 0.6,
      vorticity: 14,
      velocityDissipation: 0.8,
      dyeDissipation: 0.7,
      buoyancy: 0.9,
      ambient: 0.8,
    },
    'lava lamp': {
      ground: 'dark',
      palette: 'ember',
      gain: 1.6,
      sheen: 0.7,
      dye: 1.6,
      radius: 0.075,
      force: 0.6,
      vorticity: 4,
      velocityDissipation: 0.9,
      dyeDissipation: 0.04,
      buoyancy: 0.35,
      ambient: 1,
      calm: 0.4,
    },
    'rainbow wash': {
      ground: 'light',
      palette: 'rainbow',
      gain: 1.1,
      sheen: 0.3,
      dye: 0.8,
      radius: 0.06,
      force: 0.8,
      vorticity: 8,
      velocityDissipation: 0.5,
      dyeDissipation: 0.25,
      buoyancy: 0,
      ambient: 0.6,
    },
  },

  async create({ device, format, canvas, params, profiler }) {
    const displayUniforms = createUniformBuffer(device, DisplayParams, 'fluid-dye display params');
    const pointers = trackPointers(canvas);
    const stirrer = createFluidStirrer();
    const random = createRandom(0xf1d0);
    const reducedMotion =
      typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

    const sampler = device.createSampler({
      label: 'fluid-dye sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
    // The divergence texture is r32float: not filterable, read with textureLoad, so the layout is
    // spelled out rather than derived.
    const layout = device.createBindGroupLayout({
      label: 'fluid-dye display layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        {
          binding: 4,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'unfilterable-float' },
        },
      ],
    });
    const pipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'fluid-dye/display.wgsl',
        code: displayShader.code,
        chunks: { ...displayShader.chunks, 'display_params.wgsl': DisplayParams.wgsl },
      },
      format,
      layout: [layout],
    });

    let width = Math.max(1, canvas.width || 1);
    let height = Math.max(1, canvas.height || 1);
    /** The aspect the fluid was last sized for; it only follows the canvas when that changes a lot. */
    let simAspect = width / height;

    function cells(shortSide: number, aspect: number): number {
      return Math.round(shortSide * Math.max(aspect, 1 / aspect));
    }

    const fluid: Fluid = createFluid({
      device,
      resolution: cells(Number(params.resolution), simAspect),
      dyeResolution: cells(Number(params.dyeResolution), simAspect),
      width,
      height,
      profiler,
      label: 'fluid-dye',
    });

    const groups = new Map<string, GPUBindGroup>();
    let groupGeneration = fluid.generation;

    function displayGroup(): GPUBindGroup {
      if (groupGeneration !== fluid.generation) {
        groups.clear();
        groupGeneration = fluid.generation;
      }
      const key = String(fluid.dye.slot);
      let group = groups.get(key);
      if (!group) {
        group = device.createBindGroup({
          label: 'fluid-dye display',
          layout,
          entries: [
            { binding: 0, resource: { buffer: displayUniforms.buffer } },
            { binding: 1, resource: fluid.dye.view },
            { binding: 2, resource: sampler },
            { binding: 3, resource: fluid.fieldsView },
            { binding: 4, resource: fluid.divergence.createView() },
          ],
        });
        groups.set(key, group);
      }
      return group;
    }

    /** The dye value of a stroke, per unit amount: colour in rgb (dark ground) or optical density (light), density in a. */
    function dyeValue(index: number, age: number): [number, number, number, number] {
      const t = index * 0.381966 + age * 0.14;
      const colour = paletteColour(params.palette, t);
      const rgb = params.ground === 'light' ? absorbance(colour) : colour;
      return [rgb[0], rgb[1], rgb[2], 1];
    }

    // Idle strokes: two emitters wander the canvas on slow Lissajous paths, pushing and inking as they
    // go, and now and then a drop lands. The wake is proportional to how far an emitter moved this
    // frame, so it looks the same at any frame rate. They ease off while someone is touching the fluid.
    interface Emitter {
      phase: number;
      rate: [number, number];
      offset: number;
      last: [number, number] | undefined;
      colour: number;
    }
    const emitters: Emitter[] = [0, 1].map((i) => ({
      phase: random() * 100,
      rate: [0.52 + 0.23 * i, 0.71 - 0.17 * i],
      offset: i * 2.1,
      last: undefined,
      colour: random() * 10,
    }));
    let dropDebt = 0;
    let interaction = 0;

    // A handful of drops on the first frame, so the page is never empty while the idle strokes warm up.
    let seeded = false;
    function seed(aspect: number) {
      seeded = true;
      for (let i = 0; i < 6; i++) {
        drop(0.12 + 0.76 * random(), 0.15 + 0.7 * random(), i * 3 + random(), 1.2, aspect);
      }
    }

    /** One drop of ink landing at (x, y): a blob of dye and a few outward pushes that curl it open. */
    function drop(x: number, y: number, colour: number, size: number, aspect: number) {
      const at: [number, number] = [x, y];
      const v = dyeValue(Math.floor(colour), colour % 1);
      const amount = params.dye * 1.4 * size;
      fluid.addDye(at, at, params.radius * 1.2 * size, [
        v[0] * amount,
        v[1] * amount,
        v[2] * amount,
        amount,
      ]);
      const kick = 0.5 + 0.5 * random();
      for (let k = 0; k < 5; k++) {
        const angle = random() * Math.PI * 2;
        const p: [number, number] = [
          x + (Math.cos(angle) * params.radius * size) / aspect,
          y + Math.sin(angle) * params.radius * size,
        ];
        fluid.addForce(p, p, params.radius * size, [
          Math.cos(angle) * kick,
          Math.sin(angle) * kick,
        ]);
      }
    }

    function ambient(delta: number, aspect: number) {
      const quiet = 1 - params.calm * interaction;
      const level = params.ambient * quiet * (reducedMotion ? 0.3 : 1);
      const dt = Math.min(Math.max(delta, 1 / 240), 0.1);
      for (const e of emitters) {
        e.phase += dt;
        e.colour += dt * 0.05;
        const at: [number, number] = [
          0.5 + 0.4 * Math.sin(e.phase * e.rate[0] + e.offset) * Math.cos(e.phase * 0.13),
          0.5 + 0.38 * Math.sin(e.phase * e.rate[1] + e.offset * 0.7),
        ];
        const from = e.last;
        e.last = at;
        if (!from || level < 0.01) continue;
        const dx = (at[0] - from[0]) * aspect;
        const dy = at[1] - from[1];
        const speed = Math.min(Math.hypot(dx, dy) / dt, 1.4);
        if (speed < 1e-4) continue;
        const velocity: [number, number] = [
          (dx / Math.hypot(dx, dy)) * speed * 2.2 * level,
          (dy / Math.hypot(dx, dy)) * speed * 2.2 * level,
        ];
        const ramp = params.radius * 0.5;
        fluid.addForce(from, at, params.radius * 1.4, velocity, ramp);
        if (params.dye > 0) {
          const v = dyeValue(0, e.colour * 2.6);
          const amount = params.dye * 0.55 * level;
          fluid.addDye(
            from,
            at,
            params.radius * 0.9,
            [v[0] * amount, v[1] * amount, v[2] * amount, v[3] * amount],
            ramp,
          );
        }
      }

      dropDebt += dt * level * 0.2;
      if (dropDebt >= 1) {
        dropDebt = 0;
        drop(
          0.1 + 0.8 * random(),
          0.12 + 0.76 * random(),
          Math.floor(random() * 20) + random(),
          1,
          aspect,
        );
      }
    }

    function sync() {
      const p = fluid.params;
      p.pressureSolver = params.pressureSolver as FluidPressureSolver;
      p.pressureIterations = params.pressureIterations;
      p.velocityScheme = params.velocityScheme as FluidScheme;
      p.dyeScheme = params.dyeScheme as FluidScheme;
      p.vorticity = params.vorticity;
      p.velocityDissipation = params.velocityDissipation;
      p.dyeDissipation = params.dyeDissipation;
      p.buoyancy = params.buoyancy;
      p.boundary = params.boundary as FluidBoundary;
    }

    return {
      resize(w, h) {
        width = Math.max(1, Math.round(w));
        height = Math.max(1, Math.round(h));
        const aspect = width / height;
        // Dragging a window edge changes the aspect a little at a time; re-creating the grids (which
        // clears the picture) for each pixel would wipe the ink. Follow the canvas only when the shape
        // has really changed; in between the fluid is stretched by a few percent, which nobody sees.
        if (Math.abs(Math.log(aspect / simAspect)) > 0.06) {
          simAspect = aspect;
          fluid.resize(width, height);
        }
      },

      frame({ delta }, encoder, target) {
        sync();
        const aspect = width / height;
        fluid.setResolution(
          cells(Number(params.resolution), simAspect),
          cells(Number(params.dyeResolution), simAspect),
        );

        // Input: a press-drag stirs; touch drags scroll the page unless the visitor asked the ink to take them.
        pointers.setCapture(params.touch === 'paints');
        const stir = stirrer.stir(fluid, pointers.poll(), aspect, delta, {
          mode: params.pointer as FluidStirMode,
          radius: params.radius,
          force: params.force,
          dyeAmount: params.dye,
          colour: (stroke: FluidStroke) => dyeValue(stroke.index, stroke.age),
        });
        const blend = 1 - Math.exp(-delta * (stir.active ? 5 : 0.7));
        interaction += ((stir.active ? 1 : 0) - interaction) * blend;

        if (!seeded) seed(aspect);
        ambient(delta, aspect);

        if (params.stepsPerFrame > 0) fluid.advance(encoder, params.stepsPerFrame);
        else fluid.update(encoder, delta);

        const grid = fluid.grid;
        displayUniforms.set({
          ground: params.ground === 'light' ? [0.93, 0.92, 0.88] : [0.012, 0.014, 0.022],
          aspect,
          grid: [grid.width, grid.height],
          dye_texel: [1 / grid.dyeWidth, 1 / grid.dyeHeight],
          gain: params.gain,
          sheen: params.sheen,
          dye_scale: grid.dyeHeight / 512,
          debug_gain: params.debugGain,
          vignette: params.ground === 'light' ? 0.12 : 0.35,
          mode: params.ground === 'light' ? 1 : 0,
          view: Math.max(0, VIEWS.indexOf(params.view as (typeof VIEWS)[number])),
        });
        displayUniforms.upload();
        renderFullscreen(encoder, target, pipeline, [displayGroup()], {
          timestampWrites: profiler.pass('shade'),
        });
      },

      destroy() {
        pointers.destroy();
        fluid.destroy();
        displayUniforms.destroy();
      },
    };
  },
});
