import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { trackPointers, type TrackedPointer } from '../../gpu/pointer.ts';
import { createFluid, type Fluid } from '../../gpu/sim/fluid.ts';
import { createFluidStirrer, type FluidStirrer } from '../../gpu/sim/fluid-stir.ts';
import { createPigmentLayer, type PigmentLayer } from '../../gpu/sim/pigment.ts';
import { createRenderTarget } from '../../gpu/targets.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import compositeShader from './composite.wgsl?raw';
import displayShader from './display.wgsl?raw';
import { meta } from './meta.ts';
import paperShader from './paper.wgsl?raw';

const DisplayParams = defineUniformStruct('DisplayParams', {
  paper: 'vec3f',
  aspect: 'f32',
  dye_size: 'vec2f',
  tile: 'f32',
  strength: 'f32',
  gran: 'f32',
  grain: 'f32',
  wet_look: 'f32',
  debug_gain: 'f32',
  vignette: 'f32',
  mode: 'u32',
  view: 'u32',
});

type Rgb = [number, number, number];

const srgbToLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
/** Linear RGB from a hex colour (the form colour parameters take). */
export function linear(css: string): Rgb {
  const n = parseInt(css.replace('#', ''), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => srgbToLinear(v / 255)) as Rgb;
}

/** The per-channel optical density of a colour: -ln(colour), as fluid.wgsl's fluid_absorbance. */
export function absorbance(c: readonly number[]): Rgb {
  return [0, 1, 2].map((i) => -Math.log(Math.min(1, Math.max(0.02, c[i]!)))) as Rgb;
}

// Inks are named by the colour a medium layer of them transmits; the pigment carried in the water is
// that colour's optical density, so mixing two inks multiplies their colours (blue over yellow is green).
export const INK_COLOURS: Record<string, string> = {
  indigo: '#31469b',
  sumi: '#262a33',
  crimson: '#c0304a',
  ochre: '#d9a02e',
  viridian: '#2a9478',
  sienna: '#9a5530',
  violet: '#7442a0',
  cobalt: '#2f7ac2',
};
const INK_NAMES = Object.keys(INK_COLOURS);
/** The inks `mixed` cycles through, in an order that puts neighbours across the colour wheel. */
const MIXED = ['indigo', 'ochre', 'crimson', 'viridian', 'cobalt', 'sienna', 'violet'];
const INK_OPTIONS = [...INK_NAMES, 'mixed'] as const;
/** A medium layer's density times this is a full-strength stroke. */
const INK_DENSITY = 0.75;

export const PAPERS: Record<string, string> = {
  cream: '#f3ebdc',
  white: '#f7f6f2',
  rice: '#ebe5d3',
  kraft: '#cdb28a',
  night: '#10141b',
};
const PAPER_OPTIONS = Object.keys(PAPERS) as readonly string[];

const RESOLUTIONS = ['48', '64', '96', '128'] as const;
const DYE_RESOLUTIONS = ['256', '384', '512', '768'] as const;
const SCHEMES = ['maccormack', 'semi-lagrangian'] as const;
const BRUSHES = ['ink', 'clean water'] as const;
const POINTER_MODES = ['press only', 'hover stirs', 'off'] as const;
const TOUCH_MODES = ['scrolls the page', 'paints'] as const;
const VIEWS = ['paint', 'wetness', 'deposited', 'suspended', 'velocity'] as const;

/** Edge length of the baked paper tile, texels. One tile texel is one dye texel. */
const PAPER_SIZE = 256;
/** The water of a brush reaches this much further than its ink. */
const WET_HALO = 1.5;

/** mulberry32: a small seeded generator, so the idle drops of a given preset repeat. */
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

const smoothstep = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

export const effect = defineEffect({
  meta,
  params: {
    resolution: {
      type: 'select',
      default: '64',
      options: RESOLUTIONS,
      label: 'Velocity cells (short side; re-creates the fluid)',
    },
    dyeResolution: {
      type: 'select',
      default: '384',
      options: DYE_RESOLUTIONS,
      label: 'Pigment texels (short side; re-creates the fluid)',
    },
    ink: { type: 'select', default: 'mixed', options: INK_OPTIONS, label: 'Ink' },
    brush: { type: 'select', default: 'ink', options: BRUSHES, label: 'Brush' },
    size: {
      type: 'number',
      default: 0.035,
      min: 0.012,
      max: 0.1,
      step: 0.001,
      label: 'Brush size (canvas heights)',
    },
    load: { type: 'number', default: 1.5, min: 0, max: 2, step: 0.01, label: 'Ink on the brush' },
    water: { type: 'number', default: 1, min: 0, max: 2, step: 0.01, label: 'Water on the brush' },
    push: {
      type: 'number',
      default: 0.1,
      min: 0,
      max: 2,
      step: 0.01,
      label: 'Push (fraction of pointer velocity)',
    },
    submerged: {
      type: 'number',
      default: 0,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Submerged (0 dry paper, 1 ink in a pool)',
    },
    advection: {
      type: 'select',
      default: 'maccormack',
      options: SCHEMES,
      label: 'Pigment advection (MacCormack is crisp, semi-Lagrangian soft and cheaper)',
    },
    diffusion: {
      type: 'number',
      default: 6e-5,
      min: 0,
      max: 3e-4,
      step: 1e-6,
      label: 'Diffusion (heights² / s)',
    },
    feather: {
      type: 'number',
      default: 0.6,
      min: 0,
      max: 1.5,
      step: 0.01,
      label: 'Feathering along paper fibres',
    },
    viscosity: {
      type: 'number',
      default: 0.004,
      min: 0,
      max: 0.03,
      step: 0.0005,
      label: 'Viscosity (heights² / s)',
    },
    flowDecay: {
      type: 'number',
      default: 1.2,
      min: 0.1,
      max: 4,
      step: 0.01,
      label: 'Flow decay (1 / s)',
    },
    swirl: {
      type: 'number',
      default: 0,
      min: 0,
      max: 20,
      step: 0.5,
      label: 'Swirl (vorticity confinement)',
    },
    settling: {
      type: 'number',
      default: 0,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Settling (pigment sinks)',
    },
    absorption: {
      type: 'number',
      default: 0.7,
      min: 0,
      max: 4,
      step: 0.01,
      label: 'Absorption into dry paper (1 / s)',
    },
    wetAbsorption: {
      type: 'number',
      default: 0.2,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Absorption while wet (fraction)',
    },
    drying: {
      type: 'number',
      default: 0.15,
      min: 0,
      max: 0.5,
      step: 0.005,
      label: 'Drying speed (1 / s)',
    },
    dryDrag: {
      type: 'number',
      default: 14,
      min: 0,
      max: 40,
      step: 0.5,
      label: 'Dry paper holds the flow (1 / s)',
    },
    maxSubsteps: {
      type: 'number',
      default: 2,
      min: 1,
      max: 6,
      step: 1,
      label: 'Diffusion sub-steps at most (cost)',
    },
    ragged: {
      type: 'number',
      default: 0.5,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Ragged wet edges (uneven paper)',
    },
    wicking: {
      type: 'number',
      default: 0.03,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Wicking: diffusion on dry paper (fraction of wet)',
    },
    wetSpread: {
      type: 'number',
      default: 1e-5,
      min: 0,
      max: 1e-4,
      step: 1e-6,
      label: 'Water creep on paper (heights² / s)',
    },
    edgeDarkening: {
      type: 'number',
      default: 0.4,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Edge darkening',
    },
    granulation: { type: 'number', default: 0.5, min: 0, max: 1, step: 0.01, label: 'Granulation' },
    lift: {
      type: 'number',
      default: 0.25,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Back-runs (re-wetting lifts stain, 1 / s)',
    },
    fade: {
      type: 'number',
      default: 0.012,
      min: 0,
      max: 0.2,
      step: 0.001,
      label: 'Washing out (1 / s)',
    },
    paper: { type: 'select', default: 'cream', options: PAPER_OPTIONS, label: 'Paper' },
    paperGrain: {
      type: 'number',
      default: 0.6,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Paper grain and fibres',
    },
    wetLook: { type: 'number', default: 0.6, min: 0, max: 1, step: 0.01, label: 'Wet sheen' },
    strength: {
      type: 'number',
      default: 1,
      min: 0.3,
      max: 2.5,
      step: 0.01,
      label: 'Pigment strength',
    },
    ambient: {
      type: 'number',
      default: 0.5,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Idle drops and strokes',
    },
    pointer: { type: 'select', default: 'press only', options: POINTER_MODES, label: 'Pointer' },
    touch: {
      type: 'select',
      default: 'scrolls the page',
      options: TOUCH_MODES,
      label: 'Touch drags',
    },
    view: { type: 'select', default: 'paint', options: VIEWS, label: 'View' },
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
    'wet-on-wet watercolour': {},
    'ink drop in water': {
      submerged: 1,
      ink: 'indigo',
      paper: 'white',
      size: 0.03,
      load: 2,
      water: 0,
      push: 0.7,
      diffusion: 1.5e-5,
      feather: 0,
      viscosity: 0.001,
      flowDecay: 0.6,
      swirl: 10,
      settling: 0.7,
      absorption: 0.05,
      wetAbsorption: 0.3,
      drying: 0,
      edgeDarkening: 0,
      granulation: 0,
      lift: 0,
      fade: 0.05,
      paperGrain: 0.1,
      wetLook: 0.2,
      ambient: 0.5,
    },
    'sumi-e': {
      ink: 'sumi',
      paper: 'rice',
      size: 0.026,
      load: 1.3,
      water: 0.25,
      push: 0.05,
      diffusion: 2.5e-4,
      feather: 1.5,
      wicking: 0.6,
      maxSubsteps: 3,
      viscosity: 0.002,
      flowDecay: 3,
      absorption: 1.1,
      wetAbsorption: 0.6,
      drying: 0.4,
      dryDrag: 30,
      wetSpread: 2e-5,
      edgeDarkening: 0.1,
      granulation: 0.2,
      lift: 0.1,
      paperGrain: 0.9,
      wetLook: 0.2,
      fade: 0.01,
      ambient: 0.4,
    },
    'granulating wash': {
      ink: 'mixed',
      paper: 'white',
      size: 0.05,
      load: 1.2,
      water: 1.6,
      push: 0.1,
      diffusion: 1e-4,
      feather: 0.3,
      absorption: 0.5,
      drying: 0.06,
      edgeDarkening: 0.8,
      granulation: 1,
      lift: 0.35,
      paperGrain: 1,
      fade: 0.008,
      ambient: 0.5,
    },
    'night ink': {
      paper: 'night',
      ink: 'mixed',
      size: 0.035,
      load: 0.8,
      water: 0.7,
      push: 0.3,
      submerged: 0.4,
      diffusion: 6e-5,
      feather: 0.4,
      viscosity: 0.003,
      swirl: 6,
      absorption: 0.3,
      drying: 0.04,
      edgeDarkening: 0.15,
      granulation: 0.2,
      fade: 0.05,
      paperGrain: 0.3,
      wetLook: 0.1,
      strength: 1,
      ambient: 0.6,
    },
  },

  async create({ device, format, canvas, params, profiler }) {
    const displayUniforms = createUniformBuffer(device, DisplayParams, 'ink-water display params');
    const pointers = trackPointers(canvas);
    const random = createRandom(0x1a7e);
    const reducedMotion =
      typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

    const clampSampler = device.createSampler({
      label: 'ink-water sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
    const repeatSampler = device.createSampler({
      label: 'ink-water paper sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
    });
    const filtered = (binding: number): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: GPUShaderStage.FRAGMENT,
      texture: { sampleType: 'float' },
    });
    const layout = device.createBindGroupLayout({
      label: 'ink-water display layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        filtered(1),
        filtered(2),
        filtered(3),
        filtered(4),
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });
    const displayPipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'ink-water/display.wgsl',
        code: displayShader,
        chunks: { 'display_params.wgsl': DisplayParams.wgsl, 'composite.wgsl': compositeShader },
      },
      format,
      layout: [layout],
    });
    const paperPipeline = createFullscreenPipeline(device, {
      fragment: { label: 'ink-water/paper.wgsl', code: paperShader },
      format: 'rgba8unorm',
    });
    const paper = createRenderTarget(device, {
      label: 'ink-water paper',
      format: 'rgba8unorm',
      width: PAPER_SIZE,
      height: PAPER_SIZE,
    });
    let paperBaked = false;

    let width = Math.max(1, canvas.width || 1);
    let height = Math.max(1, canvas.height || 1);
    /** The aspect the fluid was last sized for; it only follows the canvas when that changes a lot. */
    let simAspect = width / height;

    const cells = (shortSide: number, aspect: number) =>
      Math.round(shortSide * Math.max(aspect, 1 / aspect));

    const fluid: Fluid = createFluid({
      device,
      resolution: cells(Number(params.resolution), simAspect),
      dyeResolution: cells(Number(params.dyeResolution), simAspect),
      width,
      height,
      profiler,
      label: 'ink-water',
      params: { dyeDissipation: 0, pressureIterations: 24, viscosityIterations: 8 },
    });
    const pigment: PigmentLayer = createPigmentLayer({
      device,
      width: fluid.grid.dyeWidth,
      height: fluid.grid.dyeHeight,
      profiler,
      label: 'ink-water',
    });
    pigment.setPaper(paper.view);
    pigment.attach(fluid);
    let fluidGeneration = fluid.generation;
    let needsReset = true;

    // --- Bind groups ----------------------------------------------------------------------------------

    const groups = new Map<string, GPUBindGroup>();
    let groupGeneration = `${fluid.generation}/${pigment.generation}`;

    function displayGroup(): GPUBindGroup {
      const generation = `${fluid.generation}/${pigment.generation}`;
      if (generation !== groupGeneration) {
        groups.clear();
        groupGeneration = generation;
      }
      const key = `${fluid.dye.slot}|${pigment.stain.slot}`;
      let group = groups.get(key);
      if (!group) {
        group = device.createBindGroup({
          label: 'ink-water display',
          layout,
          entries: [
            { binding: 0, resource: { buffer: displayUniforms.buffer } },
            { binding: 1, resource: fluid.dye.view },
            { binding: 2, resource: pigment.stain.view },
            { binding: 3, resource: paper.view },
            { binding: 4, resource: fluid.fieldsView },
            { binding: 5, resource: clampSampler },
            { binding: 6, resource: repeatSampler },
          ],
        });
        groups.set(key, group);
      }
      return group;
    }

    // --- Painting -------------------------------------------------------------------------------------

    /** The ink of a stroke: a name, or `mixed`, which steps through the palette per stroke. */
    function inkDensity(seed: number): Rgb {
      const name =
        params.ink === 'mixed'
          ? MIXED[Math.floor(((seed * 0.381966) % 1) * MIXED.length) % MIXED.length]!
          : params.ink;
      const colour = linear(INK_COLOURS[name] ?? INK_COLOURS.indigo!);
      return absorbance(colour);
    }

    /** Dye value (rgba) of one unit of a stroke's ink: optical density, a = its mean. */
    function dyeValue(seed: number, amount: number): [number, number, number, number] {
      const od = inkDensity(seed);
      const k = INK_DENSITY * amount;
      return [od[0] * k, od[1] * k, od[2] * k, ((od[0] + od[1] + od[2]) / 3) * k];
    }

    let waterAmount = 1;
    /** Wetness of a stroke's capsule: above 1, so the film has a flat top and a firm edge. */
    const STROKE_WATER = 1.8;
    /**
     * What the stirrer paints into: its dye splats also wet the paper (along a moving segment only: the
     * point splats it drips while the pointer is still would put a full blob of water on every frame
     * and scallop the edge of the film, which the edge drift then turns into stripes), and carry no ink
     * for clean water.
     */
    const brushTarget: Pick<Fluid, 'addForce' | 'addDye'> = {
      addForce: (from, to, radius, velocity, ramp) =>
        fluid.addForce(from, to, radius, velocity, ramp),
      addDye: (from, to, radius, value, ramp) => {
        if (value[0] + value[1] + value[2] > 1e-6) fluid.addDye(from, to, radius, value, ramp);
        if (from[0] !== to[0] || from[1] !== to[1]) {
          pigment.addWater(from, to, radius * WET_HALO, waterAmount * STROKE_WATER, ramp);
        }
        return true;
      },
    };

    const stirrers = new Map<number, FluidStirrer>();
    const strokeSeeds = new Map<number, number>();
    let strokeCounter = Math.floor(random() * 50);
    let interaction = 0;

    /** One drop of ink landing at (x, y): a blot of ink in a patch of water and a few outward pushes. */
    function drop(x: number, y: number, seed: number, scale: number, aspect: number) {
      const at: [number, number] = [x, y];
      const radius = params.size * scale;
      if (params.brush === 'ink' || scale > 1.05) {
        const value = dyeValue(seed, params.load * 1.6 * scale);
        fluid.addDye(at, at, radius * 1.2, value);
      }
      if (params.water > 0 || params.submerged < 1) {
        pigment.addWater(at, at, radius * 2.2, Math.max(params.water, 0.5) * 1.5);
      }
      if (params.push > 0) {
        const kick = (0.35 + 0.35 * random()) * Math.min(1.5, 0.5 + params.push);
        for (let k = 0; k < 5; k++) {
          const angle = random() * Math.PI * 2;
          const p: [number, number] = [
            x + (Math.cos(angle) * radius) / aspect,
            y + Math.sin(angle) * radius,
          ];
          fluid.addForce(p, p, radius, [Math.cos(angle) * kick, Math.sin(angle) * kick]);
        }
      }
    }

    function paint(delta: number, aspect: number): boolean {
      pointers.setCapture(params.touch === 'paints');
      const active = pointers.poll();
      let touched = false;
      const seen = new Set<number>();
      for (const pointer of active) {
        seen.add(pointer.id);
        let stirrer = stirrers.get(pointer.id);
        if (!stirrer) {
          stirrer = createFluidStirrer();
          stirrers.set(pointer.id, stirrer);
        }
        if (pointer.justPressed) strokeSeeds.set(pointer.id, strokeCounter++);
        const seed = strokeSeeds.get(pointer.id) ?? 0;
        const result = stirrers
          .get(pointer.id)!
          .stir(
            brushTarget,
            [pointer as TrackedPointer],
            aspect,
            delta,
            strokeOptions(pointer, seed),
          );
        touched ||= result.active;
        // A brush held still keeps wetting its own patch.
        if (pointer.down && result.speed < 0.05 && params.pointer !== 'off') {
          pigment.addWater(
            [pointer.x, pointer.y],
            [pointer.x, pointer.y],
            params.size * WET_HALO,
            waterAmount * 2 * Math.min(delta, 0.05),
          );
        }

        // A press is a dab of the brush: a drop that blooms on its own (a tap).
        if (pointer.justPressed && params.pointer !== 'off') {
          const water = params.brush !== 'ink';
          if (water) {
            pigment.addWater(
              [pointer.x, pointer.y],
              [pointer.x, pointer.y],
              params.size * WET_HALO * 1.2,
              params.water * 1.6 + 0.4,
            );
          } else {
            drop(pointer.x, pointer.y, seed, 0.6 + 0.3 * Math.min(1, pointer.pressure * 2), aspect);
          }
        }
      }
      for (const id of [...stirrers.keys()]) {
        if (!seen.has(id)) {
          stirrers.delete(id);
          strokeSeeds.delete(id);
        }
      }
      return touched;
    }

    function strokeOptions(pointer: TrackedPointer, seed: number) {
      // Pressure (a pen's; a mouse button reads 0.5) widens the stroke, speed thins and dries it.
      const pressure = 0.55 + 0.9 * Math.min(1, Math.max(0, pointer.pressure));
      const quick = smoothstep(0.3, 2.5, pointer.speed);
      const radius = params.size * pressure * (1 - 0.3 * quick);
      waterAmount = params.water * (1 - 0.35 * quick);
      const water = params.brush !== 'ink';
      return {
        mode: (params.pointer === 'hover stirs'
          ? 'hover and press'
          : params.pointer === 'off'
            ? 'off'
            : 'press only') as 'hover and press' | 'press only' | 'off',
        radius,
        force: params.push,
        dyeRadius: radius,
        // Ink per unit length of stroke; clean water paints "ink" that has no colour.
        dyeAmount: water ? 1 : Math.max(params.load, 1e-3) * (1 - 0.25 * quick),
        colour: (): [number, number, number, number] => (water ? [0, 0, 0, 0] : dyeValue(seed, 1)),
        hoverForce: 0.3,
        hoverDye: 0,
        puff: 0.35 * Math.min(1, params.push * 2),
        maxSpeed: 3,
      };
    }

    // --- Idle life ------------------------------------------------------------------------------------

    interface AutoStroke {
      p0: [number, number];
      p1: [number, number];
      p2: [number, number];
      t: number;
      duration: number;
      seed: number;
      last: [number, number] | undefined;
    }
    let autoStroke: AutoStroke | undefined;
    let dropTimer = 0.4;
    let strokeTimer = 6 + random() * 4;
    let seeded = false;

    function seed(aspect: number) {
      seeded = true;
      const count = 4;
      for (let i = 0; i < count; i++) {
        drop(
          0.14 + 0.72 * random(),
          0.2 + 0.6 * random(),
          strokeCounter++,
          0.9 + 0.5 * random(),
          aspect,
        );
      }
    }

    function startStroke() {
      const horizontal = random() < 0.65;
      const a: [number, number] = horizontal
        ? [0.12 + 0.25 * random(), 0.25 + 0.5 * random()]
        : [0.2 + 0.6 * random(), 0.15 + 0.2 * random()];
      const b: [number, number] = horizontal
        ? [a[0] + 0.3 + 0.25 * random(), a[1] + (random() - 0.5) * 0.3]
        : [a[0] + (random() - 0.5) * 0.25, a[1] + 0.3 + 0.2 * random()];
      const mid: [number, number] = [
        (a[0] + b[0]) / 2 + (random() - 0.5) * 0.18,
        (a[1] + b[1]) / 2 + (random() - 0.5) * 0.18,
      ];
      autoStroke = {
        p0: a,
        p1: mid,
        p2: [Math.min(0.95, b[0]), Math.min(0.9, Math.max(0.1, b[1]))],
        t: 0,
        duration: 1.1 + random() * 0.8,
        seed: strokeCounter++,
        last: undefined,
      };
    }

    function ambient(delta: number, aspect: number) {
      const quiet = 1 - 0.85 * interaction;
      const level = params.ambient * quiet * (reducedMotion ? 0.3 : 1);
      const dt = Math.min(Math.max(delta, 1 / 240), 0.1);
      if (level < 0.02) return;

      dropTimer -= dt * level;
      if (dropTimer <= 0) {
        dropTimer = 2.2 + random() * 3.5;
        drop(
          0.1 + 0.8 * random(),
          0.12 + 0.76 * random(),
          strokeCounter++,
          0.75 + 0.7 * random(),
          aspect,
        );
      }
      strokeTimer -= dt * level;
      if (!autoStroke && strokeTimer <= 0) {
        strokeTimer = 7 + random() * 8;
        startStroke();
      }
      if (autoStroke) {
        const s = autoStroke;
        s.t = Math.min(1, s.t + dt / s.duration);
        const u = 1 - s.t;
        const at: [number, number] = [
          u * u * s.p0[0] + 2 * u * s.t * s.p1[0] + s.t * s.t * s.p2[0],
          u * u * s.p0[1] + 2 * u * s.t * s.p1[1] + s.t * s.t * s.p2[1],
        ];
        const from = s.last;
        s.last = at;
        // A brush stroke is pressed in the middle and lifts at both ends.
        const taper = Math.sqrt(Math.sin(Math.PI * Math.min(1, Math.max(0.02, s.t))));
        if (from) {
          const dx = (at[0] - from[0]) * aspect;
          const dy = at[1] - from[1];
          const len = Math.hypot(dx, dy);
          if (len > 1e-6) {
            const radius = params.size * (0.6 + 0.6 * taper);
            const ramp = radius * 0.5;
            const speed = Math.min(len / dt, 1.5);
            const velocity: [number, number] = [
              (dx / len) * speed * params.push,
              (dy / len) * speed * params.push,
            ];
            fluid.addForce(from, at, radius * 1.4, velocity, ramp);
            if (params.brush === 'ink') {
              const value = dyeValue(s.seed, params.load * 0.9 * taper);
              fluid.addDye(from, at, radius, value, ramp);
            }
            pigment.addWater(from, at, radius * WET_HALO, params.water * STROKE_WATER * 0.8, ramp);
          }
        }
        if (s.t >= 1) autoStroke = undefined;
      }

      // Slow currents in a pool: a soft push wandering over the water.
      if (params.submerged > 0.5 && params.push > 0) {
        const phase = performance.now() * 0.001;
        const from: [number, number] = [
          0.5 + 0.35 * Math.sin(phase * 0.11),
          0.5 + 0.3 * Math.sin(phase * 0.17 + 1),
        ];
        const to: [number, number] = [
          0.5 + 0.35 * Math.sin(phase * 0.11 + 0.15),
          0.5 + 0.3 * Math.sin(phase * 0.17 + 1.2),
        ];
        fluid.addForce(from, to, params.size * 2.5, [0.04 * level, 0.03 * level], params.size);
      }
    }

    // --- Parameters ---------------------------------------------------------------------------------

    function sync() {
      const f = fluid.params;
      f.viscosity = params.viscosity;
      f.velocityDissipation = params.flowDecay;
      f.vorticity = params.swirl;
      f.buoyancy = -0.3 * params.settling;
      f.dyeDissipation = 0;
      f.dyeScheme = params.advection === 'semi-lagrangian' ? 'semi-lagrangian' : 'maccormack';
      const p = pigment.params;
      p.diffusion = params.diffusion;
      p.feather = params.feather;
      p.dryDiffusion = params.wicking;
      p.maxSubsteps = params.maxSubsteps;
      p.ragged = params.ragged;
      p.edgeSpeed = params.edgeDarkening * 0.03;
      p.absorption = params.absorption;
      p.wetAbsorption = params.wetAbsorption;
      p.granulation = params.granulation;
      p.lift = params.lift;
      p.drying = params.drying;
      p.wetSpread = params.wetSpread;
      p.wetFloor = params.submerged;
      p.dryDrag = params.dryDrag;
      p.fade = params.fade;
    }

    return {
      resize(w, h) {
        width = Math.max(1, Math.round(w));
        height = Math.max(1, Math.round(h));
        const aspect = width / height;
        // Dragging a window edge changes the aspect a little at a time; re-creating the grids (which
        // clears the picture) for each pixel would wipe the paint. Follow the canvas only when the
        // shape has really changed; in between the picture is stretched by a few percent.
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
        if (fluid.generation !== fluidGeneration) {
          fluidGeneration = fluid.generation;
          pigment.resize(fluid.grid.dyeWidth, fluid.grid.dyeHeight);
          for (const stirrer of stirrers.values()) stirrer.reset();
          needsReset = true;
          seeded = false;
        }
        if (needsReset) {
          needsReset = false;
          pigment.reset(encoder);
        }
        if (!paperBaked) {
          paperBaked = true;
          renderFullscreen(encoder, paper.view, paperPipeline, [], {
            timestampWrites: profiler.pass('bake paper (once)'),
          });
        }

        const touched = paint(delta, aspect);
        const blend = 1 - Math.exp(-delta * (touched ? 5 : 0.5));
        interaction += ((touched ? 1 : 0) - interaction) * blend;
        if (!seeded) seed(aspect);
        ambient(delta, aspect);

        pigment.beginFrame(encoder);
        if (params.stepsPerFrame > 0) fluid.advance(encoder, params.stepsPerFrame);
        else fluid.update(encoder, delta);

        const luminous = params.paper === 'night';
        displayUniforms.set({
          paper: linear(PAPERS[params.paper] ?? PAPERS.cream!),
          aspect,
          dye_size: [fluid.grid.dyeWidth, fluid.grid.dyeHeight],
          tile: PAPER_SIZE,
          strength: params.strength,
          gran: params.granulation,
          grain: params.paperGrain,
          wet_look: params.wetLook,
          debug_gain: params.debugGain,
          vignette: luminous ? 0.3 : 0.1,
          mode: luminous ? 1 : 0,
          view: Math.max(0, VIEWS.indexOf(params.view as (typeof VIEWS)[number])),
        });
        displayUniforms.upload();
        renderFullscreen(encoder, target, displayPipeline, [displayGroup()], {
          timestampWrites: profiler.pass('shade'),
        });
      },

      destroy() {
        pointers.destroy();
        pigment.detach(fluid);
        pigment.destroy();
        fluid.destroy();
        paper.destroy();
        displayUniforms.destroy();
      },
    };
  },
});
