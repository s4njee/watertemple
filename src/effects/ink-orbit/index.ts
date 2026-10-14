import { trackDomRects, type DomRectsOptions, type Obstacle } from '../../gpu/dom-rects.ts';
import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { trackPointers, type TrackedPointer } from '../../gpu/pointer.ts';
import { createShader } from '../../gpu/shader.ts';
import { createFluid, type Fluid, type FluidBoundary } from '../../gpu/sim/fluid.ts';
import {
  createFluidStirrer,
  type FluidStirMode,
  type FluidStroke,
} from '../../gpu/sim/fluid-stir.ts';
import { createUniformBuffer, defineUniformStruct, uniformArray } from '../../gpu/uniforms.ts';
// The demonstration cards are the ones ripple-dom lays over a bare canvas: plain elements marked
// `data-wt-obstacle`, one of them draggable.
import { createDemoElements, type DemoElements } from '../ripple-dom/demo.ts';
import { defineEffect, type EffectMeta } from '../types.ts';
import { createStrokeLayer } from '../orbit-variants/strokes.ts';
import { MATERIAL_PALETTES } from '../orbit-variants/catalog.ts';
import { biolumePulse, stepSpark, type EmberSpark } from '../orbit-variants/dynamics.ts';
import displayShader from './display.wgsl?raw';
import { createHead, stepHead, type Head, type HeadBox, type HeadField } from './heads.ts';
import { meta } from './meta.ts';
import orbitShader from './orbit.wgsl?raw';

/** Elements the current pass knows about. Further ones still steer the heads and are still solid. */
const MAX_LANES = 24;
/** The attribute an element uses to choose how heads swing round it: `cw`, `ccw`, or `off` to only push them away. */
export const ORBIT_ATTRIBUTE = 'data-wt-orbit';
/** Heads at most; each lays one dye and one momentum splat per frame, and the solver takes 256 of each. */
const MAX_HEADS = 48;
/** How firmly the current holds the fluid, 1 / s. Gentle: the heads' wakes should win. */
const CURRENT_GRIP = 3;
const MAX_SPARKS = 32;

const OrbitParams = defineUniformStruct('OrbitParams', {
  grid: 'vec2u',
  cell: 'f32',
  dt: 'f32',
  speed: 'f32',
  lane: 'f32',
  grip: 'f32',
  count: 'u32',
  closed: 'u32',
  /** Centre (xy) and half-size (zw), image heights. */
  boxes: uniformArray('vec4f', MAX_LANES),
  /** Corner radius, signed spin (its size scales the speed), then the element's own velocity. */
  extras: uniformArray('vec4f', MAX_LANES),
});

const DisplayParams = defineUniformStruct('DisplayParams', {
  ground: 'vec3f',
  aspect: 'f32',
  dye_texel: 'vec2f',
  gain: 'f32',
  sheen: 'f32',
  dye_scale: 'f32',
  vignette: 'f32',
  mode: 'u32',
  view: 'u32',
  /** Brightness of the picture as a multiple of SDR white: 1 except on an HDR canvas. */
  light: 'f32',
  /** One drawing-buffer pixel in image heights, for the edge of the cut-outs. */
  pixel: 'f32',
  /** Simulation time, for variations with animated light folds. */
  time: 'f32',
  style: 'u32',
  head_count: 'u32',
  spark_count: 'u32',
  orbit_shell: 'f32',
  /** Head xy, radius and transient brightness; velocity, palette phase and age. */
  heads: uniformArray('vec4f', MAX_HEADS),
  head_motion: uniformArray('vec4f', MAX_HEADS),
  head_colours: uniformArray('vec4f', MAX_HEADS),
  /** Detached ember xy, radius and remaining heat. */
  sparks: uniformArray('vec4f', MAX_SPARKS),
  /** Elements to cut out as clean paper: centre and half-size, then corner radius; 0 of them when `cutout` is off. */
  rect_count: 'u32',
  rects: uniformArray('vec4f', MAX_LANES),
  radii: uniformArray('vec4f', MAX_LANES),
});

const RESOLUTIONS = ['96', '128', '192', '256'] as const;
const DYE_RESOLUTIONS = ['256', '512', '768', '1024'] as const;
const PALETTES = [
  'rgb cycle',
  'aurora',
  'neon',
  'dichroic',
  'biolume',
  'chrome',
  'embers',
  'eclipse',
  'primary',
  'rainbow',
  'pastel',
  'sunset',
  'lagoon',
  'sumi',
] as const;
const GROUNDS = ['light', 'dark'] as const;
const SPINS = ['alternate', 'clockwise', 'counter-clockwise'] as const;
const BOUNDARIES = ['wall', 'open', 'wrap'] as const;
const POINTER_MODES = ['hover and press', 'press only', 'hover only', 'off'] as const;
const TOUCH_MODES = ['scrolls the page', 'paints'] as const;
const DEMOS = ['auto', 'always', 'never'] as const;
const VIEWS = ['ink', 'flow'] as const;

type Vec3 = readonly [number, number, number];

// Two kinds of palette, both designed in sRGB.
//
// SWATCHES: a short list of inks, and a trail is one of them from head to tail. `primary` is the four
// colours of a paint box. Each lets at least one channel of the paper through untouched and takes the
// others almost entirely, which is what makes it read as a primary and not a tint, and is the most an
// HDR canvas can show: the channel that passes is as bright as the paper, several times SDR white.
//
// COSINE (I. Quilez): colour(t) = a + b * cos(2 pi (c t + d)), and a trail drifts through it along its
// length. `rainbow` is the full hue circle, lifted a little off pure primaries so that two inks
// crossing make a third colour instead of black.
type Palette = { swatches: readonly Vec3[] } | { a: Vec3; b: Vec3; c: Vec3; d: Vec3 };

const PALETTE_TABLE: Record<
  'primary' | 'rainbow' | 'pastel' | 'sunset' | 'lagoon' | 'sumi',
  Palette
> = {
  primary: {
    swatches: [
      [1, 0.07, 0.05], // red
      [1, 0.86, 0], // yellow
      [0.04, 0.3, 1], // blue
      [0, 0.9, 0.24], // green
    ],
  },
  rainbow: { a: [0.56, 0.56, 0.56], b: [0.44, 0.44, 0.44], c: [1, 1, 1], d: [0.0, 0.33, 0.67] },
  pastel: { a: [0.74, 0.72, 0.76], b: [0.24, 0.24, 0.22], c: [1, 1, 1], d: [0.0, 0.33, 0.67] },
  sunset: { a: [0.72, 0.42, 0.42], b: [0.28, 0.32, 0.3], c: [1, 1, 1], d: [0.0, 0.12, 0.3] },
  lagoon: { a: [0.3, 0.6, 0.66], b: [0.28, 0.3, 0.3], c: [1, 1, 1], d: [0.55, 0.2, 0.0] },
  sumi: { a: [0.16, 0.16, 0.24], b: [0.12, 0.1, 0.14], c: [1, 1, 1], d: [0.1, 0.3, 0.5] },
};

function srgbToLinear(x: number): number {
  return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
}

/**
 * A colour of the palette, linear RGB. A cosine palette is read at `t` (period 1); a swatch palette
 * ignores it and gives swatch number `step`, wrapping. RGB cycle reads `t` in seconds and
 * smoothly fades red, green, blue over six seconds, ignoring the head or pointer's swatch number.
 */
export function paletteColour(name: string, t: number, step = 0): [number, number, number] {
  const material = MATERIAL_PALETTES[name];
  if (material) {
    const phase = (((t % 1) + 1) % 1) * material.length;
    const index = Math.floor(phase);
    const fraction = phase - index;
    const blend = fraction * fraction * (3 - 2 * fraction);
    const from = material[index]!;
    const to = material[(index + 1) % material.length]!;
    return [0, 1, 2].map((i) => from[i]! * (1 - blend) + to[i]! * blend) as [
      number,
      number,
      number,
    ];
  }
  if (name === 'aurora') {
    // Linear-light emerald, cyan and violet. Each head drifts through this over about 20 seconds.
    const stops: readonly Vec3[] = [
      [0.015, 0.85, 0.23],
      [0.02, 0.5, 0.9],
      [0.42, 0.035, 0.72],
    ];
    const phase = (((t % 1) + 1) % 1) * stops.length;
    const index = Math.floor(phase);
    const fraction = phase - index;
    const blend = fraction * fraction * (3 - 2 * fraction);
    const from = stops[index]!;
    const to = stops[(index + 1) % stops.length]!;
    return [0, 1, 2].map((i) => from[i]! * (1 - blend) + to[i]! * blend) as [
      number,
      number,
      number,
    ];
  }
  if (name === 'rgb cycle') {
    const phase = (((t / 2) % 3) + 3) % 3;
    const channel = Math.floor(phase);
    const next = (channel + 1) % 3;
    const fraction = phase - channel;
    // Ease at each primary so the colour and its rate of change are continuous across the loop.
    const blend = fraction * fraction * (3 - 2 * fraction);
    const colour: [number, number, number] = [0, 0, 0];
    // Cross-fade emitted light in linear RGB so transitions keep their energy.
    colour[channel] = 1 - blend;
    colour[next] = blend;
    return colour;
  }
  const p = PALETTE_TABLE[name as keyof typeof PALETTE_TABLE] ?? PALETTE_TABLE.primary;
  if ('swatches' in p) {
    const count = p.swatches.length;
    const swatch = p.swatches[((Math.floor(step) % count) + count) % count]!;
    return [srgbToLinear(swatch[0]), srgbToLinear(swatch[1]), srgbToLinear(swatch[2])];
  }
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

/** mulberry32: a small seeded generator, so the drops of a given page repeat. */
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

/** Share the simulation while allowing a variation to supply its own lighting pass. */
export function createOrbitEffect(
  options: {
    meta?: EffectMeta;
    displayShader?: string;
    darkGround?: Vec3;
    /** 1 neon, 2 glass, 3 bioluminescence, 4 chrome, 5 embers, 6 eclipse. */
    style?: number;
  } = {},
) {
  return defineEffect({
    meta: options.meta ?? meta,
    // On an HDR display the paper can be brighter than the page's own white; ctx.hdr says if it is.
    hdr: true,
    params: {
      heads: {
        type: 'number',
        default: 20,
        min: 0,
        max: MAX_HEADS,
        step: 1,
        label: 'Trail heads',
      },
      pull: {
        type: 'number',
        default: 1,
        min: 0,
        max: 2,
        step: 0.01,
        label: 'Pull of the elements (0 = heads only drift)',
      },
      swirl: {
        type: 'number',
        default: 0.48,
        min: 0,
        max: 1.5,
        step: 0.01,
        label: 'Swing round an element',
      },
      shell: {
        type: 'number',
        default: 0.045,
        min: 0.008,
        max: 0.12,
        step: 0.001,
        label: 'Orbit distance from the element (canvas heights)',
      },
      wander: { type: 'number', default: 1, min: 0, max: 4, step: 0.01, label: 'Drift' },
      chaos: {
        type: 'number',
        default: 1,
        min: 0,
        max: 3,
        step: 0.01,
        label: 'Restlessness (how often a head lets go)',
      },
      headSpeed: {
        type: 'number',
        default: 0.5,
        min: 0.05,
        max: 1.2,
        step: 0.01,
        label: 'Top speed of a head (canvas heights / s)',
      },
      wake: {
        type: 'number',
        default: 0.45,
        min: 0,
        max: 2,
        step: 0.01,
        label: "Wake (fraction of a head's velocity the fluid takes)",
      },
      stir: {
        type: 'number',
        default: 1,
        min: 0,
        max: 3,
        step: 0.01,
        label: 'Stirring (a swinging jet in each stroke that folds the inks together)',
      },
      current: {
        type: 'number',
        default: 0.05,
        min: 0,
        max: 0.5,
        step: 0.005,
        label: 'Current round each element (canvas heights / s)',
      },
      spin: { type: 'select', default: 'alternate', options: SPINS, label: 'Direction' },
      solid: {
        type: 'boolean',
        default: true,
        label: 'Elements are solid (off = ink passes under them)',
      },
      demo: {
        type: 'select',
        default: 'auto',
        options: DEMOS,
        label: 'Demo cards (auto: only where the page has no marked elements)',
      },
      ink: { type: 'number', default: 0.55, min: 0, max: 3, step: 0.01, label: 'Ink amount' },
      inkRadius: {
        type: 'number',
        default: 0.04,
        min: 0.004,
        max: 0.3,
        step: 0.001,
        label: 'Trail width (canvas heights)',
      },
      palette: { type: 'select', default: 'rgb cycle', options: PALETTES },
      ground: {
        type: 'select',
        default: 'dark',
        options: GROUNDS,
        label: 'Ground (light is ink on paper, dark glows)',
      },
      paper: {
        type: 'color',
        default: [0.913, 0.905, 0.871],
        label: 'Paper colour (light ground)',
      },
      paperLight: {
        type: 'number',
        default: 1.5,
        min: 1,
        max: 6,
        step: 0.05,
        label: 'HDR brightness (x SDR white)',
      },
      cutout: {
        type: 'boolean',
        default: false,
        label: 'Draw the elements as clean paper (for elements with no background of their own)',
      },
      gain: { type: 'number', default: 1.4, min: 0.2, max: 4, step: 0.01, label: 'Ink strength' },
      sheen: { type: 'number', default: 0, min: 0, max: 1, step: 0.01, label: 'Wet sheen' },
      vignette: { type: 'number', default: 0.1, min: 0, max: 0.6, step: 0.01 },
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
      pressureIterations: {
        type: 'number',
        default: 30,
        min: 2,
        max: 120,
        step: 1,
        label: 'Pressure iterations',
      },
      vorticity: {
        type: 'number',
        default: 9,
        min: 0,
        max: 40,
        step: 0.5,
        label: 'Vorticity confinement',
      },
      velocityDissipation: {
        type: 'number',
        default: 0.35,
        min: 0,
        max: 4,
        step: 0.01,
        label: 'Velocity dissipation (1 / s)',
      },
      dyeDissipation: {
        type: 'number',
        default: 0.22,
        min: 0,
        max: 3,
        step: 0.01,
        label: 'Ink fading (1 / s)',
      },
      boundary: { type: 'select', default: 'wall', options: BOUNDARIES, label: 'Canvas edges' },
      radius: {
        type: 'number',
        default: 0.035,
        min: 0.01,
        max: 0.15,
        step: 0.001,
        label: 'Pointer splat radius (canvas heights)',
      },
      force: {
        type: 'number',
        default: 0.6,
        min: 0,
        max: 4,
        step: 0.01,
        label: 'Pointer push (fraction of its velocity)',
      },
      pointer: { type: 'select', default: 'hover and press', options: POINTER_MODES },
      touch: {
        type: 'select',
        default: 'scrolls the page',
        options: TOUCH_MODES,
        label: 'Touch drags',
      },
      view: { type: 'select', default: 'ink', options: VIEWS, label: 'View' },
    },
    presets: {
      'loose orbits': {},
      'tidy rings': {
        heads: 8,
        inkRadius: 0.03,
        chaos: 0,
        wander: 0.3,
        swirl: 0.6,
        wake: 0.25,
        current: 0.16,
        vorticity: 2,
        dyeDissipation: 0.3,
      },
      swarm: {
        heads: 36,
        chaos: 2,
        wander: 2,
        headSpeed: 0.7,
        wake: 0.6,
        ink: 0.7,
        inkRadius: 0.008,
        shell: 0.03,
        vorticity: 10,
        dyeDissipation: 0.45,
      },
      'night orbit': {
        ground: 'dark',
        gain: 1.5,
        sheen: 0.4,
        vignette: 0.35,
        dyeDissipation: 0.3,
      },
    },

    async create({ device, format, canvas, params, profiler, hdr = false }) {
      const orbitUniforms = createUniformBuffer(device, OrbitParams, 'ink-orbit orbit params');
      const displayUniforms = createUniformBuffer(
        device,
        DisplayParams,
        'ink-orbit display params',
      );
      const strokeLayer =
        options.style === 1
          ? createStrokeLayer(device, format, displayUniforms.buffer, DisplayParams.wgsl)
          : undefined;
      const pointers = trackPointers(canvas);
      const stirrer = createFluidStirrer();
      const random = createRandom(0x0b17);
      const sparkRandom = createRandom(0xe8be7);
      const pulses = new WeakMap<Head, number>();
      const sparks: EmberSpark[] = [];
      let pointerFrame: readonly TrackedPointer[] = [];
      const reducedMotion =
        typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
      // Mutable: the change threshold follows the grid and is read by the tracker on every update.
      const domOptions: DomRectsOptions = { epsilon: 1e-3 };
      const dom = trackDomRects(canvas, domOptions);
      let demo: DemoElements | undefined;

      const sampler = device.createSampler({
        label: 'ink-orbit sampler',
        magFilter: 'linear',
        minFilter: 'linear',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
      });
      const displayPipeline = createFullscreenPipeline(device, {
        fragment: {
          label: 'ink-orbit/display.wgsl',
          code: options.displayShader ?? displayShader,
          chunks: { 'display_params.wgsl': DisplayParams.wgsl },
        },
        format,
      });

      const orbitLayout = device.createBindGroupLayout({
        label: 'ink-orbit orbit layout',
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
          {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            storageTexture: { access: 'write-only', format: 'rgba16float' },
          },
        ],
      });
      const orbitPipeline = device.createComputePipeline({
        label: 'ink-orbit/orbit.wgsl',
        layout: device.createPipelineLayout({
          label: 'ink-orbit orbit pipeline layout',
          bindGroupLayouts: [orbitLayout],
        }),
        compute: {
          module: createShader(device, {
            label: 'ink-orbit/orbit.wgsl',
            code: orbitShader,
            chunks: { 'orbit_params.wgsl': OrbitParams.wgsl },
          }),
          entryPoint: 'cs_orbit',
        },
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
        label: 'ink-orbit',
      });

      // The solver rotates its velocity through a small pool of textures, so the hook sees a handful of
      // (read, write) pairs over and over: one bind group each, dropped when the pool is re-created.
      const orbitGroups = new Map<GPUTextureView, Map<GPUTextureView, GPUBindGroup>>();
      let orbitGeneration = fluid.generation;
      let laneCount = 0;
      let orbitTimed = false;

      fluid.hooks.forces = ({ encoder, read, write, grid }) => {
        if (laneCount === 0) return false;
        if (orbitGeneration !== fluid.generation) {
          orbitGroups.clear();
          orbitGeneration = fluid.generation;
        }
        let byWrite = orbitGroups.get(read);
        if (!byWrite) orbitGroups.set(read, (byWrite = new Map()));
        let group = byWrite.get(write);
        if (!group) {
          group = device.createBindGroup({
            label: 'ink-orbit orbit',
            layout: orbitLayout,
            entries: [
              { binding: 0, resource: { buffer: orbitUniforms.buffer } },
              { binding: 1, resource: read },
              { binding: 2, resource: write },
            ],
          });
          byWrite.set(write, group);
        }
        const pass = encoder.beginComputePass({
          label: 'ink-orbit orbit',
          // Timed once per frame: the solver may run this for several steps.
          timestampWrites: orbitTimed ? undefined : profiler.pass('orbit'),
        });
        orbitTimed = true;
        pass.setPipeline(orbitPipeline);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(
          Math.ceil(grid.velocityWidth / 8),
          Math.ceil(grid.velocityHeight / 8),
        );
        pass.end();
        return true;
      };

      const displayGroups = new Map<number, GPUBindGroup>();
      let displayGeneration = fluid.generation;

      function displayGroup(): GPUBindGroup {
        if (displayGeneration !== fluid.generation) {
          displayGroups.clear();
          displayGeneration = fluid.generation;
        }
        let group = displayGroups.get(fluid.dye.slot);
        if (!group) {
          group = device.createBindGroup({
            label: 'ink-orbit display',
            layout: displayPipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: displayUniforms.buffer } },
              { binding: 1, resource: fluid.dye.view },
              { binding: 2, resource: sampler },
              { binding: 3, resource: fluid.fieldsView },
            ],
          });
          displayGroups.set(fluid.dye.slot, group);
        }
        return group;
      }

      /** Dye per unit amount at hue t: optical density on paper, colour on a dark ground; amount in a. */
      function dyeValue(t: number, step: number): [number, number, number, number] {
        const colour = paletteColour(
          params.palette,
          params.palette === 'rgb cycle' ? (reducedMotion ? 0 : clock) : t,
          step,
        );
        const rgb = params.ground === 'light' ? absorbance(colour) : colour;
        return [rgb[0], rgb[1], rgb[2], 1];
      }

      // An element keeps the number it was given when first seen, so "alternate" does not flip a
      // direction when an earlier element scrolls out of the canvas.
      const order = new WeakMap<Element, number>();
      let nextOrder = 0;

      /** One measured element: its box for the heads, and how fast it is moving itself. */
      interface Lane {
        box: HeadBox;
        velocity: [number, number];
      }

      /** This frame's elements, in DOM order, in image heights. */
      function buildLanes(obstacles: readonly Obstacle[], aspect: number): Lane[] {
        const lanes: Lane[] = [];
        for (const o of obstacles) {
          const choice = o.element?.getAttribute(ORBIT_ATTRIBUTE)?.trim().toLowerCase();
          const r = o.exact ?? o.rect;
          let index = lanes.length;
          if (o.element) {
            index = order.get(o.element) ?? nextOrder++;
            order.set(o.element, index);
          }
          let spin = index % 2 === 0 ? 1 : -1;
          if (choice === 'off' || choice === 'none') spin = 0;
          else if (choice === 'cw') spin = 1;
          else if (choice === 'ccw') spin = -1;
          else if (params.spin === 'clockwise') spin = 1;
          else if (params.spin === 'counter-clockwise') spin = -1;
          const velocity: [number, number] = [0, 0];
          if (o.motion) {
            velocity[0] = 0.5 * (o.motion.left + o.motion.right) * aspect;
            velocity[1] = 0.5 * (o.motion.top + o.motion.bottom);
            const speed = Math.hypot(velocity[0], velocity[1]);
            if (speed > 1.2) {
              velocity[0] *= 1.2 / speed;
              velocity[1] *= 1.2 / speed;
            }
          }
          lanes.push({
            box: {
              cx: (r.x + r.width / 2) * aspect,
              cy: r.y + r.height / 2,
              hx: (r.width * aspect) / 2,
              hy: r.height / 2,
              radius: r.radius ?? 0,
              spin,
            },
            velocity,
          });
        }
        return lanes;
      }

      // THE HEADS (heads.ts): points that wander the canvas and are swung round the elements. Each one
      // is treated like a pointer being dragged through the fluid: along the path it took this frame it
      // lays a capsule of ink and gives the fluid part of its velocity, with the smooth ramps of
      // fluid-stir.ts so the capsules of successive frames join into one unbroken stroke.
      const heads: Head[] = [];
      let clock = 0;
      let seeded = false;

      function fieldFor(lanes: readonly Lane[], aspect: number): HeadField {
        return {
          boxes: lanes.map((lane) => lane.box),
          aspect,
          time: clock,
          pull: params.pull,
          swirl: params.swirl,
          shell: params.shell,
          wander: params.wander,
          chaos: params.chaos,
          maxSpeed: params.headSpeed * (reducedMotion ? 0.4 : 1),
          // A broad trail may overlap an element (the solver lays no ink inside a wall); its head may not.
          clearance: Math.min(params.inkRadius * 0.9, 0.02),
        };
      }

      /** Ink strength of a head: it fades in when it is born and out before it is reborn elsewhere. */
      function strength(head: Head): number {
        const rise = Math.min(1, head.age / 0.6);
        const fall = Math.min(1, Math.max(0, head.life - head.age) / 1.2);
        return params.ink * rise * rise * (3 - 2 * rise) * fall;
      }

      function layInk(head: Head, from: readonly [number, number], aspect: number) {
        // A head that dawdles would soak one spot: it lays less ink the slower it goes.
        const pace = Math.hypot(head.vx, head.vy) / Math.max(1e-3, 0.3 * params.headSpeed);
        const excitement = options.style === 3 ? 0.65 + (pulses.get(head) ?? 0) * 0.7 : 1;
        const amount = strength(head) * (0.2 + 0.8 * Math.min(1, pace)) * excitement;
        if (amount <= 0) return;
        const radius = params.inkRadius * head.size;
        const v = dyeValue(head.hue, head.colour);
        strokeLayer?.add({
          from: [from[0], from[1]],
          to: [head.x, head.y],
          colour: [v[0], v[1], v[2]],
          radius: radius * 0.18,
          strength: Math.min(1, strength(head) * 1.4),
          born: clock,
        });
        fluid.addDye(
          [from[0] / aspect, from[1]],
          [head.x / aspect, head.y],
          radius,
          [v[0] * amount, v[1] * amount, v[2] * amount, amount],
          radius * 0.5,
        );
      }

      function moveHeads(field: HeadField, dt: number) {
        const wanted = Math.min(MAX_HEADS, Math.max(0, Math.round(params.heads)));
        heads.length = Math.min(heads.length, wanted);
        while (heads.length < wanted) heads.push(createHead(heads.length, field, random));
        for (const head of heads) {
          const from: [number, number] = [head.x, head.y];
          const previousVelocity = [head.vx, head.vy];
          const travelled = stepHead(head, field, dt, random);
          if (options.style === 3) {
            let proximity = 0;
            for (const pointer of pointerFrame) {
              if (pointer.inside) {
                const distance = Math.hypot(pointer.x * field.aspect - head.x, pointer.y - head.y);
                proximity = Math.max(proximity, 1 - distance / 0.14);
              }
            }
            pulses.set(
              head,
              travelled
                ? biolumePulse(
                    pulses.get(head) ?? 0,
                    previousVelocity,
                    [head.vx, head.vy],
                    dt,
                    proximity,
                  )
                : 0,
            );
          }
          if (!travelled || Math.hypot(head.x - from[0], head.y - from[1]) < 1e-5) continue;
          layInk(head, from, field.aspect);
          if (
            options.style === 5 &&
            !reducedMotion &&
            sparks.length < MAX_SPARKS &&
            sparkRandom() < dt * 0.75
          ) {
            sparks.push({
              x: head.x,
              y: head.y,
              vx: head.vx * 0.45 + (sparkRandom() - 0.5) * 0.12,
              vy: head.vy * 0.45 - 0.03 - sparkRandom() * 0.06,
              age: 0,
              life: 0.7 + sparkRandom() * 1.3,
              radius: 0.001 + sparkRandom() * 0.001,
            });
          }
          if (params.wake > 0) {
            const radius = params.inkRadius * head.size;
            fluid.addForce(
              [from[0] / field.aspect, from[1]],
              [head.x / field.aspect, head.y],
              radius * 1.2,
              [head.vx * params.wake, head.vy * params.wake],
              radius * 0.5,
            );
          }
          if (params.stir > 0) {
            // A fish's tail. The wake above is as broad as the stroke, and a broad push only carries
            // ink along: the colours slide past each other and stay apart. This is a narrow jet through
            // the middle of the stroke whose direction swings from side to side about once a second.
            // Every swing sheds a vortex, to the left and then to the right, a fraction of the
            // stroke's width across, and that row of vortices is what folds this trail's ink into its
            // neighbours'.
            const swing = Math.sin(clock * (4.5 + 3 * head.pace) + head.offset * 7);
            const cos = Math.cos(1.1 * swing);
            const sin = Math.sin(1.1 * swing);
            const jet = 1.5 * params.stir;
            const radius = params.inkRadius * head.size * 0.3;
            fluid.addForce(
              [from[0] / field.aspect, from[1]],
              [head.x / field.aspect, head.y],
              radius,
              [(head.vx * cos - head.vy * sin) * jet, (head.vx * sin + head.vy * cos) * jet],
              radius * 0.5,
            );
          }
        }
      }

      /**
       * Runs the heads on for a moment before the first frame and lays the trails they would have left,
       * so a page (or a paused one) never starts blank. Ink only: the water starts still.
       */
      function seedTrails(field: HeadField) {
        const wanted = Math.min(MAX_HEADS, Math.max(0, Math.round(params.heads)));
        heads.length = 0;
        sparks.length = 0;
        strokeLayer?.clear();
        for (let i = 0; i < wanted; i++) heads.push(createHead(i, field, random));
        const pieces = Math.max(4, Math.min(16, Math.floor(240 / Math.max(1, wanted))));
        for (const head of heads) {
          for (let piece = 0; piece < pieces; piece++) {
            const from: [number, number] = [head.x, head.y];
            if (stepHead(head, field, 0.09, random)) layInk(head, from, field.aspect);
          }
        }
      }

      const boxes = new Float32Array(MAX_LANES * 4);
      const extras = new Float32Array(MAX_LANES * 4);
      const cutRects = new Float32Array(MAX_LANES * 4);
      const cutRadii = new Float32Array(MAX_LANES * 4);
      const headPositions = new Float32Array(MAX_HEADS * 4);
      const headMotion = new Float32Array(MAX_HEADS * 4);
      const headColours = new Float32Array(MAX_HEADS * 4);
      const sparkPositions = new Float32Array(MAX_SPARKS * 4);
      let maskApplied = false;

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
            if (fluid.resize(width, height)) seeded = false;
          }
          dom.invalidate();
        },

        frame({ delta }, encoder, target) {
          const aspect = width / height;
          const dt = Math.min(Math.max(delta, 1 / 240), 0.1);
          if (
            fluid.setResolution(
              cells(Number(params.resolution), simAspect),
              cells(Number(params.dyeResolution), simAspect),
            )
          ) {
            seeded = false;
          }
          const grid = fluid.grid;

          // 1. Read the page first, before anything is written.
          domOptions.epsilon = 0.25 / Math.max(1, grid.width);
          const measured = dom.update(delta);

          // 2. Demonstration cards where the page has none of its own. `found` counts the demo cards
          // too when they sit under the search root, so they are taken off before judging the page.
          const demoCounted = demo && dom.root?.contains(demo.container) ? demo.count : 0;
          const wantDemo =
            params.demo === 'always' ||
            (params.demo === 'auto' && measured.found - demoCounted <= 0);
          let demoPending = false;
          if (wantDemo && !demo) {
            demo = createDemoElements(canvas);
            if (demo) {
              dom.invalidate();
              demoPending = true;
            }
          } else if (!wantDemo && demo) {
            demo.destroy();
            demo = undefined;
            dom.invalidate();
          }

          // 3. The page into the solver: walls only when they moved, lanes every frame.
          if (params.solid) {
            if (measured.changed || !maskApplied) {
              fluid.obstacles.setRects(measured.rects);
              maskApplied = true;
            }
          } else if (maskApplied) {
            fluid.obstacles.clear();
            maskApplied = false;
          }
          const lanes = buildLanes(measured.obstacles, aspect);
          // The current pass: a slow drift round each element, and the water a moving element carries.
          const moving = lanes.some((lane) => lane.velocity[0] !== 0 || lane.velocity[1] !== 0);
          laneCount = params.current > 0 || moving ? Math.min(MAX_LANES, lanes.length) : 0;
          orbitTimed = false;
          boxes.fill(0);
          extras.fill(0);
          for (let i = 0; i < laneCount; i++) {
            const { box, velocity } = lanes[i]!;
            boxes.set([box.cx, box.cy, box.hx, box.hy], i * 4);
            extras.set([box.radius, box.spin, velocity[0], velocity[1]], i * 4);
          }

          const p = fluid.params;
          p.pressureIterations = params.pressureIterations;
          p.vorticity = params.vorticity;
          p.velocityDissipation = params.velocityDissipation;
          p.dyeDissipation = params.dyeDissipation;
          p.boundary = params.boundary as FluidBoundary;
          orbitUniforms.set({
            grid: [grid.width, grid.height],
            cell: grid.cellSize,
            dt: fluid.stepSeconds,
            speed: params.current,
            lane: params.shell * 1.5,
            grip: CURRENT_GRIP,
            count: laneCount,
            closed: params.boundary === 'wall' ? 1 : 0,
            boxes,
            extras,
          });
          orbitUniforms.upload();

          // 4. The heads. On the frame the demo cards are made they have not been measured yet, so the
          // first trails wait one frame for them rather than be drawn across where the cards will be.
          clock += dt;
          pointers.setCapture(params.touch === 'paints');
          pointerFrame = pointers.poll();
          const field = fieldFor(lanes, aspect);
          if (!seeded && !demoPending) {
            seeded = true;
            seedTrails(field);
          }
          if (seeded) moveHeads(field, dt);

          // 5. The pointer stirs and paints; touch drags scroll the page unless asked to paint.
          stirrer.stir(fluid, pointerFrame, aspect, delta, {
            mode: params.pointer as FluidStirMode,
            radius: params.radius,
            force: params.force,
            dyeAmount: params.ink * 0.9,
            colour: (stroke: FluidStroke) =>
              dyeValue(stroke.index * 0.381966 + stroke.age * 0.14, stroke.index),
          });

          fluid.update(encoder, delta);

          headPositions.fill(0);
          headMotion.fill(0);
          headColours.fill(0);
          sparkPositions.fill(0);
          if (options.style) {
            for (let i = 0; i < heads.length; i++) {
              const head = heads[i]!;
              const colour = paletteColour(
                params.palette,
                params.palette === 'rgb cycle' ? (reducedMotion ? 0 : clock) : head.hue,
                head.colour,
              );
              headPositions.set(
                [head.x, head.y, params.inkRadius * head.size, pulses.get(head) ?? 0],
                i * 4,
              );
              headMotion.set([head.vx, head.vy, head.hue, head.age], i * 4);
              headColours.set([...colour, strength(head) / Math.max(params.ink, 1e-6)], i * 4);
            }
            for (let i = sparks.length - 1; i >= 0; i--) {
              if (!stepSpark(sparks[i]!, dt)) sparks.splice(i, 1);
            }
            for (let i = 0; i < sparks.length; i++) {
              const spark = sparks[i]!;
              sparkPositions.set(
                [spark.x, spark.y, spark.radius, 1 - spark.age / spark.life],
                i * 4,
              );
            }
          }

          const light = params.ground === 'light';
          let cutCount = 0;
          if (params.cutout) {
            cutCount = Math.min(MAX_LANES, lanes.length);
            for (let i = 0; i < cutCount; i++) {
              const { box } = lanes[i]!;
              cutRects.set([box.cx, box.cy, box.hx, box.hy], i * 4);
              cutRadii[i * 4] = box.radius;
            }
          }
          displayUniforms.set({
            ground: light ? params.paper : (options.darkGround ?? [0.012, 0.014, 0.022]),
            aspect,
            dye_texel: [1 / grid.dyeWidth, 1 / grid.dyeHeight],
            gain: params.gain,
            sheen: params.sheen,
            dye_scale: grid.dyeHeight / 512,
            vignette: params.vignette,
            mode: light ? 1 : 0,
            view: Math.max(0, VIEWS.indexOf(params.view as (typeof VIEWS)[number])),
            // Extended-range output makes the emitted RGB light brighter on HDR displays too.
            light: hdr ? params.paperLight : 1,
            pixel: 1 / height,
            time: reducedMotion ? 0 : clock,
            style: options.style ?? 0,
            head_count: options.style ? heads.length : 0,
            spark_count: sparks.length,
            orbit_shell: params.shell,
            heads: headPositions,
            head_motion: headMotion,
            head_colours: headColours,
            sparks: sparkPositions,
            rect_count: cutCount,
            rects: cutRects,
            radii: cutRadii,
          });
          displayUniforms.upload();
          renderFullscreen(encoder, target, displayPipeline, [displayGroup()], {
            timestampWrites: profiler.pass('shade'),
          });
          if (params.view === 'ink') strokeLayer?.render(encoder, target, clock);
        },

        destroy() {
          demo?.destroy();
          demo = undefined;
          dom.destroy();
          pointers.destroy();
          fluid.hooks.forces = undefined;
          fluid.destroy();
          orbitUniforms.destroy();
          strokeLayer?.destroy();
          displayUniforms.destroy();
        },
      };
    },
  });
}

export const effect = createOrbitEffect();
