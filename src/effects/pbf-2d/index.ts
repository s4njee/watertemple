import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import {
  createBillboardRenderer,
  createComputePass,
  createNeighborCounter,
  orthographicCamera,
  type ParticleSet,
  type Rgb,
} from '../../gpu/particles/index.ts';
import { trackPointers, type TrackedPointer } from '../../gpu/pointer.ts';
import type { PassProfiler } from '../../gpu/profiler.ts';
import type { CameraMatrices } from '../../gpu/scene/camera.ts';
import { createPbf, type Pbf, type PbfScene, type Vec3 } from '../../gpu/sim/pbf.ts';
import { createUniformBuffer, defineUniformStruct, uniformArray } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import backdropWgsl from './backdrop.wgsl';
import { meta } from './meta.ts';
import scalarsWgsl from './scalars.wgsl';

// The 2D position-based-fluids demo: a tank of water solved by src/gpu/sim/pbf.ts (read its header for the method) and
// drawn as discs. This file is the scenes, the pointer, the scene "director" (the wave-maker's wall, the drops, the
// idle strokes, the restarts) and the plumbing. HOW THE PARTICLES ARE DRAWN is one clearly marked section below,
// `createLiquidDrawer`: it takes a ParticleSet and a target and nothing else from this file, so a liquid-surface
// renderer can replace it without touching the rest.

const SCENES = ['dam break', 'pour', 'wave-maker', 'drops'] as const;
type SceneName = (typeof SCENES)[number];

/** Particle spacing in millimetres, as the select shows it, and in metres. The tank is 2 m wide. */
const RESOLUTIONS = ['12 mm', '8 mm', '6 mm', '4 mm', '3 mm', '2 mm'] as const;
const SPACING: Record<(typeof RESOLUTIONS)[number], number> = {
  '12 mm': 0.012,
  '8 mm': 0.008,
  '6 mm': 0.006,
  '4 mm': 0.004,
  '3 mm': 0.003,
  '2 mm': 0.002,
};

/**
 * "Auto" iterations and sub-steps (the controls' 0). The solver's parameters scale by construction (see pbf.ts) but how
 * many iterations and sub-steps it needs does NOT. Information crosses one kernel radius (3 spacings) per iteration, so a
 * 0.4 m pool that is twice as many spacings deep needs more of them: measured on a pool at rest, at least 4 up to about 50
 * rows and then roughly one more pair per 32 rows (fewer and it does not settle, it explodes; an odd count also leaves a
 * small residual jitter, so these are even). A particle falling at 3 m/s should not move more than a couple of spacings per
 * sub-step, which gives sub-steps growing like 1 / spacing too. See the notes for the measurements.
 */
const autoIterations = (spacing: number) => Math.max(4, 2 * Math.round(0.013125 / spacing));
const autoSubsteps = (spacing: number) => clamp(Math.round(0.025 / spacing), 2, 8);

const VIEWS = ['water', 'speed', 'density error', 'neighbours'] as const;
type ViewName = (typeof VIEWS)[number];
const RENDERS = ['opaque', 'alpha'] as const;

/** The tank, metres. Gravity 9.81 looks right at this size. z only exists to give the box a depth. */
const TANK = { min: [0, 0, -1] as Vec3, max: [2, 1.2, 1] as Vec3 };
const MAX_COLLIDERS_DRAWN = 8;
const MAX_RINGS = 4;

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

const BackdropParams = defineUniformStruct('BackdropParams', {
  center: 'vec2f',
  half_height: 'f32',
  aspect: 'f32',
  tank_min: 'vec2f',
  tank_max: 'vec2f',
  wall_min: 'vec2f',
  wall_max: 'vec2f',
  height_px: 'f32',
  collider_count: 'u32',
  pointer_count: 'u32',
  encode_srgb: 'u32',
  colliders: uniformArray('vec4f', MAX_COLLIDERS_DRAWN * 2),
  pointers: uniformArray('vec4f', MAX_RINGS),
});

// --- Scenes -------------------------------------------------------------------------------------------------------------

interface SceneBuild {
  scene: PbfScene;
  /** Area of water (m^2) the scene holds at most, which fixes the capacity at a spacing. */
  area: number;
}

/** What each scene sets up at a given spacing. Areas, not counts, so that every resolution holds the same water. */
function describeScene(name: SceneName): SceneBuild {
  const bounds = { min: TANK.min, max: TANK.max };
  if (name === 'pour') {
    return {
      area: 0.6,
      scene: {
        bounds,
        blocks: [],
        // A shelf the stream lands on, and a rock the water runs round, both thicker than the solver's 2h.
        colliders: [
          { kind: 'box', min: [0.28, 0.5, -1], max: [0.92, 0.58, 1] },
          { kind: 'sphere', center: [1.35, 0.2, 0], radius: 0.2 },
        ],
        emitter: { position: [0.45, 1.15, 0], velocity: [0, -1.3, 0], radius: 0.035 },
      },
    };
  }
  if (name === 'wave-maker') {
    return {
      area: 0.84,
      scene: { bounds, blocks: [{ min: [0, 0, 0], max: [2, 0.42, 0] }] },
    };
  }
  if (name === 'drops') {
    return {
      area: 0.9,
      scene: { bounds, blocks: [{ min: [0, 0, 0], max: [2, 0.4, 0] }] },
    };
  }
  return {
    area: 0.33,
    scene: { bounds, blocks: [{ min: [0, 0, 0], max: [0.6, 0.55, 0] }] },
  };
}

/** Drops a scene may add before it restarts, and the seconds between them. */
const DROP_BUDGET = 16;
const DROP_EVERY = 2.4;
const DROP_SIZE = 0.07;
/** Seconds of quiet (nothing touched, nothing falling) before the idle wave stroke and before a restart. */
const IDLE_BEFORE_STROKE = 4;
const STROKE_EVERY = 10;
const STROKE_LENGTH = 1.5;
const STROKE_DEPTH = 0.12;
const RESTART_AFTER = 14;

interface Director {
  /** Simulated seconds since the scene started. */
  age: number;
  /** `age` at which the idle wave stroke in progress began (-1: none), and the earliest the next may begin. */
  strokeStart: number;
  nextStroke: number;
  /** Drops dropped so far, and the `age` of the next. */
  drops: number;
  nextDrop: number;
  /** `age` when the pour ran dry or the last drop fell. */
  settledSince: number | null;
  random: () => number;
}

function createDirector(): Director {
  let seed = 0x2f6e2b1;
  return {
    age: 0,
    strokeStart: -1,
    nextStroke: 10,
    drops: 0,
    nextDrop: 1.2,
    settledSince: null,
    random: () => {
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    },
  };
}

interface World {
  key: string;
  name: SceneName;
  pbf: Pbf;
  spacing: number;
  build: SceneBuild;
  director: Director;
  /** One f32 per particle for the colour-by views, vertex-readable. */
  scalar: GPUBuffer;
  destroy(): void;
}

interface TapState {
  x: number;
  y: number;
  time: number;
}

/** How the camera frames the tank: the world point at the screen centre, half the visible height, and the aspect. */
interface Framing {
  cx: number;
  cy: number;
  halfHeight: number;
  aspect: number;
}

// ===================================================================================================================
// DRAW STEP
//
// Everything that decides how a ParticleSet looks is in this section and nothing else in the file touches the
// billboard renderer. The contract: `drawLiquid(encoder, set, target, draw)` draws the particles of `set` over what
// `target` already holds (the backdrop; it does not clear), in one render pass. The colour comes from the foam amount
// in `set.positions.w` ('water') or from `draw.scalar`, a buffer of one f32 per particle in the order of `set`. A
// surface renderer (src/gpu/liquid-2d/) replaces `createLiquidDrawer` and keeps this signature.
// ===================================================================================================================

type LiquidColour = ViewName;

interface LiquidDraw {
  camera: CameraMatrices;
  /** Drawing buffer size in pixels, and the device pixel ratio (a minimum particle size in CSS pixels). */
  width: number;
  height: number;
  dpr: number;
  mode: (typeof RENDERS)[number];
  colour: LiquidColour;
  /** Multiplies the particles' radius. */
  sizeScale: number;
  /** For every colour but 'water': one f32 per particle, VERTEX usage, in the order of the set. */
  scalar?: GPUBuffer;
}

interface LiquidDrawer {
  resize(width: number, height: number): void;
  drawLiquid(
    encoder: GPUCommandEncoder,
    set: ParticleSet,
    target: GPUTextureView,
    draw: LiquidDraw,
  ): void;
  destroy(): void;
}

// Linear RGB. Water: calm particles are blue, foam (fast and sparse) goes white.
const WATER_PALETTE: Rgb[] = [
  [0.035, 0.34, 0.74],
  [0.22, 0.66, 0.95],
  [0.95, 0.98, 1],
];
const SPEED_PALETTE: Rgb[] = [
  [0.01, 0.07, 0.25],
  [0.03, 0.35, 0.72],
  [0.35, 0.85, 1],
  [1, 1, 1],
];
/** Diverging, white at the middle stop: blue is sparser than rest, red is compressed. */
const DENSITY_PALETTE: Rgb[] = [
  [0.04, 0.12, 0.8],
  [0.1, 0.62, 0.95],
  [0.92, 0.95, 0.97],
  [0.98, 0.65, 0.12],
  [0.85, 0.06, 0.04],
];
const HEAT_PALETTE: Rgb[] = [
  [0.02, 0.03, 0.18],
  [0.05, 0.25, 0.7],
  [0.05, 0.7, 0.55],
  [0.95, 0.85, 0.15],
  [0.95, 0.2, 0.08],
  [1, 0.95, 0.95],
];

function createLiquidDrawer(
  device: GPUDevice,
  format: GPUTextureFormat,
  profiler: PassProfiler,
): LiquidDrawer {
  const renderer = createBillboardRenderer({
    device,
    format,
    profiler,
    label: 'pbf-2d',
    style: { softness: 0.5 },
  });
  return {
    resize(width, height) {
      renderer.resize(width, height);
    },
    drawLiquid(encoder, set, target, draw) {
      const style = renderer.style;
      style.mode = draw.mode;
      style.sizeScale = draw.sizeScale;
      style.minPixels = 0.7 * draw.dpr;
      style.intensity = 0.92;
      style.shading = draw.colour === 'water' ? 0.5 : 0.35;
      switch (draw.colour) {
        case 'water':
          style.palette = WATER_PALETTE;
          style.scalarRange = [0, 1];
          break;
        case 'speed':
          style.palette = SPEED_PALETTE;
          style.scalarRange = [0, 4];
          break;
        case 'density error':
          style.palette = DENSITY_PALETTE;
          style.scalarRange = [-0.2, 0.2];
          break;
        case 'neighbours':
          style.palette = HEAT_PALETTE;
          style.scalarRange = [0, 28];
          break;
      }
      renderer.draw(encoder, set, draw.camera, target, {
        width: draw.width,
        height: draw.height,
        clear: null,
        scalar: draw.colour === 'water' ? undefined : draw.scalar,
        timestamp: 'liquid draw',
      });
    },
    destroy() {
      renderer.destroy();
    },
  };
}

// ===================================================================================================================
// END OF THE DRAW STEP
// ===================================================================================================================

export const effect = defineEffect({
  meta,
  params: {
    scene: {
      type: 'select',
      default: SCENES[0],
      options: SCENES,
      label: 'Scene (restarts)',
    },
    restart: {
      type: 'boolean',
      default: false,
      label: 'Restart the scene (flip this switch)',
    },
    resolution: {
      type: 'select',
      default: RESOLUTIONS[0],
      options: RESOLUTIONS,
      label:
        'Particle spacing (restarts; the same physics at every size, 4x the particles per halving)',
    },
    iterations: {
      type: 'number',
      default: 0,
      min: 0,
      max: 16,
      step: 1,
      label:
        'Solver iterations per sub-step (0: auto, grows as the spacing shrinks; 1 is unstable)',
    },
    substeps: {
      type: 'number',
      default: 0,
      min: 0,
      max: 8,
      step: 1,
      label:
        'Sub-steps per 1/60 s (0: auto; each rebuilds the grid, the best cure for compression)',
    },
    viscosity: {
      type: 'number',
      default: 0.01,
      min: 0,
      max: 0.3,
      step: 0.005,
      label: 'Viscosity (XSPH, fraction of the velocity difference removed per 1/60 s)',
    },
    vorticity: {
      type: 'number',
      default: 0,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Vorticity confinement (puts back the swirl the solver damps)',
    },
    artificialPressure: {
      type: 'number',
      default: 0.02,
      min: 0,
      max: 0.3,
      step: 0.005,
      label: 'Artificial pressure (stops clumping at the free surface; too much makes it fizz)',
    },
    gravity: {
      type: 'number',
      default: 9.81,
      min: 0,
      max: 30,
      step: 0.1,
      label: 'Gravity (m/s²)',
    },
    tilt: {
      type: 'number',
      default: 0,
      min: -90,
      max: 90,
      step: 1,
      label: 'Gravity tilt (degrees: the water runs sideways)',
    },
    waveAmplitude: {
      type: 'number',
      default: 0.15,
      min: 0,
      max: 0.5,
      step: 0.005,
      label: 'Wave-maker stroke (m)',
    },
    waveFrequency: {
      type: 'number',
      default: 0.85,
      min: 0.2,
      max: 2.5,
      step: 0.01,
      label: 'Wave-maker frequency (Hz; the tank sloshes naturally near 0.5)',
    },
    pointerSize: {
      type: 'number',
      default: 0.13,
      min: 0.05,
      max: 0.4,
      step: 0.01,
      label: 'Pointer radius (m)',
    },
    pointerStrength: {
      type: 'number',
      default: 0.6,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Pointer strength',
    },
    captureTouch: {
      type: 'boolean',
      default: true,
      label: 'Touch drags push the water (and stop the page scrolling over the canvas)',
    },
    ambient: {
      type: 'boolean',
      default: true,
      label: 'Idle motion (a gentle wave when nobody touches the water)',
    },
    loop: {
      type: 'boolean',
      default: true,
      label: 'Restart a finished pour or drop scene after a rest',
    },
    view: {
      type: 'select',
      default: VIEWS[0],
      options: VIEWS,
      label: 'Colour by (density error: blue sparse, white at rest density, red compressed)',
    },
    render: { type: 'select', default: RENDERS[0], options: RENDERS, label: 'Render mode' },
    size: {
      type: 'number',
      default: 1,
      min: 0.4,
      max: 1.6,
      step: 0.05,
      label: 'Particle size',
    },
  },
  presets: {
    'dam break': {},
    // Artificial pressure is a repulsion that also acts on sparse water, so it fans a thin stream out: nearly none here.
    pour: { scene: SCENES[1], artificialPressure: 0.005 },
    'wave-maker': { scene: SCENES[2] },
    drops: { scene: SCENES[3] },
    'fine particles': { resolution: RESOLUTIONS[3] },
    'coarse particles': { resolution: RESOLUTIONS[0], render: RENDERS[0] },
    speed: { view: VIEWS[1] },
    'density error': { view: VIEWS[2] },
    neighbours: { view: VIEWS[3] },
    'one iteration (jelly)': { iterations: 1, substeps: 1, view: VIEWS[2] },
    'sloshing, no idle': { scene: SCENES[2], waveFrequency: 0.5, ambient: false },
  },

  async create({ device, format, canvas, params, profiler }) {
    // --- Plumbing ---------------------------------------------------------------------------------------------------
    const backdropUniform = createUniformBuffer(device, BackdropParams, 'pbf-2d backdrop');
    const countBuffer = device.createBuffer({
      label: 'pbf-2d count',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const blend: GPUBlendState = {
      color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    };
    const backdropSource = {
      label: 'pbf-2d/backdrop.wgsl',
      code: backdropWgsl.code,
      chunks: { ...backdropWgsl.chunks, 'backdrop_params.wgsl': BackdropParams.wgsl },
    };
    const backdropPipeline = createFullscreenPipeline(device, {
      label: 'pbf-2d backdrop',
      fragment: backdropSource,
      entryPoint: 'fs_backdrop',
      format,
    });
    const overlayPipeline = createFullscreenPipeline(device, {
      label: 'pbf-2d overlay',
      fragment: backdropSource,
      entryPoint: 'fs_overlay',
      format,
      blend,
    });
    const backdropGroup = device.createBindGroup({
      layout: backdropPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: backdropUniform.buffer } }],
    });
    const overlayGroup = device.createBindGroup({
      layout: overlayPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: backdropUniform.buffer } }],
    });
    const scalarPasses = {
      speed: createComputePass(device, {
        label: 'pbf-2d/scalars.wgsl',
        code: scalarsWgsl.code,
        entryPoint: 'speed',
        profiler,
        timestamp: 'speed',
      }),
      density: createComputePass(device, {
        label: 'pbf-2d/scalars.wgsl',
        code: scalarsWgsl.code,
        entryPoint: 'density_error',
        profiler,
        timestamp: 'density error',
      }),
    };
    const counter = createNeighborCounter({ device, profiler, label: 'pbf-2d neighbours' });
    const drawer = createLiquidDrawer(device, format, profiler);
    const pointers = trackPointers(canvas, { capture: params.captureTouch });
    const taps = new Map<number, TapState>();

    let width = 1;
    let height = 1;
    let dpr = 1;
    let clock = 0;
    /** Clock time of the last pointer press or drag, for the idle motion. */
    let lastTouched = -100;
    let lastRestart = params.restart;
    let world: World | undefined;

    // --- The world: solver, scene, scalar buffer ---------------------------------------------------------------------
    function createWorld(name: SceneName, resolution: string): World {
      const spacing = SPACING[resolution as (typeof RESOLUTIONS)[number]] ?? SPACING['8 mm'];
      const build = describeScene(name);
      // Headroom for the blocks a tap drops and for the drops of the 'drops' scene.
      const capacity = Math.ceil((build.area * 1.2) / (spacing * spacing)) + 256;
      const pbf = createPbf({
        device,
        dimension: 2,
        capacity,
        spacing,
        bounds: { min: TANK.min, max: TANK.max },
        profiler,
        label: 'pbf-2d',
      });
      pbf.reset(build.scene);
      const scalar = device.createBuffer({
        label: 'pbf-2d scalar',
        size: capacity * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      return {
        key: `${name}|${resolution}`,
        name,
        pbf,
        spacing,
        build,
        director: createDirector(),
        scalar,
        destroy() {
          pbf.destroy();
          scalar.destroy();
        },
      };
    }

    function restartWorld(w: World): void {
      w.pbf.reset(w.build.scene);
      w.director = createDirector();
      taps.clear();
    }

    function ensureWorld(): World {
      const name = params.scene as SceneName;
      const key = `${name}|${params.resolution}`;
      const flipped = params.restart !== lastRestart;
      lastRestart = params.restart;
      if (!world || world.key !== key) {
        world?.destroy();
        world = createWorld(name, params.resolution);
      } else if (flipped) {
        restartWorld(world);
      }
      return world;
    }

    /** The live parameters, copied into the solver every frame (they are cheap, and the solver sanitises them). */
    function syncParams(w: World): void {
      const p = w.pbf.params;
      const tilt = (params.tilt * Math.PI) / 180;
      p.gravity = [params.gravity * Math.sin(tilt), -params.gravity * Math.cos(tilt), 0];
      p.iterations = Math.round(params.iterations) || autoIterations(w.spacing);
      p.substeps = Math.round(params.substeps) || autoSubsteps(w.spacing);
      p.viscosity = params.viscosity;
      p.vorticity = params.vorticity;
      p.artificialPressure = params.artificialPressure;
      // A wall that moves at a fixed speed, not one that depends on the particle spacing.
      p.maxWallSpeed = 2;
    }

    // --- Camera ----------------------------------------------------------------------------------------------------
    function framing(): Framing {
      const aspect = width / height;
      const halfWidth = (TANK.max[0] - TANK.min[0]) / 2;
      const halfHeight = (TANK.max[1] - TANK.min[1]) / 2;
      return {
        cx: (TANK.min[0] + TANK.max[0]) / 2,
        cy: (TANK.min[1] + TANK.max[1]) / 2,
        // Keep the whole tank in view whatever the canvas shape.
        halfHeight: Math.max(halfHeight, halfWidth / aspect) * 1.08,
        aspect,
      };
    }

    const toWorld = (u: number, v: number, f: Framing): [number, number] => [
      f.cx + (2 * u - 1) * f.halfHeight * f.aspect,
      f.cy + (1 - 2 * v) * f.halfHeight,
    ];

    // --- The pointer -----------------------------------------------------------------------------------------------
    /** Pushes the water along the drag, drops a blob on a tap; returns the rings to draw (x, y, radius, intensity). */
    function handlePointers(w: World, f: Framing, dt: number): number[] {
      const tracked: TrackedPointer[] = pointers.poll();
      const rings: number[] = [];
      const pbf = w.pbf;
      for (const p of tracked) {
        if (p.justPressed) {
          taps.set(p.id, { x: p.x, y: p.y, time: clock });
          lastTouched = clock;
        }
        const tap = taps.get(p.id);
        if (tap && Math.hypot((p.x - tap.x) * f.aspect, p.y - tap.y) > 0.03) taps.delete(p.id);
        if (p.justReleased) {
          if (tap && !p.cancelled && clock - tap.time < 0.35)
            dropBlob(w, ...toWorld(tap.x, tap.y, f));
          taps.delete(p.id);
        }
        const [x, y] = toWorld(p.x, p.y, f);
        if (p.down) {
          lastTouched = clock;
          // The pointer's velocity in the world, from the tracker's smoothed canvas velocity.
          const vx = p.vx * 2 * f.halfHeight * f.aspect;
          const vy = -p.vy * 2 * f.halfHeight;
          const speed = Math.hypot(vx, vy);
          if (speed > 0.02) {
            const limit = Math.min(1, 6 / speed);
            // How much of the way to the pointer's velocity the water inside is pulled per 1/60 s: grows with the
            // pointer's speed, so a held-still pointer does nothing and a fast swipe takes the water with it.
            const pull = params.pointerStrength * clamp(speed / 0.6, 0, 1);
            pbf.impulse({
              position: [x, y, 0],
              radius: params.pointerSize,
              velocity: [vx * limit, vy * limit, 0],
              strength: 1 - (1 - pull) ** (dt * 60),
            });
          }
        }
        if ((p.down || p.inside) && rings.length < MAX_RINGS * 4) {
          rings.push(x, y, params.pointerSize, p.down ? 0.9 : 0.35);
        }
      }
      return rings;
    }

    /** A square blob of water at a point, if there is room for it. */
    function dropBlob(w: World, x: number, y: number, size = 0.1): void {
      const pbf = w.pbf;
      const need = Math.ceil(size / w.spacing) ** 2;
      if (pbf.capacity - pbf.count < need) return;
      const half = size / 2;
      const cx = clamp(x, TANK.min[0] + half, TANK.max[0] - half);
      const cy = clamp(y, TANK.min[1] + half, TANK.max[1] - half);
      pbf.addBlock({
        min: [cx - half, cy - half, 0],
        max: [cx + half, cy + half, 0],
        velocity: [0, -0.8, 0],
      });
    }

    // --- The director: what the scene does by itself ----------------------------------------------------------------
    /** Moves the left wall (the wave-maker's paddle), starts the drops and the idle strokes, restarts finished scenes. */
    function direct(w: World, dt: number): void {
      const d = w.director;
      const pbf = w.pbf;
      d.age += dt;
      let wall = 0;

      if (w.name === 'wave-maker') {
        const ramp = clamp(d.age / 2, 0, 1);
        wall =
          params.waveAmplitude *
          ramp *
          0.5 *
          (1 - Math.cos(2 * Math.PI * params.waveFrequency * d.age));
      } else if (params.ambient && w.name !== 'drops') {
        // A gentle push of the paddle every few seconds once the scene has settled and nobody is playing.
        const stroking = d.strokeStart >= 0 && d.age - d.strokeStart < STROKE_LENGTH;
        if (!stroking && d.age >= d.nextStroke && clock - lastTouched > IDLE_BEFORE_STROKE) {
          d.strokeStart = d.age;
          d.nextStroke = d.age + STROKE_EVERY;
        }
        const t = d.age - d.strokeStart;
        if (d.strokeStart >= 0 && t < STROKE_LENGTH) {
          wall = STROKE_DEPTH * Math.sin((Math.PI * t) / STROKE_LENGTH) ** 2;
        }
      }
      pbf.setBounds([wall, TANK.min[1], TANK.min[2]], [TANK.max[0], TANK.max[1], TANK.max[2]]);

      if (w.name === 'drops' && d.drops < DROP_BUDGET && d.age >= d.nextDrop) {
        const x = 0.3 + 1.4 * d.random();
        dropBlob(w, x, 0.95, DROP_SIZE);
        d.drops++;
        d.nextDrop = d.age + DROP_EVERY * (0.8 + 0.4 * d.random());
        if (d.drops >= DROP_BUDGET) d.settledSince = d.age;
      }
      if (w.name === 'pour' && d.settledSince === null && pbf.capacity - pbf.count < 48) {
        d.settledSince = d.age;
      }
      if (
        params.loop &&
        d.settledSince !== null &&
        d.age - d.settledSince > RESTART_AFTER &&
        clock - lastTouched > IDLE_BEFORE_STROKE
      ) {
        restartWorld(w);
      }
    }

    // --- Frame -------------------------------------------------------------------------------------------------------
    function writeBackdrop(w: World, f: Framing, rings: number[]): void {
      const pbf = w.pbf;
      const colliders: number[] = [];
      let count = 0;
      for (const c of w.build.scene.colliders ?? []) {
        if (count >= MAX_COLLIDERS_DRAWN) break;
        count++;
        if (c.kind === 'sphere') colliders.push(c.center[0], c.center[1], c.radius, 1, 0, 0, 0, 0);
        else colliders.push(c.min[0], c.min[1], 0, 2, c.max[0], c.max[1], 0, 0);
      }
      backdropUniform.set({
        center: [f.cx, f.cy],
        half_height: f.halfHeight,
        aspect: f.aspect,
        tank_min: [TANK.min[0], TANK.min[1]],
        tank_max: [TANK.max[0], TANK.max[1]],
        wall_min: [pbf.bounds.min[0]!, pbf.bounds.min[1]!],
        wall_max: [pbf.bounds.max[0]!, pbf.bounds.max[1]!],
        height_px: height,
        collider_count: count,
        pointer_count: Math.floor(rings.length / 4),
        encode_srgb: format.endsWith('-srgb') ? 0 : 1,
        colliders,
        pointers: rings,
      });
      backdropUniform.upload();
    }

    return {
      resize(w, h, ratio) {
        width = Math.max(1, Math.round(w));
        height = Math.max(1, Math.round(h));
        dpr = ratio;
        drawer.resize(width, height);
      },

      frame({ delta }, encoder, target) {
        const dt = clamp(delta, 0, 0.1);
        clock += dt;
        pointers.setCapture(params.captureTouch);
        const w = ensureWorld();
        const pbf = w.pbf;
        const f = framing();
        syncParams(w);
        const rings = handlePointers(w, f, dt);
        direct(w, dt);

        // 1. The solver: fixed steps of the clock it was handed, impulses queued above apply on the first of them.
        pbf.update(encoder, dt);

        // 2. The scalar a 'colour by' view needs, in the solver's current particle order.
        const view = params.view as ViewName;
        let scalar: GPUBuffer | undefined;
        if (view !== 'water' && pbf.count > 0) {
          scalar = w.scalar;
          if (view === 'neighbours') {
            counter.count(encoder, pbf.grid, pbf.system, w.scalar);
          } else {
            device.queue.writeBuffer(countBuffer, 0, new Uint32Array([pbf.count, 0, 0, 0]));
            if (view === 'speed') {
              scalarPasses.speed.runCount(
                encoder,
                [countBuffer, pbf.system.velocities, undefined, w.scalar],
                pbf.count,
              );
            } else {
              scalarPasses.density.runCount(
                encoder,
                [countBuffer, undefined, pbf.solve, w.scalar],
                pbf.count,
              );
            }
          }
        }

        // 3. The picture: the tank, then the liquid, then the pointer rings.
        writeBackdrop(w, f, rings);
        renderFullscreen(encoder, target, backdropPipeline, [backdropGroup], {
          label: 'pbf-2d backdrop',
          timestampWrites: profiler.pass('backdrop'),
        });
        drawer.drawLiquid(encoder, pbf.particles, target, {
          camera: orthographicCamera({
            center: [f.cx, f.cy],
            halfHeight: f.halfHeight,
            aspect: f.aspect,
          }),
          width,
          height,
          dpr,
          mode: params.render as (typeof RENDERS)[number],
          colour: view,
          sizeScale: params.size,
          scalar,
        });
        if (rings.length > 0) {
          renderFullscreen(encoder, target, overlayPipeline, [overlayGroup], {
            label: 'pbf-2d pointer',
            load: true,
            timestampWrites: profiler.pass('pointer'),
          });
        }
      },

      destroy() {
        world?.destroy();
        world = undefined;
        pointers.destroy();
        drawer.destroy();
        counter.destroy();
        scalarPasses.speed.destroy();
        scalarPasses.density.destroy();
        backdropUniform.destroy();
        countBuffer.destroy();
      },
    };
  },
});
