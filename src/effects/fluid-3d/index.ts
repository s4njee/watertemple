import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { trackPointers } from '../../gpu/pointer.ts';
import {
  CameraUniform,
  computeCamera,
  createCameraMatrices,
  fitFovY,
  setCameraUniform,
  unprojectPoint,
} from '../../gpu/scene/camera.ts';
import {
  DEFAULT_ORBIT_OPTIONS,
  createOrbitState,
  orbitEye,
  stepOrbit,
} from '../../gpu/scene/orbit.ts';
import { createFluid3d, rayBox, type Fluid3d, type Fluid3dScheme } from '../../gpu/sim/fluid3d.ts';
import { createRenderTarget } from '../../gpu/targets.ts';
import { createShader } from '../../gpu/shader.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import compositeShader from './composite.wgsl';
import inkShader from './ink.wgsl';
import marchShader from './march.wgsl';
import { meta } from './meta.ts';
import prepShader from './volume-prep.wgsl';

/** Everything the three shaders read about the scene, in one block (generated into march_params.wgsl). */
const MarchParams = defineUniformStruct('MarchParams', {
  box_min: 'vec3f',
  abs_gain: 'f32',
  box_max: 'vec3f',
  scatter: 'f32',
  light_dir: 'vec3f',
  phase_g: 'f32',
  light_color: 'vec3f',
  edge_strength: 'f32',
  ambient: 'vec3f',
  exposure: 'f32',
  water: 'vec3f',
  steps: 'u32',
  light_size: 'vec3u',
  light_steps: 'u32',
  sphere: 'vec4f',
  jitter: 'f32',
  frame: 'u32',
  lighting: 'u32',
  occupancy: 'u32',
  view: 'u32',
  slice: 'f32',
  debug_gain: 'f32',
});

const RESOLUTIONS = ['32', '48', '64', '80', '96', '128'] as const;
const SCALAR_RESOLUTIONS = ['64', '96', '128', '160'] as const;
const SOLVERS = ['multigrid', 'jacobi'] as const;
const SCHEMES = ['semi-lagrangian', 'maccormack'] as const;
const INKS = ['mixed inks', 'chosen colour'] as const;
const EMITTERS = ['drops from above', 'plume from below'] as const;
const LIGHTING = ['off', 'light volume', 'per-sample march'] as const;
const DRAGS = ['pushes the fluid', 'orbits the camera'] as const;
const VIEWS = ['ink', 'velocity slice', 'ink slice'] as const;

/** The tank's relative extents; the solver makes the longest side 1 length unit. */
const TANK: readonly [number, number, number] = [1, 1.25, 1];
const DEG = Math.PI / 180;
const CAMERA_FOV_Y = 36 * DEG;
const MIN_HORIZONTAL_FOV = 50 * DEG;
/** Most steps the solver runs per frame; a slow frame drops simulation time rather than spiralling. */
const MAX_STEPS = 3;
/** A pointer faster than this (tank heights per second) pushes no harder. */
const POINTER_SPEED_CAP = 3;

type Vec3 = [number, number, number];

// Inks are designed in sRGB and converted to linear; the solver stores the optical density -ln(colour).
const INK_PALETTE: readonly Vec3[] = [
  [0.1, 0.14, 0.62], // indigo
  [0.8, 0.07, 0.12], // crimson
  [0.05, 0.58, 0.6], // teal
  [0.96, 0.58, 0.06], // amber
  [0.82, 0.1, 0.56], // magenta
  [0.08, 0.62, 0.3], // emerald
];

function srgbToLinear(x: number): number {
  return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
}

/** Optical density per unit amount of a linear colour: -ln(colour), floored so a black ink absorbs a lot, not infinitely. */
export function inkDensity(colour: readonly number[]): Vec3 {
  return [0, 1, 2].map((i) => -Math.log(Math.min(1, Math.max(0.02, colour[i] ?? 1)))) as Vec3;
}

/** The linear colour of palette ink number `index` (cycling). */
export function paletteInk(index: number): Vec3 {
  const c =
    INK_PALETTE[
      ((Math.floor(index) % INK_PALETTE.length) + INK_PALETTE.length) % INK_PALETTE.length
    ]!;
  return c.map(srgbToLinear) as Vec3;
}

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

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const normalize = (a: Vec3): Vec3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

/** Where a ray meets the plane through `point` with unit `normal`, or null when it is parallel or behind. */
export function rayPlane(origin: Vec3, dir: Vec3, point: Vec3, normal: Vec3): Vec3 | null {
  const denominator = dot(dir, normal);
  if (Math.abs(denominator) < 1e-6) return null;
  const t = dot(sub(point, origin), normal) / denominator;
  if (t < 0) return null;
  return [origin[0] + dir[0] * t, origin[1] + dir[1] * t, origin[2] + dir[2] * t];
}

export const effect = defineEffect({
  meta,
  params: {
    resolution: {
      type: 'select',
      default: '64',
      options: RESOLUTIONS,
      label: 'Velocity cells (long side; re-creates the fluid)',
    },
    scalarResolution: {
      type: 'select',
      default: '96',
      options: SCALAR_RESOLUTIONS,
      label: 'Ink texels (long side; re-creates the fluid)',
    },
    pressureSolver: {
      type: 'select',
      default: 'multigrid',
      options: SOLVERS,
      label: 'Pressure solver',
    },
    pressureIterations: {
      type: 'number',
      default: 2,
      min: 1,
      max: 12,
      step: 1,
      label: 'Pressure work per step (V-cycles, or Jacobi sweeps)',
    },
    scalarScheme: {
      type: 'select',
      default: 'maccormack',
      options: SCHEMES,
      label: 'Ink advection',
    },
    vorticity: {
      type: 'number',
      default: 4,
      min: 0,
      max: 24,
      step: 0.5,
      label: 'Vorticity confinement (keeps the rings tight)',
    },
    buoyancy: {
      type: 'number',
      default: 1.2,
      min: -4,
      max: 6,
      step: 0.05,
      label: 'Ink weight (positive sinks, negative rises like smoke)',
    },
    velocityDissipation: {
      type: 'number',
      default: 0.25,
      min: 0,
      max: 3,
      step: 0.01,
      label: 'Water friction (1 / s)',
    },
    inkFade: {
      type: 'number',
      default: 0.2,
      min: 0,
      max: 1,
      step: 0.005,
      label: 'Ink fade (1 / s)',
    },
    speed: {
      type: 'number',
      default: 1,
      min: 0.2,
      max: 2,
      step: 0.01,
      label:
        'Flow speed (below 1 is slow motion and costs less; steps are skipped, so it judders a little)',
    },
    inks: { type: 'select', default: 'mixed inks', options: INKS, label: 'Ink colours' },
    inkColor: {
      type: 'color',
      default: [0.012, 0.03, 0.3],
      label: 'Ink colour when chosen (linear)',
    },
    emitter: { type: 'select', default: 'drops from above', options: EMITTERS, label: 'Ambient' },
    dropRate: {
      type: 'number',
      default: 0.3,
      min: 0,
      max: 3,
      step: 0.05,
      label: 'Ambient drops per second',
    },
    dropSize: {
      type: 'number',
      default: 0.07,
      min: 0.025,
      max: 0.14,
      step: 0.001,
      label: 'Drop radius (tank height = 1)',
    },
    dropAmount: {
      type: 'number',
      default: 3,
      min: 0.2,
      max: 4,
      step: 0.05,
      label: 'Ink in a drop',
    },
    dropSpeed: {
      type: 'number',
      default: 0.05,
      min: 0,
      max: 1.5,
      step: 0.01,
      label: 'Starting speed of a drop',
    },
    push: {
      type: 'number',
      default: 1.2,
      min: 0,
      max: 4,
      step: 0.05,
      label: 'Push (fraction of pointer speed)',
    },
    trail: {
      type: 'number',
      default: 0,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Ink left by a push',
    },
    drag: { type: 'select', default: DRAGS[0], options: DRAGS, label: 'Dragging' },
    inkDensity: {
      type: 'number',
      default: 8,
      min: 0.5,
      max: 30,
      step: 0.1,
      label: 'Ink strength (absorption)',
    },
    scatter: {
      type: 'number',
      default: 3,
      min: 0,
      max: 20,
      step: 0.1,
      label: 'Milkiness (scattering)',
    },
    steps: {
      type: 'number',
      default: 64,
      min: 12,
      max: 256,
      step: 1,
      label: 'March steps per ray',
    },
    renderScale: {
      type: 'number',
      default: 0.5,
      min: 0.2,
      max: 1,
      step: 0.01,
      label: 'March resolution (fraction of the canvas)',
    },
    lighting: { type: 'select', default: LIGHTING[1], options: LIGHTING, label: 'Lighting' },
    shadowSteps: {
      type: 'number',
      default: 12,
      min: 3,
      max: 32,
      step: 1,
      label: 'Steps toward the light',
    },
    skipEmpty: { type: 'boolean', default: true, label: 'Skip empty space' },
    jitter: { type: 'boolean', default: true, label: 'Jitter the ray starts' },
    sunAzimuth: { type: 'number', default: 35, min: 0, max: 360, step: 1, label: 'Sun direction' },
    sunElevation: {
      type: 'number',
      default: 55,
      min: 10,
      max: 88,
      step: 1,
      label: 'Sun elevation (degrees)',
    },
    obstacle: {
      type: 'number',
      default: 0,
      min: 0,
      max: 0.2,
      step: 0.005,
      label: 'Sphere obstacle radius (0 = none; uses Jacobi pressure)',
    },
    exposure: { type: 'number', default: 1.1, min: 0.3, max: 3, step: 0.01 },
    autoRotate: { type: 'boolean', default: true, label: 'Camera drifts when left alone' },
    view: { type: 'select', default: VIEWS[0], options: VIEWS, label: 'View' },
    slice: { type: 'number', default: 0.5, min: 0, max: 1, step: 0.01, label: 'Slice position' },
    debugGain: {
      type: 'number',
      default: 1,
      min: 0.1,
      max: 20,
      step: 0.1,
      label: 'Debug view gain',
    },
    stepsPerFrame: {
      type: 'number',
      default: 0,
      min: 0,
      max: 32,
      step: 1,
      label: 'Solver steps per frame (0 = by clock; for benchmarks)',
    },
    marchRepeat: {
      type: 'number',
      default: 1,
      min: 1,
      max: 16,
      step: 1,
      label: 'Repeat the ray march (for benchmarks)',
    },
  },
  presets: {
    'ink drop': {},
    'smoke plume': {
      emitter: 'plume from below',
      inks: 'chosen colour',
      inkColor: [0.86, 0.88, 0.9],
      buoyancy: -3,
      vorticity: 8,
      velocityDissipation: 0.15,
      inkFade: 0.35,
      dropRate: 0.5,
      dropSize: 0.06,
      dropAmount: 1.6,
      dropSpeed: 0.05,
      inkDensity: 1.5,
      scatter: 6,
      exposure: 1.3,
    },
    'many inks': {
      dropRate: 1,
      dropSize: 0.05,
      dropAmount: 2,
      inkFade: 0.3,
      vorticity: 5,
      buoyancy: 1.4,
      inkDensity: 7,
    },
    'slow ring': {
      speed: 0.55,
      dropRate: 0.2,
      dropSize: 0.08,
      dropSpeed: 0.1,
      vorticity: 3,
      inks: 'chosen colour',
    },
    'low-cost': {
      resolution: '32',
      scalarResolution: '64',
      pressureIterations: 1,
      scalarScheme: 'semi-lagrangian',
      vorticity: 2,
      steps: 32,
      renderScale: 0.35,
      lighting: 'off',
      skipEmpty: true,
    },
    'high quality': {
      resolution: '96',
      scalarResolution: '160',
      pressureIterations: 2,
      steps: 112,
      renderScale: 0.75,
      shadowSteps: 16,
    },
  },

  async create({ device, format, canvas, params, profiler }) {
    const random = createRandom(0x1c3d);
    const pointers = trackPointers(canvas, { capture: true });
    // A drag on the tank must not start selecting the page's text: restored on destroy, like touch-action.
    const previousUserSelect = canvas.style.userSelect;
    canvas.style.userSelect = 'none';
    const reducedMotion =
      typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

    // --- The solver ---------------------------------------------------------------------------------
    const fluid: Fluid3d = createFluid3d({
      device,
      resolution: Number(params.resolution),
      scalarResolution: Number(params.scalarResolution),
      size: TANK,
      maxStepsPerFrame: MAX_STEPS,
      profiler,
      label: 'fluid-3d',
    });
    // The box in world space: centred on the origin in x and z, standing on the table (y = 0).
    const size = (): Vec3 => [...fluid.grid.size] as Vec3;
    const toSolver = (p: Vec3): Vec3 => [p[0] + size()[0] / 2, p[1], p[2] + size()[2] / 2];
    const toWorld = (p: Vec3): Vec3 => [p[0] - size()[0] / 2, p[1], p[2] - size()[2] / 2];

    // --- Pipelines ------------------------------------------------------------------------------------
    const chunks = {
      'march_params.wgsl': MarchParams.wgsl,
      'ink.wgsl': inkShader.code,
      ...inkShader.chunks,
    };
    const marchPipeline = createFullscreenPipeline(device, {
      fragment: { label: 'fluid-3d/march.wgsl', code: marchShader.code, chunks },
      format: ['rgba16float', 'rgba16float'],
    });
    const compositePipeline = createFullscreenPipeline(device, {
      fragment: { label: 'fluid-3d/composite.wgsl', code: compositeShader.code, chunks },
      format,
    });
    const prepModule = createShader(device, {
      label: 'fluid-3d/volume-prep.wgsl',
      code: prepShader.code,
      chunks,
    });
    const lightPipeline = device.createComputePipeline({
      label: 'fluid-3d light volume',
      layout: 'auto',
      compute: { module: prepModule, entryPoint: 'cs_light' },
    });
    const occupancyPipeline = device.createComputePipeline({
      label: 'fluid-3d occupancy',
      layout: 'auto',
      compute: { module: prepModule, entryPoint: 'cs_occupancy' },
    });

    // --- Uniforms and textures --------------------------------------------------------------------------
    const cameraBuffer = createUniformBuffer(device, CameraUniform, 'fluid-3d camera');
    const paramsBuffer = createUniformBuffer(device, MarchParams, 'fluid-3d params');
    const sampler = device.createSampler({
      label: 'fluid-3d sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
    });
    const radiance = createRenderTarget(device, {
      label: 'fluid-3d radiance',
      format: 'rgba16float',
      width: 2,
      height: 2,
    });
    const transmittance = createRenderTarget(device, {
      label: 'fluid-3d transmittance',
      format: 'rgba16float',
      width: 2,
      height: 2,
    });
    let lightTexture: GPUTexture | undefined;
    let lightView: GPUTextureView | undefined;
    let occupancyTexture: GPUTexture | undefined;
    let occupancyView: GPUTextureView | undefined;
    let lightSize: Vec3 = [16, 16, 16];
    let preparedFor = -1;
    /** Bumps when the march targets are re-created, so cached bind groups that hold them are dropped. */
    let targetVersion = 0;

    function ensureVolumes() {
      if (preparedFor === fluid.generation && lightTexture && occupancyTexture) return;
      preparedFor = fluid.generation;
      lightTexture?.destroy();
      occupancyTexture?.destroy();
      const [sx, sy, sz] = fluid.grid.scalarSize;
      lightSize = [sx, sy, sz].map((n) => Math.max(16, Math.ceil(n / 2 / 4) * 4)) as Vec3;
      const make = (label: string, dims: readonly number[]) =>
        device.createTexture({
          label,
          dimension: '3d',
          size: [dims[0]!, dims[1]!, dims[2]!],
          format: 'rgba16float',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
        });
      lightTexture = make('fluid-3d light volume', lightSize);
      lightView = lightTexture.createView({ dimension: '3d' });
      occupancyTexture = make('fluid-3d occupancy', [
        Math.ceil(sx / 4),
        Math.ceil(sy / 4),
        Math.ceil(sz / 4),
      ]);
      occupancyView = occupancyTexture.createView({ dimension: '3d' });
    }

    // Bind groups reference textures that change: cached by the keys that identify them.
    const groups = new Map<string, GPUBindGroup>();
    function group(key: string, make: () => GPUBindGroup): GPUBindGroup {
      let found = groups.get(key);
      if (!found) {
        if (groups.size > 64) groups.clear();
        found = make();
        groups.set(key, found);
      }
      return found;
    }

    // --- Camera and interaction -------------------------------------------------------------------------
    const orbit = createOrbitState({
      yaw: 0.65,
      pitch: 0.28,
      distance: 2.6,
      target: [0, 0.5, 0],
    });
    const orbitOptions = {
      ...DEFAULT_ORBIT_OPTIONS,
      minPitch: 0.04,
      maxPitch: 1.4,
      autoSpeed: 0.1,
    };
    const camera = createCameraMatrices();
    let width = Math.max(1, canvas.width || 1);
    let height = Math.max(1, canvas.height || 1);
    let clock = 0;

    const unproject = (u: number, v: number, depth: number) =>
      unprojectPoint(camera.invViewProj, u, v, depth);
    function pointerRay(u: number, v: number): { origin: Vec3; dir: Vec3 } {
      const near = unproject(u, v, 0);
      const far = unproject(u, v, 1);
      const origin: Vec3 = [near[0]!, near[1]!, near[2]!];
      return { origin, dir: normalize(sub([far[0]!, far[1]!, far[2]!], origin)) };
    }
    function cameraForward(): Vec3 {
      const eye = camera.position;
      return normalize([
        orbit.target[0] - eye[0]!,
        orbit.target[1] - eye[1]!,
        orbit.target[2] - eye[2]!,
      ]);
    }

    interface Gesture {
      id: number;
      mode: 'push' | 'orbit';
      /** The plane the push lives in: through the point where the press entered the tank, facing the camera. */
      plane?: { point: Vec3; normal: Vec3 };
      last?: Vec3;
      travel: number;
      start: number;
      /** The press landed on the tank (so a tap releases a drop). */
      onTank: boolean;
      pressed: { x: number; y: number };
    }
    let gesture: Gesture | undefined;
    let interaction = 0;
    /** The colour a push leaves: one ink for a whole stroke. */
    let strokeInk: Vec3 | undefined;

    // --- Ink -------------------------------------------------------------------------------------------------
    let inkIndex = Math.floor(random() * 6);

    /** The ink of the next drop: linear colour. */
    function nextInk(): Vec3 {
      if (params.inks === 'chosen colour') return [...params.inkColor] as Vec3;
      inkIndex += 1 + Math.floor(random() * 2);
      return paletteInk(inkIndex);
    }

    function inkValue(colour: Vec3, amount: number): [number, number, number, number] {
      const od = inkDensity(colour);
      return [od[0] * amount, od[1] * amount, od[2] * amount, amount];
    }

    /** A lumpy drop at `at` (solver coordinates): a main blob and a few satellites, so it breaks up like a real one. */
    function drop(at: Vec3, scale = 1, ink: Vec3 = nextInk()) {
      const r = params.dropSize * scale;
      const amount = params.dropAmount;
      const [sx, sy, sz] = size();
      const clampP = (p: Vec3): Vec3 => [
        Math.min(Math.max(p[0], 0.04), sx - 0.04),
        Math.min(Math.max(p[1], 0.04), sy - 0.04),
        Math.min(Math.max(p[2], 0.04), sz - 0.04),
      ];
      const centre = clampP(at);
      fluid.addDensity(centre, centre, r, inkValue(ink, amount));
      for (let i = 0; i < 4; i++) {
        const angle = random() * Math.PI * 2;
        const lift = random() * 2 - 1;
        const reach = r * (0.5 + 0.5 * random());
        const p = clampP([
          centre[0] + Math.cos(angle) * Math.sqrt(1 - lift * lift) * reach,
          centre[1] + lift * reach,
          centre[2] + Math.sin(angle) * Math.sqrt(1 - lift * lift) * reach,
        ]);
        fluid.addDensity(p, p, r * (0.45 + 0.3 * random()), inkValue(ink, amount * 0.6));
      }
      // Starts moving: downward for ink that sinks, upward for smoke.
      const speed = params.dropSpeed * (params.buoyancy < 0 ? 1 : -1);
      if (speed !== 0) fluid.addImpulse(centre, centre, r * 1.4, [0, speed, 0]);
    }

    let dropDebt = 0.7;
    let seeded = false;
    function ambient(delta: number) {
      const sz = size();
      if (!seeded) {
        seeded = true;
        if (params.emitter === 'drops from above') {
          drop([sz[0] * 0.5, sz[1] * 0.8, sz[2] * 0.5], 1.15);
        }
      }
      const rate = params.dropRate * (reducedMotion ? 0.3 : 1) * (1 - 0.7 * interaction);
      if (params.emitter === 'plume from below') {
        // A steady source near the floor: a puff every so often, drifting a little.
        dropDebt += delta * rate * 4;
        let count = 0;
        while (dropDebt >= 1 && count < 3) {
          dropDebt -= 1;
          count++;
          const wobble = clock * 0.7;
          drop(
            [
              sz[0] * (0.5 + 0.12 * Math.sin(wobble)),
              sz[1] * 0.1,
              sz[2] * (0.5 + 0.12 * Math.cos(wobble * 1.3)),
            ],
            0.6 + 0.3 * random(),
          );
        }
      } else {
        dropDebt += delta * rate;
        if (dropDebt >= 1) {
          dropDebt = Math.min(dropDebt - 1, 1);
          drop(
            [
              sz[0] * (0.25 + 0.5 * random()),
              sz[1] * (0.72 + 0.18 * random()),
              sz[2] * (0.25 + 0.5 * random()),
            ],
            0.75 + 0.5 * random(),
          );
        }
      }
      dropDebt = Math.min(dropDebt, 4);
    }

    function handlePointers(delta: number) {
      const frame = pointers.poll();
      const sz = size();
      let orbitDx = 0;
      let orbitDy = 0;
      let dragging = false;
      let touched = false;

      for (const pointer of frame) {
        const ray = pointerRay(pointer.x, pointer.y);
        const solverOrigin = toSolver(ray.origin);
        const span = rayBox(solverOrigin, ray.dir, [0, 0, 0], sz);

        if (!gesture && pointer.justPressed) {
          const onTank = span !== null;
          const push = onTank && params.drag === DRAGS[0];
          gesture = {
            id: pointer.id,
            mode: push ? 'push' : 'orbit',
            travel: 0,
            start: clock,
            onTank,
            pressed: { x: pointer.x, y: pointer.y },
          };
          if (push && span) {
            // The push plane passes through a point about a third of the way into the tank.
            const t = span.near + 0.35 * (span.far - span.near);
            gesture.plane = {
              point: [
                solverOrigin[0] + ray.dir[0] * t,
                solverOrigin[1] + ray.dir[1] * t,
                solverOrigin[2] + ray.dir[2] * t,
              ],
              normal: cameraForward(),
            };
          }
        }
        if (!gesture || pointer.id !== gesture.id) continue;
        touched = true;

        if (pointer.down || pointer.justReleased) {
          gesture.travel += Math.abs(pointer.dx) + Math.abs(pointer.dy);
        }

        if (gesture.mode === 'push' && pointer.down && gesture.plane) {
          const q = rayPlane(solverOrigin, ray.dir, gesture.plane.point, gesture.plane.normal);
          if (q && gesture.travel > 0.01) {
            const from = gesture.last ?? q;
            const move = sub(q, from);
            const length = Math.hypot(...move);
            const seconds = Math.max(delta, 1 / 240);
            const speed = length / seconds;
            if (speed > 1e-3 && length > 1e-5) {
              const radius = 0.075;
              // The fluid is given the pointer's own velocity (capped) times Push, along the stroke's capsule.
              const gain = (params.push * Math.min(speed, POINTER_SPEED_CAP)) / length;
              const velocity: Vec3 = [move[0] * gain, move[1] * gain, move[2] * gain];
              fluid.addImpulse(from, q, radius, velocity, radius * 0.5);
              if (params.trail > 0) {
                strokeInk ??= nextInk();
                fluid.addDensity(
                  from,
                  q,
                  radius * 0.6,
                  inkValue(strokeInk, params.dropAmount * params.trail * 0.4),
                  radius * 0.3,
                );
              }
            }
            gesture.last = q;
          }
        } else if (gesture.mode === 'orbit' && pointer.down) {
          orbitDx += pointer.dx;
          orbitDy += pointer.dy;
          dragging = true;
        }

        if (pointer.justReleased || !pointer.down) {
          const wasTap =
            !pointer.cancelled && gesture.travel < 0.012 && clock - gesture.start < 0.55;
          if (wasTap && gesture.onTank && span) {
            // A tap releases a drop where the pointer ray enters the tank, a little inside it.
            const t = span.near + 0.04 * (span.far - span.near);
            drop(
              [
                solverOrigin[0] + ray.dir[0] * t,
                Math.min(solverOrigin[1] + ray.dir[1] * t, sz[1] - 0.12),
                solverOrigin[2] + ray.dir[2] * t,
              ],
              1.1,
            );
          }
          gesture = undefined;
          strokeInk = undefined;
        }
      }
      if (gesture && !frame.some((p) => p.id === gesture!.id)) gesture = undefined;

      const autoRotate = params.autoRotate && !reducedMotion;
      stepOrbit(
        orbit,
        { yaw: dragging ? -orbitDx * 3 : 0, pitch: dragging ? orbitDy * 2 : 0, dragging },
        delta,
        autoRotate ? orbitOptions : { ...orbitOptions, autoSpeed: 0 },
      );
      const blend = 1 - Math.exp(-delta * (touched ? 6 : 0.8));
      interaction += ((touched ? 1 : 0) - interaction) * blend;
    }

    function sync() {
      const p = fluid.params;
      p.pressureSolver = params.pressureSolver as 'multigrid' | 'jacobi';
      p.pressureIterations = params.pressureIterations;
      p.scalarScheme = params.scalarScheme as Fluid3dScheme;
      p.vorticity = params.vorticity;
      p.buoyancy = params.buoyancy;
      p.velocityDissipation = params.velocityDissipation;
      p.scalarDissipation = params.inkFade;
      const sz = fluid.grid.size;
      p.obstacle = {
        center: [sz[0] * 0.5, sz[1] * 0.3, sz[2] * 0.5],
        radius: params.obstacle,
      };
    }

    return {
      resize(w, h) {
        width = Math.max(1, Math.round(w));
        height = Math.max(1, Math.round(h));
      },

      frame({ delta }, encoder, target) {
        const dt = Math.min(Math.max(delta, 0), 0.1);
        clock += dt;
        const aspect = width / height;

        fluid.setResolution(Number(params.resolution), Number(params.scalarResolution));
        sync();
        ensureVolumes();

        // The camera first: picking needs this frame's matrices.
        handlePointers(dt);
        const sz = size();
        orbit.target = [0, sz[1] * 0.5, 0];
        computeCamera(
          {
            position: orbitEye(orbit),
            target: orbit.target,
            fovY: fitFovY(CAMERA_FOV_Y, aspect, MIN_HORIZONTAL_FOV),
            aspect,
            near: 0.05,
            far: 50,
          },
          camera,
        );
        setCameraUniform(cameraBuffer, camera, width, height);
        cameraBuffer.upload();

        // `speed` scales the time the fluid and the drops see, not the camera.
        const simDelta = dt * params.speed;
        ambient(simDelta);
        if (params.stepsPerFrame > 0) fluid.advance(encoder, params.stepsPerFrame);
        else fluid.update(encoder, simDelta);
        if (params.view === 'velocity slice') fluid.resolve(encoder);

        // --- Scene parameters -----------------------------------------------------------------------------
        const azimuth = params.sunAzimuth * DEG;
        const elevation = params.sunElevation * DEG;
        const lightDir: Vec3 = [
          Math.cos(elevation) * Math.sin(azimuth),
          Math.sin(elevation),
          Math.cos(elevation) * Math.cos(azimuth),
        ];
        const lightingMode = Math.max(
          0,
          LIGHTING.indexOf(params.lighting as (typeof LIGHTING)[number]),
        );
        const obstacleCentre = toWorld([sz[0] * 0.5, sz[1] * 0.3, sz[2] * 0.5]);
        paramsBuffer.set({
          box_min: [-sz[0] / 2, 0, -sz[2] / 2],
          abs_gain: params.inkDensity,
          box_max: [sz[0] / 2, sz[1], sz[2] / 2],
          scatter: params.scatter,
          light_dir: lightDir,
          phase_g: 0.35,
          light_color: [2.6, 2.5, 2.3],
          edge_strength: 1,
          ambient: [0.55, 0.62, 0.72],
          exposure: params.exposure,
          water: [0.18, 0.06, 0.04],
          steps: Math.round(params.steps),
          light_size: lightSize,
          light_steps: Math.round(params.shadowSteps),
          sphere: [...obstacleCentre, params.obstacle],
          jitter: params.jitter ? 1 : 0,
          frame: 0,
          lighting: lightingMode,
          occupancy: params.skipEmpty ? 1 : 0,
          view: Math.max(0, VIEWS.indexOf(params.view as (typeof VIEWS)[number])),
          slice: params.slice,
          debug_gain: params.debugGain,
        });
        paramsBuffer.upload();

        // --- Preparation: occupancy and light volume --------------------------------------------------------
        const density = fluid.density;
        const key = `${fluid.generation}:${density.slot}`;
        const needsPrep = params.view === 'ink';
        if (needsPrep && (params.skipEmpty || lightingMode === 1)) {
          const pass = encoder.beginComputePass({
            label: 'fluid-3d prepare',
            timestampWrites: profiler.pass('light + occupancy') as
              GPUComputePassTimestampWrites | undefined,
          });
          if (lightingMode === 1) {
            pass.setPipeline(lightPipeline);
            pass.setBindGroup(
              0,
              group(`light:${key}:${preparedFor}`, () =>
                device.createBindGroup({
                  label: 'fluid-3d light',
                  layout: lightPipeline.getBindGroupLayout(0),
                  entries: [
                    { binding: 0, resource: { buffer: paramsBuffer.buffer } },
                    { binding: 1, resource: density.view },
                    { binding: 2, resource: sampler },
                    { binding: 3, resource: lightView! },
                  ],
                }),
              ),
            );
            pass.dispatchWorkgroups(
              Math.ceil(lightSize[0] / 4),
              Math.ceil(lightSize[1] / 4),
              Math.ceil(lightSize[2] / 4),
            );
          }
          if (params.skipEmpty) {
            pass.setPipeline(occupancyPipeline);
            pass.setBindGroup(
              0,
              group(`occ:${key}:${preparedFor}`, () =>
                device.createBindGroup({
                  label: 'fluid-3d occupancy',
                  layout: occupancyPipeline.getBindGroupLayout(0),
                  entries: [
                    { binding: 1, resource: density.view },
                    { binding: 4, resource: occupancyView! },
                  ],
                }),
              ),
            );
            const [ox, oy, oz] = fluid.grid.scalarSize.map((n) => Math.ceil(n / 4));
            pass.dispatchWorkgroups(Math.ceil(ox! / 4), Math.ceil(oy! / 4), Math.ceil(oz! / 4));
          }
          pass.end();
        }

        // --- The march, at a fraction of the canvas, then the composite at full size -------------------------
        const scale = Math.min(Math.max(params.renderScale, 0.1), 1);
        const mw = Math.max(1, Math.round(width * scale));
        const mh = Math.max(1, Math.round(height * scale));
        if (radiance.resize(mw, mh)) targetVersion++;
        transmittance.resize(mw, mh);
        const marchGroup = group(`march:${key}:${preparedFor}`, () =>
          device.createBindGroup({
            label: 'fluid-3d march',
            layout: marchPipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: paramsBuffer.buffer } },
              { binding: 1, resource: { buffer: cameraBuffer.buffer } },
              { binding: 2, resource: density.view },
              { binding: 3, resource: lightView! },
              { binding: 4, resource: occupancyView! },
              { binding: 5, resource: sampler },
            ],
          }),
        );
        if (params.view === 'ink') {
          const repeats = Math.max(1, Math.round(params.marchRepeat));
          for (let i = 0; i < repeats; i++) {
            renderFullscreen(
              encoder,
              [radiance.view, transmittance.view],
              marchPipeline,
              [marchGroup],
              {
                label: 'fluid-3d march',
                timestampWrites:
                  i === 0
                    ? profiler.pass('march')
                    : i === repeats - 1
                      ? profiler.pass('march (last)')
                      : undefined,
              },
            );
          }
        }
        const compositeGroup = group(`composite:${key}:${targetVersion}`, () =>
          device.createBindGroup({
            label: 'fluid-3d composite',
            layout: compositePipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: paramsBuffer.buffer } },
              { binding: 1, resource: { buffer: cameraBuffer.buffer } },
              { binding: 2, resource: radiance.view },
              { binding: 3, resource: transmittance.view },
              { binding: 4, resource: sampler },
              { binding: 5, resource: fluid.fieldsView },
              { binding: 6, resource: density.view },
            ],
          }),
        );
        renderFullscreen(encoder, target, compositePipeline, [compositeGroup], {
          label: 'fluid-3d composite',
          timestampWrites: profiler.pass('composite'),
        });
      },

      destroy() {
        pointers.destroy();
        canvas.style.userSelect = previousUserSelect;
        fluid.destroy();
        cameraBuffer.destroy();
        paramsBuffer.destroy();
        radiance.destroy();
        transmittance.destroy();
        lightTexture?.destroy();
        occupancyTexture?.destroy();
        groups.clear();
      },
    };
  },
});
