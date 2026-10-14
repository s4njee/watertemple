import { createFixedStepper, type FixedStepper } from '../loop.ts';
import { noProfiler, type PassProfiler } from '../profiler.ts';
import { renderFullscreen } from '../fullscreen.ts';
import { createShader } from '../shader.ts';
import { createUniformBuffer, defineUniformStruct, uniformArray } from '../uniforms.ts';
import computeSource from './heightfield-compute.wgsl';
import coreSource from './heightfield-core.wgsl';
import renderSource from './heightfield-render.wgsl';
import { createObstacleMask, type ObstacleMask } from './obstacles.ts';

// The heightfield solver (story 5.1): a discrete 2D wave equation on a grid whose state lives in
// textures. Two interchangeable implementations of the same maths, `render` (fragment passes over
// ping-pong render targets) and `compute` (compute passes over storage textures).
//
// STATE. Two `rgba16float` ping-pong textures hold (height, velocity) per cell; a third, `output`,
// holds what consumers read: (height, velocity, slope x, slope y). `rgba16float` is renderable,
// filterable and write-only-storage in core WebGPU on every device, so it is the one format both
// variants and every consumer can share. See heightfield-core.wgsl for the scheme and
// src/wgsl/heightfield.wgsl for the decode helpers.
//
// UNITS. Positions are canvas uv (0..1, top-left origin). Lengths (`radius`, `cellSize`, `waveSpeed`)
// are in IMAGE HEIGHTS (canvas height = 1), the unit refraction.wgsl uses. The grid's cells are
// square; the stored height is in cells.
//
// STABILITY. With the 9-point laplacian the scheme is stable while the Courant number
// C = waveSpeed * dt / cellSize is at most 2 / sqrt(32/6) = 0.866. The solver never lets C exceed
// COURANT_LIMIT (0.8): it picks the step dt itself, dt = C * cellSize / waveSpeed, so a faster wave
// or a finer grid means MORE steps per second, never a larger C. Hitting the per-frame step cap
// makes the simulation run slower than real time (slow motion) instead of unstable.

export type HeightfieldKind = 'render' | 'compute';
export type BoundaryMode = 'reflect' | 'absorb';

/** Format of the solver's state and output textures. */
export const HEIGHTFIELD_FORMAT: GPUTextureFormat = 'rgba16float';
/** Largest stable Courant number the solver will use (the scheme's limit is 0.866). */
export const COURANT_LIMIT = 0.8;
/** Impulses (drops plus displacements) accepted between two updates; more are dropped. */
export const MAX_IMPULSES = 64;
/** Smallest footprint radius, in cells: anything narrower aliases on the grid. */
export const MIN_RADIUS_CELLS = 1.5;
/** Cells of height a strength-1 impulse makes per cell of radius (so strength ~ a peak slope). */
export const IMPULSE_GAIN = 0.2;
const MAX_AMPLITUDE_CELLS = 24;
const WAVE_SPEED_RANGE = [0.02, 8] as const;
// 16 x 16 = 256 invocations, the default per-workgroup maximum. Measured on Apple GPU: 10 to 20 percent
// faster than 8 x 8 and the same as 32 x 8 (cost study in tests/heightfield.test.ts).
const WORKGROUP = 16;

/** Live, mutable simulation parameters: change them any time, they are read on every update. */
export interface WaveParams {
  /** Wave speed, image heights per second. A ring crosses the canvas height in 1 / waveSpeed seconds. */
  waveSpeed: number;
  /** Velocity decay rate, 1 / s. Amplitude falls as exp(-damping * t / 2); 0 never loses energy. */
  damping: number;
  /** Rate at which the mean level relaxes back to flat, 1 / s. Keeps rain from lifting the whole pond. */
  leak: number;
  /** `reflect`: walls at the grid edge. `absorb`: a sponge layer eats waves before they reflect. */
  boundary: BoundaryMode;
  /** Courant number c dt / cellSize per step, clamped to [0.1, COURANT_LIMIT]. Smaller is more accurate and costs more steps. */
  courant: number;
  /** Width of the absorbing sponge, in cells. */
  spongeCells: number;
  /** Strength of the sponge (1 = nominal). */
  spongeStrength: number;
}

export const DEFAULT_WAVE_PARAMS: Readonly<WaveParams> = {
  waveSpeed: 0.45,
  damping: 0.6,
  leak: 0.05,
  boundary: 'absorb',
  courant: 0.5,
  spongeCells: 14,
  spongeStrength: 1,
};

/** The grid and what a shading pass needs to know about it. */
export interface HeightfieldGrid {
  /** Cells across and down. */
  readonly width: number;
  readonly height: number;
  /** width / height of the grid (the canvas aspect, to within one cell). */
  readonly aspect: number;
  /** Side of a cell in image heights: 1 / height. Multiply a stored height (cells) by this for image heights. */
  readonly cellSize: number;
  /** One cell in uv: [1 / width, 1 / height]. */
  readonly texelSize: readonly [number, number];
}

export interface HeightfieldOptions {
  device: GPUDevice;
  /** Which implementation to run. Default `render`. Fixed for the life of the solver. */
  kind?: HeightfieldKind;
  /** Cells along the longer side of the grid. Default 256. */
  resolution?: number;
  /** Canvas size (any units) that gives the grid its aspect ratio. Default square. */
  width?: number;
  height?: number;
  /** Initial parameters, merged over DEFAULT_WAVE_PARAMS. */
  params?: Partial<WaveParams>;
  /** Most steps one `update` runs. Default 16. */
  maxStepsPerFrame?: number;
  /** Receives a timestamp slot for the first and last step, the splat and the resolve pass. */
  profiler?: PassProfiler;
  label?: string;
}

export interface Heightfield {
  readonly kind: HeightfieldKind;
  /** Live parameters. Mutate freely. */
  readonly params: WaveParams;
  /** The current grid. A new object after every re-creation. */
  readonly grid: HeightfieldGrid;
  /** Increments every time the grid is re-created: rebuild bind groups that hold `texture` or `view`. */
  readonly generation: number;
  /** The output texture: (height, velocity, slope x, slope y), filterable. Stable between updates. */
  readonly texture: GPUTexture;
  readonly view: GPUTextureView;
  /** Solid-versus-water mask the waves reflect off. */
  readonly obstacles: ObstacleMask;
  /** Seconds of simulated time per step at the current wave speed and Courant number. */
  readonly stepSeconds: number;
  /** Steps run by the last `update` / `advance`. */
  readonly lastSteps: number;

  /**
   * Re-fits the grid to a canvas of this size (any unit: only the aspect matters). Returns true if
   * the grid was re-created, which clears the surface to flat and bumps `generation`.
   */
  resize(width: number, height: number): boolean;
  /** Changes the cells along the long side; re-creates the grid (surface cleared) if that changes it. */
  setResolution(cells: number): boolean;

  /** Queues a drop: a smooth bump of radius `radius` (image heights) at uv (x, y). Positive strength raises. */
  addDrop(x: number, y: number, radius: number, strength: number): boolean;
  /**
   * Queues the displacement of a moving footprint (a disc or sphere going from uv `from` to uv `to`):
   * the surface drops where the shape arrives and rises where it leaves, so a moving shape pushes
   * water aside and leaves a wake while a stationary one does nothing. Volume is conserved: what
   * drops at `to` rises at `from`. Call it once per frame per shape, with the position the shape had
   * last frame and the one it has now. Do not cut a path into several calls inside one frame: their
   * ends cancel each other exactly, and only the first `from` and last `to` would remain. (The
   * wake comes from the water moving between frames.) A move longer than two radii leaves two
   * separate bumps, which is right for a shape that jumps and too coarse for one that races across
   * the surface; a pointer trail is better made of drops (see stir.ts).
   */
  displace(
    from: readonly [number, number],
    to: readonly [number, number],
    radius: number,
    strength: number,
  ): boolean;
  /** Impulses queued and not yet applied. */
  readonly pendingImpulses: number;

  /**
   * Advances the simulation by `delta` seconds of wall time at a fixed internal timestep and
   * applies queued impulses. Call at most once per submitted command buffer. Returns the steps run.
   */
  update(encoder: GPUCommandEncoder, delta: number): number;
  /** Runs exactly `steps` steps, whatever the clock says (benchmarks, tests), applying queued impulses first. */
  advance(encoder: GPUCommandEncoder, steps: number): void;
  /** Flattens the surface (state and output). */
  reset(encoder: GPUCommandEncoder): void;
  destroy(): void;
}

// --- Uniforms ------------------------------------------------------------------------------------

const SimParams = defineUniformStruct('SimParams', {
  size: 'vec2u',
  k: 'f32',
  v_decay: 'f32',
  h_decay: 'f32',
  sponge_cells: 'f32',
  sponge_rate: 'f32',
  solid: 'u32',
  impulse_count: 'u32',
  impulses: uniformArray('vec4f', MAX_IMPULSES * 2),
});

// --- Pipelines, cached per device -----------------------------------------------------------------

type Pipeline = GPURenderPipeline | GPUComputePipeline;

interface Pipelines {
  layout: GPUBindGroupLayout;
  splat: Pipeline;
  step: Pipeline;
  resolve: Pipeline;
}

const pipelineCache = new WeakMap<GPUDevice, Partial<Record<HeightfieldKind, Pipelines>>>();

function createPipelines(device: GPUDevice, kind: HeightfieldKind): Pipelines {
  const visibility = kind === 'render' ? GPUShaderStage.FRAGMENT : GPUShaderStage.COMPUTE;
  const entries: GPUBindGroupLayoutEntry[] = [
    { binding: 0, visibility, buffer: { type: 'uniform' } },
    { binding: 1, visibility, texture: { sampleType: 'unfilterable-float' } },
    { binding: 2, visibility, texture: { sampleType: 'unfilterable-float' } },
  ];
  if (kind === 'compute') {
    entries.push({
      binding: 3,
      visibility,
      storageTexture: { access: 'write-only', format: HEIGHTFIELD_FORMAT },
    });
  }
  const layout = device.createBindGroupLayout({ label: `heightfield ${kind} layout`, entries });
  const pipelineLayout = device.createPipelineLayout({
    label: `heightfield ${kind} pipeline layout`,
    bindGroupLayouts: [layout],
  });
  const shader = kind === 'render' ? renderSource : computeSource;
  const module = createShader(device, {
    label: `sim/heightfield-${kind}.wgsl`,
    code: shader.code,
    chunks: {
      ...shader.chunks,
      'sim_params.wgsl': SimParams.wgsl,
      'sim_core.wgsl': coreSource.code,
      ...coreSource.chunks,
    },
  });

  const make = (name: 'splat' | 'step' | 'resolve'): Pipeline =>
    kind === 'render'
      ? device.createRenderPipeline({
          label: `heightfield render ${name}`,
          layout: pipelineLayout,
          vertex: { module, entryPoint: 'fullscreen_vs' },
          fragment: {
            module,
            entryPoint: `fs_${name}`,
            targets: [{ format: HEIGHTFIELD_FORMAT }],
          },
        })
      : device.createComputePipeline({
          label: `heightfield compute ${name}`,
          layout: pipelineLayout,
          compute: { module, entryPoint: `cs_${name}` },
        });

  return { layout, splat: make('splat'), step: make('step'), resolve: make('resolve') };
}

function getPipelines(device: GPUDevice, kind: HeightfieldKind): Pipelines {
  let perDevice = pipelineCache.get(device);
  if (!perDevice) pipelineCache.set(device, (perDevice = {}));
  return (perDevice[kind] ??= createPipelines(device, kind));
}

// --- Helpers -------------------------------------------------------------------------------------

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

/** Grid size for `resolution` cells on the long side of a canvas with this aspect ratio. */
export function computeGridSize(
  resolution: number,
  aspect: number,
  maxDimension = 4096,
): { width: number; height: number } {
  const long = Math.round(clamp(Number.isFinite(resolution) ? resolution : 256, 16, maxDimension));
  const a = Number.isFinite(aspect) && aspect > 0 ? aspect : 1;
  const width = a >= 1 ? long : Math.max(16, Math.round(long * a));
  const height = a >= 1 ? Math.max(16, Math.round(long / a)) : long;
  return { width: Math.min(width, maxDimension), height: Math.min(height, maxDimension) };
}

function makeGrid(width: number, height: number): HeightfieldGrid {
  return Object.freeze({
    width,
    height,
    aspect: width / height,
    cellSize: 1 / height,
    texelSize: Object.freeze([1 / width, 1 / height] as const),
  });
}

interface Resources {
  grid: HeightfieldGrid;
  /** Ping-pong state: (height, velocity). */
  state: [GPUTexture, GPUTexture];
  stateViews: [GPUTextureView, GPUTextureView];
  output: GPUTexture;
  outputView: GPUTextureView;
  /** Bind groups indexed by the parity of the texture being read. */
  stepGroups: [GPUBindGroup, GPUBindGroup];
  resolveGroups: [GPUBindGroup, GPUBindGroup];
}

export function createHeightfield(options: HeightfieldOptions): Heightfield {
  const {
    device,
    kind = 'render',
    maxStepsPerFrame = 16,
    profiler = noProfiler,
    label = 'heightfield',
  } = options;
  let resolution = options.resolution ?? 256;
  let aspect = (options.width ?? 1) / (options.height ?? 1);
  const maxDimension = Math.min(device.limits.maxTextureDimension2D, 4096);
  const params: WaveParams = { ...DEFAULT_WAVE_PARAMS, ...options.params };
  const pipelines = getPipelines(device, kind);
  const uniforms = createUniformBuffer(device, SimParams, `${label} params`);

  const usage =
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.COPY_SRC |
    GPUTextureUsage.COPY_DST |
    (kind === 'compute' ? GPUTextureUsage.STORAGE_BINDING : 0);

  let destroyed = false;
  let generation = 0;
  let parity: 0 | 1 = 0;
  let lastSteps = 0;
  let stepper: FixedStepper | undefined;
  let stepperSeconds = 0;
  let resources: Resources;

  const queue = new Float32Array(MAX_IMPULSES * 8);
  let queued = 0;

  const mask = createObstacleMask(device, label);

  function allocate(width: number, height: number): Resources {
    const grid = makeGrid(width, height);
    const make = (name: string) =>
      device.createTexture({
        label: `${label} ${name}`,
        size: [width, height],
        format: HEIGHTFIELD_FORMAT,
        usage,
      });
    const state: [GPUTexture, GPUTexture] = [make('state A'), make('state B')];
    const output = make('output');
    mask.resize(width, height);

    const group = (read: GPUTexture, write: GPUTexture, name: string) => {
      const entries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: uniforms.buffer } },
        { binding: 1, resource: read.createView() },
        { binding: 2, resource: mask.view },
      ];
      if (kind === 'compute') entries.push({ binding: 3, resource: write.createView() });
      return device.createBindGroup({
        label: `${label} ${name}`,
        layout: pipelines.layout,
        entries,
      });
    };
    return {
      grid,
      state,
      stateViews: [state[0].createView(), state[1].createView()],
      output,
      outputView: output.createView({ label: `${label} output view` }),
      stepGroups: [
        group(state[0], state[1], 'step A to B'),
        group(state[1], state[0], 'step B to A'),
      ],
      resolveGroups: [group(state[0], output, 'resolve A'), group(state[1], output, 'resolve B')],
    };
  }

  function release(r: Resources) {
    r.state[0].destroy();
    r.state[1].destroy();
    r.output.destroy();
  }

  function rebuild(): boolean {
    const size = computeGridSize(resolution, aspect, maxDimension);
    if (resources && size.width === resources.grid.width && size.height === resources.grid.height) {
      return false;
    }
    if (resources) release(resources);
    resources = allocate(size.width, size.height);
    parity = 0;
    generation++;
    stepper = undefined;
    return true;
  }

  rebuild();

  // --- Time ----------------------------------------------------------------------------------

  const courant = () =>
    clamp(Number.isFinite(params.courant) ? params.courant : 0.5, 0.1, COURANT_LIMIT);
  const waveSpeed = () =>
    clamp(
      Number.isFinite(params.waveSpeed) ? params.waveSpeed : DEFAULT_WAVE_PARAMS.waveSpeed,
      WAVE_SPEED_RANGE[0],
      WAVE_SPEED_RANGE[1],
    );
  const stepSeconds = () => (courant() * resources.grid.cellSize) / waveSpeed();

  function writeUniforms() {
    const { grid } = resources;
    const c = courant();
    const dt = stepSeconds();
    const absorbing = params.boundary === 'absorb';
    const sponge = clamp(
      Math.round(params.spongeCells),
      0,
      Math.floor(Math.min(grid.width, grid.height) / 3),
    );
    // Per-step multipliers from rates per second, so damping looks the same at every resolution
    // and wave speed (dt differs between them).
    const damping = Math.max(0, Number.isFinite(params.damping) ? params.damping : 0);
    const leak = Math.max(0, Number.isFinite(params.leak) ? params.leak : 0);
    const spongeStrength = Math.max(
      0,
      Number.isFinite(params.spongeStrength) ? params.spongeStrength : 1,
    );

    const data = new Float32Array(MAX_IMPULSES * 8);
    const radiusBase = grid.height;
    for (let i = 0; i < queued; i++) {
      const o = i * 8;
      const radius = Math.max(MIN_RADIUS_CELLS, (queue[o + 2] ?? 0) * radiusBase);
      const amplitude = clamp(
        (queue[o + 3] ?? 0) * IMPULSE_GAIN * radius,
        -MAX_AMPLITUDE_CELLS,
        MAX_AMPLITUDE_CELLS,
      );
      data[o] = (queue[o] ?? 0) * grid.width;
      data[o + 1] = (queue[o + 1] ?? 0) * grid.height;
      data[o + 2] = radius;
      data[o + 3] = amplitude;
      data[o + 4] = (queue[o + 4] ?? 0) * grid.width;
      data[o + 5] = (queue[o + 5] ?? 0) * grid.height;
      data[o + 6] = queue[o + 6] ?? 0;
    }
    uniforms.set({
      size: [grid.width, grid.height],
      k: c * c,
      v_decay: Math.exp(-damping * dt),
      h_decay: Math.exp(-leak * dt),
      sponge_cells: absorbing ? sponge : 0,
      sponge_rate: clamp(spongeStrength * c, 0, 0.5),
      solid: mask.enabled ? 1 : 0,
      impulse_count: queued,
      impulses: data,
    });
    uniforms.upload();
  }

  // --- Passes --------------------------------------------------------------------------------

  function runPass(
    encoder: GPUCommandEncoder,
    pipeline: Pipeline,
    group: GPUBindGroup,
    target: GPUTextureView,
    passLabel: string,
    timestampLabel: string | undefined,
  ) {
    const timestampWrites = timestampLabel ? profiler.pass(timestampLabel) : undefined;
    if (kind === 'render') {
      renderFullscreen(encoder, target, pipeline as GPURenderPipeline, [group], {
        label: passLabel,
        timestampWrites,
      });
    } else {
      const pass = encoder.beginComputePass({
        label: passLabel,
        timestampWrites: timestampWrites as GPUComputePassTimestampWrites | undefined,
      });
      pass.setPipeline(pipeline as GPUComputePipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(
        Math.ceil(resources.grid.width / WORKGROUP),
        Math.ceil(resources.grid.height / WORKGROUP),
      );
      pass.end();
    }
  }

  function run(encoder: GPUCommandEncoder, steps: number) {
    if (destroyed) return;
    const count = Math.max(0, Math.floor(steps));
    lastSteps = count;
    if (count === 0 && queued === 0) return;
    writeUniforms();
    const r = resources;

    if (queued > 0) {
      runPass(
        encoder,
        pipelines.splat,
        r.stepGroups[parity],
        r.stateViews[1 - parity]!,
        'sim splat',
        'sim splat',
      );
      parity = (1 - parity) as 0 | 1;
      queued = 0;
    }
    for (let i = 0; i < count; i++) {
      // Time the first and the last step; the HUD's frame span then covers everything between.
      const timed = i === 0 || i === count - 1 ? 'sim step' : undefined;
      runPass(
        encoder,
        pipelines.step,
        r.stepGroups[parity],
        r.stateViews[1 - parity]!,
        'sim step',
        timed,
      );
      parity = (1 - parity) as 0 | 1;
    }
    runPass(
      encoder,
      pipelines.resolve,
      r.resolveGroups[parity],
      r.outputView,
      'sim resolve',
      'sim resolve',
    );
  }

  function enqueue(
    x: number,
    y: number,
    radius: number,
    strength: number,
    x1: number,
    y1: number,
    isDisplacement: boolean,
  ): boolean {
    if (queued >= MAX_IMPULSES) return false;
    if (![x, y, radius, strength, x1, y1].every(Number.isFinite)) return false;
    const o = queued * 8;
    queue[o] = x;
    queue[o + 1] = y;
    queue[o + 2] = Math.max(0, radius);
    queue[o + 3] = strength;
    queue[o + 4] = x1;
    queue[o + 5] = y1;
    queue[o + 6] = isDisplacement ? 1 : 0;
    queued++;
    return true;
  }

  return {
    kind,
    params,
    get grid() {
      return resources.grid;
    },
    get generation() {
      return generation;
    },
    get texture() {
      return resources.output;
    },
    get view() {
      return resources.outputView;
    },
    obstacles: mask,
    get stepSeconds() {
      return stepSeconds();
    },
    get lastSteps() {
      return lastSteps;
    },
    get pendingImpulses() {
      return queued;
    },

    resize(width, height) {
      if (!(width > 0 && height > 0)) return false;
      aspect = width / height;
      return rebuild();
    },
    setResolution(cells) {
      if (!Number.isFinite(cells)) return false;
      resolution = cells;
      return rebuild();
    },

    addDrop(x, y, radius, strength) {
      return enqueue(x, y, radius, strength, 0, 0, false);
    },
    displace(from, to, radius, strength) {
      return enqueue(from[0], from[1], radius, strength, to[0], to[1], true);
    },

    update(encoder, delta) {
      if (destroyed) return 0;
      const dt = stepSeconds();
      if (!stepper || Math.abs(dt - stepperSeconds) > dt * 1e-3) {
        stepper = createFixedStepper(dt, { maxSteps: maxStepsPerFrame });
        stepperSeconds = dt;
      }
      const { steps } = stepper.advance(Number.isFinite(delta) ? delta : 0);
      run(encoder, steps);
      return steps;
    },
    advance(encoder, steps) {
      run(encoder, steps);
    },
    reset(encoder) {
      queued = 0;
      for (const texture of [...resources.state, resources.output]) {
        encoder
          .beginRenderPass({
            label: `${label} clear`,
            colorAttachments: [{ view: texture.createView(), loadOp: 'clear', storeOp: 'store' }],
          })
          .end();
      }
      parity = 0;
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      release(resources);
      mask.destroy();
      uniforms.destroy();
    },
  };
}
