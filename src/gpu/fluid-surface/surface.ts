import type { CameraMatrices } from '../scene/camera.ts';
import { CameraUniform, setCameraUniform } from '../scene/camera.ts';
import { createDepthTarget } from '../scene/depth.ts';
import { createFullscreenPipeline } from '../fullscreen.ts';
import { linearSampler, perDevice } from '../passes/common.ts';
import type { ParticleSet } from '../particles/types.ts';
import { noProfiler, type PassProfiler } from '../profiler.ts';
import { createShader } from '../shader.ts';
import { createRenderTarget, type RenderTarget } from '../targets.ts';
import { createUniformBuffer } from '../uniforms.ts';
import compositeSource from './composite.wgsl';
import {
  clampQuality,
  DEBUG_VIEWS,
  DEFAULT_QUALITY,
  FILTERS,
  FLUID_STRUCT_CHUNKS,
  FluidFrame,
  FluidPass,
  resolveParams,
  STAGES,
  SURFACES,
  type FluidStage,
  type FluidSurfaceParams,
  type FluidSurfaceQuality,
} from './frame.ts';
import normalsSource from './normals.wgsl';
import smoothSource from './smooth.wgsl';
import sphereDepthSource from './sphere-depth.wgsl';
import thicknessBlurSource from './thickness-blur.wgsl';
import thicknessSource from './thickness.wgsl';

/** Formats of the chain. See `createFluidSurface` for why each was chosen. */
export const FLUID_FORMATS = {
  /** Linear view-space depth. 32-bit: 16 bits cannot carry normals. Read with `textureLoad` only. */
  depth: 'r32float',
  /** The depth buffer of the sphere pass. */
  depthBuffer: 'depth32float',
  /** View-space normal (xyz) and a validity flag (w). Filterable, so it is safe on every device. */
  normals: 'rgba16float',
  /** Thickness (r) and thickness times the particle scalar (g): additive blending needs a blendable format. */
  thickness: 'rg16float',
} as const satisfies Record<string, GPUTextureFormat>;

const READABLE = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
/** Every chain image can be read back, and seeded by a test that feeds the filters a depth image of its own. */
const INSPECTABLE = READABLE | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;

export interface FluidSurfaceOptions extends Partial<FluidSurfaceQuality> {
  device: GPUDevice;
  /** Format of the `target` views passed to `render`. */
  format: GPUTextureFormat;
  /** Size of the target in pixels. */
  width: number;
  height: number;
  label?: string;
}

export interface FluidRenderOptions {
  /**
   * The particles to draw. Only `positions`, `count`, `capacity` and `radius` are read. Pass the
   * set the simulation exposes this frame: a ping-ponged set names a different buffer each frame
   * and the renderer binds whatever it names, with nothing cached per buffer.
   */
  particles: ParticleSet;
  /** The camera the scene was drawn with. */
  camera: CameraMatrices;
  /** Linear colour of the scene, `width x height`, filterable (rgba16float). Not written. */
  sceneColor: GPUTextureView;
  /** The depth buffer that scene was drawn with, `width x height`. Not written. */
  sceneDepth: GPUTextureView;
  /** Receives scene plus liquid. May not be the same texture as `sceneColor`. */
  target: GPUTextureView;
  params?: FluidSurfaceParams;
  profiler?: PassProfiler;
  /**
   * Run only these stages (default all). For measurement: draw once with everything, then repeat
   * one stage N times in a single submit and take the slope. A stage that is left out reuses what
   * the last full render left in its targets.
   */
  stages?: readonly FluidStage[];
}

/** The chain's images, readable for tests and debug tools. Views change when the surface resizes. */
export interface FluidSurfaceTextures {
  /** Linear depth of the sphere pass, 0 where there is no fluid. */
  readonly rawDepth: RenderTarget;
  /** Linear depth after smoothing. */
  readonly smoothDepth: RenderTarget;
  /** View-space normals of the raw depth (computed when a mode or view needs them). */
  readonly rawNormals: RenderTarget;
  /** View-space normals of the smoothed depth. */
  readonly smoothNormals: RenderTarget;
  /** Thickness (r) and thickness times scalar (g), blurred. */
  readonly thickness: RenderTarget;
}

export interface FluidSurface {
  readonly quality: Readonly<FluidSurfaceQuality>;
  /** Merges into the quality. Resolution changes reallocate on the next `render`. */
  setQuality(quality: Partial<FluidSurfaceQuality>): void;
  /** Size of the target, in pixels. */
  resize(width: number, height: number): void;
  /**
   * Draws the liquid over the scene into `target`: depth, smoothing, normals, thickness and the
   * composite, each as its own labelled pass. One call per submit: the uniforms are written with
   * `queue.writeBuffer`, so two calls before one submit would both use the last call's values.
   */
  render(encoder: GPUCommandEncoder, options: FluidRenderOptions): void;
  /** Bumps whenever the internal targets are re-created. */
  readonly generation: number;
  /** Chain and thickness sizes in pixels, as of the last `render` or `resize`. */
  readonly sizes: { chain: readonly [number, number]; thickness: readonly [number, number] };
  readonly textures: FluidSurfaceTextures;
  /** Counters for tests and the HUD. */
  readonly stats: {
    bindGroupsCreated: number;
    particlesDrawn: number;
    lastPasses: readonly string[];
  };
  destroy(): void;
}

// --- Pipelines, cached per device ----------------------------------------------------------------

const PARTICLE_LAYOUT: GPUVertexBufferLayout = {
  arrayStride: 16,
  stepMode: 'instance',
  attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }],
};

const ADD: GPUBlendComponent = { operation: 'add', srcFactor: 'one', dstFactor: 'one' };

interface Pipelines {
  /** Layout shared by the two particle passes: camera, frame, scene depth. */
  particleLayout: GPUBindGroupLayout;
  sphereDepth: GPURenderPipeline;
  /** One pipeline per stride: the stride of the instance buffer is part of the pipeline. */
  thickness(stride: number): GPURenderPipeline;
  smooth: GPURenderPipeline;
  thicknessBlur: GPURenderPipeline;
  normals: GPURenderPipeline;
  composite: Map<GPUTextureFormat, GPURenderPipeline>;
}

const pipelinesFor = perDevice((device): Pipelines => {
  const chunks = FLUID_STRUCT_CHUNKS;
  const sphereModule = createShader(device, {
    label: 'fluid-surface/sphere-depth.wgsl',
    code: sphereDepthSource.code,
    chunks,
  });
  const thicknessModule = createShader(device, {
    label: 'fluid-surface/thickness.wgsl',
    code: thicknessSource.code,
    chunks,
  });
  const particleLayout = device.createBindGroupLayout({
    label: 'fluid particle passes',
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
    ],
  });
  const particleLayoutPipeline = device.createPipelineLayout({
    label: 'fluid particle passes layout',
    bindGroupLayouts: [particleLayout],
  });
  const thicknessPipelines = new Map<number, GPURenderPipeline>();
  const fullscreen = (
    label: string,
    source: { code: string; chunks: Readonly<Record<string, string>> },
    format: GPUTextureFormat,
  ) =>
    createFullscreenPipeline(device, {
      label: `fluid ${label}`,
      fragment: {
        label: `fluid-surface/${label}.wgsl`,
        code: source.code,
        chunks: { ...source.chunks, ...chunks },
      },
      format,
    });
  return {
    particleLayout,
    sphereDepth: device.createRenderPipeline({
      label: 'fluid sphere depth',
      layout: particleLayoutPipeline,
      vertex: { module: sphereModule, entryPoint: 'vs', buffers: [PARTICLE_LAYOUT] },
      fragment: {
        module: sphereModule,
        entryPoint: 'fs',
        targets: [{ format: FLUID_FORMATS.depth }],
      },
      primitive: { topology: 'triangle-strip', cullMode: 'none' },
      depthStencil: {
        format: FLUID_FORMATS.depthBuffer,
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    }),
    thickness(stride) {
      let pipeline = thicknessPipelines.get(stride);
      if (!pipeline) {
        pipeline = device.createRenderPipeline({
          label: `fluid thickness (every ${stride})`,
          layout: particleLayoutPipeline,
          vertex: {
            module: thicknessModule,
            entryPoint: 'vs',
            buffers: [{ ...PARTICLE_LAYOUT, arrayStride: PARTICLE_LAYOUT.arrayStride * stride }],
          },
          fragment: {
            module: thicknessModule,
            entryPoint: 'fs',
            targets: [{ format: FLUID_FORMATS.thickness, blend: { color: ADD, alpha: ADD } }],
          },
          primitive: { topology: 'triangle-strip', cullMode: 'none' },
        });
        thicknessPipelines.set(stride, pipeline);
      }
      return pipeline;
    },
    smooth: fullscreen('smooth', smoothSource, FLUID_FORMATS.depth),
    thicknessBlur: fullscreen('thickness-blur', thicknessBlurSource, FLUID_FORMATS.thickness),
    normals: fullscreen('normals', normalsSource, FLUID_FORMATS.normals),
    composite: new Map(),
  };
});

function compositePipeline(device: GPUDevice, format: GPUTextureFormat): GPURenderPipeline {
  const pipelines = pipelinesFor(device);
  let pipeline = pipelines.composite.get(format);
  if (!pipeline) {
    pipeline = createFullscreenPipeline(device, {
      label: `fluid composite ${format}`,
      fragment: {
        label: 'fluid-surface/composite.wgsl',
        code: compositeSource.code,
        chunks: FLUID_STRUCT_CHUNKS,
      },
      format,
    });
    pipelines.composite.set(format, pipeline);
  }
  return pipeline;
}

// --- The renderer --------------------------------------------------------------------------------

const scaled = (size: number, scale: number) => Math.max(1, Math.round(size * scale));

/**
 * Screen-space fluid rendering (van der Laan, Green and Sainz 2009; Truong and Yuksel 2018): turns
 * the positions of a `ParticleSet` into a continuous liquid surface composited over a scene.
 *
 * Passes, each labelled for the profiler:
 *  - `fluid depth`      particles as sphere impostors into an r32float linear-depth image with a
 *                       depth32float buffer (nearest sphere wins, exact intersections), at `scale`
 *  - `fluid smooth`     separable bilateral or narrow-range filter of that depth, `iterations` rounds
 *  - `fluid normals`    view-space normals from the depth, one-sided differences at silhouettes
 *  - `fluid thickness`  additive sprites of sphere chord length into rg16float at `thicknessScale`
 *  - `fluid blur`       separable Gaussian over the thickness
 *  - `fluid composite`  full resolution: refraction, absorption, Fresnel sky reflection, sun
 *                       highlight, foam tint, depth test against the scene
 */
export function createFluidSurface(options: FluidSurfaceOptions): FluidSurface {
  const { device, format } = options;
  const label = options.label ?? 'fluid';
  const pipelines = pipelinesFor(device);
  const composite = compositePipeline(device, format);

  let quality = clampQuality({
    ...DEFAULT_QUALITY,
    ...Object.fromEntries(Object.entries(options).filter(([key]) => key in DEFAULT_QUALITY)),
  } as FluidSurfaceQuality);
  let width = scaled(options.width, 1);
  let height = scaled(options.height, 1);
  let generation = 0;
  let destroyed = false;

  const chainSize = (): [number, number] => [
    scaled(width, quality.scale),
    scaled(height, quality.scale),
  ];
  const thicknessSize = (): [number, number] => [
    scaled(width, quality.thicknessScale),
    scaled(height, quality.thicknessScale),
  ];

  const [cw, ch] = chainSize();
  const [tw, th] = thicknessSize();
  const target = (name: string, fmt: GPUTextureFormat, w: number, h: number) =>
    createRenderTarget(device, {
      label: `${label} ${name}`,
      format: fmt,
      width: w,
      height: h,
      usage: INSPECTABLE,
    });
  const rawDepth = target('raw depth', FLUID_FORMATS.depth, cw, ch);
  const smoothA = target('smooth A', FLUID_FORMATS.depth, cw, ch);
  const smoothB = target('smooth B', FLUID_FORMATS.depth, cw, ch);
  const rawNormals = target('raw normals', FLUID_FORMATS.normals, cw, ch);
  const smoothNormals = target('smooth normals', FLUID_FORMATS.normals, cw, ch);
  const thickA = target('thickness', FLUID_FORMATS.thickness, tw, th);
  const thickB = target('thickness scratch', FLUID_FORMATS.thickness, tw, th);
  const depthBuffer = createDepthTarget(device, {
    label: `${label} depth buffer`,
    width: cw,
    height: ch,
    format: FLUID_FORMATS.depthBuffer,
    sampled: false,
  });

  const cameraUniform = createUniformBuffer(device, CameraUniform, `${label} camera`);
  const frame = createUniformBuffer(device, FluidFrame, `${label} frame`);
  const horizontal = createUniformBuffer(device, FluidPass, `${label} horizontal`);
  const vertical = createUniformBuffer(device, FluidPass, `${label} vertical`);
  horizontal.set({ direction: [1, 0] });
  vertical.set({ direction: [0, 1] });
  horizontal.upload();
  vertical.upload();

  const stats = { bindGroupsCreated: 0, particlesDrawn: 0, lastPasses: [] as string[] };

  // --- Bind groups: rebuilt when a view they name changes, never per particle buffer. ---
  interface InternalGroups {
    key: readonly unknown[];
    smoothFromRaw: GPUBindGroup;
    smoothFromB: GPUBindGroup;
    smoothVertical: GPUBindGroup;
    normalsRaw: GPUBindGroup;
    normalsSmooth: GPUBindGroup;
    blurHorizontal: GPUBindGroup;
    blurVertical: GPUBindGroup;
  }
  interface SceneGroups {
    key: readonly unknown[];
    depth: GPUBindGroup;
    thickness: GPUBindGroup;
    composite: GPUBindGroup;
  }
  let internal: InternalGroups | undefined;
  let sceneGroups: SceneGroups | undefined;
  const sameKey = (a: readonly unknown[] | undefined, b: readonly unknown[]) =>
    a !== undefined && a.every((view, i) => view === b[i]);

  function bind(
    pipeline: GPURenderPipeline | GPUBindGroupLayout,
    name: string,
    resources: readonly GPUBindingResource[],
  ): GPUBindGroup {
    stats.bindGroupsCreated++;
    return device.createBindGroup({
      label: `${label} ${name}`,
      layout: 'getBindGroupLayout' in pipeline ? pipeline.getBindGroupLayout(0) : pipeline,
      entries: resources.map((resource, binding) => ({ binding, resource })),
    });
  }
  const buffer = (b: { buffer: GPUBuffer }): GPUBindingResource => ({ buffer: b.buffer });

  /** Groups that name only the chain's own images: rebuilt when a target is re-created. */
  function internalGroups(): InternalGroups {
    const key = [rawDepth.view, smoothA.view, smoothB.view, thickA.view, thickB.view];
    if (internal && sameKey(internal.key, key)) return internal;
    const camera = buffer(cameraUniform);
    const fr = buffer(frame);
    internal = {
      key,
      smoothFromRaw: bind(pipelines.smooth, 'smooth H (raw)', [
        camera,
        fr,
        buffer(horizontal),
        rawDepth.view,
      ]),
      smoothFromB: bind(pipelines.smooth, 'smooth H (B)', [
        camera,
        fr,
        buffer(horizontal),
        smoothB.view,
      ]),
      smoothVertical: bind(pipelines.smooth, 'smooth V', [
        camera,
        fr,
        buffer(vertical),
        smoothA.view,
      ]),
      normalsRaw: bind(pipelines.normals, 'normals (raw)', [camera, fr, rawDepth.view]),
      normalsSmooth: bind(pipelines.normals, 'normals (smooth)', [camera, fr, smoothB.view]),
      blurHorizontal: bind(pipelines.thicknessBlur, 'blur H', [
        fr,
        buffer(horizontal),
        thickA.view,
      ]),
      blurVertical: bind(pipelines.thicknessBlur, 'blur V', [fr, buffer(vertical), thickB.view]),
    };
    return internal;
  }

  /** Groups that also name the caller's scene: rebuilt when the caller passes other views. */
  function groupsForScene(sceneColor: GPUTextureView, sceneDepth: GPUTextureView): SceneGroups {
    const key = [
      sceneColor,
      sceneDepth,
      rawDepth.view,
      smoothB.view,
      rawNormals.view,
      smoothNormals.view,
      thickA.view,
    ];
    if (sceneGroups && sameKey(sceneGroups.key, key)) return sceneGroups;
    const camera = buffer(cameraUniform);
    const fr = buffer(frame);
    sceneGroups = {
      key,
      depth: bind(pipelines.particleLayout, 'depth', [camera, fr, sceneDepth]),
      thickness: bind(pipelines.particleLayout, 'thickness', [camera, fr, sceneDepth]),
      composite: bind(composite, 'composite', [
        camera,
        fr,
        sceneColor,
        sceneDepth,
        linearSampler(device),
        rawDepth.view,
        rawNormals.view,
        smoothB.view,
        smoothNormals.view,
        thickA.view,
      ]),
    };
    return sceneGroups;
  }

  function ensureSizes(): void {
    const [nw, nh] = chainSize();
    const [mw, mh] = thicknessSize();
    let changed = false;
    for (const t of [rawDepth, smoothA, smoothB, rawNormals, smoothNormals, depthBuffer]) {
      changed = t.resize(nw, nh) || changed;
    }
    for (const t of [thickA, thickB]) changed = t.resize(mw, mh) || changed;
    if (changed) generation++;
  }

  function colorPass(
    encoder: GPUCommandEncoder,
    name: string,
    view: GPUTextureView,
    profiler: PassProfiler,
    clear: GPUColor = { r: 0, g: 0, b: 0, a: 0 },
    depth?: GPUTextureView,
  ): GPURenderPassEncoder {
    stats.lastPasses.push(name);
    return encoder.beginRenderPass({
      label: name,
      colorAttachments: [{ view, loadOp: 'clear', clearValue: clear, storeOp: 'store' }],
      depthStencilAttachment: depth
        ? { view: depth, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' }
        : undefined,
      timestampWrites: profiler.pass(name),
    });
  }

  function fullscreenPass(
    encoder: GPUCommandEncoder,
    name: string,
    pipeline: GPURenderPipeline,
    view: GPUTextureView,
    group: GPUBindGroup,
    profiler: PassProfiler,
  ): void {
    const pass = colorPass(encoder, name, view, profiler);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.draw(3);
    pass.end();
  }

  return {
    get quality() {
      return quality;
    },
    setQuality(change) {
      const defined = Object.fromEntries(Object.entries(change).filter(([, v]) => v !== undefined));
      quality = clampQuality({ ...quality, ...defined });
    },
    resize(w, h) {
      width = scaled(w, 1);
      height = scaled(h, 1);
      ensureSizes();
    },
    get generation() {
      return generation;
    },
    get sizes() {
      return { chain: chainSize(), thickness: thicknessSize() };
    },
    textures: {
      get rawDepth() {
        return rawDepth;
      },
      get smoothDepth() {
        return smoothB;
      },
      get rawNormals() {
        return rawNormals;
      },
      get smoothNormals() {
        return smoothNormals;
      },
      get thickness() {
        return thickA;
      },
    },
    stats,

    render(encoder, o) {
      if (destroyed) throw new Error('FluidSurface: render after destroy');
      const profiler = o.profiler ?? noProfiler;
      const stages = new Set<FluidStage>(o.stages ?? STAGES);
      const p = resolveParams(o.params);
      const set = o.particles;
      const count = Math.max(0, Math.min(Math.floor(set.count), set.capacity));
      stats.lastPasses = [];
      stats.particlesDrawn = 0;

      ensureSizes();
      const [cwNow, chNow] = chainSize();
      const [twNow, thNow] = thicknessSize();
      const q = quality;

      setCameraUniform(cameraUniform, o.camera, width, height);
      cameraUniform.upload();
      const surfaceMode = SURFACES.indexOf(p.surface);
      const debug = DEBUG_VIEWS.indexOf(p.debug);
      const [sx, sy, sz] = p.sunDirection;
      const sunLength = Math.hypot(sx, sy, sz) || 1;
      frame.set({
        chain_size: [cwNow, chNow],
        thick_size: [twNow, thNow],
        full_size: [width, height],
        radius: set.radius,
        thick_radius: set.radius * p.thicknessRadius,
        thick_gain: p.thicknessGain * q.thicknessStride,
        live: count > 0 ? 1 : 0,
        filter_kind: FILTERS.indexOf(q.filter),
        filter_scale: q.filterRadius,
        max_radius: q.maxFilterRadius,
        range: q.rangeScale * set.radius,
        blur_radius: q.thicknessBlur,
        surface: Math.max(0, surfaceMode),
        split: p.split,
        debug: Math.max(0, debug),
        output_srgb: p.srgb ? 1 : 0,
        sigma: p.absorption,
        density: p.density,
        scatter: p.scatter,
        turbidity: p.turbidity,
        sun_dir: [sx / sunLength, sy / sunLength, sz / sunLength],
        ior: p.ior,
        foam_color: p.foamColor,
        foam_amount: p.foamAmount,
        refraction: p.refraction,
        reflection: p.reflection,
        roughness: p.roughness,
        specular: p.specular,
        haze: p.haze,
        water_type: Math.round(p.waterType),
        max_refract: p.maxRefractionThickness,
      });
      frame.upload();

      const g = { ...internalGroups(), ...groupsForScene(o.sceneColor, o.sceneDepth) };

      if (count > 0) {
        stats.particlesDrawn = count;
        const needSmooth = p.surface !== 'raw' || p.debug === 'smooth-depth';
        const needSmoothNormals = p.surface !== 'raw';
        const needRawNormals = p.surface !== 'smooth';

        if (stages.has('depth')) {
          const pass = colorPass(
            encoder,
            'fluid depth',
            rawDepth.view,
            profiler,
            { r: 0, g: 0, b: 0, a: 0 },
            depthBuffer.view,
          );
          pass.setPipeline(pipelines.sphereDepth);
          pass.setBindGroup(0, g.depth);
          pass.setVertexBuffer(0, set.positions);
          pass.draw(4, count);
          pass.end();
        }

        if (needSmooth && stages.has('smooth')) {
          for (let i = 0; i < q.iterations; i++) {
            fullscreenPass(
              encoder,
              'fluid smooth',
              pipelines.smooth,
              smoothA.view,
              i === 0 ? g.smoothFromRaw : g.smoothFromB,
              profiler,
            );
            fullscreenPass(
              encoder,
              'fluid smooth',
              pipelines.smooth,
              smoothB.view,
              g.smoothVertical,
              profiler,
            );
          }
        }

        if (stages.has('normals')) {
          if (needSmoothNormals) {
            fullscreenPass(
              encoder,
              'fluid normals',
              pipelines.normals,
              smoothNormals.view,
              g.normalsSmooth,
              profiler,
            );
          }
          if (needRawNormals) {
            fullscreenPass(
              encoder,
              'fluid normals',
              pipelines.normals,
              rawNormals.view,
              g.normalsRaw,
              profiler,
            );
          }
        }

        if (stages.has('thickness')) {
          const pass = colorPass(encoder, 'fluid thickness', thickA.view, profiler);
          pass.setPipeline(pipelines.thickness(q.thicknessStride));
          pass.setBindGroup(0, g.thickness);
          pass.setVertexBuffer(0, set.positions);
          pass.draw(4, Math.ceil(count / q.thicknessStride));
          pass.end();
          if (q.thicknessBlur > 0) {
            fullscreenPass(
              encoder,
              'fluid blur',
              pipelines.thicknessBlur,
              thickB.view,
              g.blurHorizontal,
              profiler,
            );
            fullscreenPass(
              encoder,
              'fluid blur',
              pipelines.thicknessBlur,
              thickA.view,
              g.blurVertical,
              profiler,
            );
          }
        }
      }

      if (stages.has('composite')) {
        fullscreenPass(encoder, 'fluid composite', composite, o.target, g.composite, profiler);
      }
    },

    destroy() {
      if (destroyed) return;
      destroyed = true;
      for (const t of [rawDepth, smoothA, smoothB, rawNormals, smoothNormals, thickA, thickB]) {
        t.destroy();
      }
      depthBuffer.destroy();
      for (const u of [cameraUniform, frame, horizontal, vertical]) u.destroy();
      internal = undefined;
      sceneGroups = undefined;
    },
  };
}
