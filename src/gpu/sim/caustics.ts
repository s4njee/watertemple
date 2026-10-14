import { createBlur, type Blur } from '../passes/blur.ts';
import { renderFullscreen } from '../fullscreen.ts';
import { noProfiler, type PassProfiler } from '../profiler.ts';
import { createShader } from '../shader.ts';
import { createUniformBuffer, defineUniformStruct, type UniformBuffer } from '../uniforms.ts';
import causticsCore from './caustics-core.wgsl';
import causticsJacobian from './caustics-jacobian.wgsl';
import causticsMesh from './caustics-mesh.wgsl';
import { computeGridSize, type Heightfield } from './heightfield.ts';

// Caustics from the live heightfield (story 5.4): the bright network on a pool floor is where the wavy
// surface focuses sunlight. This module turns the heightfield solver's output (height and slope per
// cell) into a texture of that light, one value per floor texel, for any shading pass to multiply in.
//
// OUTPUT. `view` is an rgba16float texture over the same domain as the heightfield (uv 0..1, the
// top-left origin, the aspect of the canvas), indexed by FLOOR position. rgb is light intensity
// relative to a flat surface: 1 where the sun lights the floor as it would through still water, above 1
// where the surface focuses it, below 1 where it spreads it out (0 in shadow); r, g and b differ only
// when dispersion is on. a is 1. The mean over a floor that is lit everywhere stays at 1: caustics
// move light around, they do not make any (the tests check it).
//
// FRAME AND UNITS. The heightfield's: x right, y down in image heights (canvas height = 1, so the
// domain is `aspect` wide and 1 high), z up out of the water with z = 0 the mean surface. `lightDir`
// points toward the light. `depth` is how far below the mean surface the floor is, in image heights.
//
// TWO METHODS, one texture, interchangeable per frame:
//  - `mesh`: the projected light mesh (Evan Wallace's WebGL Water). A grid of triangles over the
//    surface; the vertex stage refracts a sunbeam at each vertex and moves it to where the beam hits
//    the floor; the fragment stage shades by original area / projected area from screen-space
//    derivatives; the draws are added together. Faithful past the first focus (folded triangles add
//    their light twice) and conserves energy by construction, but it rasterises 6 triangles per cell
//    and the per-triangle area is a little noisy (optional light blur, `smooth`).
//  - `jacobian`: a per-pixel backward estimate, brightness = 1 / |det(I + dD/dp)| where D is the
//    sideways travel of the refracted beam. One fullscreen pass, no vertices, no blending, no
//    derivatives. Exact to first order and until the surface focuses light twice, after which it
//    overshoots the energy. On the Apple GPU measured it costs the same as the mesh.
// See src/effects/ripple-caustics/notes.md for the comparison.

export type CausticsMethod = 'mesh' | 'jacobian';

/** Format of the caustics texture: filterable, renderable and blendable on every WebGPU device. */
export const CAUSTICS_FORMAT: GPUTextureFormat = 'rgba16float';
/** Largest mesh edge, in cells: 6 * 768^2 indices is 14 MB. */
export const MAX_MESH_CELLS = 768;
/** The sun is never lower than this above the horizon (sine of the elevation): glancing light is meaningless here. */
const MIN_LIGHT_Z = 0.05;

/** A sphere that shades the floor: centre and radius in the heightfield frame (image heights). */
export interface CausticsOccluder {
  /** x and y as for a surface point (x up to the aspect), z relative to the mean surface (negative is under water). */
  center: readonly [number, number, number];
  radius: number;
  /** Width of the shadow's soft edge, image heights. Default: the larger of a tenth of the radius and 1.5 mesh cells. */
  softness?: number;
}

export interface CausticsSettings {
  /** Direction TOWARD the light in the heightfield frame (x right, y down, z up). Normalised; z is kept above 0.05. */
  lightDir: readonly [number, number, number];
  /** Floor depth below the mean surface, image heights. */
  depth: number;
  /** Index of refraction of the water. Default 1.333. */
  ior?: number;
  /** Chromatic spread: the red channel uses ior - dispersion and blue ior + dispersion. 0 (default) is one grey pass. */
  dispersion?: number;
  /** Gain on the slope before it is used (1 = the physical surface, larger exaggerates). Default 1. */
  strength?: number;
  /** A ball in the way of the light, or null/undefined for none. */
  occluder?: CausticsOccluder | null;
  /**
   * The surface continues beyond the domain (an endless pool): the floor under the strip the beam
   * shifts into is lit. false (default for a pool with walls would be) leaves that strip in the walls'
   * shadow. Default true.
   */
  extend?: boolean;
  /** Intensity cap per texel. Default 12. */
  maxIntensity?: number;
  /** Include the surface height when working out where a beam lands (mesh only; the jacobian ignores it). Default true. */
  useHeight?: boolean;
  /** Radiometric tilt and Fresnel factors (see caustics-core.wgsl). Default true. */
  radiometry?: boolean;
  /** Gaussian blur of the result, sigma in texels; 0 (default for jacobian) is off. */
  smooth?: number;
  /** jacobian only: regularisation of 1 / |det|. Larger is softer, caps sooner and keeps the mean nearer 1 on a violent surface. Default 0.15. */
  epsilon?: number;
  /** jacobian only: fixed-point rounds that find the surface point behind a floor texel. One is visibly worse; more than three changes little. Default 3. */
  iterations?: number;
}

export interface HeightfieldCausticsOptions {
  device: GPUDevice;
  heightfield: Heightfield;
  /** Texels along the longer side of the caustics texture. Default 512. */
  resolution?: number;
  /** Default `mesh`. Changeable at any time through `method`. */
  method?: CausticsMethod;
  /** Mesh cells along the longer side as a multiple of the heightfield's cells. Default 1. */
  meshScale?: number;
  profiler?: PassProfiler;
  label?: string;
}

export interface HeightfieldCaustics {
  /** The method the next `update` runs. Assign to switch. */
  method: CausticsMethod;
  /** Floor-space light intensity: rgba16float, filterable. Stable between `generation` bumps. */
  readonly texture: GPUTexture;
  readonly view: GPUTextureView;
  /** Texture size in texels. */
  readonly width: number;
  readonly height: number;
  /** Mesh size in cells. */
  readonly meshCells: Readonly<{ x: number; y: number }>;
  /** Increments whenever `texture`/`view` are re-created (resolution, aspect or heightfield change): rebuild bind groups. */
  readonly generation: number;
  /** Encodes this frame's caustics pass(es). Call after the heightfield's `update`, before shading. */
  update(encoder: GPUCommandEncoder, settings: CausticsSettings): void;
  /** Re-fits the texture to a canvas of this size (any unit: only the aspect matters). Returns whether it was re-created. */
  resize(width: number, height: number): boolean;
  setResolution(cells: number): boolean;
  setMeshScale(scale: number): boolean;
  destroy(): void;
}

// --- Pure helpers (also used by the tests and the demos) -----------------------------------------

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

/** Unit vector toward the sun in the heightfield frame from an elevation above the horizon and an azimuth on the canvas (0 = light from the right, counter-clockwise on screen = increasing). */
export function lightDirection(elevation: number, azimuth: number): [number, number, number] {
  const horizontal = Math.cos(elevation);
  // The canvas y axis points down, so a counter-clockwise azimuth on screen subtracts.
  return [horizontal * Math.cos(azimuth), -horizontal * Math.sin(azimuth), Math.sin(elevation)];
}

/** Snell refraction of the unit vector `incident` through unit normal `normal` (facing the incident side) at relative index `eta` = n1 / n2. Zero vector on total internal reflection. */
export function refractVector(
  incident: readonly [number, number, number],
  normal: readonly [number, number, number],
  eta: number,
): [number, number, number] {
  const d = incident[0] * normal[0] + incident[1] * normal[1] + incident[2] * normal[2];
  const k = 1 - eta * eta * (1 - d * d);
  if (k < 0) return [0, 0, 0];
  const f = eta * d + Math.sqrt(k);
  return [
    eta * incident[0] - f * normal[0],
    eta * incident[1] - f * normal[1],
    eta * incident[2] - f * normal[2],
  ];
}

/**
 * Where a beam through a FLAT surface lands relative to where it entered, in image heights
 * (x right, y down): the shift of the whole caustic pattern from the geometry above it. Zero for
 * overhead light, depth * tan(refracted angle) otherwise.
 */
export function flatLandingShift(
  lightDir: readonly [number, number, number],
  depth: number,
  ior = 1.333,
): [number, number] {
  const l = normaliseLight(lightDir);
  const r = refractVector([-l[0], -l[1], -l[2]], [0, 0, 1], 1 / ior);
  return [(r[0] / -r[2]) * depth, (r[1] / -r[2]) * depth];
}

function normaliseLight(l: readonly [number, number, number]): [number, number, number] {
  let [x, y, z] = l;
  if (![x, y, z].every(Number.isFinite)) [x, y, z] = [0, 0, 1];
  const length = Math.hypot(x, y, z) || 1;
  [x, y, z] = [x / length, y / length, z / length];
  if (z < MIN_LIGHT_Z) {
    const horizontal = Math.hypot(x, y);
    const scale = horizontal > 0 ? Math.sqrt(1 - MIN_LIGHT_Z * MIN_LIGHT_Z) / horizontal : 0;
    x = horizontal > 0 ? x * scale : 0;
    y = horizontal > 0 ? y * scale : 0;
    z = MIN_LIGHT_Z;
  }
  return [x, y, z];
}

/**
 * Triangle list for a grid of `cellsX` x `cellsY` cells whose vertices are numbered row by row,
 * `cellsX + 1` to a row. Winding does not matter to the caustics (nothing is culled).
 */
export function gridIndices(cellsX: number, cellsY: number): Uint32Array {
  const row = cellsX + 1;
  const indices = new Uint32Array(cellsX * cellsY * 6);
  let o = 0;
  for (let j = 0; j < cellsY; j++) {
    for (let i = 0; i < cellsX; i++) {
      const a = j * row + i;
      indices[o++] = a;
      indices[o++] = a + row;
      indices[o++] = a + 1;
      indices[o++] = a + 1;
      indices[o++] = a + row;
      indices[o++] = a + row + 1;
    }
  }
  return indices;
}

// --- GPU -----------------------------------------------------------------------------------------

const Params = defineUniformStruct('CausticsParams', {
  size: 'vec2f',
  aspect: 'f32',
  depth: 'f32',
  light: 'vec3f',
  strength: 'f32',
  channel: 'vec3f',
  max_intensity: 'f32',
  occluder: 'vec4f',
  ior_rgb: 'vec3f',
  mesh_ior: 'f32',
  mesh_lo: 'vec2f',
  mesh_hi: 'vec2f',
  mesh_cells: 'vec2u',
  texel: 'vec2f',
  cell_size: 'f32',
  soft: 'f32',
  eps: 'f32',
  radiometry: 'f32',
  extend: 'f32',
  height_scale: 'f32',
  iterations: 'u32',
  dispersive: 'u32',
});

interface Pipelines {
  layout: GPUBindGroupLayout;
  mesh: GPURenderPipeline;
  jacobian: GPURenderPipeline;
}

const pipelineCache = new WeakMap<GPUDevice, Pipelines>();

function getPipelines(device: GPUDevice): Pipelines {
  const cached = pipelineCache.get(device);
  if (cached) return cached;
  const both = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
  const layout = device.createBindGroupLayout({
    label: 'caustics layout',
    entries: [
      { binding: 0, visibility: both, buffer: { type: 'uniform' } },
      { binding: 1, visibility: both, texture: { sampleType: 'float' } },
      { binding: 2, visibility: both, sampler: { type: 'filtering' } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({
    label: 'caustics pipeline layout',
    bindGroupLayouts: [layout],
  });
  const chunks = {
    'caustics_params.wgsl': Params.wgsl,
    'caustics_core.wgsl': causticsCore.code,
    ...causticsCore.chunks,
  };
  const meshModule = createShader(device, {
    label: 'sim/caustics-mesh.wgsl',
    code: causticsMesh.code,
    chunks,
  });
  const jacobianModule = createShader(device, {
    label: 'sim/caustics-jacobian.wgsl',
    code: causticsJacobian.code,
    chunks,
  });
  const add: GPUBlendComponent = { srcFactor: 'one', dstFactor: 'one', operation: 'add' };
  const pipelines: Pipelines = {
    layout,
    mesh: device.createRenderPipeline({
      label: 'caustics mesh',
      layout: pipelineLayout,
      vertex: { module: meshModule, entryPoint: 'caustics_mesh_vs' },
      fragment: {
        module: meshModule,
        entryPoint: 'caustics_mesh_fs',
        targets: [{ format: CAUSTICS_FORMAT, blend: { color: add, alpha: add } }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    }),
    jacobian: device.createRenderPipeline({
      label: 'caustics jacobian',
      layout: pipelineLayout,
      vertex: { module: jacobianModule, entryPoint: 'fullscreen_vs' },
      fragment: {
        module: jacobianModule,
        entryPoint: 'caustics_jacobian_fs',
        targets: [{ format: CAUSTICS_FORMAT }],
      },
    }),
  };
  pipelineCache.set(device, pipelines);
  return pipelines;
}

type ParamsBuffer = UniformBuffer<typeof Params.fields>;

/** `CausticsSettings` with every default filled in and every number clamped. */
interface Resolved {
  depth: number;
  ior: number;
  dispersion: number;
  strength: number;
  occluder: CausticsOccluder | null;
  extend: boolean;
  maxIntensity: number;
  useHeight: boolean;
  radiometry: boolean;
  smooth: number;
  epsilon: number;
  iterations: number;
}

export function createHeightfieldCaustics(
  options: HeightfieldCausticsOptions,
): HeightfieldCaustics {
  const { device, heightfield, profiler = noProfiler, label = 'caustics' } = options;
  let resolution = options.resolution ?? 512;
  let meshScale = options.meshScale ?? 1;
  let aspectHint: number | undefined;
  const pipelines = getPipelines(device);
  const maxDimension = Math.min(device.limits.maxTextureDimension2D, 4096);

  const sampler = device.createSampler({
    label: `${label} sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });
  // One parameter buffer per channel draw: writeBuffer lands before the whole submit, so a single
  // buffer would show all three draws the last channel's values.
  const buffers: ParamsBuffer[] = [0, 1, 2].map((i) =>
    createUniformBuffer(device, Params, `${label} params ${i}`),
  );

  interface Resources {
    texture: GPUTexture;
    view: GPUTextureView;
    width: number;
    height: number;
    meshX: number;
    meshY: number;
    indices: GPUBuffer;
    groups: GPUBindGroup[];
    blur: Blur;
    heightfieldGeneration: number;
  }
  let resources: Resources | undefined;
  let generation = 0;
  let destroyed = false;

  function sizes() {
    const grid = heightfield.grid;
    const aspect = aspectHint ?? grid.aspect;
    const texture = computeGridSize(resolution, aspect, maxDimension);
    // The mesh follows the heightfield's own grid (so a vertex is about one cell), scaled.
    const long = Math.max(grid.width, grid.height);
    const wanted = clamp(Math.round(long * meshScale), 16, MAX_MESH_CELLS);
    const mesh = computeGridSize(wanted, grid.aspect, MAX_MESH_CELLS);
    return { texture, mesh };
  }

  function release(r: Resources) {
    r.texture.destroy();
    r.indices.destroy();
    r.blur.destroy();
  }

  function build(): void {
    const { texture: size, mesh } = sizes();
    const previous = resources;
    const sameTexture =
      previous && previous.width === size.width && previous.height === size.height;
    const sameMesh = previous && previous.meshX === mesh.width && previous.meshY === mesh.height;
    if (
      previous &&
      sameTexture &&
      sameMesh &&
      previous.heightfieldGeneration === heightfield.generation
    ) {
      return;
    }

    let texture = previous?.texture;
    let view = previous?.view;
    let blur = previous?.blur;
    if (!previous || !sameTexture) {
      texture = device.createTexture({
        label: `${label} texture`,
        size: [size.width, size.height],
        format: CAUSTICS_FORMAT,
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC,
      });
      view = texture.createView({ label: `${label} view` });
      blur = createBlur(device, {
        label: `${label} blur`,
        format: CAUSTICS_FORMAT,
        width: size.width,
        height: size.height,
      });
      previous?.texture.destroy();
      previous?.blur.destroy();
      generation++;
    }
    let indices = previous?.indices;
    if (!previous || !sameMesh) {
      const data = gridIndices(mesh.width, mesh.height);
      indices = device.createBuffer({
        label: `${label} mesh indices`,
        size: data.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(indices, 0, data);
      previous?.indices.destroy();
    }
    const groups = buffers.map((buffer, i) =>
      device.createBindGroup({
        label: `${label} bind group ${i}`,
        layout: pipelines.layout,
        entries: [
          { binding: 0, resource: { buffer: buffer.buffer } },
          { binding: 1, resource: heightfield.view },
          { binding: 2, resource: sampler },
        ],
      }),
    );
    resources = {
      texture: texture!,
      view: view!,
      width: size.width,
      height: size.height,
      meshX: mesh.width,
      meshY: mesh.height,
      indices: indices!,
      groups,
      blur: blur!,
      heightfieldGeneration: heightfield.generation,
    };
  }

  build();

  /** Writes the settings every draw shares into `buffer`. */
  function fill(
    buffer: ParamsBuffer,
    s: Resolved,
    light: [number, number, number],
    iors: [number, number, number],
    channel: [number, number, number],
    meshIor: number,
    meshRange: { lo: [number, number]; hi: [number, number] },
  ) {
    const r = resources!;
    const grid = heightfield.grid;
    const occ = s.occluder;
    // The shadow's edge is at least a cell and a half wide at the resolution that draws it, so a
    // mesh with one light value per vertex shows no triangles in the rim.
    const unit =
      api.method === 'mesh'
        ? Math.max(grid.aspect / r.meshX, 1 / r.meshY)
        : Math.max(grid.aspect / r.width, 1 / r.height);
    const soft = occ ? (occ.softness ?? Math.max(occ.radius * 0.1, 1.5 * unit)) : 0;
    buffer.set({
      size: [r.width, r.height],
      aspect: grid.aspect,
      depth: s.depth,
      light,
      strength: s.strength,
      channel,
      max_intensity: s.maxIntensity,
      occluder: occ
        ? [occ.center[0], occ.center[1], occ.center[2], Math.max(0, occ.radius)]
        : [0, 0, 0, 0],
      ior_rgb: iors,
      mesh_ior: meshIor,
      mesh_lo: meshRange.lo,
      mesh_hi: meshRange.hi,
      mesh_cells: [r.meshX, r.meshY],
      texel: [grid.texelSize[0], grid.texelSize[1]],
      cell_size: grid.cellSize,
      soft,
      eps: Math.max(s.epsilon, 1e-3),
      radiometry: s.radiometry ? 1 : 0,
      extend: s.extend ? 1 : 0,
      height_scale: s.useHeight ? 1 : 0,
      iterations: Math.max(0, Math.round(s.iterations)),
      dispersive: s.dispersion > 0 ? 1 : 0,
    });
    buffer.upload();
  }

  const api: HeightfieldCaustics = {
    method: options.method ?? 'mesh',
    get texture() {
      return resources!.texture;
    },
    get view() {
      return resources!.view;
    },
    get width() {
      return resources!.width;
    },
    get height() {
      return resources!.height;
    },
    get meshCells() {
      return { x: resources!.meshX, y: resources!.meshY };
    },
    get generation() {
      return generation;
    },

    update(encoder, settings) {
      if (destroyed) return;
      build();
      const r = resources!;
      const finite = (x: number | undefined, fallback: number) =>
        x !== undefined && Number.isFinite(x) ? x : fallback;
      const s: Resolved = {
        depth: clamp(finite(settings.depth, 0.35), 1e-3, 16),
        ior: clamp(finite(settings.ior, 1.333), 1.0001, 3),
        dispersion: clamp(finite(settings.dispersion, 0), 0, 0.5),
        strength: clamp(finite(settings.strength, 1), 0, 32),
        occluder: settings.occluder ?? null,
        extend: settings.extend ?? true,
        maxIntensity: clamp(finite(settings.maxIntensity, 12), 1, 1000),
        useHeight: settings.useHeight ?? true,
        radiometry: settings.radiometry ?? true,
        smooth: clamp(finite(settings.smooth, 0), 0, 8),
        epsilon: finite(settings.epsilon, 0.15),
        iterations: clamp(finite(settings.iterations, 3), 0, 6),
      };
      const light = normaliseLight(settings.lightDir);
      const iors: [number, number, number] = [
        Math.max(1.0001, s.ior - s.dispersion),
        s.ior,
        s.ior + s.dispersion,
      ];
      const grid = heightfield.grid;

      // Where the whole pattern is shifted to, per channel, so the mesh can cover just the surface
      // that feeds the floor (plus a margin for the waves' own sideways travel).
      let range = { lo: [0, 0] as [number, number], hi: [1, 1] as [number, number] };
      if (s.extend) {
        const pad = 0.05;
        let sx = 0;
        let sy = 0;
        for (const ior of iors) {
          const [dx, dy] = flatLandingShift(light, s.depth, ior);
          if (Math.abs(dx) > Math.abs(sx)) sx = dx;
          if (Math.abs(dy) > Math.abs(sy)) sy = dy;
        }
        const ux = sx / grid.aspect;
        range = { lo: [-ux - pad, -sy - pad], hi: [1 - ux + pad, 1 - sy + pad] };
      }

      if (api.method === 'jacobian') {
        fill(buffers[0]!, s, light, iors, [1, 1, 1], s.ior, range);
        renderFullscreen(encoder, r.view, pipelines.jacobian, [r.groups[0]!], {
          label: `${label} jacobian`,
          timestampWrites: profiler.pass(`${label} jacobian`),
        });
      } else {
        const draws: { channel: [number, number, number]; ior: number }[] =
          s.dispersion > 0
            ? [
                { channel: [1, 0, 0], ior: iors[0] },
                { channel: [0, 1, 0], ior: iors[1] },
                { channel: [0, 0, 1], ior: iors[2] },
              ]
            : [{ channel: [1, 1, 1], ior: s.ior }];
        draws.forEach((draw, i) => {
          fill(buffers[i]!, s, light, iors, draw.channel, draw.ior, range);
        });
        const pass = encoder.beginRenderPass({
          label: `${label} mesh`,
          colorAttachments: [
            { view: r.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] },
          ],
          timestampWrites: profiler.pass(`${label} mesh`),
        });
        pass.setPipeline(pipelines.mesh);
        pass.setIndexBuffer(r.indices, 'uint32');
        draws.forEach((_, i) => {
          pass.setBindGroup(0, r.groups[i]!);
          pass.drawIndexed(r.meshX * r.meshY * 6);
        });
        pass.end();
      }

      if (s.smooth > 0.05) {
        const radius = Math.min(16, Math.ceil(s.smooth * 2.5));
        r.blur.encode(encoder, r.view, r.view, { radius, sigma: s.smooth });
      }
    },

    resize(width, height) {
      if (!(width > 0 && height > 0)) return false;
      aspectHint = width / height;
      const before = generation;
      build();
      return generation !== before;
    },
    setResolution(cells) {
      if (!Number.isFinite(cells)) return false;
      resolution = cells;
      const before = generation;
      build();
      return generation !== before;
    },
    setMeshScale(scale) {
      if (!Number.isFinite(scale)) return false;
      meshScale = scale;
      build();
      return false;
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      if (resources) release(resources);
      for (const buffer of buffers) buffer.destroy();
    },
  };
  return api;
}
