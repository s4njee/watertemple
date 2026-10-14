import { mat4, type Mat4 } from 'wgpu-matrix';
import { createShader } from '../shader.ts';
import { createUniformBuffer, defineUniformStruct, type UniformBuffer } from '../uniforms.ts';
import {
  CameraUniform,
  setCameraUniform,
  type CameraMatrices,
  type CameraUniformBuffer,
} from './camera.ts';
import forwardShader from './forward.wgsl?raw';
import { MESH_VERTEX_LAYOUT, type Mesh } from './mesh.ts';

/** Sun, sky and fog for the forward renderer (and for any effect shader that wants the same look). */
export const SceneLightUniform = defineUniformStruct('SceneLight', {
  /** Unit vector from the scene towards the sun. */
  sun_dir: 'vec3f',
  sun_intensity: 'f32',
  sun_color: 'vec3f',
  /** Exponential fog: 1 - exp(-density * distance). */
  fog_density: 'f32',
  /** Colour of the hemisphere above, and the ambient strength. */
  sky_color: 'vec3f',
  ambient: 'f32',
  /** Colour of the hemisphere below. */
  ground_color: 'vec3f',
  /** Fog colour, and the sky at the horizon. */
  horizon_color: 'vec3f',
});

/** One draw: model matrix, colour and normal matrix. Padded to the dynamic-offset alignment. */
export const SceneObjectUniform = defineUniformStruct('SceneObject', {
  model: 'mat4x4f',
  /** Inverse transpose of the model's upper 3x3, so non-uniform scale keeps normals correct. */
  normal: 'mat3x3f',
  /** rgb is linear albedo; a is how much of it is emissive (0 lit, 1 unlit). */
  color: 'vec4f',
});

/** Effect-local include chunks that carry the struct declarations. */
export const SCENE_STRUCT_CHUNKS: Readonly<Record<string, string>> = {
  'scene-light.wgsl': SceneLightUniform.wgsl,
  'scene-object.wgsl': SceneObjectUniform.wgsl,
};

export interface SceneLight {
  /** Direction towards the sun; normalised on upload. */
  sunDirection: readonly [number, number, number];
  sunColor: readonly [number, number, number];
  sunIntensity: number;
  skyColor: readonly [number, number, number];
  groundColor: readonly [number, number, number];
  horizonColor: readonly [number, number, number];
  ambient: number;
  fogDensity: number;
}

export interface SceneObject {
  mesh: Mesh;
  /** Column-major model matrix. */
  model: Mat4;
  /** Linear RGB albedo. */
  color: readonly [number, number, number];
  /** 0 = lit (default), 1 = flat colour. */
  emissive?: number;
}

/** What `upload` returns: pass it to `draw` for every pass that renders this frame's objects. */
export interface ObjectBatch {
  readonly objects: readonly SceneObject[];
}

/** Per-view state: a camera and a light, and the bind group a pass binds to see them. */
export interface SceneView {
  readonly camera: CameraUniformBuffer;
  readonly light: UniformBuffer<typeof SceneLightUniform.fields>;
  readonly bindGroup: GPUBindGroup;
  /** Stores and uploads both uniforms. `width` and `height` are the pixel size of the target. */
  update(camera: CameraMatrices, width: number, height: number, light: SceneLight): void;
  destroy(): void;
}

export interface ForwardTarget {
  color: GPUTextureView;
  depth: GPUTextureView;
}

export interface ForwardDrawOptions extends ForwardTarget {
  view: SceneView;
  /**
   * The view is a mirror image (see `mirrorCamera`), which flips triangle winding: draw with
   * clockwise front faces so culling still removes the back faces.
   */
  mirrored?: boolean;
  /** Called inside the pass before any object, e.g. to draw a sky. Must not leave state assumed. */
  background?: (pass: GPURenderPassEncoder) => void;
  /** Colour the target is cleared to (linear). Default black. */
  clearColor?: GPUColor;
  label?: string;
  timestampWrites?: GPURenderPassTimestampWrites;
}

export interface ForwardRendererOptions {
  colorFormat: GPUTextureFormat;
  depthFormat: GPUTextureFormat;
  /** Capacity of the object buffer. Default 64. */
  maxObjects?: number;
}

export interface ForwardRenderer {
  /** Layout of group 0 (camera, light), for pipelines that draw in the same pass, e.g. a sky. */
  readonly viewLayout: GPUBindGroupLayout;
  createView(label: string): SceneView;
  /** Writes every object's matrices and colour into the shared buffer. Call once per frame. */
  upload(objects: readonly SceneObject[]): ObjectBatch;
  draw(encoder: GPUCommandEncoder, batch: ObjectBatch, options: ForwardDrawOptions): void;
  destroy(): void;
}

const roundUp = (multiple: number, value: number) => Math.ceil(value / multiple) * multiple;

/**
 * A small forward pass: draws a list of meshes with one directional light, hemisphere ambient and
 * fog into a colour and depth target. All objects share one uniform buffer addressed with dynamic
 * offsets, so a frame costs one `writeBuffer` for the objects and no bind group per object, however
 * many passes (main, mirrored, shadow) draw them.
 */
export function createForwardRenderer(
  device: GPUDevice,
  options: ForwardRendererOptions,
): ForwardRenderer {
  const { colorFormat, depthFormat, maxObjects = 64 } = options;
  const stride = roundUp(device.limits.minUniformBufferOffsetAlignment, SceneObjectUniform.size);
  const objectBytes = stride * maxObjects;
  const objectBuffer = device.createBuffer({
    label: 'scene objects',
    size: objectBytes,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const staging = new Float32Array(objectBytes / 4);

  const viewLayout = device.createBindGroupLayout({
    label: 'scene view layout',
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
    ],
  });
  const objectLayout = device.createBindGroupLayout({
    label: 'scene object layout',
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: {
          type: 'uniform',
          hasDynamicOffset: true,
          minBindingSize: SceneObjectUniform.size,
        },
      },
    ],
  });
  const objectGroup = device.createBindGroup({
    label: 'scene objects',
    layout: objectLayout,
    entries: [{ binding: 0, resource: { buffer: objectBuffer, size: SceneObjectUniform.size } }],
  });

  const module = createShader(device, {
    label: 'scene/forward.wgsl',
    code: forwardShader,
    chunks: SCENE_STRUCT_CHUNKS,
  });
  const pipelineLayout = device.createPipelineLayout({
    label: 'scene forward layout',
    bindGroupLayouts: [viewLayout, objectLayout],
  });
  const makePipeline = (frontFace: GPUFrontFace) =>
    device.createRenderPipeline({
      label: `scene forward (${frontFace === 'ccw' ? 'normal' : 'mirrored'})`,
      layout: pipelineLayout,
      vertex: { module, entryPoint: 'vs', buffers: [MESH_VERTEX_LAYOUT] },
      fragment: { module, entryPoint: 'fs', targets: [{ format: colorFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'back', frontFace },
      depthStencil: { format: depthFormat, depthWriteEnabled: true, depthCompare: 'less' },
    });
  const pipelines = { normal: makePipeline('ccw'), mirrored: makePipeline('cw') };

  const scratchInverse = mat4.create();
  const views = new Set<SceneView>();
  const objectLayoutFields = Object.fromEntries(
    SceneObjectUniform.layout.map((field) => [field.name, field.offset / 4]),
  ) as Record<'model' | 'normal' | 'color', number>;

  return {
    viewLayout,

    createView(label) {
      const camera = createUniformBuffer(device, CameraUniform, `${label} camera`);
      const light = createUniformBuffer(device, SceneLightUniform, `${label} light`);
      const bindGroup = device.createBindGroup({
        label: `${label} view`,
        layout: viewLayout,
        entries: [
          { binding: 0, resource: { buffer: camera.buffer } },
          { binding: 1, resource: { buffer: light.buffer } },
        ],
      });
      const view: SceneView = {
        camera,
        light,
        bindGroup,
        update(matrices, width, height, lighting) {
          setCameraUniform(camera, matrices, width, height);
          const [sx, sy, sz] = lighting.sunDirection;
          const length = Math.hypot(sx, sy, sz) || 1;
          light.set({
            sun_dir: [sx / length, sy / length, sz / length],
            sun_intensity: lighting.sunIntensity,
            sun_color: lighting.sunColor,
            fog_density: lighting.fogDensity,
            sky_color: lighting.skyColor,
            ambient: lighting.ambient,
            ground_color: lighting.groundColor,
            horizon_color: lighting.horizonColor,
          });
          camera.upload();
          light.upload();
        },
        destroy() {
          camera.destroy();
          light.destroy();
          views.delete(view);
        },
      };
      views.add(view);
      return view;
    },

    upload(objects) {
      if (objects.length > maxObjects) {
        throw new RangeError(`Forward renderer holds ${maxObjects} objects, got ${objects.length}`);
      }
      objects.forEach((object, index) => {
        const base = (index * stride) >> 2;
        staging.set(object.model, base + objectLayoutFields.model);
        // Normal matrix = transpose(inverse(upper 3x3)), written as padded columns.
        mat4.invert(object.model, scratchInverse);
        const normalBase = base + objectLayoutFields.normal;
        for (let column = 0; column < 3; column++) {
          for (let row = 0; row < 3; row++) {
            staging[normalBase + column * 4 + row] = scratchInverse[row * 4 + column]!;
          }
        }
        const colorBase = base + objectLayoutFields.color;
        staging[colorBase] = object.color[0];
        staging[colorBase + 1] = object.color[1];
        staging[colorBase + 2] = object.color[2];
        staging[colorBase + 3] = object.emissive ?? 0;
      });
      if (objects.length > 0) {
        device.queue.writeBuffer(objectBuffer, 0, staging.buffer, 0, objects.length * stride);
      }
      return { objects };
    },

    draw(encoder, batch, drawOptions) {
      const pass = encoder.beginRenderPass({
        label: drawOptions.label ?? 'scene forward',
        colorAttachments: [
          {
            view: drawOptions.color,
            clearValue: drawOptions.clearColor ?? { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
        depthStencilAttachment: {
          view: drawOptions.depth,
          depthClearValue: 1,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
        timestampWrites: drawOptions.timestampWrites,
      });
      drawOptions.background?.(pass);
      pass.setPipeline(drawOptions.mirrored ? pipelines.mirrored : pipelines.normal);
      pass.setBindGroup(0, drawOptions.view.bindGroup);
      let bound: Mesh | undefined;
      batch.objects.forEach((object, index) => {
        pass.setBindGroup(1, objectGroup, [index * stride]);
        if (object.mesh !== bound) {
          pass.setVertexBuffer(0, object.mesh.vertexBuffer);
          pass.setIndexBuffer(object.mesh.indexBuffer, object.mesh.indexFormat);
          bound = object.mesh;
        }
        pass.drawIndexed(object.mesh.indexCount);
      });
      pass.end();
    },

    destroy() {
      objectBuffer.destroy();
      for (const view of [...views]) view.destroy();
    },
  };
}
