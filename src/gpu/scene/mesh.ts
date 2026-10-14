import { VERTEX_STRIDE, type Geometry } from './geometry.ts';

/** A mesh on the GPU: one vertex buffer (position, normal) and one index buffer. */
export interface Mesh {
  readonly label: string;
  readonly vertexBuffer: GPUBuffer;
  readonly indexBuffer: GPUBuffer;
  readonly indexFormat: GPUIndexFormat;
  readonly indexCount: number;
  destroy(): void;
}

/** Vertex layout shared by every mesh and by the forward renderer's pipelines. */
export const MESH_VERTEX_LAYOUT: GPUVertexBufferLayout = {
  arrayStride: VERTEX_STRIDE,
  attributes: [
    { shaderLocation: 0, offset: 0, format: 'float32x3' },
    { shaderLocation: 1, offset: 12, format: 'float32x3' },
  ],
};

function upload(
  device: GPUDevice,
  label: string,
  data: Float32Array | Uint16Array | Uint32Array,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  // writeBuffer needs a multiple of 4 bytes; a Uint16 index list with an odd count is padded.
  const size = Math.ceil(data.byteLength / 4) * 4;
  const buffer = device.createBuffer({ label, size, usage: usage | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
  return buffer;
}

export function createMesh(device: GPUDevice, geometry: Geometry, label: string): Mesh {
  const vertexBuffer = upload(
    device,
    `${label} vertices`,
    geometry.vertices,
    GPUBufferUsage.VERTEX,
  );
  const indexBuffer = upload(device, `${label} indices`, geometry.indices, GPUBufferUsage.INDEX);
  return {
    label,
    vertexBuffer,
    indexBuffer,
    indexFormat: geometry.indices instanceof Uint16Array ? 'uint16' : 'uint32',
    indexCount: geometry.indexCount,
    destroy() {
      vertexBuffer.destroy();
      indexBuffer.destroy();
    },
  };
}
