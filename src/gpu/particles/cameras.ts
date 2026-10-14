import { mat4 } from 'wgpu-matrix';
import { createCameraMatrices, type CameraMatrices } from '../scene/camera.ts';

// An orthographic camera for two-dimensional particle systems (z = 0), producing the same CameraMatrices as
// computeCamera so it feeds the same uniform and the same renderers. World +x is right and +y is up on screen,
// the camera looks down -z from z = +distance, and depth runs 0 (near) .. 1 (far) like the perspective camera.

export interface OrthographicDescription {
  /** World point at the centre of the screen (x, y). */
  center: readonly [number, number];
  /** Half the visible height in world units; the width follows from `aspect`. */
  halfHeight: number;
  /** Canvas width / height. */
  aspect: number;
  /** Depth range along -z from the camera; particles at z = 0 must lie between. Defaults 0.1 and 20. */
  near?: number;
  far?: number;
}

export function orthographicCamera(
  description: OrthographicDescription,
  dst: CameraMatrices = createCameraMatrices(),
): CameraMatrices {
  const { center, halfHeight, aspect, near = 0.1, far = 20 } = description;
  const distance = (near + far) / 2;
  const halfWidth = halfHeight * aspect;
  const eye: [number, number, number] = [center[0], center[1], distance];
  mat4.lookAt(eye, [center[0], center[1], 0], [0, 1, 0], dst.view);
  mat4.ortho(-halfWidth, halfWidth, -halfHeight, halfHeight, near, far, dst.proj);
  mat4.multiply(dst.proj, dst.view, dst.viewProj);
  mat4.invert(dst.view, dst.invView);
  mat4.invert(dst.proj, dst.invProj);
  mat4.invert(dst.viewProj, dst.invViewProj);
  dst.position[0] = eye[0];
  dst.position[1] = eye[1];
  dst.position[2] = eye[2];
  dst.near = near;
  dst.far = far;
  return dst;
}
