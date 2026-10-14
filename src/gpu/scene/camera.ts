import { mat4, vec3, type Mat4, type Vec3 } from 'wgpu-matrix';
import { defineUniformStruct, type UniformBuffer } from '../uniforms.ts';

/*
 * Conventions, fixed once for everything that draws a 3D scene (pools, oceans, particle fluids):
 *
 * - World space is right-handed with +Y up. The camera looks down -Z in view space.
 * - Clip space follows WebGPU: x and y in -1..1 (y up), depth 0..1 (0 = near plane, 1 = far plane),
 *   which is what wgpu-matrix's `perspective` produces. The depth buffer is not reversed.
 * - Screen `uv` has its origin at the top-left with v growing downwards, exactly like
 *   `fullscreen.wgsl` and texture coordinates: uv = (ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5).
 * - Triangles are front-facing when counter-clockwise as seen from outside (WebGPU's default).
 * - Matrices are column-major Float32Arrays (wgpu-matrix), so they upload to WGSL `mat4x4f` as is.
 * - A `Plane` is a unit normal plus an offset: signed distance of p is dot(normal, p) + offset.
 */

export type Vec3Like = readonly [number, number, number] | Float32Array;

export interface CameraDescription {
  position: Vec3Like;
  target: Vec3Like;
  up?: Vec3Like;
  /** Vertical field of view in radians. */
  fovY: number;
  aspect: number;
  near: number;
  far: number;
}

/** Everything the GPU and the CPU need to know about a camera. All matrices are column-major. */
export interface CameraMatrices {
  view: Mat4;
  proj: Mat4;
  viewProj: Mat4;
  invView: Mat4;
  invProj: Mat4;
  invViewProj: Mat4;
  /** World-space eye position. */
  position: Vec3;
  near: number;
  far: number;
}

export function createCameraMatrices(): CameraMatrices {
  return {
    view: mat4.create(),
    proj: mat4.create(),
    viewProj: mat4.create(),
    invView: mat4.create(),
    invProj: mat4.create(),
    invViewProj: mat4.create(),
    position: vec3.create(),
    near: 0.1,
    far: 100,
  };
}

/** Fills the derived matrices from `view` and `proj`. */
function finish(camera: CameraMatrices): CameraMatrices {
  mat4.multiply(camera.proj, camera.view, camera.viewProj);
  mat4.invert(camera.view, camera.invView);
  mat4.invert(camera.proj, camera.invProj);
  mat4.invert(camera.viewProj, camera.invViewProj);
  return camera;
}

export function computeCamera(
  description: CameraDescription,
  dst: CameraMatrices = createCameraMatrices(),
): CameraMatrices {
  const { position, target, up = [0, 1, 0], fovY, aspect, near, far } = description;
  mat4.lookAt(position, target, up, dst.view);
  mat4.perspective(fovY, aspect, near, far, dst.proj);
  vec3.copy(position, dst.position);
  dst.near = near;
  dst.far = far;
  return finish(dst);
}

/**
 * Vertical field of view that keeps at least `minHorizontalFov` visible on tall canvases: a phone
 * in portrait would otherwise crop the sides of a scene framed for a landscape canvas.
 */
export function fitFovY(baseFovY: number, aspect: number, minHorizontalFov: number): number {
  const fromHorizontal = Math.tan(minHorizontalFov / 2) / Math.max(aspect, 1e-3);
  return 2 * Math.atan(Math.max(Math.tan(baseFovY / 2), fromHorizontal));
}

// --- Maths helpers used by the tests and by effects ----------------------------------------------

export interface ProjectedPoint {
  /** Screen uv, origin top-left. */
  u: number;
  v: number;
  /** Depth buffer value, 0..1. */
  depth: number;
  /** Clip-space w, i.e. the view-space distance along the view axis. Negative behind the camera. */
  w: number;
}

/** Projects a world point the way the GPU does. */
export function projectPoint(viewProj: Mat4, point: Vec3Like): ProjectedPoint {
  const [x = 0, y = 0, z = 0] = point;
  const m = viewProj;
  const cx = m[0]! * x + m[4]! * y + m[8]! * z + m[12]!;
  const cy = m[1]! * x + m[5]! * y + m[9]! * z + m[13]!;
  const cz = m[2]! * x + m[6]! * y + m[10]! * z + m[14]!;
  const cw = m[3]! * x + m[7]! * y + m[11]! * z + m[15]!;
  return { u: (cx / cw) * 0.5 + 0.5, v: 0.5 - (cy / cw) * 0.5, depth: cz / cw, w: cw };
}

/** World point under screen uv at a given depth-buffer value. */
export function unprojectPoint(invViewProj: Mat4, u: number, v: number, depth: number): Vec3 {
  const m = invViewProj;
  const x = u * 2 - 1;
  const y = 1 - v * 2;
  const wx = m[0]! * x + m[4]! * y + m[8]! * depth + m[12]!;
  const wy = m[1]! * x + m[5]! * y + m[9]! * depth + m[13]!;
  const wz = m[2]! * x + m[6]! * y + m[10]! * depth + m[14]!;
  const ww = m[3]! * x + m[7]! * y + m[11]! * depth + m[15]!;
  return vec3.create(wx / ww, wy / ww, wz / ww);
}

/** View-space distance along the view axis for a depth-buffer value (the same as `camera_linear_depth`). */
export function linearizeDepth(near: number, far: number, depth: number): number {
  return (near * far) / (far - depth * (far - near));
}

// --- Planes and mirroring ------------------------------------------------------------------------

export interface Plane {
  /** Unit length. */
  normal: Vec3Like;
  /** dot(normal, p) + offset is the signed distance of p; positive on the side the normal faces. */
  offset: number;
}

/** The horizontal plane y = height, facing up. */
export function horizontalPlane(height: number): Plane {
  return { normal: [0, 1, 0], offset: -height };
}

export function signedDistance(plane: Plane, point: Vec3Like): number {
  const [nx = 0, ny = 0, nz = 0] = plane.normal;
  const [x = 0, y = 0, z = 0] = point;
  return nx * x + ny * y + nz * z + plane.offset;
}

/**
 * The matrix that reflects world points about the plane: I - 2nn^T for directions plus
 * -2·offset·n for position. Points on the plane stay put; applying it twice is the identity.
 */
export function reflectionMatrix(plane: Plane, dst: Mat4 = mat4.create()): Mat4 {
  const [nx = 0, ny = 0, nz = 0] = plane.normal;
  const d = plane.offset;
  dst[0] = 1 - 2 * nx * nx;
  dst[1] = -2 * nx * ny;
  dst[2] = -2 * nx * nz;
  dst[3] = 0;
  dst[4] = -2 * nx * ny;
  dst[5] = 1 - 2 * ny * ny;
  dst[6] = -2 * ny * nz;
  dst[7] = 0;
  dst[8] = -2 * nx * nz;
  dst[9] = -2 * ny * nz;
  dst[10] = 1 - 2 * nz * nz;
  dst[11] = 0;
  dst[12] = -2 * d * nx;
  dst[13] = -2 * d * ny;
  dst[14] = -2 * d * nz;
  dst[15] = 1;
  return dst;
}

/**
 * Replaces the near plane of `proj` with `plane` (a world-space plane; `invViewOfWorld` is the inverse
 * of the matrix taking those world coordinates into the view space `proj` projects from),
 * so the clipper discards everything on the negative side of the plane for free. This is Eric
 * Lengyel's oblique near-plane clipping, re-derived for WebGPU's 0..1 depth: row 2 of the
 * projection becomes lambda * C, with lambda chosen so the frustum corner farthest from the plane
 * still lands exactly on the far plane (z = w).
 *
 * Valid only when the eye is on the negative side of the plane, which is the case for a mirrored
 * camera looking up through the surface it was mirrored in. Otherwise `proj` is returned
 * unchanged. The depth range of the result is skewed: do not use it with `camera_linear_depth`.
 */
export function obliqueProjection(
  proj: Mat4,
  invProj: Mat4,
  invViewOfWorld: Mat4,
  plane: Plane,
  dst: Mat4 = mat4.create(),
): Mat4 {
  const [nx = 0, ny = 0, nz = 0] = plane.normal;
  const p = [nx, ny, nz, plane.offset] as const;
  // The plane as a row vector times the inverse transform: dot(C, viewPoint) == signed distance.
  const c = [0, 0, 0, 0];
  for (let k = 0; k < 4; k++) {
    for (let j = 0; j < 4; j++) c[k]! += p[j]! * invViewOfWorld[k * 4 + j]!;
  }
  mat4.copy(proj, dst);
  // The eye is the view-space origin, so its signed distance is c[3]: it must be behind the plane.
  if (!(c[3]! < 0)) return dst;

  // Which corner of the far plane is farthest from the plane: the sign of C in clip space.
  const cx = c[0]! * invProj[0]! + c[1]! * invProj[1]! + c[2]! * invProj[2]! + c[3]! * invProj[3]!;
  const cy = c[0]! * invProj[4]! + c[1]! * invProj[5]! + c[2]! * invProj[6]! + c[3]! * invProj[7]!;
  const qx = Math.sign(cx) || 1;
  const qy = Math.sign(cy) || 1;
  // Q = invProj * (qx, qy, 1, 1), a homogeneous view-space point.
  const q = [0, 0, 0, 0];
  for (let r = 0; r < 4; r++) {
    q[r] = invProj[r]! * qx + invProj[4 + r]! * qy + invProj[8 + r]! + invProj[12 + r]!;
  }
  const lambda = 1 / (c[0]! * q[0]! + c[1]! * q[1]! + c[2]! * q[2]! + c[3]! * q[3]!);
  dst[2] = c[0]! * lambda;
  dst[6] = c[1]! * lambda;
  dst[10] = c[2]! * lambda;
  dst[14] = c[3]! * lambda;
  return dst;
}

/**
 * The camera as seen in a mirror: the view is reflected about the plane and the near plane is
 * moved onto the plane (see `obliqueProjection`), so geometry below the surface never reaches the
 * reflection image.
 *
 * Reflecting flips handedness, so triangles that were counter-clockwise on screen become clockwise:
 * draw the mirrored pass with `frontFace: 'cw'` (the forward renderer does this for `mirrored`).
 * The result has the same aspect, field of view and `uv` meaning as `camera`, so a world point on
 * the plane lands at its own mirrored position in the reflection image.
 */
export function mirrorCamera(
  camera: CameraMatrices,
  plane: Plane,
  dst: CameraMatrices = createCameraMatrices(),
): CameraMatrices {
  const mirror = reflectionMatrix(plane);
  // world -> mirrored world -> view
  mat4.multiply(camera.view, mirror, dst.view);
  mat4.invert(dst.view, dst.invView);
  obliqueProjection(camera.proj, camera.invProj, dst.invView, plane, dst.proj);
  // The eye in the mirror: the point the reflected view looks from.
  const [x = 0, y = 0, z = 0] = camera.position;
  dst.position[0] = mirror[0]! * x + mirror[4]! * y + mirror[8]! * z + mirror[12]!;
  dst.position[1] = mirror[1]! * x + mirror[5]! * y + mirror[9]! * z + mirror[13]!;
  dst.position[2] = mirror[2]! * x + mirror[6]! * y + mirror[10]! * z + mirror[14]!;
  dst.near = camera.near;
  dst.far = camera.far;
  mat4.multiply(dst.proj, dst.view, dst.viewProj);
  mat4.invert(dst.proj, dst.invProj);
  mat4.invert(dst.viewProj, dst.invViewProj);
  return dst;
}

// --- GPU uniform ---------------------------------------------------------------------------------

/**
 * The uniform block behind `struct Camera` in `src/wgsl/camera.wgsl`. That chunk holds a copy of
 * `CameraUniform.wgsl`; tests/scene.test.ts fails if the two drift apart.
 */
export const CameraUniform = defineUniformStruct('Camera', {
  view: 'mat4x4f',
  proj: 'mat4x4f',
  view_proj: 'mat4x4f',
  inv_view: 'mat4x4f',
  inv_proj: 'mat4x4f',
  inv_view_proj: 'mat4x4f',
  position: 'vec3f',
  near: 'f32',
  far: 'f32',
  viewport: 'vec2f',
});

export type CameraUniformBuffer = UniformBuffer<typeof CameraUniform.fields>;

/** Stores a camera in a `CameraUniform` buffer (call `upload()` afterwards). */
export function setCameraUniform(
  buffer: CameraUniformBuffer,
  camera: CameraMatrices,
  width: number,
  height: number,
): void {
  buffer.set({
    view: camera.view,
    proj: camera.proj,
    view_proj: camera.viewProj,
    inv_view: camera.invView,
    inv_proj: camera.invProj,
    inv_view_proj: camera.invViewProj,
    position: camera.position,
    near: camera.near,
    far: camera.far,
    viewport: [width, height],
  });
}
