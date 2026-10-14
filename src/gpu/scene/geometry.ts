/*
 * Mesh generators. Pure data, no GPU: they return interleaved vertices (position xyz, normal xyz;
 * 6 floats, 24 bytes per vertex) and indexed triangle lists, so tests can check them without a
 * device and `createMesh` uploads them as is.
 *
 * Every primitive is centred on the origin, has outward unit normals, and is wound
 * counter-clockwise when seen from outside (the WebGPU default front face, see camera.ts).
 */

export const VERTEX_FLOATS = 6;
export const VERTEX_STRIDE = VERTEX_FLOATS * 4;

export interface Geometry {
  /** Interleaved position (3) and normal (3) per vertex. */
  vertices: Float32Array;
  indices: Uint16Array | Uint32Array;
  vertexCount: number;
  indexCount: number;
}

/** Growable builder that picks the narrowest index type that fits. */
class Builder {
  readonly vertices: number[] = [];
  readonly indices: number[] = [];

  vertex(px: number, py: number, pz: number, nx: number, ny: number, nz: number): number {
    this.vertices.push(px, py, pz, nx, ny, nz);
    return this.vertices.length / VERTEX_FLOATS - 1;
  }

  triangle(a: number, b: number, c: number): void {
    this.indices.push(a, b, c);
  }

  build(): Geometry {
    const vertexCount = this.vertices.length / VERTEX_FLOATS;
    const indices =
      vertexCount <= 65536 ? new Uint16Array(this.indices) : new Uint32Array(this.indices);
    return {
      vertices: new Float32Array(this.vertices),
      indices,
      vertexCount,
      indexCount: indices.length,
    };
  }
}

export interface PlaneOptions {
  /** Extent along X. Default 1. */
  width?: number;
  /** Extent along Z. Default 1. */
  depth?: number;
  segmentsX?: number;
  segmentsZ?: number;
}

/** A horizontal grid in the XZ plane at y = 0 facing +Y. */
export function planeGeometry(options: PlaneOptions = {}): Geometry {
  const { width = 1, depth = 1 } = options;
  const sx = Math.max(1, Math.floor(options.segmentsX ?? 1));
  const sz = Math.max(1, Math.floor(options.segmentsZ ?? 1));
  const b = new Builder();
  for (let j = 0; j <= sz; j++) {
    for (let i = 0; i <= sx; i++) {
      b.vertex((i / sx - 0.5) * width, 0, (j / sz - 0.5) * depth, 0, 1, 0);
    }
  }
  const row = sx + 1;
  for (let j = 0; j < sz; j++) {
    for (let i = 0; i < sx; i++) {
      const a = j * row + i;
      // (x0,z0) -> (x0,z1) -> (x1,z0) is counter-clockwise seen from +Y.
      b.triangle(a, a + row, a + 1);
      b.triangle(a + 1, a + row, a + row + 1);
    }
  }
  return b.build();
}

export interface BoxOptions {
  width?: number;
  height?: number;
  depth?: number;
}

// Face normal n and two tangents with u x v = n, so (-u-v, +u-v, +u+v, -u+v) is counter-clockwise.
const BOX_FACES: readonly (readonly [number[], number[], number[]])[] = [
  [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
  [
    [-1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
  ],
  [
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
  ],
  [
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1],
  ],
  [
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
  ],
  [
    [0, 0, -1],
    [0, 1, 0],
    [1, 0, 0],
  ],
];

/** An axis-aligned box with flat shading: 24 vertices, 36 indices. */
export function boxGeometry(options: BoxOptions = {}): Geometry {
  const half = [(options.width ?? 1) / 2, (options.height ?? 1) / 2, (options.depth ?? 1) / 2];
  const b = new Builder();
  for (const [n, u, v] of BOX_FACES) {
    const corners = [
      [-1, -1],
      [1, -1],
      [1, 1],
      [-1, 1],
    ] as const;
    const first = b.vertices.length / VERTEX_FLOATS;
    for (const [cu, cv] of corners) {
      const p = [0, 1, 2].map((k) => (n[k]! + u[k]! * cu + v[k]! * cv) * half[k]!) as [
        number,
        number,
        number,
      ];
      b.vertex(p[0], p[1], p[2], n[0]!, n[1]!, n[2]!);
    }
    b.triangle(first, first + 1, first + 2);
    b.triangle(first, first + 2, first + 3);
  }
  return b.build();
}

export interface SphereOptions {
  radius?: number;
  /** Segments around the Y axis (at least 3). */
  widthSegments?: number;
  /** Segments from pole to pole (at least 2). */
  heightSegments?: number;
}

/**
 * A UV sphere. (w + 1)(h + 1) vertices (the seam and the poles are duplicated so normals stay
 * exact), and 6w(h - 1) indices because the degenerate triangles at the poles are left out.
 */
export function sphereGeometry(options: SphereOptions = {}): Geometry {
  const { radius = 0.5 } = options;
  const w = Math.max(3, Math.floor(options.widthSegments ?? 24));
  const h = Math.max(2, Math.floor(options.heightSegments ?? 16));
  const b = new Builder();
  for (let j = 0; j <= h; j++) {
    const theta = (j / h) * Math.PI;
    for (let i = 0; i <= w; i++) {
      const phi = (i / w) * Math.PI * 2;
      const nx = Math.sin(theta) * Math.cos(phi);
      const ny = Math.cos(theta);
      const nz = Math.sin(theta) * Math.sin(phi);
      b.vertex(nx * radius, ny * radius, nz * radius, nx, ny, nz);
    }
  }
  const row = w + 1;
  for (let j = 0; j < h; j++) {
    for (let i = 0; i < w; i++) {
      const a = j * row + i;
      if (j !== 0) b.triangle(a, a + 1, a + row);
      if (j !== h - 1) b.triangle(a + 1, a + row + 1, a + row);
    }
  }
  return b.build();
}

export interface CylinderOptions {
  radiusTop?: number;
  radiusBottom?: number;
  height?: number;
  radialSegments?: number;
  /** Close the ends. Default true. */
  caps?: boolean;
}

/** A cylinder (or cone frustum) along Y, from y = -height/2 to +height/2. */
export function cylinderGeometry(options: CylinderOptions = {}): Geometry {
  const { radiusTop = 0.5, radiusBottom = 0.5, height = 1, caps = true } = options;
  const n = Math.max(3, Math.floor(options.radialSegments ?? 24));
  const b = new Builder();
  const slope = (radiusBottom - radiusTop) / height;
  const norm = 1 / Math.hypot(1, slope);

  for (let j = 0; j <= 1; j++) {
    const radius = j === 0 ? radiusTop : radiusBottom;
    const y = height / 2 - j * height;
    for (let i = 0; i <= n; i++) {
      const phi = (i / n) * Math.PI * 2;
      const c = Math.cos(phi);
      const s = Math.sin(phi);
      b.vertex(c * radius, y, s * radius, c * norm, slope * norm, s * norm);
    }
  }
  const row = n + 1;
  for (let i = 0; i < n; i++) {
    // The sides are two rows; a degenerate triangle appears only for a cone's zero-radius end.
    if (radiusTop > 0) b.triangle(i, i + 1, i + row);
    if (radiusBottom > 0) b.triangle(i + 1, i + row + 1, i + row);
  }

  if (caps) {
    for (const top of [true, false]) {
      const radius = top ? radiusTop : radiusBottom;
      if (radius <= 0) continue;
      const y = top ? height / 2 : -height / 2;
      const ny = top ? 1 : -1;
      const centre = b.vertex(0, y, 0, 0, ny, 0);
      const ring = b.vertices.length / VERTEX_FLOATS;
      for (let i = 0; i <= n; i++) {
        const phi = (i / n) * Math.PI * 2;
        b.vertex(Math.cos(phi) * radius, y, Math.sin(phi) * radius, 0, ny, 0);
      }
      for (let i = 0; i < n; i++) {
        if (top) b.triangle(centre, ring + i + 1, ring + i);
        else b.triangle(centre, ring + i, ring + i + 1);
      }
    }
  }
  return b.build();
}
