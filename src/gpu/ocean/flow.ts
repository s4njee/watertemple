export const RIVER_ROCKS = [
  [1.8, -3, 1.3],
  [-1.5, -9, 1.05],
  [0.7, -18, 1.8],
  [-3, -25, 1.3],
] as const;
/** Potential-flow disturbance around circular rocks; the baked RG texture stores signed velocity. */
export function riverFlow(
  x: number,
  z: number,
  rocks: readonly (readonly [number, number, number])[] = RIVER_ROCKS,
): [number, number] {
  let vx = 0,
    vz = -1;
  for (const [rx, rz, radius] of rocks) {
    const dx = x - rx,
      dz = z - rz,
      r2 = Math.max(dx * dx + dz * dz, radius * radius);
    const a = (radius * radius) / r2;
    vx += (2 * a * dz * dx) / r2;
    vz += -a + (2 * a * dz * dz) / r2;
  }
  const speed = Math.hypot(vx, vz);
  if (speed > 2) {
    vx *= 2 / speed;
    vz *= 2 / speed;
  }
  return [vx, vz];
}
export function createRiverFlowMap(device: GPUDevice) {
  const size = 128;
  const data = new Uint16Array(size * size * 4);
  // IEEE binary16 encoder, enough for bounded signed velocities.
  const half = (v: number) => {
    const f = new Float32Array([v]);
    const u = new Uint32Array(f.buffer)[0]!;
    const e = ((u >>> 23) & 255) - 127 + 15;
    const sign = (u >>> 16) & 0x8000;
    if (e <= 0) return sign;
    if (e >= 31) return sign | 0x7bff;
    return sign | (e << 10) | ((u >>> 13) & 1023);
  };
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const v = riverFlow(((x + 0.5) / size) * 32 - 16, ((y + 0.5) / size) * 64 - 40);
      const i = (y * size + x) * 4;
      data[i] = half(v[0]);
      data[i + 1] = half(v[1]);
    }
  const texture = device.createTexture({
    label: 'river signed flow map',
    size: [size, size],
    format: 'rgba16float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture({ texture }, data, { bytesPerRow: size * 8 }, [size, size]);
  return texture;
}
