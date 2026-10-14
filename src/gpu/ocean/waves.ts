/** Packed Gerstner waves shared verbatim with ocean-waves.wgsl: dx, dz, amplitude, wavenumber. */
export const OCEAN_WAVES = new Float32Array([
  0.94, 0.342, 0.52, 0.24, 0.77, -0.638, 0.28, 0.43, -0.24, 0.971, 0.15, 0.78, 0.61, 0.792, 0.085,
  1.31, -0.88, 0.475, 0.042, 2.37, 0.36, -0.933, 0.024, 4.17,
]);
export interface WaveSample {
  position: [number, number, number];
  normal: [number, number, number];
  dx: [number, number, number];
  dz: [number, number, number];
  jacobian: number;
}
export function sampleWaves(
  x: number,
  z: number,
  time: number,
  strength = 1,
  chop = 0.8,
): WaveSample {
  const position: WaveSample['position'] = [x, 0, z];
  const dx: WaveSample['dx'] = [1, 0, 0];
  const dz: WaveSample['dz'] = [0, 0, 1];
  for (let i = 0; i < OCEAN_WAVES.length; i += 4) {
    const length = Math.hypot(OCEAN_WAVES[i]!, OCEAN_WAVES[i + 1]!);
    const ux = OCEAN_WAVES[i]! / length;
    const uz = OCEAN_WAVES[i + 1]! / length;
    const a = OCEAN_WAVES[i + 2]! * strength;
    const k = OCEAN_WAVES[i + 3]!;
    const phase = k * (ux * x + uz * z) - Math.sqrt(9.81 * k) * time + i * 0.31;
    const s = Math.sin(phase),
      c = Math.cos(phase);
    position[0] += chop * a * ux * c;
    position[1] += a * s;
    position[2] += chop * a * uz * c;
    dx[0] -= chop * a * k * ux * ux * s;
    dx[1] += a * k * ux * c;
    dx[2] -= chop * a * k * ux * uz * s;
    dz[0] -= chop * a * k * ux * uz * s;
    dz[1] += a * k * uz * c;
    dz[2] -= chop * a * k * uz * uz * s;
  }
  const n: WaveSample['normal'] = [
    dz[1] * dx[2] - dz[2] * dx[1],
    dz[2] * dx[0] - dz[0] * dx[2],
    dz[0] * dx[1] - dz[1] * dx[0],
  ];
  const len = Math.hypot(...n);
  return {
    position,
    dx,
    dz,
    normal: n.map((v) => v / len) as WaveSample['normal'],
    jacobian: dx[0] * dz[2] - dx[2] * dz[0],
  };
}
/** Invert horizontal displacement; reading h(x,z) directly makes floating objects visibly slide. */
export function sampleWorldWaves(
  x: number,
  z: number,
  time: number,
  strength = 1,
  chop = 0.8,
): WaveSample {
  let qx = x,
    qz = z;
  for (let i = 0; i < 8; i++) {
    const w = sampleWaves(qx, qz, time, strength, chop);
    const ex = w.position[0] - x,
      ez = w.position[2] - z;
    if (Math.abs(w.jacobian) < 0.1) break;
    qx -= Math.max(-1, Math.min(1, (w.dz[2] * ex - w.dz[0] * ez) / w.jacobian));
    qz -= Math.max(-1, Math.min(1, (-w.dx[2] * ex + w.dx[0] * ez) / w.jacobian));
  }
  return sampleWaves(qx, qz, time, strength, chop);
}
