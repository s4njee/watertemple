export interface SpectrumOptions {
  size: number;
  lengths?: readonly number[];
  wind?: number;
  seed?: number;
}
export const OCEAN_LENGTHS = [192, 48, 12] as const;
/** Phillips spectrum in disjoint frequency bands. Values include N², cancelled by the IFFT. */
export function oceanSpectrum({
  size,
  lengths = OCEAN_LENGTHS,
  wind = 12,
  seed = 19,
}: SpectrumOptions): Float32Array {
  if (size < 8 || size > 256 || (size & (size - 1)) !== 0)
    throw new Error('FFT size must be a power of two in [8,256]');
  let state = seed >>> 0;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state + 0.5) / 4294967296;
  };
  const out = new Float32Array(size * size * lengths.length * 4);
  for (let layer = 0; layer < lengths.length; layer++) {
    const length = lengths[layer]!,
      dk = (2 * Math.PI) / length;
    const minK = layer === 0 ? 0 : (Math.PI * size) / lengths[layer - 1]!;
    const maxK = layer === lengths.length - 1 ? Infinity : (Math.PI * size) / length;
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++) {
        const kx = (x <= size / 2 ? x : x - size) * dk,
          kz = (y <= size / 2 ? y : y - size) * dk;
        const k = Math.hypot(kx, kz);
        const index = (layer * size * size + y * size + x) * 4;
        if (k === 0 || k < minK || k >= maxK) continue;
        const L = (wind * wind) / 9.81;
        const direction = (kx * 0.94 + kz * 0.342) / k;
        const phillips =
          (0.0035 *
            Math.exp(-1 / (k * k * L * L)) *
            direction ** 2 *
            Math.exp(-k * k * 0.03 ** 2)) /
          k ** 4;
        const sigma = Math.sqrt((phillips * dk * dk) / 2) * size * size;
        const r = Math.sqrt(-2 * Math.log(random())),
          angle = 2 * Math.PI * random();
        out[index] = sigma * r * Math.cos(angle);
        out[index + 1] = sigma * r * Math.sin(angle);
      }
  }
  // Store the opposite independent seed alongside h0. Evolution uses h0(k)+conj(h0(-k)).
  for (let l = 0; l < lengths.length; l++)
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++) {
        const i = (l * size * size + y * size + x) * 4,
          j = (l * size * size + ((size - y) % size) * size + ((size - x) % size)) * 4;
        out[i + 2] = out[j]!;
        out[i + 3] = out[j + 1]!;
      }
  return out;
}
