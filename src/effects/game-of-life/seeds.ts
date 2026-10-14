export type Cell = readonly [number, number];
export const GLIDER: readonly Cell[] = [
  [1, 0],
  [2, 1],
  [0, 2],
  [1, 2],
  [2, 2],
];
const GUN = [
  '........................O...........',
  '......................O.O...........',
  '............OO......OO............OO',
  '...........O...O....OO............OO',
  'OO........O.....O...OO..............',
  'OO........O...O.OO....O.O...........',
  '..........O.....O.......O...........',
  '...........O...O....................',
  '............OO......................',
];
export function makeSeed(width: number, height: number, pattern: string, seed = 1) {
  const data = new Uint32Array(width * height);
  const put = (x: number, y: number) => {
    data[(((y % height) + height) % height) * width + (((x % width) + width) % width)] =
      1 | (1 << 8);
  };
  const stamp = (cells: readonly Cell[], x: number, y: number) =>
    cells.forEach(([a, b]) => put(x + a, y + b));
  let state = seed >>> 0 || 1;
  const random = () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 4294967296;
  };
  if (pattern === 'soup') {
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) if (random() < 0.23) put(x, y);
  } else if (pattern === 'oscillators') {
    for (let y = 8; y < height - 8; y += 20)
      for (let x = 8; x < width - 8; x += 20) {
        stamp(
          [
            [0, 0],
            [1, 0],
            [2, 0],
          ],
          x,
          y,
        );
        stamp(
          [
            [0, 0],
            [1, 0],
            [0, 1],
            [3, 2],
            [2, 3],
            [3, 3],
          ],
          x + 8,
          y + 7,
        );
      }
  } else {
    // Gosper guns keep producing gliders without automatic reseeding or changing Life's rules.
    for (let y = 6; y < height - 12; y += 70)
      for (let x = 6; x < width - 40; x += 96) {
        GUN.forEach((row, b) =>
          [...row].forEach((c, a) => {
            if (c === 'O') put(x + a, y + b);
          }),
        );
      }
    for (let i = 0; i < Math.max(12, (width * height) / 1500); i++) {
      stamp(GLIDER, Math.floor(random() * width), Math.floor(random() * height));
    }
  }
  return data;
}
