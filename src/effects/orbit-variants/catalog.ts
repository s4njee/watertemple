/** Shared page, palette and motion settings for the orbit material studies. */
export type OrbitVariationId =
  | 'ink-orbit'
  | 'aurora'
  | 'neon-calligraphy'
  | 'dichroic-glass'
  | 'bioluminescent-currents'
  | 'liquid-chrome'
  | 'silk-embers'
  | 'eclipse-halos';

export interface OrbitVariation {
  title: string;
  words: readonly [string, string];
  page: string;
  style: number;
  background: string;
  accent: string;
  width: number;
  shell: number;
  light: number;
  description: string;
  card: string;
  palettes: readonly [string, string][];
  params: Readonly<Record<string, string | number | boolean>>;
}

export const MATERIAL_PALETTES: Record<string, readonly (readonly [number, number, number])[]> = {
  neon: [
    [0.9, 0.02, 0.34],
    [0.25, 0.025, 1],
    [0.015, 0.75, 1],
  ],
  dichroic: [
    [0.015, 0.75, 0.62],
    [0.45, 0.035, 0.7],
    [0.03, 0.36, 0.85],
  ],
  biolume: [
    [0.01, 0.58, 0.95],
    [0.015, 0.75, 0.48],
    [0.015, 0.4, 0.56],
  ],
  chrome: [
    [0.45, 0.5, 0.6],
    [0.7, 0.74, 0.8],
    [0.3, 0.35, 0.42],
  ],
  embers: [
    [0.72, 0.015, 0.002],
    [0.9, 0.18, 0.012],
    [1, 0.56, 0.065],
  ],
  eclipse: [
    [0.04, 0.3, 0.9],
    [0.33, 0.06, 0.7],
    [0.65, 0.35, 0.09],
  ],
};

export const ORBIT_VARIATIONS: Record<OrbitVariationId, OrbitVariation> = {
  'ink-orbit': {
    title: 'Ink orbit',
    words: ['Loose', 'orbit.'],
    page: 'ink-orbit.html',
    style: 0,
    background: '#1d1f29',
    accent: '#75d4ff',
    width: 32,
    shell: 40,
    light: 1.5,
    description: 'Restless RGB trails in a real fluid, swung around the page.',
    card: 'Take it with you.',
    palettes: [
      ['rgb cycle', 'RGB cycle'],
      ['primary', 'Primary'],
      ['rainbow', 'Rainbow'],
      ['pastel', 'Pastel'],
      ['sunset', 'Sunset'],
      ['lagoon', 'Lagoon'],
      ['sumi', 'Sumi'],
    ],
    params: {},
  },
  aurora: {
    title: 'Aurora ribbons',
    words: ['Aurora', 'ribbons.'],
    page: 'aurora.html',
    style: 0,
    background: '#050917',
    accent: '#57edb3',
    width: 24,
    shell: 56,
    light: 1.25,
    description: 'Silken ribbons of emerald, cyan and violet, folding gently around the page.',
    card: 'Draw the light closer.',
    palettes: [
      ['aurora', 'Northern lights'],
      ['lagoon', 'Emerald lagoon'],
      ['pastel', 'Pale violet'],
    ],
    params: {
      palette: 'aurora',
      heads: 7,
      ink: 0.42,
      gain: 1.2,
      headSpeed: 0.28,
      chaos: 0.15,
      wander: 0.35,
      swirl: 0.35,
      current: 0.07,
      wake: 0.28,
      stir: 0.16,
      vorticity: 2.5,
      dyeDissipation: 0.19,
    },
  },
  'neon-calligraphy': {
    title: 'Neon calligraphy',
    words: ['Neon', 'script.'],
    page: 'neon-calligraphy.html',
    style: 1,
    background: '#080811',
    accent: '#eb73d9',
    width: 5,
    shell: 24,
    light: 1.25,
    description:
      'Glowing ink blots draw fine luminous lines around the page. Each stroke softens into a lingering afterimage.',
    card: 'Write around me.',
    palettes: [
      ['neon', 'Electric neon'],
      ['rgb cycle', 'RGB cycle'],
      ['aurora', 'Cool spectrum'],
    ],
    params: {
      palette: 'neon',
      heads: 16,
      ink: 0.65,
      gain: 1.5,
      headSpeed: 0.44,
      chaos: 0.35,
      wander: 0.3,
      swirl: 0.5,
      current: 0.055,
      wake: 0.12,
      stir: 0.03,
      vorticity: 0.7,
      dyeDissipation: 0.42,
    },
  },
  'dichroic-glass': {
    title: 'Dichroic glass',
    words: ['Dichroic', 'glass.'],
    page: 'dichroic-glass.html',
    style: 2,
    background: '#080e18',
    accent: '#7cead6',
    width: 28,
    shell: 52,
    light: 1.1,
    description:
      'Transparent ribbons catch teal and magenta at different angles. Light bends inside the folds and gathers where they cross.',
    card: 'Bend the light.',
    palettes: [
      ['dichroic', 'Teal / magenta'],
      ['aurora', 'Northern glass'],
      ['pastel', 'Opal'],
    ],
    params: {
      palette: 'dichroic',
      heads: 7,
      ink: 0.7,
      gain: 1.15,
      headSpeed: 0.25,
      chaos: 0.25,
      wander: 0.35,
      swirl: 0.32,
      current: 0.04,
      wake: 0.2,
      stir: 0.1,
      vorticity: 2,
      dyeDissipation: 0.28,
    },
  },
  'bioluminescent-currents': {
    title: 'Bioluminescent currents',
    words: ['Living', 'currents.'],
    page: 'bioluminescent-currents.html',
    style: 3,
    background: '#020d1b',
    accent: '#54e1bb',
    width: 7,
    shell: 35,
    light: 1.25,
    description:
      'Cyan and green organisms drift through deep blue water. A sharp turn or a nearby pointer wakes a brief flash of light.',
    card: 'Wake the water.',
    palettes: [
      ['biolume', 'Ocean life'],
      ['aurora', 'Violet plankton'],
      ['lagoon', 'Green tide'],
    ],
    params: {
      palette: 'biolume',
      heads: 28,
      ink: 0.24,
      gain: 1.15,
      headSpeed: 0.24,
      chaos: 0.8,
      wander: 0.75,
      swirl: 0.32,
      current: 0.04,
      wake: 0.25,
      stir: 0.14,
      vorticity: 3,
      dyeDissipation: 0.55,
      force: 0.35,
    },
  },
  'liquid-chrome': {
    title: 'Liquid chrome',
    words: ['Liquid', 'chrome.'],
    page: 'liquid-chrome.html',
    style: 4,
    background: '#08090b',
    accent: '#cbd5e5',
    width: 22,
    shell: 42,
    light: 1.1,
    description:
      'Black and silver currents reflect a moving studio light. Sharp highlights carry small flashes of spectral colour.',
    card: 'Catch a reflection.',
    palettes: [
      ['chrome', 'Polished silver'],
      ['dichroic', 'Spectral chrome'],
      ['embers', 'Copper alloy'],
    ],
    params: {
      palette: 'chrome',
      heads: 9,
      ink: 0.8,
      gain: 1.1,
      headSpeed: 0.32,
      chaos: 0.3,
      wander: 0.35,
      swirl: 0.4,
      current: 0.06,
      wake: 0.3,
      stir: 0.18,
      vorticity: 3,
      dyeDissipation: 0.25,
    },
  },
  'silk-embers': {
    title: 'Silk embers',
    words: ['Silk', 'embers.'],
    page: 'silk-embers.html',
    style: 5,
    background: '#140806',
    accent: '#e9a565',
    width: 6,
    shell: 32,
    light: 1.2,
    description:
      'Copper, crimson and gold filaments weave through the dark. Sparks break free while the trails cool and disappear.',
    card: 'Carry a little fire.',
    palettes: [
      ['embers', 'Copper / crimson / gold'],
      ['sunset', 'Rose embers'],
      ['neon', 'Electric fire'],
    ],
    params: {
      palette: 'embers',
      heads: 18,
      ink: 0.48,
      gain: 1.3,
      headSpeed: 0.35,
      chaos: 0.5,
      wander: 0.55,
      swirl: 0.42,
      current: 0.04,
      wake: 0.12,
      stir: 0.07,
      vorticity: 1.5,
      dyeDissipation: 0.6,
    },
  },
  'eclipse-halos': {
    title: 'Eclipse halos',
    words: ['Eclipse', 'halos.'],
    page: 'eclipse-halos.html',
    style: 6,
    background: '#070912',
    accent: '#92aaff',
    width: 5,
    shell: 14,
    light: 1.25,
    description:
      'Thin rings of light trace each outline, escape into small arcs, then return. The page becomes a constellation of quiet eclipses.',
    card: 'Hold an eclipse.',
    palettes: [
      ['eclipse', 'Indigo / amber'],
      ['aurora', 'Emerald eclipse'],
      ['neon', 'Electric eclipse'],
    ],
    params: {
      palette: 'eclipse',
      heads: 8,
      ink: 0.35,
      gain: 1.25,
      headSpeed: 0.32,
      chaos: 0.65,
      wander: 0.2,
      swirl: 0.65,
      current: 0.13,
      wake: 0.05,
      stir: 0,
      vorticity: 0,
      dyeDissipation: 0.85,
      pull: 1.4,
    },
  },
};

export function orbitVariation(id: string | undefined): OrbitVariationId {
  return id && Object.hasOwn(ORBIT_VARIATIONS, id) ? (id as OrbitVariationId) : 'ink-orbit';
}

export function linearColour(hex: string): [number, number, number] {
  const linear = (value: number) =>
    value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  return [1, 3, 5].map((offset) => linear(parseInt(hex.slice(offset, offset + 2), 16) / 255)) as [
    number,
    number,
    number,
  ];
}
