/** Data-only discovery: shared weather source is visible alongside each small effect wrapper. */
export const WEATHER_EFFECT_IDS = [
  'rain-streaks',
  'rain-rings',
  'bubbles',
  'splash-burst',
  'fogged-glass',
  'frost-melt',
] as const;
export function weatherSourceFiles(paths: readonly string[], id: string) {
  if (!(WEATHER_EFFECT_IDS as readonly string[]).includes(id)) return [];
  return [
    'particles.wgsl',
    'display.wgsl',
    'scene.wgsl',
    'backdrop.wgsl',
    ...(id === 'fogged-glass' || id === 'frost-melt' ? ['memory.wgsl', 'memory.ts'] : []),
    'effect.ts',
    'dynamics.ts',
    'params.ts',
  ]
    .map((name) => ({ name: `weather/${name}`, path: `../gpu/weather/${name}` }))
    .filter((file) => paths.includes(file.path));
}
