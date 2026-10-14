/** Data-only source listing: keeps the gallery independent of the renderer. */
export const UI_EFFECT_IDS = [
  'liquid-glass',
  'gooey-metaballs',
  'liquid-buttons',
  'liquid-transitions',
  'liquid-image-hover',
  'liquid-typography',
  'fill-level',
  'liquid-blob',
  'water-cursor',
  'scroll-water',
  'ui-droplets',
] as const;
export function uiSourceFiles(paths: readonly string[], id: string) {
  if (!(UI_EFFECT_IDS as readonly string[]).includes(id)) return [];
  return [
    'display.wgsl',
    'backdrop.wgsl',
    'text.wgsl',
    ...(id === 'water-cursor' ? ['trail.wgsl'] : []),
    'effect.ts',
    'dynamics.ts',
    'atlas.ts',
    'capture.ts',
  ]
    .map((name) => ({ name: `ui/${name}`, path: `../gpu/ui/${name}` }))
    .filter((file) => paths.includes(file.path));
}
