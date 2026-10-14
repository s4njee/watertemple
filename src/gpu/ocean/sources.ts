/** Effects using the shared ocean renderer. Kept data-only so the source viewer stays lazy. */
export const OCEAN_EFFECT_IDS = [
  'gerstner-ocean',
  'fft-ocean',
  'shoreline',
  'river',
  'waterfall',
  'waterline',
  'calm-lake',
] as const;
export function oceanSourceFiles(paths: readonly string[], id: string) {
  if (!(OCEAN_EFFECT_IDS as readonly string[]).includes(id)) return [];
  const names = ['scene.wgsl', 'effect.ts', 'mesh.ts', 'waves.ts'];
  if (id === 'gerstner-ocean') names.push('buoyancy.ts');
  if (id === 'fft-ocean') names.push('fft.wgsl', 'fft.ts', 'spectrum.ts');
  if (id === 'river') names.push('flow.ts');
  if (id === 'waterfall') names.push('spray.ts');
  return names
    .map((name) => ({ name: `ocean/${name}`, path: `../gpu/ocean/${name}` }))
    .filter((file) => paths.includes(file.path));
}
