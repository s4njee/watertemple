// One registry over the generated per-effect entries. See scripts/registry.mjs (M3 story 3.3).
import type { EffectMeta, EffectModule } from './types.ts';
import { loaders, metas, posters } from './registry.generated.ts';

export function effectPoster(id: string): string | undefined {
  return posters[id];
}

export function listEffects(): EffectMeta[] {
  return [...metas].sort((a, b) => Number(a.epic) - Number(b.epic) || a.id.localeCompare(b.id));
}

export function hasEffect(id: string): boolean {
  return id in loaders;
}

export async function loadEffect(id: string): Promise<EffectModule> {
  const load = loaders[id];
  if (!load) throw new Error(`Unknown effect "${id}"`);
  return load();
}

const listeners = new Set<() => void>();

/** Fires after the generated registry changes during development. */
export function onEffectsChanged(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

if (import.meta.hot) {
  import.meta.hot.accept(() => {
    for (const listener of [...listeners]) listener();
  });
}
