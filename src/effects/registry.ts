import type { EffectMeta, EffectModule } from './types.ts';

// Each effect lives in src/effects/<id>/ and exports `meta` from meta.ts and `effect` from
// index.ts. Metadata is bundled eagerly so the gallery can list every effect; the effect
// itself (code + shaders) is a lazy chunk loaded only when it is shown.
const metaFiles = import.meta.glob<EffectMeta>('./*/meta.ts', { import: 'meta', eager: true });
const effectFiles = import.meta.glob<EffectModule>('./*/index.ts', { import: 'effect' });
const posterFiles = import.meta.glob<string>('./*/poster.webp', {
  query: '?url',
  import: 'default',
  eager: true,
});

export function effectPoster(id: string): string | undefined {
  return posterFiles[`./${id}/poster.webp`];
}

interface RegistryState {
  metas: Map<string, EffectMeta>;
  loaders: Map<string, () => Promise<EffectModule>>;
  listeners: Set<() => void>;
}

// During development Vite re-executes this module whenever an effect (or anything only an
// effect imports, such as its .wgsl files) changes. Other modules still hold functions from
// the first execution, so the state lives in `hot.data`, which every execution shares.
const previous = import.meta.hot?.data.registry as RegistryState | undefined;
const state: RegistryState = previous ?? {
  metas: new Map(),
  loaders: new Map(),
  listeners: new Set(),
};

const directoryOf = (path: string) => path.split('/')[1]!;

state.metas.clear();
for (const [path, meta] of Object.entries(metaFiles)) {
  const directory = directoryOf(path);
  if (meta.id !== directory) {
    throw new Error(`Effect id "${meta.id}" must match its directory "${directory}"`);
  }
  state.metas.set(meta.id, meta);
}

state.loaders.clear();
for (const [path, load] of Object.entries(effectFiles)) {
  state.loaders.set(directoryOf(path), load);
}

// Vite only recognises a self-accepting module by the literal text `import.meta.hot.accept(`,
// so this must not go through an alias.
if (import.meta.hot) {
  import.meta.hot.data.registry = state;
  import.meta.hot.accept();
  if (previous) for (const listener of [...state.listeners]) listener();
}

export function listEffects(): EffectMeta[] {
  return [...state.metas.values()].sort(
    (a, b) => Number(a.epic) - Number(b.epic) || a.id.localeCompare(b.id),
  );
}

export function hasEffect(id: string): boolean {
  return state.loaders.has(id);
}

export async function loadEffect(id: string): Promise<EffectModule> {
  const load = state.loaders.get(id);
  if (!load) throw new Error(`Unknown effect "${id}"`);
  return load();
}

/** Fires after an effect's code or shaders are edited during development. */
export function onEffectsChanged(listener: () => void): () => void {
  state.listeners.add(listener);
  return () => state.listeners.delete(listener);
}
