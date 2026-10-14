// The shared WGSL library: every .wgsl file in this directory is an include target,
// addressed by file name (`#include "noise.wgsl"`).
const files = import.meta.glob<string>('./*.wgsl', {
  query: '?raw',
  import: 'default',
  eager: true,
});

interface ChunkStore {
  chunks: Map<string, string>;
  listeners: Set<() => void>;
}

// During development an edited .wgsl file propagates to this module, which accepts the update
// below, so Vite runs a fresh copy of it instead of reloading the page. Every other module still
// holds functions from the first copy, so the state lives in `hot.data`, which all copies share.
const previous = import.meta.hot?.data.chunkStore as ChunkStore | undefined;
const store: ChunkStore = previous ?? { chunks: new Map(), listeners: new Set() };

store.chunks.clear();
for (const [path, source] of Object.entries(files)) {
  store.chunks.set(path.replace(/^\.\//, ''), source);
}

// Vite only recognises a self-accepting module by the literal text `import.meta.hot.accept(`,
// so this must not go through an alias. Production builds replace `import.meta.hot` and drop it.
if (import.meta.hot) {
  import.meta.hot.data.chunkStore = store;
  import.meta.hot.accept();
  if (previous) {
    for (const listener of [...store.listeners]) {
      try {
        listener();
      } catch (error) {
        console.error(error);
      }
    }
  }
}

export function getChunk(name: string): string | undefined {
  return store.chunks.get(name);
}

export function chunkNames(): string[] {
  return [...store.chunks.keys()].sort();
}

/** Fires after a shared chunk is edited during development. Returns an unsubscribe function. */
export function onChunksChanged(listener: () => void): () => void {
  store.listeners.add(listener);
  return () => store.listeners.delete(listener);
}
