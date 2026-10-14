// M3.5: shared temple core, extracted to @mk7s/hologpu-temple-core. Water Temple
// resolves #include from its own src/wgsl library, so register that with the shared shader.
export * from '@mk7s/hologpu-temple-core/gpu/shader';
import { setShaderLibrary } from '@mk7s/hologpu-temple-core/gpu/shader';
import { getChunk } from '../wgsl/index.ts';
setShaderLibrary(getChunk);
