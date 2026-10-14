// A `.wgsl` import is a ShaderSource built by vite-plugin-wgsl.ts: the file's text plus every
// file it includes. (`?raw` imports stay plain strings, typed by vite/client.)
declare module '*.wgsl' {
  const source: {
    readonly label: string;
    readonly code: string;
    readonly chunks: Readonly<Record<string, string>>;
  };
  export default source;
}
