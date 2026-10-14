/**
 * Memoises `create` per device. Keyed weakly by the device object, so a lost and recreated
 * device starts from scratch instead of reusing objects that belong to the dead one.
 */
export function perDevice<T>(create: (device: GPUDevice) => T): (device: GPUDevice) => T {
  const cache = new WeakMap<GPUDevice, T>();
  return (device) => {
    let value = cache.get(device);
    if (value === undefined) {
      value = create(device);
      cache.set(device, value);
    }
    return value;
  };
}

export const linearSampler = perDevice((device) =>
  device.createSampler({
    label: 'passes linear sampler',
    magFilter: 'linear',
    minFilter: 'linear',
  }),
);

const LINEAR_RENDERABLE: ReadonlySet<GPUTextureFormat> = new Set([
  'r8unorm',
  'rg8unorm',
  'rgba8unorm',
  'rgba8unorm-srgb',
  'bgra8unorm',
  'bgra8unorm-srgb',
  'r16float',
  'rg16float',
  'rgba16float',
  'rgb10a2unorm',
]);
const FLOAT32 = new Set<GPUTextureFormat>(['r32float', 'rg32float', 'rgba32float']);

/**
 * Throws a readable error for formats a linear-filtered render pass cannot use, instead of
 * letting the device raise a validation error at draw time.
 */
export function assertLinearRenderable(
  device: GPUDevice,
  format: GPUTextureFormat,
  who: string,
): void {
  if (LINEAR_RENDERABLE.has(format)) return;
  if (FLOAT32.has(format)) {
    if (device.features.has('float32-filterable')) return;
    throw new Error(
      `${who}: ${format} is not filterable on this device (it needs the 'float32-filterable' feature). Use rgba16float, or request the feature when creating the device.`,
    );
  }
  if (format === 'rg11b10ufloat' && device.features.has('rg11b10ufloat-renderable')) return;
  throw new Error(
    `${who}: ${format} cannot be filtered and rendered to. Use rgba8unorm, rgba16float or another filterable colour format.`,
  );
}
