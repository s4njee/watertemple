let devicePromise: Promise<GPUDevice> | undefined;

/** One device shared by every test in a file. */
export function getTestDevice(): Promise<GPUDevice> {
  devicePromise ??= (async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter available to the test runner');
    return adapter.requestDevice();
  })();
  return devicePromise;
}

/** Runs `run` and throws if it raised a WebGPU validation error. */
export async function expectValid<T>(device: GPUDevice, run: () => T | Promise<T>): Promise<T> {
  device.pushErrorScope('validation');
  const outcome = await Promise.resolve()
    .then(run)
    .then(
      (value) => ({ ok: true as const, value }),
      (error: unknown) => ({ ok: false as const, error }),
    );
  const gpuError = await device.popErrorScope();
  if (gpuError) throw new Error(`WebGPU validation error: ${gpuError.message}`);
  if (!outcome.ok) throw outcome.error;
  return outcome.value;
}

/** Runs `run` and returns the validation error message it raised, or null if it raised none. */
export async function captureValidationError(
  device: GPUDevice,
  run: () => unknown,
): Promise<string | null> {
  device.pushErrorScope('validation');
  await run();
  const error = await device.popErrorScope();
  return error ? error.message : null;
}

/**
 * Reads back mip level 0 of a texture as tightly packed bytes (row padding removed).
 * The texture needs COPY_SRC usage. Wrap the result in a Float32Array etc. for float formats.
 */
export async function readTexture(
  device: GPUDevice,
  texture: GPUTexture,
  bytesPerPixel = 4,
): Promise<Uint8Array> {
  const { width, height } = texture;
  const rowBytes = width * bytesPerPixel;
  const bytesPerRow = Math.ceil(rowBytes / 256) * 256;
  const buffer = device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow }, { width, height });
  device.queue.submit([encoder.finish()]);

  await buffer.mapAsync(GPUMapMode.READ);
  const padded = new Uint8Array(buffer.getMappedRange());
  const pixels = new Uint8Array(rowBytes * height);
  for (let y = 0; y < height; y++) {
    pixels.set(padded.subarray(y * bytesPerRow, y * bytesPerRow + rowBytes), y * rowBytes);
  }
  buffer.unmap();
  buffer.destroy();
  return pixels;
}
