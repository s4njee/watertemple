import {
  uploadBackdrop,
  rasteriseForeignObject,
  htmlCanvasCaptureSupport,
} from '../../gpu/ui/capture.ts';
import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import './page.css';
const text = (id: string, value: string) => {
  document.getElementById(id)!.textContent = value;
};
const button = document.getElementById('run') as HTMLButtonElement;
button.addEventListener('click', () => {
  void run();
});
async function run() {
  button.disabled = true;
  text('status', 'Running…');
  let device: GPUDevice | undefined;
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU is unavailable');
    device = await adapter.requestDevice();
    text(
      'html-result',
      `GPUQueue HTML capture: ${htmlCanvasCaptureSupport(device.queue)}. No flags or origin-trial token were enabled by this page.`,
    );
    const format = navigator.gpu.getPreferredCanvasFormat();
    const pipeline = createFullscreenPipeline(device, {
      format,
      fragment: {
        label: 'capture-spike/display.wgsl',
        code: `
#include "fullscreen.wgsl"
#include "color.wgsl"
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var linear_sampler: sampler;
@fragment fn fs(in:FullscreenVarying)->@location(0) vec4f {let c=textureSampleLevel(image,linear_sampler,in.uv,0.0);return vec4f(linear_to_srgb(c.rgb),c.a);}`,
      },
    });
    const show = async (id: string, source: GPUCopyExternalImageSource) => {
      device!.pushErrorScope('validation');
      const texture = uploadBackdrop(device!, source);
      const canvas = document.getElementById(id) as HTMLCanvasElement;
      const context = canvas.getContext('webgpu')!;
      context.configure({ device: device!, format, alphaMode: 'opaque' });
      const group = device!.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: texture.createView() },
          { binding: 1, resource: device!.createSampler({ magFilter: 'linear' }) },
        ],
      });
      const encoder = device!.createCommandEncoder();
      renderFullscreen(encoder, context.getCurrentTexture().createView(), pipeline, [group]);
      device!.queue.submit([encoder.finish()]);
      await device!.queue.onSubmittedWorkDone();
      const error = await device!.popErrorScope();
      texture.destroy();
      if (error) throw new Error(error.message);
    };
    const source = document.createElement('canvas');
    source.width = 480;
    source.height = 240;
    const ctx = source.getContext('2d')!;
    ctx.fillStyle = '#dce9de';
    ctx.fillRect(0, 0, 480, 240);
    ctx.fillStyle = '#145c67';
    ctx.beginPath();
    ctx.arc(360, 140, 80, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = 'bold 44px system-ui';
    ctx.fillStyle = '#163137';
    ctx.fillText('WATER', 30, 110);
    ctx.font = '18px system-ui';
    ctx.fillText('Canvas content, uploaded once', 30, 170);
    const start = performance.now();
    await show('upload', source);
    text(
      'upload-result',
      `Pass. 480 × 240 sRGB source; ${(performance.now() - start).toFixed(1)} ms wall time including first pipeline use and GPU completion.`,
    );
    let bitmap: ImageBitmap | undefined;
    try {
      const begin = performance.now();
      bitmap = await rasteriseForeignObject(
        '<div style="width:480px;height:240px;background:#dce9de;color:#163137;font:700 44px system-ui;padding:30px;box-sizing:border-box">WATER<p style="font:18px system-ui">Self-contained XHTML snapshot</p></div>',
        480,
        240,
      );
      await show('foreign', bitmap);
      text(
        'foreign-result',
        `Pass. ${(performance.now() - begin).toFixed(1)} ms decode + upload + completion. Check for text in the image: decode success alone does not prove foreignObject painted.`,
      );
    } catch (error) {
      text(
        'foreign-result',
        `Unavailable here: ${error instanceof Error ? error.message : String(error)}`,
      );
    } finally {
      bitmap?.close();
    }
    text(
      'status',
      'Capture probes complete. Results apply to this browser and this self-contained source.',
    );
  } catch (error) {
    text('status', error instanceof Error ? error.message : String(error));
  } finally {
    device?.destroy();
    button.disabled = false;
  }
}
