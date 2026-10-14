/*
 * Demo media that needs no asset files: a still painted onto a 2D canvas, and a live "video" made by
 * turning an animated canvas into a real <video> element with captureStream(), so the genuine video
 * path (src/gpu/media.ts) is exercised. Both are DOM-only; the effect falls back to a procedural
 * texture when there is no document.
 *
 * The picture is a small harbour, chosen to make distortion easy to judge: long straight lines (pier
 * planks, posts, the horizon, the lighthouse stripes, a ruler along the frame), recognisable shapes
 * (a boat, a sun, houses), readable text (so a mirrored or flipped upload is obvious) and fine grain
 * (so softening and aliasing show). A row of known swatches at the bottom left checks colour.
 */

export const STILL_SIZE = { width: 1280, height: 720 } as const;
export const VIDEO_SIZE = { width: 640, height: 360 } as const;

/** Small deterministic PRNG so the picture is the same every time. */
function random(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gradient(
  ctx: CanvasRenderingContext2D,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  stops: [number, string][],
): CanvasGradient {
  const g = ctx.createLinearGradient(x0, y0, x1, y1);
  for (const [at, colour] of stops) g.addColorStop(at, colour);
  return g;
}

/** Paints the harbour still. */
export function paintStill(ctx: CanvasRenderingContext2D, width: number, height: number): void {
  const rand = random(7);
  const horizon = height * 0.5;

  // Sky and sun.
  ctx.fillStyle = gradient(ctx, 0, 0, 0, horizon, [
    [0, '#1d4f91'],
    [0.55, '#5b9bd0'],
    [1, '#f6d9a8'],
  ]);
  ctx.fillRect(0, 0, width, horizon);
  const sun = ctx.createRadialGradient(
    width * 0.74,
    horizon - 70,
    6,
    width * 0.74,
    horizon - 70,
    230,
  );
  sun.addColorStop(0, 'rgba(255,250,225,1)');
  sun.addColorStop(0.08, 'rgba(255,240,190,0.95)');
  sun.addColorStop(1, 'rgba(255,214,150,0)');
  ctx.fillStyle = sun;
  ctx.fillRect(0, 0, width, horizon);

  // Clouds: stacks of soft ellipses.
  for (let c = 0; c < 7; c++) {
    const cx = rand() * width;
    const cy = 40 + rand() * 170;
    for (let i = 0; i < 9; i++) {
      ctx.fillStyle = `rgba(255,255,255,${0.1 + rand() * 0.14})`;
      ctx.beginPath();
      ctx.ellipse(
        cx + (rand() - 0.5) * 150,
        cy + (rand() - 0.5) * 20,
        40 + rand() * 80,
        9 + rand() * 14,
        0,
        0,
        Math.PI * 2,
      );
      ctx.fill();
    }
  }

  // Distant hills.
  for (const [colour, base, rough] of [
    ['#7d93ab', 0.42, 26],
    ['#4f6a86', 0.46, 18],
  ] as const) {
    ctx.fillStyle = colour;
    ctx.beginPath();
    ctx.moveTo(0, horizon);
    for (let x = 0; x <= width; x += 16) {
      const ridge =
        Math.sin(x * 0.006 + base * 9) * 34 + Math.sin(x * 0.021 + base) * 14 + rand() * rough;
      ctx.lineTo(x, height * base - ridge * 0.5 + 8);
    }
    ctx.lineTo(width, horizon);
    ctx.closePath();
    ctx.fill();
  }

  // Sea, with the sun's glitter path and fine horizontal streaks.
  ctx.fillStyle = gradient(ctx, 0, horizon, 0, height, [
    [0, '#8dc3d6'],
    [0.35, '#2a7fa6'],
    [1, '#0b3b63'],
  ]);
  ctx.fillRect(0, horizon, width, height - horizon);
  for (let i = 0; i < 700; i++) {
    const y = horizon + 2 + Math.pow(rand(), 1.6) * (height - horizon);
    const depth = (y - horizon) / (height - horizon);
    const x = rand() * width;
    const near = Math.abs(x - width * 0.74) < 40 + depth * 150;
    ctx.fillStyle = near
      ? `rgba(255,244,210,${0.15 + rand() * 0.3})`
      : `rgba(255,255,255,${0.04 + rand() * 0.1})`;
    ctx.fillRect(x, y, 6 + rand() * (14 + depth * 70), 1 + depth * 1.4);
  }

  // Houses on the right: boxes with roofs and a grid of windows.
  const houses = [
    [830, 80, 96, '#d86a4a'],
    [930, 104, 78, '#f0c05a'],
    [1014, 66, 90, '#4f9d8a'],
    [1108, 92, 82, '#e7e0d2'],
    [1194, 74, 70, '#8a6bb5'],
  ] as const;
  for (const [x, h, w, colour] of houses) {
    const top = horizon - h + 14;
    ctx.fillStyle = colour;
    ctx.fillRect(x, top, w, h);
    ctx.fillStyle = 'rgba(0,0,0,0.28)';
    ctx.beginPath();
    ctx.moveTo(x - 5, top);
    ctx.lineTo(x + w / 2, top - 26);
    ctx.lineTo(x + w + 5, top);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = '#fbf2c9';
    for (let wx = x + 10; wx < x + w - 14; wx += 22) {
      for (let wy = top + 12; wy < horizon - 12; wy += 24) ctx.fillRect(wx, wy, 11, 14);
    }
  }
  ctx.fillStyle = 'rgba(8,40,70,0.55)';
  ctx.fillRect(820, horizon + 14, 440, 6);

  // Lighthouse on a rock.
  const lx = 210;
  const baseY = horizon + 22;
  ctx.fillStyle = '#4a4036';
  ctx.beginPath();
  ctx.moveTo(lx - 120, baseY + 18);
  ctx.lineTo(lx - 70, baseY - 22);
  ctx.lineTo(lx + 60, baseY - 26);
  ctx.lineTo(lx + 130, baseY + 18);
  ctx.closePath();
  ctx.fill();
  const stripes = ['#f4f1ea', '#c43b2f', '#f4f1ea', '#c43b2f', '#f4f1ea'];
  for (let s = 0; s < stripes.length; s++) {
    const y0 = baseY - 24 - s * 44;
    const half0 = 34 - s * 3.4;
    const half1 = 34 - (s + 1) * 3.4;
    ctx.fillStyle = stripes[s]!;
    ctx.beginPath();
    ctx.moveTo(lx - half0, y0);
    ctx.lineTo(lx + half0, y0);
    ctx.lineTo(lx + half1, y0 - 44);
    ctx.lineTo(lx - half1, y0 - 44);
    ctx.closePath();
    ctx.fill();
  }
  const topY = baseY - 24 - stripes.length * 44;
  ctx.fillStyle = '#2d2f36';
  ctx.fillRect(lx - 24, topY - 8, 48, 8);
  ctx.fillStyle = '#ffe9a0';
  ctx.fillRect(lx - 17, topY - 40, 34, 32);
  ctx.fillStyle = '#2d2f36';
  ctx.beginPath();
  ctx.moveTo(lx - 24, topY - 40);
  ctx.lineTo(lx, topY - 66);
  ctx.lineTo(lx + 24, topY - 40);
  ctx.closePath();
  ctx.fill();
  ctx.fillRect(lx - 20, topY - 41, 40, 4);

  // Sailboat.
  const bx = 560;
  const by = horizon + 44;
  ctx.fillStyle = '#f7f3e8';
  ctx.beginPath();
  ctx.moveTo(bx, by - 120);
  ctx.lineTo(bx, by - 12);
  ctx.lineTo(bx + 62, by - 12);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = '#e8503c';
  ctx.beginPath();
  ctx.moveTo(bx - 6, by - 98);
  ctx.lineTo(bx - 6, by - 12);
  ctx.lineTo(bx - 52, by - 12);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = '#25303c';
  ctx.fillRect(bx - 3, by - 126, 3, 116);
  ctx.beginPath();
  ctx.moveTo(bx - 66, by - 8);
  ctx.lineTo(bx + 76, by - 8);
  ctx.lineTo(bx + 56, by + 12);
  ctx.lineTo(bx - 48, by + 12);
  ctx.closePath();
  ctx.fillStyle = '#1b2a3a';
  ctx.fill();
  ctx.fillStyle = 'rgba(25,40,55,0.35)';
  ctx.fillRect(bx - 60, by + 14, 130, 4);

  // Pier in the foreground: deck planks in perspective, rails, posts.
  const deckTop = height * 0.8;
  ctx.fillStyle = '#6b4a2e';
  ctx.beginPath();
  ctx.moveTo(0, deckTop + 20);
  ctx.lineTo(width, deckTop - 6);
  ctx.lineTo(width, height);
  ctx.lineTo(0, height);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = 'rgba(20,10,0,0.55)';
  ctx.lineWidth = 2;
  for (let i = 0; i < 40; i++) {
    const x = i * 34 - 6;
    ctx.beginPath();
    ctx.moveTo(x, deckTop + 20 - (x / width) * 26);
    ctx.lineTo(x - 18 - i * 0.9, height);
    ctx.stroke();
  }
  ctx.strokeStyle = 'rgba(255,230,190,0.18)';
  for (let i = 0; i < 6; i++) {
    const y = deckTop + 36 + i * 24;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y - 24);
    ctx.stroke();
  }
  ctx.fillStyle = '#3f2c1a';
  for (let x = 40; x < width; x += 160) {
    const y = deckTop + 20 - (x / width) * 26;
    ctx.fillRect(x, y - 78, 10, 80);
  }
  ctx.fillStyle = '#4c3520';
  ctx.beginPath();
  ctx.moveTo(0, deckTop - 56);
  ctx.lineTo(width, deckTop - 82);
  ctx.lineTo(width, deckTop - 72);
  ctx.lineTo(0, deckTop - 46);
  ctx.closePath();
  ctx.fill();

  // Sign (text shows orientation) and a swatch row (colour check).
  ctx.fillStyle = '#fbf6e6';
  ctx.fillRect(width * 0.5 - 190, height * 0.84, 380, 74);
  ctx.strokeStyle = '#25303c';
  ctx.lineWidth = 4;
  ctx.strokeRect(width * 0.5 - 190, height * 0.84, 380, 74);
  ctx.fillStyle = '#12304f';
  ctx.font = '700 40px system-ui, Helvetica, Arial, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('WATER TEMPLE', width * 0.5, height * 0.84 + 28);
  ctx.font = '500 20px system-ui, Helvetica, Arial, sans-serif';
  ctx.fillText('harbour  ·  pier 7  ·  →', width * 0.5, height * 0.84 + 55);
  const swatches = ['#ff0000', '#00c000', '#0000ff', '#ffff00', '#808080', '#ffffff', '#000000'];
  swatches.forEach((colour, i) => {
    ctx.fillStyle = colour;
    ctx.fillRect(24 + i * 30, height - 50, 28, 28);
  });

  // Film grain, then a ruler along the frame: straight, fine, and right at the borders.
  for (let i = 0; i < 40000; i++) {
    ctx.fillStyle = rand() < 0.5 ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';
    ctx.fillRect(rand() * width, rand() * height, 2, 2);
  }
  ctx.fillStyle = 'rgba(255,255,255,0.85)';
  for (let x = 0; x <= width; x += 10) {
    const tall = x % 100 === 0;
    ctx.fillRect(x, 0, 1.5, tall ? 16 : 7);
    ctx.fillRect(x, height - (tall ? 16 : 7), 1.5, tall ? 16 : 7);
  }
  for (let y = 0; y <= height; y += 10) {
    const tall = y % 100 === 0;
    ctx.fillRect(0, y, tall ? 16 : 7, 1.5);
    ctx.fillRect(width - (tall ? 16 : 7), y, tall ? 16 : 7, 1.5);
  }
}

/** The still as a canvas, or undefined when there is no DOM. */
export function createStillCanvas(): HTMLCanvasElement | undefined {
  if (typeof document === 'undefined') return undefined;
  const canvas = document.createElement('canvas');
  canvas.width = STILL_SIZE.width;
  canvas.height = STILL_SIZE.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return undefined;
  paintStill(ctx, canvas.width, canvas.height);
  return canvas;
}

/** One frame of the animated canvas behind the live video. `t` is in seconds. */
export function paintLiveFrame(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  t: number,
): void {
  ctx.fillStyle = gradient(ctx, 0, 0, width, height, [
    [0, `hsl(${(210 + t * 12) % 360} 60% 22%)`],
    [1, `hsl(${(280 + t * 12) % 360} 55% 38%)`],
  ]);
  ctx.fillRect(0, 0, width, height);

  // A drifting grid of straight lines.
  ctx.strokeStyle = 'rgba(255,255,255,0.22)';
  ctx.lineWidth = 1.5;
  const step = 40;
  const shift = (t * 30) % step;
  for (let x = -step + shift; x < width; x += step) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 0; y < height; y += step) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  // Bars scrolling along the bottom.
  const colours = ['#ff5a4f', '#ffd24a', '#5bd67a', '#4fb3ff', '#c27bff'];
  for (let i = -1; i < 12; i++) {
    ctx.fillStyle = colours[(i + 120) % colours.length]!;
    ctx.fillRect(((i * 60 + t * 90) % 720) - 60, height - 34, 54, 34);
  }

  // A bouncing ball and a clock hand.
  const bx = width * 0.5 + Math.sin(t * 1.3) * width * 0.34;
  const by = height * 0.62 - Math.abs(Math.sin(t * 2.6)) * height * 0.34;
  ctx.fillStyle = '#fff3b0';
  ctx.beginPath();
  ctx.arc(bx, by, 26, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.arc(width - 90, 90, 56, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(width - 90, 90);
  ctx.lineTo(width - 90 + Math.sin(t * 2) * 50, 90 - Math.cos(t * 2) * 50);
  ctx.stroke();

  ctx.fillStyle = '#ffffff';
  ctx.font = '700 44px system-ui, Helvetica, Arial, sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText('LIVE', 24, 20);
  ctx.font = '500 26px ui-monospace, Menlo, monospace';
  ctx.fillText(`t = ${t.toFixed(1)} s`, 24, 74);
}

export interface LiveVideo {
  readonly video: HTMLVideoElement;
  /** Draw the next frame of the source canvas (call every rAF; it throttles itself to ~30 fps). */
  tick(time: number): void;
  destroy(): void;
}

/** A playing <video> fed by an animated canvas through captureStream, or undefined when unsupported. */
export async function createLiveVideo(): Promise<LiveVideo | undefined> {
  if (typeof document === 'undefined') return undefined;
  const canvas = document.createElement('canvas');
  canvas.width = VIDEO_SIZE.width;
  canvas.height = VIDEO_SIZE.height;
  const ctx = canvas.getContext('2d');
  if (!ctx || typeof canvas.captureStream !== 'function') return undefined;

  paintLiveFrame(ctx, canvas.width, canvas.height, 0);
  const stream = canvas.captureStream(30);
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.srcObject = stream;
  let lastDrawn = -1;
  await video.play();

  return {
    video,
    tick(time) {
      if (time - lastDrawn < 1 / 30) return;
      lastDrawn = time;
      paintLiveFrame(ctx, canvas.width, canvas.height, time);
    },
    destroy() {
      video.pause();
      for (const track of stream.getTracks()) track.stop();
      video.srcObject = null;
    },
  };
}
