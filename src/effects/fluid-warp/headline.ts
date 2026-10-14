// The headline of the fluid-warp effect, drawn into a 2D canvas at device resolution and uploaded as a
// texture. The font and the strings are constants here because the control schema cannot carry text;
// the effect lets the visitor choose among HEADLINES. Everything is DOM-only: `renderHeadline` returns
// undefined where there is no document or no 2D context (the tests), and the effect then draws a
// procedural picture instead.

/** The strings on offer, shown in the `headline` select (one to three words each). */
export const HEADLINES = ['Everything flows', 'Make some waves', 'Liquid type'] as const;

/** CSS font weight and family; the size is chosen to fill the canvas. */
export const HEADLINE_WEIGHT = 800;
export const HEADLINE_FAMILY =
  '"Inter", "Helvetica Neue", "Segoe UI", Helvetica, Arial, system-ui, sans-serif';
/** Share of the canvas width and height the type may fill. */
export const HEADLINE_FILL = { width: 0.88, height: 0.62 } as const;
export const HEADLINE_LINE_HEIGHT = 1.02;

const PAPER = ['#0a2c4d', '#14606f', '#1f8a8a'] as const;
const INK = '#fff4de';
const ACCENT = '#ff6a4d';

/** The ways to cut `words` into at most `max` consecutive lines. */
export function splitLines(words: readonly string[], max = 3): string[][] {
  const out: string[][] = [[words.join(' ')]];
  if (words.length >= 2 && max >= 2) {
    for (let i = 1; i < words.length; i++) {
      out.push([words.slice(0, i).join(' '), words.slice(i).join(' ')]);
    }
  }
  if (words.length >= 3 && max >= 3) {
    for (let i = 1; i < words.length - 1; i++) {
      for (let j = i + 1; j < words.length; j++) {
        out.push([
          words.slice(0, i).join(' '),
          words.slice(i, j).join(' '),
          words.slice(j).join(' '),
        ]);
      }
    }
  }
  return out;
}

/**
 * The layout that makes the type biggest: font size in px and the lines. `measure` gives the width of a
 * string at a font size of 100 px.
 */
export function layoutHeadline(
  text: string,
  width: number,
  height: number,
  measure: (line: string) => number,
): { size: number; lines: string[] } {
  const words = text.split(/\s+/).filter(Boolean);
  let best = { size: 0, lines: [text] };
  for (const lines of splitLines(words.length > 0 ? words : [text])) {
    const widest = Math.max(...lines.map(measure), 1);
    const byWidth = (HEADLINE_FILL.width * width * 100) / widest;
    const byHeight = (HEADLINE_FILL.height * height) / (lines.length * HEADLINE_LINE_HEIGHT);
    const size = Math.min(byWidth, byHeight);
    if (size > best.size) best = { size, lines };
  }
  return best;
}

/** Paints the headline card at `width` x `height` pixels, or returns undefined without a DOM. */
export function renderHeadline(
  text: string,
  width: number,
  height: number,
): HTMLCanvasElement | undefined {
  if (typeof document === 'undefined') return undefined;
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(2, Math.round(width));
  canvas.height = Math.max(2, Math.round(height));
  const ctx = canvas.getContext('2d');
  if (!ctx) return undefined;
  const w = canvas.width;
  const h = canvas.height;

  const gradient = ctx.createLinearGradient(0, 0, w, h);
  gradient.addColorStop(0, PAPER[0]);
  gradient.addColorStop(0.55, PAPER[1]);
  gradient.addColorStop(1, PAPER[2]);
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, w, h);

  // Faint ruled lines and a dotted grid in the empty parts: they show the flow where there is no type.
  const pitch = Math.max(12, Math.round(h / 14));
  ctx.strokeStyle = 'rgba(255,255,255,0.09)';
  ctx.lineWidth = Math.max(1, h / 540);
  ctx.beginPath();
  for (let y = pitch / 2; y < h; y += pitch) {
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
  }
  ctx.stroke();
  ctx.fillStyle = 'rgba(255,255,255,0.14)';
  const dot = Math.max(1.2, h / 380);
  for (let x = pitch / 2; x < w; x += pitch * 2) {
    for (let y = pitch / 2; y < h; y += pitch * 2) ctx.fillRect(x - dot / 2, y - dot / 2, dot, dot);
  }

  const font = (size: number) => `${HEADLINE_WEIGHT} ${size}px ${HEADLINE_FAMILY}`;
  ctx.font = font(100);
  const layout = layoutHeadline(text, w, h, (line) => ctx.measureText(line).width);
  ctx.font = font(layout.size);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  const lineHeight = layout.size * HEADLINE_LINE_HEIGHT;
  const block = lineHeight * layout.lines.length;
  const top = (h - block) / 2;
  ctx.fillStyle = INK;
  layout.lines.forEach((line, i) => {
    ctx.fillText(line, w / 2, top + lineHeight * (i + 0.8));
  });

  // An underline bar and a caption, so there is something small to read as well.
  ctx.fillStyle = ACCENT;
  const barY = Math.min(h - layout.size * 0.3, top + block + layout.size * 0.12);
  ctx.fillRect(w / 2 - layout.size * 0.9, barY, layout.size * 1.8, Math.max(3, layout.size * 0.06));
  ctx.font = `600 ${Math.max(11, layout.size * 0.16)}px ${HEADLINE_FAMILY}`;
  ctx.fillStyle = 'rgba(255,244,222,0.85)';
  ctx.fillText('drag across the type', w / 2, barY + Math.max(20, layout.size * 0.3));
  return canvas;
}
