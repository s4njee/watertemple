import { describe, expect, it } from 'vitest';
import {
  bandBins,
  classifyAudioFile,
  createAudioAnalyser,
  createAutoGain,
  createOnsetDetector,
  groupLevels,
  logBandEdges,
  smoothLevel,
  splitBands,
  synthStep,
} from '../src/gpu/audio.ts';

// No AudioContext here: the pure maths is tested directly and the browser wrapper only for being inert.

const SAMPLE_RATE = 48000;
const FFT = 2048;
const BIN_HZ = SAMPLE_RATE / FFT;

/** A byte spectrum with `value` in the bin nearest `hz` (and nothing elsewhere). */
function spike(hz: number, value = 255): Uint8Array {
  const spectrum = new Uint8Array(FFT / 2);
  spectrum[Math.round(hz / BIN_HZ)] = value;
  return spectrum;
}

describe('band edges and bins', () => {
  it('is log spaced: equal ratios between neighbouring edges', () => {
    const edges = logBandEdges(6, 40, 12000);
    expect(edges).toHaveLength(7);
    expect(edges[0]).toBeCloseTo(40);
    expect(edges[6]).toBeCloseTo(12000);
    const ratios = edges.slice(1).map((e, i) => e / edges[i]!);
    for (const r of ratios) expect(r).toBeCloseTo(ratios[0]!, 6);
    expect(ratios[0]).toBeGreaterThan(2);
  });

  it('survives silly arguments', () => {
    expect(logBandEdges(0)).toHaveLength(2);
    expect(logBandEdges(3, 100, 50).every(Number.isFinite)).toBe(true);
  });

  it('gives every band at least one bin, in order, and never bin 0', () => {
    const bins = bandBins(logBandEdges(6), SAMPLE_RATE, FFT);
    expect(bins).toHaveLength(6);
    for (const [lo, hi] of bins) {
      expect(lo).toBeGreaterThanOrEqual(1);
      expect(hi).toBeGreaterThanOrEqual(lo);
      expect(hi).toBeLessThan(FFT / 2);
    }
    for (let i = 1; i < bins.length; i++) {
      expect(bins[i]![0]).toBeGreaterThanOrEqual(bins[i - 1]![0]);
      expect(bins[i]![0]).toBeGreaterThan(bins[i - 1]![1] - 1);
    }
  });
});

describe('splitBands', () => {
  const edges = logBandEdges(6, 40, 12000);
  const bins = bandBins(edges, SAMPLE_RATE, FFT);
  const bandOf = (hz: number) => edges.findIndex((e, i) => hz >= e && hz < edges[i + 1]!);

  it.each([60, 150, 400, 1000, 3000, 8000])('puts the energy of a %d Hz tone in its band', (hz) => {
    const bands = splitBands(spike(hz), bins);
    const loudest = bands.indexOf(Math.max(...bands));
    expect(loudest).toBe(bandOf(hz));
    // And nothing leaks into a band two or more away.
    bands.forEach((b, i) => {
      if (Math.abs(i - loudest) >= 2) expect(b).toBe(0);
    });
  });

  it('is the mean over the band, scaled to 0..1', () => {
    const flat = new Uint8Array(FFT / 2).fill(255);
    for (const b of splitBands(flat, bins)) expect(b).toBeCloseTo(1);
    const half = new Uint8Array(FFT / 2).fill(51);
    for (const b of splitBands(half, bins)) expect(b).toBeCloseTo(0.2);
    for (const b of splitBands(new Uint8Array(FFT / 2), bins)) expect(b).toBe(0);
  });

  it('copes with a spectrum shorter than the bins', () => {
    expect(splitBands([], bins).every((b) => b === 0)).toBe(true);
  });
});

describe('groupLevels', () => {
  it('splits the bands into thirds', () => {
    const g = groupLevels([1, 1, 0.5, 0.5, 0, 0]);
    expect(g).toEqual({ bass: 1, mids: 0.5, highs: 0 });
  });
  it('handles few and no bands', () => {
    expect(groupLevels([])).toEqual({ bass: 0, mids: 0, highs: 0 });
    const two = groupLevels([0.2, 0.8]);
    expect(two.bass).toBeCloseTo(0.2);
    expect(two.highs).toBeCloseTo(0.8);
  });
});

describe('smoothing', () => {
  it('rises faster than it falls', () => {
    const up = smoothLevel(0, 1, 1 / 60, 0.015, 0.22);
    const down = 1 - smoothLevel(1, 0, 1 / 60, 0.015, 0.22);
    expect(up).toBeGreaterThan(down * 4);
  });

  it('does not depend on the frame rate', () => {
    let a = 0;
    for (let i = 0; i < 60; i++) a = smoothLevel(a, 1, 1 / 60, 0.1, 0.1);
    let b = 0;
    for (let i = 0; i < 120; i++) b = smoothLevel(b, 1, 1 / 120, 0.1, 0.1);
    expect(a).toBeCloseTo(b, 6);
  });

  it('follows instantly with a zero time constant and holds for a zero dt', () => {
    expect(smoothLevel(0, 0.7, 0.016, 0, 0)).toBe(0.7);
    expect(smoothLevel(0.3, 1, 0, 0.1, 0.1)).toBe(0.3);
  });
});

describe('auto gain', () => {
  const settle = (gain: ReturnType<typeof createAutoGain>, level: number, seconds: number) => {
    let out: number[] = [];
    for (let i = 0; i < seconds * 60; i++) out = gain.process([level, level / 2], 1 / 60);
    return out;
  };

  it('normalises a quiet and a loud signal to the same range', () => {
    const quiet = settle(createAutoGain(2), 0.3, 2);
    const loud = settle(createAutoGain(2), 0.9, 2);
    expect(quiet[0]).toBeCloseTo(1);
    expect(loud[0]).toBeCloseTo(1);
    // Relative balance between bands survives (per-band gain, so it is 1 each; the point is no clipping).
    expect(quiet.every((v) => v >= 0 && v <= 1)).toBe(true);
  });

  it('tracks a drop in volume over seconds, not instantly', () => {
    const gain = createAutoGain(1, { releaseSeconds: 4 });
    for (let i = 0; i < 120; i++) gain.process([0.9], 1 / 60);
    const justAfter = gain.process([0.45], 1 / 60)[0]!;
    expect(justAfter).toBeCloseTo(0.5, 1);
    let later = 0;
    for (let i = 0; i < 60 * 20; i++) later = gain.process([0.45], 1 / 60)[0]!;
    expect(later).toBeGreaterThan(0.95);
  });

  it('never amplifies silence or noise below the floor', () => {
    const gain = createAutoGain(1, { floor: 0.12 });
    let out = 0;
    for (let i = 0; i < 600; i++) out = gain.process([0.03], 1 / 60)[0]!;
    expect(out).toBeCloseTo(0.25, 2);
    expect(gain.process([0], 1 / 60)[0]).toBe(0);
  });

  it('clamps out-of-range and missing input', () => {
    const gain = createAutoGain(2);
    const out = gain.process([5, Number.NaN], 1 / 60);
    expect(out[0]).toBe(1);
    expect(out[1]).toBe(0);
    gain.reset();
    expect(gain.peaks.every((p) => p === 0.12)).toBe(true);
  });
});

describe('onset detection', () => {
  const dt = 1 / 60;
  /** Runs a signal (one value per frame) through a detector and returns the frame indexes that fired. */
  function run(signal: number[], detector = createOnsetDetector()) {
    const fired: number[] = [];
    signal.forEach((v, i) => {
      if (detector.process(v, dt).fired) fired.push(i);
    });
    return fired;
  }
  const repeat = (value: number, frames: number) => new Array<number>(frames).fill(value);

  it('fires on a jump, once', () => {
    const fired = run([...repeat(0.05, 60), ...repeat(0.9, 120)]);
    expect(fired).toEqual([60]);
  });

  it('does not fire on a plateau, silence, or a slow swell', () => {
    expect(run(repeat(0.6, 300))).toEqual([]);
    expect(run(repeat(0, 300))).toEqual([]);
    const swell = Array.from({ length: 240 }, (_, i) => 0.1 + (0.8 * i) / 240);
    expect(run(swell)).toEqual([]);
  });

  it('does not fire on the first frame, whatever the level', () => {
    expect(run(repeat(1, 5))).toEqual([]);
  });

  it('respects the refractory period, then fires again', () => {
    // Two jumps 6 frames (0.1 s) apart: the second is inside the 0.14 s refractory period.
    const close = [...repeat(0.05, 60), ...repeat(0.5, 6), ...repeat(0.95, 60)];
    expect(run(close)).toEqual([60]);
    // The same two jumps 40 frames apart both fire.
    const apart = [...repeat(0.05, 60), ...repeat(0.5, 40), ...repeat(0.99, 60)];
    expect(run(apart)).toEqual([60, 100]);
  });

  it('finds every hit of a drum-like pulse train, one event each', () => {
    // 120 BPM-ish: a sharp hit decaying over 0.3 s, every 0.5 s, for 6 hits.
    const signal: number[] = [];
    for (let beat = 0; beat < 6; beat++) {
      for (let f = 0; f < 30; f++) signal.push(0.08 + 0.85 * Math.exp(-f / 6));
    }
    const fired = run(signal);
    // The first sits at frame 0, where the follower is seeded, so five of six are found, each on its beat.
    expect(fired).toHaveLength(5);
    fired.forEach((frame, i) => expect(frame).toBe((i + 1) * 30));
  });

  it('reports a strength between 0 and 1 that grows with the jump', () => {
    const strengthOf = (to: number) => {
      const d = createOnsetDetector();
      for (let i = 0; i < 30; i++) d.process(0.05, dt);
      return d.process(to, dt).strength;
    };
    expect(strengthOf(0.3)).toBeGreaterThan(0);
    expect(strengthOf(1)).toBeGreaterThan(strengthOf(0.3));
    expect(strengthOf(1)).toBeLessThanOrEqual(1);
  });

  it('is not thrown by NaN or a zero interval', () => {
    const d = createOnsetDetector();
    expect(d.process(Number.NaN, dt).fired).toBe(false);
    for (const [value, step] of [
      [0.5, 0],
      [0.5, Number.NaN],
      [Number.NaN, 0.01],
      [0.2, -1],
    ] as const) {
      const result = d.process(value, step);
      expect(Number.isFinite(result.strength)).toBe(true);
    }
  });
});

describe('the built-in pattern', () => {
  const count = (voice: string, from: number, to: number) => {
    let n = 0;
    for (let s = from; s < to; s++) n += synthStep(s).filter((e) => e.voice === voice).length;
    return n;
  };

  it('has a kick on the downbeats and a snare on the backbeats', () => {
    expect(synthStep(0).some((e) => e.voice === 'kick')).toBe(true);
    expect(synthStep(8).some((e) => e.voice === 'kick')).toBe(true);
    expect(synthStep(4).some((e) => e.voice === 'snare')).toBe(true);
    expect(synthStep(12).some((e) => e.voice === 'snare')).toBe(true);
    expect(synthStep(1).some((e) => e.voice === 'kick')).toBe(false);
  });

  it('plays every voice over the loop, and loops after four bars', () => {
    for (const voice of ['kick', 'snare', 'hat', 'bass', 'arp'])
      expect(count(voice, 0, 64)).toBeGreaterThan(0);
    expect(synthStep(64)).toEqual(synthStep(0));
    expect(synthStep(-1)).toEqual(synthStep(63));
  });

  it('gives tonal voices a finite pitch inside the audible range', () => {
    for (let s = 0; s < 64; s++) {
      for (const e of synthStep(s)) {
        if (e.voice === 'bass' || e.voice === 'arp') {
          expect(e.frequency).toBeGreaterThan(30);
          expect(e.frequency).toBeLessThan(2000);
        }
        expect(e.accent).toBeGreaterThan(0);
        expect(e.accent).toBeLessThanOrEqual(1);
      }
    }
  });
});

describe('file classification', () => {
  it('uses the MIME type, then the extension', () => {
    expect(classifyAudioFile({ type: 'audio/mpeg' })).toBe('audio');
    expect(classifyAudioFile({ type: 'video/mp4' })).toBe('video');
    expect(classifyAudioFile({ name: 'loop.WAV' })).toBe('audio');
    expect(classifyAudioFile({ name: 'clip.webm' })).toBe('video');
    expect(classifyAudioFile({ name: 'notes.txt', type: 'text/plain' })).toBeUndefined();
    expect(classifyAudioFile({})).toBeUndefined();
  });
});

describe('without Web Audio', () => {
  it('is inert: waiting-free, silent and never throws', async () => {
    const audio = createAudioAnalyser();
    expect(audio.state).toBe('unsupported');
    expect(audio.source).toBe('synth');
    await audio.resume();
    audio.setActive(true);
    audio.setAudible(true);
    expect(audio.audible).toBe(true);
    audio.useSynth();
    const frame = audio.update(1 / 60);
    expect(frame.running).toBe(false);
    expect(frame.onset).toBe(false);
    expect(frame.bands).toHaveLength(6);
    expect(frame.bands.every((b) => b === 0)).toBe(true);
    expect(frame.level).toBe(0);
    await expect(audio.useFile({ name: 'a.mp3', type: 'audio/mpeg' } as File)).rejects.toThrow();
    expect(() => audio.destroy()).not.toThrow();
    expect(audio.state).toBe('closed');
    expect(audio.update(1 / 60).level).toBe(0);
  });

  it('honours the band count and edges', () => {
    const audio = createAudioAnalyser({ bands: 4, lowHz: 50, highHz: 8000 });
    expect(audio.edges).toHaveLength(5);
    expect(audio.update(0.016).bands).toHaveLength(4);
    audio.destroy();
  });
});
