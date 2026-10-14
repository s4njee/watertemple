// Sound as an input for effects (story 5.8): a small Web Audio analysis helper.
//
//   const audio = createAudioAnalyser();
//   canvas.addEventListener('pointerdown', () => audio.resume());   // the autoplay policy, below
//   // each frame:
//   const a = audio.update(time.delta);     // a.bands, a.bass / mids / highs, a.onset ...
//   audio.setAudible(false);                // analysing does not need audible output
//   await audio.useFile(file);              // a dropped audio or video file; the built-in synth stops
//   audio.destroy();                        // closes the context, revokes the object URL
//
// WHAT COMES OUT. Per frame, from an AnalyserNode (2048-point FFT): a handful of log-spaced bands
// (six, 40 Hz to 12 kHz), each smoothed (fast attack, slow release) and auto-gained so that a quiet
// recording and a loud one drive an effect equally; the bands grouped into bass / mids / highs; an
// overall level; and onsets, which are a rising edge in the low end (a kick) or the high end (a hat),
// with a refractory period so one hit is one event. The maths is in pure functions (band edges, bin
// ranges, splitting, smoothing, auto-gain, onset detection) that are tested without a browser.
//
// SOURCES. (a) A built-in synthesised pattern: a kick, a snare burst, hats, a bass line and a
// pentatonic arpeggio from oscillators and noise bursts, scheduled a little ahead of the audio clock
// from `update` (no timers to leak; if the tab stops drawing, the music stops). No file, no network.
// (b) `useFile`: an audio or video File the visitor chose or dropped, played from a local object URL
// through a media element; nothing leaves the machine. The microphone is never requested.
//
// OUTPUT. Every source goes into one bus, then the AnalyserNode, then a speaker gain that is 0 by
// default and a destination. The analyser sits BEFORE the speaker gain, so muting changes what the
// visitor hears and nothing the effect sees. `setAudible(true)` fades the speakers in.
//
// THE AUTOPLAY POLICY. A browser keeps a new AudioContext suspended until the page has had a user
// gesture, and warns in the console if one is created before. So no context exists until `resume()`
// is called, which an effect does from `pointerdown` / `pointerup` on its canvas: the first touch of
// the canvas creates the context and starts it. Until then (`state` is 'waiting') and in any browser
// without Web Audio (`'unsupported'`), `update` returns silence: all zeros, no onsets, never
// throwing, so an effect built on it is simply calm. A context that stays suspended (the gesture was
// not one the browser counts, such as a synthetic event) keeps reporting 'waiting', and the next real
// gesture resumes it. `useFile` is called from a drop, which is a gesture too.
//
// WITHOUT WEB AUDIO (tests, workers, old browsers) every method is inert.

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const clamp01 = (x: number) => (Number.isFinite(x) ? clamp(x, 0, 1) : 0);

// --- Pure analysis ---------------------------------------------------------------------------------

/** `count + 1` band edges in Hz, log-spaced from `lowHz` to `highHz` (equal ratios between neighbours). */
export function logBandEdges(count: number, lowHz = 40, highHz = 12000): number[] {
  const n = Math.max(1, Math.round(count));
  const low = Math.max(1, lowHz);
  const ratio = Math.pow(Math.max(highHz, low * 1.01) / low, 1 / n);
  return Array.from({ length: n + 1 }, (_, i) => low * Math.pow(ratio, i));
}

/**
 * FFT bin ranges (inclusive) for each band: bin `k` is centred on `k * sampleRate / fftSize` Hz.
 * Every band owns at least one bin, bin 0 (DC) is never used, and neighbouring bands do not share
 * bins unless the band is narrower than one bin (the lowest bands at a small FFT size).
 */
export function bandBins(
  edges: readonly number[],
  sampleRate: number,
  fftSize: number,
): Array<[number, number]> {
  const binHz = sampleRate / fftSize;
  const last = fftSize / 2 - 1;
  const ranges: Array<[number, number]> = [];
  for (let i = 0; i + 1 < edges.length; i++) {
    const lo = clamp(Math.round(edges[i]! / binHz), 1, last);
    const hi = clamp(Math.round(edges[i + 1]! / binHz) - 1, lo, last);
    ranges.push([lo, hi]);
  }
  return ranges;
}

/** Mean of `spectrum[lo..hi]` per band, times `scale` (1 / 255 for byte data), so 0..1. */
export function splitBands(
  spectrum: ArrayLike<number>,
  bins: readonly (readonly [number, number])[],
  scale = 1 / 255,
): number[] {
  return bins.map(([lo, hi]) => {
    let sum = 0;
    let count = 0;
    for (let k = lo; k <= hi && k < spectrum.length; k++) {
      sum += spectrum[k] ?? 0;
      count++;
    }
    return count > 0 ? (sum / count) * scale : 0;
  });
}

/**
 * Exponential smoothing with separate attack (rising) and release (falling) time constants in
 * seconds, independent of the frame rate. A time constant of 0 follows instantly.
 */
export function smoothLevel(
  previous: number,
  next: number,
  dt: number,
  attackSeconds: number,
  releaseSeconds: number,
): number {
  const tau = next > previous ? attackSeconds : releaseSeconds;
  if (!(tau > 0)) return next;
  if (!(dt > 0)) return previous;
  return previous + (next - previous) * (1 - Math.exp(-dt / tau));
}

export interface BandGroups {
  bass: number;
  mids: number;
  highs: number;
}

/** Mean level of the lowest, middle and highest third of the bands. One or two bands share. */
export function groupLevels(bands: readonly number[]): BandGroups {
  const n = bands.length;
  if (n === 0) return { bass: 0, mids: 0, highs: 0 };
  const mean = (from: number, to: number) => {
    const lo = clamp(Math.floor(from), 0, n - 1);
    const hi = clamp(Math.max(Math.ceil(to), lo + 1), lo + 1, n);
    let sum = 0;
    for (let i = lo; i < hi; i++) sum += bands[i] ?? 0;
    return sum / (hi - lo);
  };
  return { bass: mean(0, n / 3), mids: mean(n / 3, (2 * n) / 3), highs: mean((2 * n) / 3, n) };
}

export interface AutoGainOptions {
  /** Seconds for a band's remembered peak to fall to 1/e of itself. Default 8. */
  releaseSeconds?: number;
  /** The peak never falls below this (in the input's units, 0..1), so silence is not amplified. Default 0.12. */
  floor?: number;
}

export interface AutoGain {
  /** Divides each band by its own remembered peak. Returns a new array, each value in 0..1. */
  process(levels: readonly number[], dt: number): number[];
  readonly peaks: readonly number[];
  reset(): void;
}

/**
 * Per-band automatic gain: each band is divided by the loudest it has recently been, so a quiet
 * track fills the 0..1 range as a loud one does. The peak is raised at once and released slowly. The
 * floor stops a band that is only room noise from being blown up to full scale.
 */
export function createAutoGain(count: number, options: AutoGainOptions = {}): AutoGain {
  const releaseSeconds = options.releaseSeconds ?? 8;
  const floor = options.floor ?? 0.12;
  const peaks = new Array<number>(Math.max(1, Math.round(count))).fill(floor);
  return {
    peaks,
    process(levels, dt) {
      const decay = dt > 0 && releaseSeconds > 0 ? Math.exp(-dt / releaseSeconds) : 1;
      return peaks.map((_, i) => {
        const level = clamp01(levels[i] ?? 0);
        peaks[i] = Math.max(floor, level, (peaks[i] ?? floor) * decay);
        return clamp01(level / peaks[i]!);
      });
    },
    reset() {
      peaks.fill(floor);
    },
  };
}

export interface OnsetOptions {
  /** Time constant of the follower the signal is compared to, seconds. Default 0.04. */
  followSeconds?: number;
  /** Time constant of the running mean of the rise, seconds. Default 1.5. */
  meanSeconds?: number;
  /** Smallest rise above the follower that counts, in the signal's units. Default 0.08. */
  minRise?: number;
  /** A rise must also exceed this multiple of the mean rise. Default 1.6. */
  adaptive?: number;
  /** Seconds after an onset during which no other fires. Default 0.14. */
  refractorySeconds?: number;
  /** The signal must still be rising by at least this much per second, so the tail of a rise that outlived the refractory period is not a second onset. Default 0.8. */
  minSlope?: number;
}

export interface OnsetResult {
  fired: boolean;
  /** 0..1, how far the rise cleared the threshold; 0 when nothing fired. */
  strength: number;
}

export interface OnsetDetector {
  /** Feed one value per frame (an energy, 0..1) and the time since the previous one. */
  process(value: number, dt: number): OnsetResult;
  reset(): void;
}

/**
 * Fires when a signal rises sharply: the rise is the signal minus a fast follower of itself, so a
 * jump is a large rise for a few frames, a plateau is none, and a slow swell never reaches the
 * threshold. The threshold adapts to the track (a multiple of the recent mean rise) with a floor,
 * and an onset blocks further ones for a refractory period, so one drum hit is one event even though
 * its rise lasts several frames.
 */
export function createOnsetDetector(options: OnsetOptions = {}): OnsetDetector {
  const {
    followSeconds = 0.04,
    meanSeconds = 1.5,
    minRise = 0.08,
    adaptive = 1.6,
    refractorySeconds = 0.14,
    minSlope = 0.8,
  } = options;
  let follower: number | undefined;
  let previous = 0;
  let meanRise = 0;
  let sinceOnset = Infinity;
  const none: OnsetResult = { fired: false, strength: 0 };
  return {
    process(value, dt) {
      const v = clamp01(Number.isFinite(value) ? value : 0);
      const step = Math.max(0, Number.isFinite(dt) ? dt : 0);
      if (follower === undefined) {
        follower = previous = v;
        return none;
      }
      sinceOnset += step;
      const rise = Math.max(0, v - follower);
      const threshold = Math.max(minRise, adaptive * meanRise);
      const rising = step > 0 ? (v - previous) / step >= minSlope : v > previous;
      previous = v;
      const fired = rise > threshold && rising && sinceOnset >= refractorySeconds;
      follower += (v - follower) * (step > 0 ? 1 - Math.exp(-step / followSeconds) : 0);
      meanRise += (rise - meanRise) * (step > 0 ? 1 - Math.exp(-step / meanSeconds) : 0);
      if (!fired) return none;
      sinceOnset = 0;
      return { fired: true, strength: clamp01((rise - threshold) / 0.4 + 0.35) };
    },
    reset() {
      follower = undefined;
      previous = 0;
      meanRise = 0;
      sinceOnset = Infinity;
    },
  };
}

/** What a dropped or chosen file is, from its MIME type or, failing that, its extension. */
export function classifyAudioFile(file: {
  name?: string;
  type?: string;
}): 'audio' | 'video' | undefined {
  const type = (file.type ?? '').toLowerCase();
  if (type.startsWith('audio/')) return 'audio';
  if (type.startsWith('video/')) return 'video';
  const extension = /\.([a-z0-9]+)$/i.exec(file.name ?? '')?.[1]?.toLowerCase() ?? '';
  if (['mp3', 'wav', 'ogg', 'oga', 'opus', 'flac', 'm4a', 'aac', 'weba'].includes(extension)) {
    return 'audio';
  }
  if (['mp4', 'webm', 'mov', 'm4v', 'ogv'].includes(extension)) return 'video';
  return undefined;
}

// --- The built-in pattern -------------------------------------------------------------------------

export type SynthVoice = 'kick' | 'snare' | 'hat' | 'bass' | 'arp';

export interface SynthEvent {
  voice: SynthVoice;
  /** Pitch in Hz for tonal voices. */
  frequency?: number;
  /** 0..1 loudness within the voice. */
  accent: number;
  /** Length in steps (sixteenths). */
  steps: number;
}

export const SYNTH_BPM = 104;
/** Bass roots for four bars: A1, F1, C2, G1. */
const BASS_ROOTS = [55, 43.65, 65.41, 49];
/** A minor pentatonic, A3 to E5. */
const ARP_NOTES = [220, 261.63, 293.66, 329.63, 392, 440, 523.25, 659.25];

/**
 * The events on one sixteenth-note step of a four-bar loop (64 steps; `step` may count past it).
 * Pure and deterministic, so a loop sounds the same every time round and a test can check it.
 */
export function synthStep(step: number): SynthEvent[] {
  const s = ((Math.floor(step) % 64) + 64) % 64;
  const bar = Math.floor(s / 16);
  const beat = s % 16;
  const events: SynthEvent[] = [];
  if (beat === 0 || beat === 8 || (beat === 10 && bar % 2 === 1)) {
    events.push({ voice: 'kick', accent: beat === 0 ? 1 : 0.8, steps: 2 });
  }
  if (beat === 4 || beat === 12) events.push({ voice: 'snare', accent: 0.9, steps: 1 });
  if (beat % 2 === 0) {
    const offbeat = beat % 4 === 2;
    events.push({ voice: 'hat', accent: offbeat ? 1 : 0.45, steps: 1 });
  }
  const root = BASS_ROOTS[bar]!;
  if (beat === 0 || beat === 3 || beat === 8 || (beat === 11 && bar === 3)) {
    events.push({ voice: 'bass', frequency: beat === 3 ? root * 2 : root, accent: 1, steps: 3 });
  }
  if (beat % 2 === 1 || beat % 4 === 0) {
    const note = ARP_NOTES[(beat * 3 + bar * 2) % ARP_NOTES.length]!;
    events.push({ voice: 'arp', frequency: note, accent: beat % 4 === 0 ? 1 : 0.6, steps: 2 });
  }
  return events;
}

// --- The analyser ----------------------------------------------------------------------------------

export type AudioState = 'unsupported' | 'waiting' | 'running' | 'closed';
export type AudioSourceKind = 'synth' | 'file';

/** One frame of analysis. All numbers are finite and in 0..1. */
export interface AudioFrame {
  /** Smoothed, auto-gained band levels, lowest first. */
  readonly bands: readonly number[];
  readonly bass: number;
  readonly mids: number;
  readonly highs: number;
  /** Overall level: the mean of the bands. */
  readonly level: number;
  /** An onset in the low or the high end happened this frame. */
  readonly onset: boolean;
  /** Strength of the strongest onset this frame, 0 when none. */
  readonly onsetStrength: number;
  readonly lowOnset: number;
  readonly highOnset: number;
  /** The context is running and a source is playing into it. */
  readonly running: boolean;
}

export interface AudioAnalyserOptions {
  /** Number of log-spaced bands. Default 6. */
  bands?: number;
  lowHz?: number;
  highHz?: number;
  /** Seconds. Defaults 0.015 and 0.22. */
  attackSeconds?: number;
  releaseSeconds?: number;
  /** Speaker volume when audible, 0..1. Default 0.5. */
  volume?: number;
}

export interface AudioAnalyser {
  readonly state: AudioState;
  readonly source: AudioSourceKind;
  readonly audible: boolean;
  /** The frequencies the bands split at, Hz (`bands + 1` values). */
  readonly edges: readonly number[];
  /** Name of the dropped file while one is playing. */
  readonly fileName: string | undefined;
  /**
   * Creates and starts the audio context. Call from a user gesture (`pointerdown`, `pointerup`,
   * `keydown`, a drop). Cheap and safe to call on every gesture; does nothing once running.
   */
  resume(): Promise<void>;
  /** Whether the analysis is wanted at all: false stops the synth and pauses a file. */
  setActive(active: boolean): void;
  /** Fades the speakers in or out. The analysis is unaffected. */
  setAudible(audible: boolean): void;
  /** Plays the built-in pattern (the default), dropping a file if one was playing. */
  useSynth(): void;
  /** Plays this audio or video file instead of the pattern. Rejects if the browser cannot decode it. */
  useFile(file: File): Promise<void>;
  /** Reads the analyser. Call once per frame. */
  update(deltaSeconds: number): AudioFrame;
  destroy(): void;
}

const SILENCE = (count: number): AudioFrame =>
  Object.freeze({
    bands: Object.freeze(new Array<number>(count).fill(0)),
    bass: 0,
    mids: 0,
    highs: 0,
    level: 0,
    onset: false,
    onsetStrength: 0,
    lowOnset: 0,
    highOnset: 0,
    running: false,
  });

type ContextConstructor = new (options?: AudioContextOptions) => AudioContext;

function audioContextConstructor(): ContextConstructor | undefined {
  const scope = globalThis as unknown as {
    AudioContext?: ContextConstructor;
    webkitAudioContext?: ContextConstructor;
  };
  return scope.AudioContext ?? scope.webkitAudioContext;
}

const FFT_SIZE = 2048;
/** How far ahead of the audio clock the pattern is scheduled, seconds. */
const LOOKAHEAD = 0.3;

export function createAudioAnalyser(options: AudioAnalyserOptions = {}): AudioAnalyser {
  const bandCount = Math.max(2, Math.round(options.bands ?? 6));
  const edges = logBandEdges(bandCount, options.lowHz, options.highHz);
  const attack = options.attackSeconds ?? 0.015;
  const release = options.releaseSeconds ?? 0.22;
  const volume = clamp01(options.volume ?? 0.5);
  const Context = audioContextConstructor();
  const silence = SILENCE(bandCount);

  let destroyed = false;
  let audible = false;
  let active = true;
  let source: AudioSourceKind = 'synth';
  let fileName: string | undefined;

  let context: AudioContext | undefined;
  let bus: GainNode | undefined;
  let analyser: AnalyserNode | undefined;
  let speakers: GainNode | undefined;
  let noise: AudioBuffer | undefined;
  let bytes: Uint8Array<ArrayBuffer> | undefined;
  let bins: Array<[number, number]> = [];

  let element: HTMLMediaElement | undefined;
  let elementSource: MediaElementAudioSourceNode | undefined;
  let objectUrl: string | undefined;

  let nextTime = 0;
  let step = 0;

  const smoothed = new Array<number>(bandCount).fill(0);
  const gain = createAutoGain(bandCount);
  const lowOnsets = createOnsetDetector();
  const highOnsets = createOnsetDetector();

  function setSpeakerGain(on: boolean) {
    if (!context || !speakers) return;
    speakers.gain.setTargetAtTime(on ? volume : 0, context.currentTime, 0.05);
  }

  function ensureContext(): AudioContext | undefined {
    if (destroyed || !Context) return undefined;
    if (context) return context;
    try {
      context = new Context({ latencyHint: 'interactive' });
    } catch {
      return undefined;
    }
    bus = context.createGain();
    analyser = context.createAnalyser();
    analyser.fftSize = FFT_SIZE;
    analyser.smoothingTimeConstant = 0.4;
    analyser.minDecibels = -90;
    analyser.maxDecibels = -20;
    speakers = context.createGain();
    speakers.gain.value = audible ? volume : 0;
    bus.connect(analyser);
    analyser.connect(speakers);
    speakers.connect(context.destination);
    bytes = new Uint8Array(analyser.frequencyBinCount);
    bins = bandBins(edges, context.sampleRate, FFT_SIZE);
    return context;
  }

  // --- The synth -------------------------------------------------------------------------------------

  function noiseBuffer(c: AudioContext): AudioBuffer {
    if (noise) return noise;
    noise = c.createBuffer(1, c.sampleRate, c.sampleRate);
    const data = noise.getChannelData(0);
    // A fixed seed: the same hiss every run.
    let seed = 0x2545f491;
    for (let i = 0; i < data.length; i++) {
      seed ^= seed << 13;
      seed ^= seed >>> 17;
      seed ^= seed << 5;
      data[i] = ((seed >>> 0) / 4294967296) * 2 - 1;
    }
    return noise;
  }

  function envelope(c: AudioContext, at: number, peak: number, decay: number, attackTime = 0.003) {
    const g = c.createGain();
    g.gain.setValueAtTime(0.0001, at);
    g.gain.linearRampToValueAtTime(peak, at + attackTime);
    g.gain.exponentialRampToValueAtTime(0.0001, at + attackTime + decay);
    g.connect(bus!);
    return g;
  }

  function tone(
    c: AudioContext,
    type: OscillatorType,
    frequency: number,
    at: number,
    length: number,
    out: AudioNode,
  ) {
    const osc = c.createOscillator();
    osc.type = type;
    osc.frequency.setValueAtTime(frequency, at);
    osc.connect(out);
    osc.start(at);
    osc.stop(at + length);
    return osc;
  }

  function burst(c: AudioContext, at: number, length: number, filter: BiquadFilterNode) {
    const hiss = c.createBufferSource();
    hiss.buffer = noiseBuffer(c);
    hiss.connect(filter);
    hiss.start(at);
    hiss.stop(at + length);
  }

  function play(c: AudioContext, event: SynthEvent, at: number, stepSeconds: number) {
    const length = event.steps * stepSeconds;
    switch (event.voice) {
      case 'kick': {
        const g = envelope(c, at, 0.95 * event.accent, 0.34);
        const osc = tone(c, 'sine', 150, at, 0.42, g);
        osc.frequency.exponentialRampToValueAtTime(42, at + 0.14);
        break;
      }
      case 'snare': {
        const filter = c.createBiquadFilter();
        filter.type = 'highpass';
        filter.frequency.value = 1400;
        filter.connect(envelope(c, at, 0.34 * event.accent, 0.16));
        burst(c, at, 0.25, filter);
        tone(c, 'triangle', 190, at, 0.12, envelope(c, at, 0.18, 0.09));
        break;
      }
      case 'hat': {
        const filter = c.createBiquadFilter();
        filter.type = 'highpass';
        filter.frequency.value = 7200;
        filter.connect(envelope(c, at, 0.1 * event.accent + 0.025, 0.05, 0.001));
        burst(c, at, 0.1, filter);
        break;
      }
      case 'bass': {
        const filter = c.createBiquadFilter();
        filter.type = 'lowpass';
        filter.frequency.value = 380;
        filter.connect(envelope(c, at, 0.34, length * 1.1, 0.012));
        tone(c, 'sawtooth', event.frequency ?? 55, at, length * 1.3, filter);
        break;
      }
      case 'arp': {
        const filter = c.createBiquadFilter();
        filter.type = 'lowpass';
        filter.frequency.value = 3600;
        filter.connect(envelope(c, at, 0.1 * event.accent + 0.03, length * 1.4, 0.004));
        tone(c, 'triangle', event.frequency ?? 440, at, length * 2, filter);
        break;
      }
    }
  }

  function schedule(c: AudioContext) {
    const stepSeconds = 60 / SYNTH_BPM / 4;
    // First start, or the page stopped drawing for a while: begin again just ahead of the clock
    // instead of playing the backlog in a burst.
    if (nextTime < c.currentTime - 0.1) nextTime = c.currentTime + 0.05;
    let guard = 0;
    while (nextTime < c.currentTime + LOOKAHEAD && guard++ < 16) {
      for (const event of synthStep(step)) play(c, event, nextTime, stepSeconds);
      nextTime += stepSeconds;
      step++;
    }
  }

  // --- The file ----------------------------------------------------------------------------------------

  function dropFile() {
    if (element) {
      element.pause();
      element.removeAttribute('src');
      element.load();
    }
    try {
      elementSource?.disconnect();
    } catch {
      // Already disconnected.
    }
    elementSource = undefined;
    element = undefined;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = undefined;
    fileName = undefined;
  }

  const analysis = (partial: Partial<AudioFrame> & { bands: readonly number[] }): AudioFrame => {
    const groups = groupLevels(partial.bands);
    return {
      ...groups,
      level: partial.bands.reduce((a, b) => a + b, 0) / Math.max(1, partial.bands.length),
      onset: false,
      onsetStrength: 0,
      lowOnset: 0,
      highOnset: 0,
      running: true,
      ...partial,
    };
  };

  return {
    get state() {
      if (destroyed) return 'closed';
      if (!Context) return 'unsupported';
      return context?.state === 'running' ? 'running' : 'waiting';
    },
    get source() {
      return source;
    },
    get audible() {
      return audible;
    },
    get fileName() {
      return fileName;
    },
    edges,

    async resume() {
      const c = ensureContext();
      if (!c || c.state === 'running') return;
      try {
        await c.resume();
      } catch {
        // Not allowed yet: the next gesture tries again.
      }
    },

    setActive(on) {
      active = on;
      if (!on) {
        element?.pause();
        return;
      }
      if (element && context?.state === 'running') void element.play().catch(() => undefined);
    },

    setAudible(on) {
      audible = on;
      setSpeakerGain(on);
    },

    useSynth() {
      source = 'synth';
      dropFile();
      nextTime = 0;
    },

    async useFile(file) {
      if (destroyed || !Context || typeof document === 'undefined') {
        throw new Error('Web Audio is not available here');
      }
      const kind = classifyAudioFile(file);
      if (!kind) throw new Error(`${file.name || 'That file'} is not an audio or video file`);
      const c = ensureContext();
      if (!c) throw new Error('Web Audio is not available here');
      const media = document.createElement(kind === 'video' ? 'video' : 'audio');
      media.loop = true;
      media.preload = 'auto';
      media.setAttribute('playsinline', '');
      const url = URL.createObjectURL(file);
      media.src = url;
      try {
        await new Promise<void>((resolve, reject) => {
          media.addEventListener('canplay', () => resolve(), { once: true });
          media.addEventListener(
            'error',
            () => reject(new Error('The file could not be decoded')),
            {
              once: true,
            },
          );
        });
        if (destroyed) throw new Error('Closed');
        const node = c.createMediaElementSource(media);
        node.connect(bus!);
        dropFile();
        element = media;
        elementSource = node;
        objectUrl = url;
        fileName = file.name;
        source = 'file';
        if (c.state !== 'running') await c.resume().catch(() => undefined);
        if (active) await media.play().catch(() => undefined);
      } catch (error) {
        URL.revokeObjectURL(url);
        throw error;
      }
    },

    update(deltaSeconds) {
      const dt = Math.max(0, Number.isFinite(deltaSeconds) ? deltaSeconds : 0);
      if (!context || !analyser || !bytes || context.state !== 'running' || !active || destroyed) {
        // Let the levels fall away rather than snapping, in case the source just stopped.
        for (let i = 0; i < smoothed.length; i++) {
          smoothed[i] = smoothLevel(smoothed[i]!, 0, dt, attack, release);
        }
        return smoothed.some((v) => v > 0.002)
          ? analysis({ bands: [...smoothed], running: false })
          : silence;
      }
      if (source === 'synth') schedule(context);
      analyser.getByteFrequencyData(bytes);
      const raw = splitBands(bytes, bins);
      for (let i = 0; i < smoothed.length; i++) {
        smoothed[i] = smoothLevel(smoothed[i]!, raw[i] ?? 0, dt, attack, release);
      }
      const bands = gain.process(smoothed, dt);
      const groups = groupLevels(bands);
      const low = lowOnsets.process(groups.bass, dt);
      const high = highOnsets.process(groups.highs, dt);
      return analysis({
        bands,
        onset: low.fired || high.fired,
        onsetStrength: Math.max(low.strength, high.strength),
        lowOnset: low.strength,
        highOnset: high.strength,
      });
    },

    destroy() {
      if (destroyed) return;
      destroyed = true;
      dropFile();
      const c = context;
      context = undefined;
      bus = analyser = speakers = undefined;
      if (c && c.state !== 'closed') void c.close().catch(() => undefined);
    },
  };
}
