import type { FrameTime } from './types.ts';

export interface FrameLoopOptions {
  /** Longest delta reported, in seconds. Default 0.1. */
  maxDelta?: number;
  /** Milliseconds. Default performance.now. */
  now?: () => number;
  requestFrame?: (callback: () => void) => number;
  cancelFrame?: (handle: number) => void;
  /** Receives the error that stopped the loop. Default console.error. */
  onError?: (error: unknown) => void;
}

export interface FrameLoop {
  start(): void;
  stop(): void;
  readonly running: boolean;
}

export function createFrameLoop(
  onFrame: (time: FrameTime) => void,
  options: FrameLoopOptions = {},
): FrameLoop {
  const {
    maxDelta = 0.1,
    now = () => performance.now(),
    requestFrame = (callback) => requestAnimationFrame(callback),
    cancelFrame = (handle) => cancelAnimationFrame(handle),
    onError = (error) => console.error('[watertemple] frame loop stopped:', error),
  } = options;

  let running = false;
  let handle: number | undefined;
  let last = 0;
  // Both advance by the clamped delta, so a stall is skipped rather than replayed, and
  // time always equals the sum of the deltas the callback has seen.
  let time = 0;
  let frame = 0;

  function start() {
    if (running) return;
    running = true;
    // Re-basing here is what keeps time spent stopped out of the clock.
    last = now();
    handle = requestFrame(tick);
  }

  function stop() {
    running = false;
    if (handle !== undefined) cancelFrame(handle);
    handle = undefined;
  }

  function tick() {
    handle = undefined;
    const current = now();
    const delta = Math.min(Math.max(0, (current - last) / 1000), maxDelta);
    last = current;
    time += delta;

    try {
      onFrame({ time, delta, frame: frame++ });
    } catch (error) {
      stop();
      onError(error);
      return;
    }
    // onFrame may have stopped the loop, or stopped and restarted it (which already scheduled).
    if (running && handle === undefined) handle = requestFrame(tick);
  }

  return {
    start,
    stop,
    get running() {
      return running;
    },
  };
}

export interface FixedStepperOptions {
  /** Most steps one advance() reports. Default 8. */
  maxSteps?: number;
}

export interface FixedStep {
  /** Whole simulation steps to run this frame. */
  steps: number;
  /** Fraction of a step left over, 0..1: how far the render state is past the last step. */
  alpha: number;
}

export interface FixedStepper {
  advance(delta: number): FixedStep;
}

export function createFixedStepper(
  stepSeconds: number,
  options: FixedStepperOptions = {},
): FixedStepper {
  const { maxSteps = 8 } = options;
  if (!(stepSeconds > 0) || !Number.isFinite(stepSeconds)) {
    throw new RangeError(`stepSeconds must be a positive finite number, got ${stepSeconds}`);
  }
  if (!(maxSteps >= 1)) throw new RangeError(`maxSteps must be at least 1, got ${maxSteps}`);

  let accumulator = 0;
  return {
    advance(delta) {
      accumulator += Math.max(0, delta);
      // The epsilon lets sixty deltas of 1/60 add up to sixty steps instead of fifty-nine.
      const whole = Math.floor(accumulator / stepSeconds + 1e-9);
      // Subtracting every whole step, not just those reported, drops the backlog beyond the cap
      // instead of letting a slow frame owe more steps next frame (the spiral of death).
      accumulator = Math.max(0, accumulator - whole * stepSeconds);
      return { steps: Math.min(whole, maxSteps), alpha: accumulator / stepSeconds };
    },
  };
}
