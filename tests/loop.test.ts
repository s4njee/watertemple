import { describe, expect, it, vi } from 'vitest';
import { createFixedStepper, createFrameLoop, type FrameLoopOptions } from '../src/gpu/loop.ts';
import type { FrameTime } from '../src/gpu/types.ts';

/** A manual clock and requestAnimationFrame stand-in. */
function createFakeFrames() {
  let clock = 1000;
  let nextHandle = 1;
  const pending = new Map<number, () => void>();
  const cancelled: number[] = [];
  const options = {
    now: () => clock,
    requestFrame: (callback: () => void) => {
      const handle = nextHandle++;
      pending.set(handle, callback);
      return handle;
    },
    cancelFrame: (handle: number) => {
      cancelled.push(handle);
      pending.delete(handle);
    },
  } satisfies FrameLoopOptions;
  return {
    options,
    cancelled,
    get pending() {
      return pending.size;
    },
    advanceClock(ms: number) {
      clock += ms;
    },
    /** Advance the clock, then run the callbacks that were pending when the frame began. */
    tick(ms: number) {
      clock += ms;
      const due = [...pending.entries()];
      pending.clear();
      for (const [, callback] of due) callback();
    },
  };
}

function collect(overrides: FrameLoopOptions = {}) {
  const frames = createFakeFrames();
  const seen: FrameTime[] = [];
  const loop = createFrameLoop((time) => seen.push(time), { ...frames.options, ...overrides });
  return { frames, seen, loop };
}

describe('createFrameLoop', () => {
  it('does nothing until started and schedules exactly one frame when started twice', () => {
    const { frames, seen, loop } = collect();
    expect(loop.running).toBe(false);
    expect(frames.pending).toBe(0);

    loop.start();
    loop.start();
    expect(loop.running).toBe(true);
    expect(frames.pending).toBe(1);
    expect(seen).toHaveLength(0);
  });

  it('reports time, delta and a zero-based frame counter in seconds', () => {
    const { frames, seen, loop } = collect();
    loop.start();
    frames.tick(16);
    frames.tick(20);
    frames.tick(10);

    expect(seen.map((t) => t.frame)).toEqual([0, 1, 2]);
    expect(seen[0]?.delta).toBeCloseTo(0.016);
    expect(seen[1]?.delta).toBeCloseTo(0.02);
    expect(seen[2]?.delta).toBeCloseTo(0.01);
    expect(seen[2]?.time).toBeCloseTo(0.046);
  });

  it('clamps delta, and time follows the clamped delta', () => {
    const { frames, seen, loop } = collect();
    loop.start();
    frames.tick(16);
    frames.tick(60_000); // a backgrounded tab coming back
    frames.tick(16);

    expect(seen[1]?.delta).toBe(0.1);
    expect(seen[1]?.time).toBeCloseTo(0.116);
    expect(seen[2]?.delta).toBeCloseTo(0.016);
    expect(seen[2]?.time).toBeCloseTo(0.132);
  });

  it('honours a custom maxDelta', () => {
    const { frames, seen, loop } = collect({ maxDelta: 0.05 });
    loop.start();
    frames.tick(500);
    expect(seen[0]?.delta).toBe(0.05);
  });

  it('never reports a negative delta if the clock steps backwards', () => {
    const { frames, seen, loop } = collect();
    loop.start();
    frames.tick(-50);
    expect(seen[0]?.delta).toBe(0);
  });

  it('excludes time spent stopped and keeps counting frames across a restart', () => {
    const { frames, seen, loop } = collect();
    loop.start();
    frames.tick(16);
    loop.stop();
    expect(loop.running).toBe(false);
    expect(frames.pending).toBe(0);

    frames.advanceClock(10_000);
    loop.start();
    frames.tick(16);

    expect(seen[1]?.delta).toBeCloseTo(0.016);
    expect(seen[1]?.time).toBeCloseTo(0.032);
    expect(seen[1]?.frame).toBe(1);
  });

  it('cancels the pending frame on stop and ignores a second stop', () => {
    const { frames, seen, loop } = collect();
    loop.start();
    loop.stop();
    loop.stop();
    expect(frames.cancelled).toHaveLength(1);
    frames.tick(16);
    expect(seen).toHaveLength(0);
  });

  it('can be stopped from inside the frame callback', () => {
    const frames = createFakeFrames();
    const loop = createFrameLoop(() => loop.stop(), frames.options);
    loop.start();
    frames.tick(16);
    expect(loop.running).toBe(false);
    expect(frames.pending).toBe(0);
  });

  it('does not double-schedule when the callback stops and restarts the loop', () => {
    const frames = createFakeFrames();
    const loop = createFrameLoop(() => {
      loop.stop();
      loop.start();
    }, frames.options);
    loop.start();
    frames.tick(16);
    expect(frames.pending).toBe(1);
  });

  it('stops and reports once when the callback throws', () => {
    const frames = createFakeFrames();
    const onError = vi.fn();
    const failure = new Error('bad shader');
    let calls = 0;
    const loop = createFrameLoop(
      () => {
        if (++calls === 2) throw failure;
      },
      { ...frames.options, onError },
    );
    loop.start();
    frames.tick(16);
    frames.tick(16);

    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenCalledWith(failure);
    expect(loop.running).toBe(false);
    expect(frames.pending).toBe(0);

    frames.tick(16);
    expect(calls).toBe(2);
  });

  it('can be started again after an error stopped it', () => {
    const frames = createFakeFrames();
    let shouldThrow = true;
    const onError = vi.fn();
    const loop = createFrameLoop(
      () => {
        if (shouldThrow) throw new Error('once');
      },
      { ...frames.options, onError },
    );
    loop.start();
    frames.tick(16);
    shouldThrow = false;
    loop.start();
    frames.tick(16);
    expect(loop.running).toBe(true);
    expect(frames.pending).toBe(1);
  });

  it('logs to console.error by default when the callback throws', () => {
    const frames = createFakeFrames();
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});
    try {
      const loop = createFrameLoop(() => {
        throw new Error('boom');
      }, frames.options);
      loop.start();
      frames.tick(16);
      expect(spy).toHaveBeenCalledTimes(1);
    } finally {
      spy.mockRestore();
    }
  });
});

describe('createFixedStepper', () => {
  it('accumulates partial steps and reports the remainder as alpha', () => {
    const stepper = createFixedStepper(0.25);
    expect(stepper.advance(0.1)).toEqual({ steps: 0, alpha: 0.4 });

    const second = stepper.advance(0.2); // 0.3 accumulated
    expect(second.steps).toBe(1);
    expect(second.alpha).toBeCloseTo(0.2);

    const third = stepper.advance(1); // 1.05 accumulated
    expect(third.steps).toBe(4);
    expect(third.alpha).toBeCloseTo(0.2);
  });

  it('runs exactly one step per frame when deltas equal the step', () => {
    const stepper = createFixedStepper(1 / 60);
    let steps = 0;
    for (let i = 0; i < 60; i++) steps += stepper.advance(1 / 60).steps;
    expect(steps).toBe(60);
  });

  it('caps steps per frame and drops the backlog beyond the cap', () => {
    const stepper = createFixedStepper(0.25, { maxSteps: 3 });
    const stalled = stepper.advance(10.1); // 40 whole steps and 0.1 s over
    expect(stalled.steps).toBe(3);
    expect(stalled.alpha).toBeCloseTo(0.4);

    // Nothing is owed from the stall: the next frame starts from the 0.1 s remainder.
    const next = stepper.advance(0.15);
    expect(next.steps).toBe(1);
    expect(next.alpha).toBeCloseTo(0);
  });

  it('treats a negative delta as zero', () => {
    const stepper = createFixedStepper(0.25);
    expect(stepper.advance(-1)).toEqual({ steps: 0, alpha: 0 });
  });

  it('keeps alpha within [0, 1)', () => {
    const stepper = createFixedStepper(1 / 120);
    for (let i = 0; i < 1000; i++) {
      const { alpha } = stepper.advance(0.001 + (i % 7) * 0.003);
      expect(alpha).toBeGreaterThanOrEqual(0);
      expect(alpha).toBeLessThan(1);
    }
  });

  it('rejects steps that would loop forever or never run', () => {
    expect(() => createFixedStepper(0)).toThrow(RangeError);
    expect(() => createFixedStepper(-1)).toThrow(RangeError);
    expect(() => createFixedStepper(Number.NaN)).toThrow(RangeError);
    expect(() => createFixedStepper(Infinity)).toThrow(RangeError);
    expect(() => createFixedStepper(0.1, { maxSteps: 0 })).toThrow(RangeError);
  });

  it('feeds from the frame loop delta end to end', () => {
    const frames = createFakeFrames();
    const stepper = createFixedStepper(1 / 60);
    let steps = 0;
    const loop = createFrameLoop((t) => {
      steps += stepper.advance(t.delta).steps;
    }, frames.options);
    loop.start();
    for (let i = 0; i < 30; i++) frames.tick(1000 / 30); // 30 fps display, 60 Hz simulation
    expect(steps).toBe(60);
  });
});
