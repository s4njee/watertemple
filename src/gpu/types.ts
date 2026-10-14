/** Timing handed to every frame callback. All values are in seconds. */
export interface FrameTime {
  /** Time since the loop started, excluding time spent paused. */
  time: number;
  /** Time since the previous frame, clamped so a long stall cannot explode a simulation. */
  delta: number;
  /** Frames rendered since the loop started. */
  frame: number;
}
