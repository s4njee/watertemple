// Shared by the front camera and the DOM demo so resizing never separates the spray and bowl.
export const LANTERN_CAMERA = { center: [0, 1.3] as const, halfHeight: 1.65 };
export const LANTERN_EMITTER_Y = 0.03;
export const LANTERN_NOZZLE_TOP =
  0.5 + (LANTERN_CAMERA.center[1] - LANTERN_EMITTER_Y) / (2 * LANTERN_CAMERA.halfHeight);
