// Low-resolution pass: the density of three mist layers, one per channel (r = far, g = middle,
// b = near), with the vertical profile already applied. Alpha unused. rgba16float.
#include "fullscreen.wgsl"
#include "mist.wgsl"
#include "layers_params.wgsl"

@group(0) @binding(0) var<uniform> mist: LayerParams;

// Distance from the source edge as a fraction of the canvas: 0 at the edge, 1 on the far side.
fn edge_distance(uv: vec2f, edge: u32) -> f32 {
  switch edge {
    case 0u: { return 1.0 - uv.y; }  // bottom
    case 1u: { return uv.y; }        // top
    case 2u: { return uv.x; }        // left
    case 3u: { return 1.0 - uv.x; }  // right
    default: { return 0.0; }         // none: the same everywhere
  }
}

fn layer(uv: vec2f, i: i32) -> f32 {
  let depth = f32(i) / 2.0;                       // 0 far, 1 near
  let z = mix(0.3, 1.0, depth);                   // parallax weight
  let scale = mist.scale * mix(0.75, 1.5, depth); // far wisps are bigger on screen
  let travel = mix(0.45, 1.15, depth);            // and drift slower
  let seed = vec2f(f32(i) * 17.3, f32(i) * 5.9);

  // Features per length of the canvas's short side; `stretch` > 1 elongates them along x (fog), < 1 along y (steam).
  let p = uv * mist.extent * vec2f(1.0 / mist.stretch, 1.0) * scale + seed;
  let drift = mist.drift * mist.phase * travel * scale;
  // Scrolling the page moves the pattern up, nearer layers further (parallax).
  let scroll = vec2f(0.0, mist.scroll * z * scale);
  let shift = scroll - drift;
  let density = mist_layer(p, shift, shift * 0.6, mist.time + f32(i) * 3.1, mist.warp, mist.coverage);

  let height = mist.height * mix(1.15, 0.85, depth);
  let s = edge_distance(uv, mist.source_edge);
  return density * select(mist_profile(s, height, mist.falloff), 1.0, mist.source_edge > 3u);
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  return vec4f(layer(in.uv, 0), layer(in.uv, 1), layer(in.uv, 2), 0.0);
}
