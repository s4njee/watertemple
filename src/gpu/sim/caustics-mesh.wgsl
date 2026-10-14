// caustics-mesh.wgsl -- the projected light mesh (the method popularised by Evan Wallace's WebGL Water),
// written from the idea: draw a regular grid of triangles over the water surface, move every vertex to
// where a refracted ray of sunlight from it lands on the floor, and let each triangle's brightness be
// the ratio of its original area to its projected area (light squeezed into a smaller area is
// brighter). Draws are added together in a floor-sized texture.
//
// No vertex buffer: the draw is indexed and the vertex index IS the grid position (column = index %
// (cells.x + 1), row = index / (cells.x + 1)), so the GPU's post-transform cache shares each vertex
// between its six triangles. The vertex stage fetches the surface (one filtered texel), refracts, and
// writes the landing point as the clip position. The fragment stage needs no area arithmetic of its
// own: `old_uv` is the vertex's position on the SURFACE, interpolated across the triangle, and the
// screen-space derivatives of it give the exact area ratio. A pixel of the floor texture covers
// 1 / (size.x * size.y) of uv area; the surface area that maps into it is |det(d old_uv / d pixel)|, so
//     intensity = |det(dpdx(old_uv), dpdy(old_uv))| * size.x * size.y.
// A flat surface translates the mesh without changing a triangle, which makes the ratio exactly 1.
// Energy is conserved by construction: a triangle of projected area A covers A / pixel pixels, each of
// intensity A_old / A, so it deposits A_old of light whatever its shape; folded triangles (the surface
// focused past a caustic) deposit it twice over, which is the extra brightness of a real caustic.
//
// Output: P.channel * intensity, added (blend one + one) onto a texture cleared to (0, 0, 0, 1).
// Three draws with channel = (1,0,0), (0,1,0), (0,0,1) and a different `mesh_ior` give a chromatic
// caustic; one draw with channel = (1,1,1) gives the grey one.
#include "caustics_core.wgsl"

struct MeshVarying {
  @builtin(position) position: vec4f,
  // Where the vertex sits on the surface, uv. Linear in screen space, so its derivatives are the
  // inverse of the surface -> floor map.
  @location(0) old_uv: vec2f,
  // Radiometry (tilt, Fresnel) times the occluder's light factor.
  @location(1) weight: f32,
}

@vertex fn caustics_mesh_vs(@builtin(vertex_index) index: u32) -> MeshVarying {
  let row = P.mesh_cells.x + 1u;
  let cell = vec2f(f32(index % row), f32(index / row));
  let uv = mix(P.mesh_lo, P.mesh_hi, cell / vec2f(P.mesh_cells));

  let s = caustics_surface(uv);
  let t = caustics_transmit(s, P.mesh_ior);
  let z = s.x * P.cell_size * P.height_scale;
  let origin = vec3f(uv.x * P.aspect, uv.y, z);
  let travel = max(z + P.depth, 0.0) / (-t.dir.z);
  let landing = origin.xy + t.dir.xy * travel;
  let landing_uv = vec2f(landing.x / P.aspect, landing.y);

  var out: MeshVarying;
  out.position = vec4f(landing_uv.x * 2.0 - 1.0, 1.0 - landing_uv.y * 2.0, 0.0, 1.0);
  out.old_uv = uv;
  out.weight = t.weight * caustics_lit(origin, t.dir, travel);
  return out;
}

@fragment fn caustics_mesh_fs(in: MeshVarying) -> @location(0) vec4f {
  let dx = dpdx(in.old_uv);
  let dy = dpdy(in.old_uv);
  let ratio = abs(dx.x * dy.y - dx.y * dy.x) * P.size.x * P.size.y;
  let intensity = min(ratio * in.weight, P.max_intensity);
  return vec4f(P.channel * intensity, 0.0);
}
