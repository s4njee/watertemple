// sss.wgsl -- a cheap subsurface-scattering look for wave crests: the glow you see in a wave that
// is lit from behind, when the sun shines through its thin, tall top toward your eye.
//
// Approach: the "fast translucency" model of Barre-Brisebois & Bouchard (GDC 2011): take the
// direction the light travels, bend it by the surface normal, and raise how directly it points at
// the viewer to a power. The bend is outward along the normal, standing in for light leaving the
// water through the front of the wave and being refracted up toward the eye. The result is weighted
// by how crest-like the point is (thin water scatters more) and by how much the surface faces away
// from the light, the height-driven weighting common in open-source ocean shaders. No thickness
// map, no extra samples, no ray work.
//
// Conventions
//  * All vectors are UNIT, world space, pointing AWAY from the surface: `view` toward the eye,
//    `light` toward the sun, `normal` the (wave) normal.
//  * `crest` is a 0..1 factor: 0 for flat or low water, 1 for the top of a tall wave. Take it from
//    waterplane_crest(height, strength, scale) or any height/curvature measure you have.
//  * The result is a scatter intensity, not a colour: multiply by the light colour and a scatter
//    colour (a saturated turquoise or green for clean water). It is view dependent, so it replaces
//    nothing: ADD it to the reflected and body colour, and weight by (1 - Fresnel) at the call
//    site, since light scattered out of the water competes with the surface reflection.
//
//   sss_crest(view, light, normal, crest, distortion, power) -> f32
//        distortion  [0, 1] (clamped): how far the surface normal bends the transmitted light outward.
//                    0 sends the light straight on along its path, which only reaches an eye below
//                    the surface; larger values lift it into the view of an eye above the water, as
//                    refraction at the exit face does. Try 0.4 to 0.8. Above 1 a viewer on the sunny
//                    side could see it, so it is clamped
//        power       [1, 16]: concentration of the view cone. Low is a wide soft glow, high is a tight
//                    bloom around the direction of the sun. Try 3 to 6
//        Returns [0, 1]. Exactly 0 when the viewer is on the sunny side of the surface (looking
//        away from the light), when the sun is below the horizon, and never negative
//   sss_crest_color(view, light, normal, crest, scatter_color, light_color, distortion, power) -> vec3f
//        sss_crest(..) x scatter_color x light_color, ready to add to the shaded colour
//
// Cost: ~35 ALU, one normalize, one pow. No textures.

fn sss_crest(
  view: vec3f, light: vec3f, normal: vec3f, crest: f32, distortion: f32, power: f32,
) -> f32 {
  // The light ray as it leaves the wave: the incoming direction -light, bent outward by the normal.
  let leaving = normal * clamp(distortion, 0.0, 1.0) - light;
  let through = dot(view, leaving) / max(length(leaving), 1e-3);
  let forward = pow(saturate(through), max(power, 1.0));
  // Thin, tall water scatters most; keep a little everywhere so a calm surface is not dead.
  let thickness = 0.06 + 0.94 * saturate(crest);
  // Faces turned away from the sun are the thin back side of a wave as seen from the sun's far side.
  let facing_away = 0.35 + 0.65 * saturate(0.5 - 0.5 * dot(normal, light));
  let sun_up = smoothstep(-0.02, 0.1, light.y);
  return forward * thickness * facing_away * sun_up;
}

fn sss_crest_color(
  view: vec3f, light: vec3f, normal: vec3f, crest: f32,
  scatter_color: vec3f, light_color: vec3f, distortion: f32, power: f32,
) -> vec3f {
  return scatter_color * light_color * sss_crest(view, light, normal, crest, distortion, power);
}
