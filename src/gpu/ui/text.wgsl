// MSDF atlas is linear data: median of RGB distances reconstructs sharp glyph corners.
fn median3(v: vec3f) -> f32  {
  return max(min(v.r,v.g),min(max(v.r,v.g),v.b));
}
fn glyph(p: vec2f, index: u32) -> f32  {
  let local=clamp(p,vec2f(0.001),vec2f(0.999));
  let uv=(vec2f(f32(index%12u),f32(index/12u))+local)/vec2f(12.0,3.0);
  let sd=median3(textureSampleLevel(atlas,linear_sampler,uv,0.0).rgb)-0.5;
  let range=max(1.0,4.0/48.0/max(length(fwidth(p)),0.0001));
  let in_box=select(0.0,1.0,all(p>=vec2f(0.0))&&all(p<=vec2f(1.0)));
  return clamp(sd*range+0.5,0.0,1.0)*in_box;
}
fn word(p: vec2f, center: vec2f, cell: vec2f, kind: u32) -> f32  {
  var letters=array<u32,6>(11u,8u,16u,20u,8u,3u);
  // LIQUID
  var count=6u;
  if(kind==1u)  {
    letters=array<u32,6>(22u,0u,19u,4u,17u,0u);
    count=5u;
  }
  if(kind==2u)  {
    letters=array<u32,6>(5u,11u,14u,22u,0u,0u);
    count=4u;
  }
  if(kind==3u)  {
    letters=array<u32,6>(26u,26u,26u,0u,0u,0u);
    count=3u;
  }
  var alpha=0.0;
  for(var i=0u;i<6u;i++)  {
    if(i<count)  {
      let q=(p-center)/cell+vec2f(f32(count)*0.5-f32(i),0.5);
      alpha=max(alpha,glyph(q,letters[i]));
    }
  }
  return alpha;
}
