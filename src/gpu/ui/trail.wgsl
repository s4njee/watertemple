#include "fullscreen.wgsl"
#include "params.wgsl"
@group(0) @binding(0) var<uniform> u: UIParams;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var wet: texture_2d<f32>;
fn distance_segment(p:vec2f,a:vec2f,b:vec2f)->f32  {
  let ab=b-a;
  return length(p-a-ab*clamp(dot(p-a,ab)/max(dot(ab,ab),0.000001),0.0,1.0));
}
@fragment fn fs(in:FullscreenVarying)->@location(0) f32  {
  let p=in.uv;
  let px=1.0/vec2f(textureDimensions(wet));
  let center=textureSampleLevel(wet,linear_sampler,p,0.0).r;
  let neighbors=textureSampleLevel(wet,linear_sampler,p+vec2f(px.x,0.0),0.0).r+textureSampleLevel(wet,linear_sampler,p-vec2f(px.x,0.0),0.0).r+textureSampleLevel(wet,linear_sampler,p+vec2f(0.0,px.y),0.0).r+textureSampleLevel(wet,linear_sampler,p-vec2f(0.0,px.y),0.0).r;
  let spread=min(0.24,u.delta*u.extra*6.0);
  let old=max(0.0,mix(center,neighbors*0.25,spread))*exp(-u.delta*u.dry);
  let aspect=vec2f(u.resolution.x/u.resolution.y,1.0);
  var d=10.0;
  for(var i=0u;i<16u;i++)  {
    if(i<u.segment_count)  {
      let segment=u.segments[i];
      d=min(d,distance_segment(p*aspect,segment.xy*aspect,segment.zw*aspect));
    }
  }
  let stroke=exp(-d*d/max(0.00001,u.radius*u.radius))*u.interaction*u.strength*min(1.0,u.delta*60.0);
  return clamp(max(old,stroke),0.0,1.0);
}
