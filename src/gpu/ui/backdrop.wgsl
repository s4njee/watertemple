#include "fullscreen.wgsl"
#include "color.wgsl"
#include "params.wgsl"
@group(0) @binding(0) var<uniform> u: BackdropParams;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var atlas: texture_2d<f32>;
#include "text.wgsl"
@fragment fn fs(in: FullscreenVarying)->@location(0) vec4f  {
  let p=in.uv;
  let alternate=u.page>0.5;
  var c=mix(vec3f(0.83,0.82,0.74),vec3f(0.04,0.11,0.14),u.page);
  let ink=mix(vec3f(0.023,0.065,0.074),vec3f(0.83,0.89,0.79),u.page);
  let photo=p.x>0.48&&p.x<0.94&&p.y>0.14&&p.y<0.9;
  if(photo)  {
    let q=(p-vec2f(0.48,0.14))/vec2f(0.46,0.76);
    let coast=0.36+0.16*sin(q.y*5.0)+0.045*sin(q.y*19.0);
    let sand=smoothstep(coast-0.02,coast+0.025,q.x);
    let ripples=0.04*sin(q.x*63.0+q.y*13.0)+0.02*sin(q.y*99.0-q.x*34.0);
    let sea=mix(vec3f(0.005,0.10,0.14),vec3f(0.05,0.58,0.58),exp(-abs(q.x-coast)*4.0))+ripples;
    c=mix(sea,vec3f(0.78,0.62,0.40),sand);
    c=mix(c,vec3f(0.85,0.94,0.80),exp(-abs(q.x-coast-0.018)*180.0)*0.8);
    if(alternate)  {
      c=mix(c,vec3f(0.13,0.18,0.32),0.42);
    }
  }
  let title=word(p,vec2f(0.29,0.42),vec2f(0.078,0.26),1u);
  c=mix(c,ink,title);
  let caption=word(p,vec2f(0.24,0.12),vec2f(0.018,0.057),2u);
  c=mix(c,ink,caption);
  let line=step(abs(p.y-0.76),0.0015)*step(p.x,0.42)*step(0.08,p.x);
  c=mix(c,ink,line);
  for(var i=0u;i<3u;i++)  {
    let bar=step(abs(p.y-0.63-f32(i)*0.026),0.002)*step(p.x,0.39-f32(i)*0.04)*step(0.08,p.x);
    c=mix(c,ink,bar*0.3);
  }
  let grid=step(fract(p.x*30.0),0.035)*step(fract(p.y*20.0),0.035);
  c=mix(c,ink,grid*0.17);
  return vec4f(c,1.0);
  // Linear rgba16float backdrop; output encoding happens in the display pass.
}
