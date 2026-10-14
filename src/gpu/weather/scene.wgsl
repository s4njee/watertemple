// Self-contained procedural backdrops; these are authored scenes, not captures of live DOM.
#include "hash.wgsl"
fn weather_scene(uv: vec2f, aspect: f32, kind: u32) -> vec3f {
  let p=uv*vec2f(aspect,1.0);
  if(kind==1u) {
    var c=mix(vec3f(0.018,0.14,0.2),vec3f(0.002,0.012,0.04),uv.y);
    for(var i=0;i<5;i++) {
      let x=(f32(i)+0.35)*aspect/5.0;
      let ray=exp(-pow((p.x-x-uv.y*0.15)/0.065,2.0))*pow(1.0-uv.y,2.0);
      c+=vec3f(0.05,0.14,0.12)*ray;
    }
    let reeds=sin(p.x*35.0+sin(uv.y*9.0)*0.7);
    c+=vec3f(0.003,0.055,0.023)*smoothstep(0.72,1.0,uv.y)*smoothstep(0.65,0.9,reeds);
    return c;
  }
  if(kind==2u) {
    var c=vec3f(0.025,0.024,0.045)+vec3f(0.14,0.11,0.15)*uv.y;
    c+=vec3f(0.55,0.11,0.12)*exp(-dot(p-vec2f(aspect*0.2,0.25),p-vec2f(aspect*0.2,0.25))*12.0);
    c+=vec3f(0.02,0.24,0.55)*exp(-dot(p-vec2f(aspect*0.8,0.6),p-vec2f(aspect*0.8,0.6))*10.0);
    let stripe=1.0-smoothstep(0.045,0.07,abs(p.x-aspect*0.65-uv.y*0.1));
    return c+vec3f(0.3,0.35,0.45)*stripe;
  }
  if(kind==3u) {
    let grid=fract(p*vec2f(7.0,9.0));
    let tile=smoothstep(0.02,0.04,min(min(grid.x,1.0-grid.x),min(grid.y,1.0-grid.y)));
    let waves=sin(p.x*14.0+uv.y*10.0)*sin(p.x*5.0-uv.y*17.0);
    return mix(vec3f(0.008,0.014,0.02),vec3f(0.025,0.10,0.095),tile)+vec3f(0.035,0.045,0.04)*(waves*0.5+0.5);
  }
  var c=mix(vec3f(0.012,0.024,0.05),vec3f(0.035,0.065,0.085),uv.y);
  let skyline=0.42+ihash_2to1(vec2i(i32(floor(p.x*8.0)),17))*0.22;
  if(uv.y>skyline && uv.y<0.82) {
    c*=0.3;
    let cell=floor(p*vec2f(48.0,40.0)); let local=fract(p*vec2f(48.0,40.0));
    let lit=step(0.65,ihash_2to1(vec2i(cell)));
    let window=smoothstep(0.15,0.25,local.x)*(1.0-smoothstep(0.75,0.85,local.x))*smoothstep(0.2,0.3,local.y)*(1.0-smoothstep(0.7,0.8,local.y));
    c+=vec3f(0.45,0.23,0.06)*window*lit;
  }
  for(var i=0;i<8;i++) {
    let center=vec2f((f32(i)+0.4)*aspect/8.0,0.6+0.12*sin(f32(i)*2.3));
    let d=p-center;
    let glow=exp(-dot(d,d)*1800.0);
    let colour=select(vec3f(0.6,0.23,0.045),vec3f(0.02,0.22,0.45),i%3==0);
    c+=colour*glow;
    if(uv.y>0.82) {c+=colour*exp(-pow(d.x/0.024,2.0))*exp(-(uv.y-0.82)*7.0)*0.17;}
  }
  return c;
}
