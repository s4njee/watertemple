#include "fullscreen.wgsl"
#include "color.wgsl"
#include "hash.wgsl"
#include "noise.wgsl"
#include "weather_params.wgsl"
@group(0) @binding(0) var<uniform> P: WeatherParams;
@group(0) @binding(1) var backdrop:texture_2d<f32>;
@group(0) @binding(2) var memory:texture_2d<f32>;
@group(0) @binding(3) var samp:sampler;
fn scene(uv:vec2f,lod:f32) -> vec3f {return textureSampleLevel(backdrop,samp,clamp(uv,vec2f(0),vec2f(1)),lod).rgb;}
fn rain_height(p:vec2f) -> f32 {
  let cell_size=0.18;let grid=floor(p/cell_size);var h=0.0;
  for(var y=-1;y<=1;y++) {for(var x=-1;x<=1;x++) {
    let cell=vec2i(grid)+vec2i(x,y);let seed=ihash_2to1(cell);
    let phase=P.clock*0.65+seed*9.0;let event=i32(floor(phase));let age=fract(phase);
    let random=ihash_3to3(vec3i(cell,event));
    let center=(vec2f(cell)+0.15+random.xy*0.7)*cell_size;
    let distance=length(p-center);
    let radius=age*cell_size*0.82;
    let envelope=sin(age*3.14159265)*exp(-age*2.0);
    let ring=exp(-pow((distance-radius)/0.009,2.0))*cos((distance-radius)*320.0);
    h+=ring*envelope*step(random.z,P.density);
  }}
  return h;
}
@fragment fn fs(in:FullscreenVarying) -> @location(0) vec4f {
  let uv=in.uv;let p=uv*vec2f(P.aspect,1.0);var c=scene(uv,0);
  if(P.mode==1u) {
    let h=rain_height(p);let grad=vec2f(dpdx(h),dpdy(h))*P.resolution.y;
    let normal=normalize(vec3f(-grad*P.strength*0.012,1.0));
    c=scene(uv+normal.xy*P.refraction*0.006,0);
    c+=vec3f(0.2,0.45,0.5)*pow(max(0.0,dot(normal,normalize(vec3f(-0.4,-0.5,1)))),22.0)*0.24;
    c+=P.tint*max(h,0.0)*P.strength*0.08;
  } else if(P.mode>=4u) {
    let state=textureSampleLevel(memory,samp,uv,0);
    let noise=vnoise_2d(p*180.0);
    let fog=clamp(state.r*P.strength,0.0,1.0);
    let ice=clamp(state.g*P.strength,0.0,1.0);
    let step_uv=1.0/vec2f(textureDimensions(memory));
    let gx=textureSampleLevel(memory,samp,uv+vec2f(step_uv.x,0),0).g-textureSampleLevel(memory,samp,uv-vec2f(step_uv.x,0),0).g;
    let gy=textureSampleLevel(memory,samp,uv+vec2f(0,step_uv.y),0).g-textureSampleLevel(memory,samp,uv-vec2f(0,step_uv.y),0).g;
    let distorted=uv+vec2f(gx,gy)*P.refraction*0.012;
    c=mix(scene(distorted,0),scene(distorted,4.0)*0.55+P.tint*(0.09+noise*0.025),fog);
    c=mix(c,scene(distorted,3.0)*0.34+vec3f(0.35,0.5,0.58)*(0.3+noise*0.14),ice*0.88);
    c+=vec3f(0.4,0.65,0.8)*pow(clamp(length(vec2f(gx,gy))*4.0,0.0,1.0),2.0)*ice*0.5;
    // Actual melted water is stored in B and advected downward. Accumulated water becomes bead lenses.
    let cells=p*26.0;let grid=floor(cells);let rnd=ihash_2to2(vec2i(grid));
    let center=(grid+0.25+rnd*0.5)/26.0;let sample_uv=center/vec2f(P.aspect,1);
    let water=textureSampleLevel(memory,samp,sample_uv,0).b;
    let radius=clamp(sqrt(max(water,0.0))*0.03*P.strength,0.0,0.017);
    if(radius>0.0004) {
      let q=(p-center)/radius;let r2=dot(q,q);
      if(r2<1.0) {
        let z=sqrt(max(0.0,1.0-r2));let n=normalize(vec3f(q,z));
        c=mix(c,scene(uv+q*z*radius*P.refraction/vec2f(P.aspect,1),0),0.84);
        c+=pow(max(0.0,dot(n,normalize(vec3f(-0.5,-0.6,1)))),40.0)*vec3f(0.7);
      }
    }
  }
  let vignette=1.0-0.22*smoothstep(0.35,0.9,length(uv-0.5));
  return vec4f(linear_to_srgb(max(c,vec3f(0))*vignette),1);
}
