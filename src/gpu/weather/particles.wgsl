#include "color.wgsl"
#include "weather_params.wgsl"
struct Drop { position:vec4f, detail:vec4f, kind:vec4f };
@group(0) @binding(0) var<uniform> P: WeatherParams;
@group(0) @binding(1) var<storage,read> drops:array<Drop>;
@group(0) @binding(2) var backdrop:texture_2d<f32>;
@group(0) @binding(3) var samp:sampler;
struct Varying { @builtin(position) position:vec4f, @location(0) local:vec2f, @location(1) uv:vec2f,
  @location(2) @interpolate(flat) data:vec4f, @location(3) @interpolate(flat) info:vec4f };
@vertex fn vs(@builtin(vertex_index) vertex:u32,@builtin(instance_index) instance:u32) -> Varying {
  let corners=array<vec2f,6>(vec2f(-1,-1),vec2f(1,-1),vec2f(-1,1),vec2f(-1,1),vec2f(1,-1),vec2f(1,1));
  let d=drops[instance]; let local=corners[vertex]; let radius=d.detail.x;
  let age=d.detail.y; let life=d.detail.z; let kind=d.kind.x;
  var offset=local*radius;
  if(kind<0.5) {
    let velocity=d.position.zw; let along=normalize(velocity+vec2f(0,0.001)); let across=vec2f(-along.y,along.x);
    offset=across*local.x*radius+along*local.y*(0.009+length(velocity)*0.012);
  } else if(kind>2.5) { offset=local*vec2f(radius+age*0.05,(radius+age*0.05)*0.25); }
  let p=d.position.xy+offset; let uv=p/vec2f(P.aspect,1.0);
  var out:Varying; out.position=vec4f(uv*vec2f(2,-2)+vec2f(-1,1),0,1);out.local=local;out.uv=uv;
  out.data=vec4f(radius,age/max(life,0.001),d.detail.w,kind);out.info=vec4f(d.position.xy,age,0);return out;
}
@fragment fn fs(in:Varying) -> @location(0) vec4f {
  let q=in.local; let r2=dot(q,q); let kind=in.data.w;
  var alpha=0.0;var colour=P.tint;
  if(kind<0.5) {
    alpha=(1.0-smoothstep(0.15,1.0,abs(q.x)))*(1.0-smoothstep(0.65,1.0,abs(q.y)))*(0.16+in.data.z*0.42);
  } else if(kind<1.5) {
    if(r2>1.0) { discard; }
    let z=sqrt(max(0.0,1.0-r2));
    let refracted=textureSampleLevel(backdrop,samp,in.uv+q*in.data.x*P.refraction/vec2f(P.aspect,1),0).rgb;
    colour=mix(refracted,P.tint,0.18)+pow(max(0.0,dot(normalize(vec3f(q,z)),normalize(vec3f(-0.4,-0.6,1)))),28.0)*0.7;
    alpha=(1.0-smoothstep(0.65,1.0,r2))*(1.0-smoothstep(0.5,1.0,in.data.y))*0.9;
  } else if(kind<2.5) {
    if(r2>1.0) {discard;}
    let z=sqrt(max(0.0,1.0-r2));let normal=normalize(vec3f(q,z));
    let lens=textureSampleLevel(backdrop,samp,in.uv+q*z*in.data.x*P.refraction/vec2f(P.aspect,1),0).rgb;
    let rim=pow(1.0-z,2.2);let highlight=pow(max(0.0,dot(normal,normalize(vec3f(-0.4,-0.55,1)))),48.0);
    var film=vec3f(0.75,0.93,1.0);
    if(P.mode==2u && P.background==2u) {
      // Thin-film phase in radians, using representative RGB wavelengths in nanometres.
      let thickness=P.film*(0.7+q.y*0.14+sin(in.info.z*0.3+q.x*3.0)*0.1);
      film=0.5+0.5*cos(12.5663706*1.33*thickness*max(0.08,z)/vec3f(650,510,475)+3.14159265);
    }
    colour=lens*(0.9+z*0.1)+film*rim*0.8+vec3f(highlight)*0.75;
    alpha=(1.0-smoothstep(0.96,1.0,r2))*clamp(0.38+rim*0.58,0.0,0.95);
  } else {
    let r=length(q);alpha=(1.0-smoothstep(0.025,0.15,abs(r-0.8)))*(1.0-in.data.y)*0.5;
  }
  alpha=clamp(alpha*P.strength,0.0,1.0);
  return vec4f(linear_to_srgb(max(colour,vec3f(0)))*alpha,alpha);
}
