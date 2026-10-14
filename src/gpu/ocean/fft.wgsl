#include "fft-params.wgsl"
#include "fft-step.wgsl"
@group(0) @binding(0) var<uniform> p: FftParams;
@group(0) @binding(1) var<storage,read> input_data: array<vec4f>;
@group(0) @binding(2) var<storage,read_write> output_data: array<vec4f>;
@group(0) @binding(3) var<uniform> step: FftStep;
@group(0) @binding(4) var displacement: texture_storage_2d_array<rgba16float,write>;
@group(0) @binding(5) var derivatives: texture_storage_2d_array<rgba16float,write>;
@group(0) @binding(6) var previous: texture_2d_array<f32>;
fn cmul(a:vec2f,b:vec2f)->vec2f {
  return vec2f(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);
}
fn reverse_n(i:u32)->u32 {
  return reverseBits(i)>>(32u-p.bits);
}
@compute @workgroup_size(8,8) fn evolve(@builtin(global_invocation_id) id:vec3u) {
  if(id.x>=p.size || id.y>=p.size || id.z>=3u) {
    return;
  }
  let n=p.size;
  let area=n*n;
  let i=id.z*area+id.y*n+id.x;
  let f=vec2i(select(i32(id.x),i32(id.x)-i32(n),id.x>n/2u),select(i32(id.y),i32(id.y)-i32(n),id.y>n/2u));
  let k=vec2f(f)*6.2831853/p.lengths[id.z];
  let len=length(k);
  let phase=sqrt(9.81*len)*p.time;
  let e=vec2f(cos(phase),sin(phase));
  let h0=input_data[i];
  let h=cmul(h0.xy,e)+cmul(h0.zw*vec2f(1,-1),e*vec2f(1,-1));
  var dx=vec2f(0);
  var dz=vec2f(0);
  if(len>.00001) {
    dx=vec2f(h.y,-h.x)*k.x/len;
    dz=vec2f(h.y,-h.x)*k.y/len;
  }
  // Odd derivative multipliers must vanish on the self-conjugate Nyquist axes.
  if(id.x==n/2u) {
    dx=vec2f(0);
  }
  if(id.y==n/2u) {
    dz=vec2f(0);
  }
  let j=id.z*area+reverse_n(id.y)*n+reverse_n(id.x);
  output_data[j]=vec4f(h,dx);
  output_data[3u*area+j]=vec4f(dz,0,0);
}
@compute @workgroup_size(8,8) fn butterfly(@builtin(global_invocation_id) id:vec3u) {
  if(id.x>=p.size || id.y>=p.size || id.z>=6u) {
    return;
  }
  let n=p.size;
  let coord=select(id.x,id.y,step.axis==1u);
  let half=step.span/2u;
  let j=coord%half;
  let base=(coord/step.span)*step.span;
  let a=base+j;
  let b=a+half;
  let ia=id.z*n*n+select(id.y*n+a,a*n+id.x,step.axis==1u);
  let ib=id.z*n*n+select(id.y*n+b,b*n+id.x,step.axis==1u);
  let angle=6.2831853*f32(j)/f32(step.span);
  let tw=vec2f(cos(angle),sin(angle));
  let av=input_data[ia];
  let bv=input_data[ib];
  let t=vec4f(cmul(bv.xy,tw),cmul(bv.zw,tw));
  output_data[id.z*n*n+id.y*n+id.x]=av+select(t,-t,coord%step.span>=half);
}
fn field(q:vec2i,layer:u32)->vec3f {
  let n=i32(p.size);
  let xy=(q%vec2i(n)+vec2i(n))%vec2i(n);
  let i=layer*p.size*p.size+u32(xy.y)*p.size+u32(xy.x);
  let a=input_data[i];
  let b=input_data[3u*p.size*p.size+i];
  return vec3f(a.z*p.chop,a.x,b.x*p.chop)*(p.strength/f32(p.size*p.size));
}
@compute @workgroup_size(8,8) fn pack(@builtin(global_invocation_id) id:vec3u) {
  if(id.x>=p.size || id.y>=p.size || id.z>=3u) {
    return;
  }
  let q=vec2i(id.xy);
  let cell=p.lengths[id.z]/f32(p.size);
  let d=field(q,id.z);
  let dx=(field(q+vec2i(1,0),id.z)-field(q-vec2i(1,0),id.z))/(2*cell)+vec3f(1,0,0);
  let dz=(field(q+vec2i(0,1),id.z)-field(q-vec2i(0,1),id.z))/(2*cell)+vec3f(0,0,1);
  let jac=dx.x*dz.z-dx.z*dz.x;
  let normal=normalize(cross(dz,dx));
  let old=textureLoad(previous,q,i32(id.z),0).w;
  let source=(1-smoothstep(p.foam_threshold-.35,p.foam_threshold,jac));
  let foam=max(old*exp(-p.delta/p.foam_lifetime),source*(1-exp(-p.delta*18)));
  textureStore(displacement,q,i32(id.z),vec4f(d,jac));
  textureStore(derivatives,q,i32(id.z),vec4f(normal.x/max(normal.y,.05),1,normal.z/max(normal.y,.05),foam));
}
