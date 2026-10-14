#include "fullscreen.wgsl"
#include "hash.wgsl"
#include "noise.wgsl"
#include "weather_params.wgsl"
@group(0) @binding(0) var<uniform> P: WeatherParams;
@group(0) @binding(1) var previous:texture_2d<f32>;
@group(0) @binding(2) var samp:sampler;
fn segment_distance(p:vec2f,a:vec2f,b:vec2f) -> f32 {
  let ab=b-a;let t=clamp(dot(p-a,ab)/max(dot(ab,ab),1e-8),0.0,1.0);return length(p-a-ab*t);
}
fn frost_pattern(p:vec2f) -> vec2f {
  // Deterministic six-armed dendrites: position-dependent front delay and narrow side branches.
  let grid=floor(p/0.28);let local=fract(p/0.28)-0.5;let random=ihash_2to2(vec2i(grid));
  let q=local-(random-0.5)*0.2;let r=length(q);
  let angle=atan2(q.y,q.x)+random.x*6.2831853;
  let sector=floor(angle/1.04719755+0.5);
  let axis=sector*1.04719755-random.x*6.2831853;
  let direction=vec2f(cos(axis),sin(axis));
  let along=dot(q,direction);let across=abs(dot(q,vec2f(-direction.y,direction.x)));
  let taper=clamp(1.0-along/0.52,0.0,1.0);
  let main_arm=(1.0-smoothstep(0.008*taper,0.013*taper+0.002,across))*(1.0-smoothstep(0.38,0.5,along));
  var twigs=0.0;
  let branch_id=floor(along/0.07);
  for(var j=-1;j<=1;j++) {
    let origin=(branch_id+f32(j))*0.07+0.04;
    let reach=0.085*clamp(1.0-origin/0.52,0.0,1.0);
    let distance=segment_distance(vec2f(along,across),vec2f(origin,0),vec2f(origin+reach,reach*0.85));
    twigs=max(twigs,(1.0-smoothstep(0.004,0.009,distance))*step(0.02,origin));
  }
  let crystal=max(main_arm,twigs)*(1.0-smoothstep(0.38,0.54,r));
  let base=0.13+vnoise_2d(p*30.0)*0.16;
  let delay=r*5.0+random.y*1.7;
  return vec2f(clamp(base+crystal*0.78,0.0,1.0),delay);
}
@fragment fn fs(in:FullscreenVarying) -> @location(0) vec4f {
  let uv=in.uv;let p=uv*vec2f(P.aspect,1);let old=textureSampleLevel(previous,samp,uv,0);
  var fog=old.r;var ice=old.g;var water=old.b;
  if(P.initial!=0u) {fog=select(0.0,P.density,P.mode==4u);let initial_pattern=frost_pattern(p);ice=select(0.0,initial_pattern.x*smoothstep(initial_pattern.y,initial_pattern.y+2.0,P.cycle),P.mode==5u);water=0.0;}
  let dt=P.delta;
  if(P.mode==4u) {fog=P.density+(fog-P.density)*exp(-P.refog*dt);}
  if(P.mode==5u) {
    let pattern=frost_pattern(p);
    let coverage_target=pattern.x*smoothstep(pattern.y,pattern.y+2.0,P.cycle);
    let before=ice;
    ice=mix(ice,coverage_target,1.0-exp(-P.growth*dt*(1.0-P.heat)))*exp(-P.heat*dt*1.8);
    let melted=max(before-ice,0.0)*P.heat;
    let transported=textureSampleLevel(previous,samp,uv-vec2f(0,dt*0.08),0).b;
    water=clamp(transported*exp(-dt*0.12)+melted,0.0,1.0);
  }
  for(var i=0u;i<P.stroke_count;i++) {
    let s=P.strokes[i];let d=segment_distance(p,s.xy*vec2f(P.aspect,1),s.zw*vec2f(P.aspect,1));
    let wiped=1.0-smoothstep(P.brush*0.7,P.brush,d);
    fog*=1.0-wiped;ice*=1.0-wiped;
  }
  return vec4f(clamp(fog,0.0,1.0),clamp(ice,0.0,1.0),water,1);
}
