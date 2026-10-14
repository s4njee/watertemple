#include "fullscreen.wgsl"
#include "color.wgsl"
#include "sky.wgsl"
#include "specular.wgsl"
#include "fresnel.wgsl"
#include "sss.wgsl"
#include "noise.wgsl"
#include "ocean-waves.wgsl"
#include "scene-params.wgsl"
@group(0) @binding(0) var<uniform> p: OceanParams;
@group(0) @binding(1) var disp: texture_2d_array<f32>;
@group(0) @binding(2) var deriv: texture_2d_array<f32>;
@group(0) @binding(3) var linear_sampler: sampler;
@group(0) @binding(4) var planar: texture_2d<f32>;
@group(0) @binding(5) var flow_map: texture_2d<f32>;
fn sun()->vec3f {
  return normalize(vec3f(.28,p.sun,-.85));
}
fn ray(uv:vec2f, inv:mat4x4f,eye:vec3f)->vec3f {
  let q=inv*vec4f(uv.x*2-1,1-uv.y*2,.99,1);
  return normalize(q.xyz/q.w-eye);
}
fn noise2(q:vec2f)->f32 {
  return vnoise_2d(q);
}
fn channel(z:f32)->f32 {
  return sin(z*.12)*.6;
}
fn rock(i:u32)->vec4f {
  let r=array<vec4f,4>(vec4f(1.8,.15,-3,1.3),vec4f(-1.5,.05,-9,1.05),vec4f(.7,.15,-18,1.8),vec4f(-3,.1,-25,1.3));
  return r[i];
}
struct Env {
  color:vec3f, distance:f32
}
fn environment(eye:vec3f,dir:vec3f)->Env {
  var color=sky_radiance(dir,sun());
  var distance=10000.;
  let cloud=noise2(dir.xz/max(.13,dir.y+.25)*1.8+vec2f(p.time*.014,0));
  color=mix(color,vec3f(.8,.85,.9),smoothstep(.65,.92,cloud)*smoothstep(.01,.22,dir.y)*.35);
  if(p.mode==6u) {
    let mountains=.07+.12*noise2(vec2f(atan2(dir.x,-dir.z)*3.,1))+.08*noise2(vec2f(dir.x*9,4));
    let mountain=1-smoothstep(mountains-.004,mountains+.004,dir.y);
    let ridge=vec3f(.035,.10,.12)+vec3f(.05,.08,.09)*smoothstep(0.,.25,dir.y);
    color=mix(color,ridge,mountain);
    let trees=mountains*.48+.03*sin(dir.x*151)*sin(dir.x*81);
    color=mix(color,vec3f(.018,.05,.035),1-smoothstep(trees-.003,trees+.003,dir.y));
  }
  if(p.mode==2u || p.mode==3u || p.mode==4u || p.mode==5u) {
    var t=-1.;
    if(p.mode==2u) {
      t=(.15*eye.x-.5-eye.y)/(dir.y-.15*dir.x);
    }
    else if(dir.y<-.0001) {
      t=(-1.2-eye.y)/dir.y;
    }
    if(t>0 && t<800) {
      var hit=eye+dir*t;
      if(p.mode==3u) {
        let upper=(.3-eye.y)/dir.y;
        let bank=eye+dir*upper;
        if(upper>0 && abs(bank.x-channel(bank.z))>4.3) {
          t=upper;
          hit=bank;
        }
      }
      distance=t;
      let grains=noise2(hit.xz*5);
      let pattern=noise2(hit.xz*.7);
      color=mix(vec3f(.38,.28,.15),vec3f(.68,.54,.31),pattern)*(.85+.15*grains);
      if(p.mode==2u) {
        let height=.15*hit.x-.5;
        let wet=1-smoothstep(.02,.45,height+.08*sin(p.time*.6));
        color*=1-.48*wet;
        color+=sky_radiance(reflect(dir,normalize(vec3f(-.15,1,0))),sun())*.08*wet;
      }
      if(p.mode==3u || p.mode==4u) {
        color=mix(vec3f(.07,.10,.065),vec3f(.24,.25,.18),pattern)*(.65+.35*grains);
      }
      if(p.mode==5u) {
        color=vec3f(.035,.17,.18)*(1+.4*sin(hit.x*3+sin(hit.z*3+p.time))*sin(hit.z*4+p.time));
      }
    }
  }
  if(p.mode==4u && dir.z<-.0001) {
    let t=(-9.-eye.z)/dir.z;
    let hit=eye+dir*t;
    if(t>0 && t<distance && abs(hit.x)<60. && hit.y<10.5+2*noise2(hit.xz*.25) && hit.y>-.5) {
      distance=t;
      let stone=noise2(hit.xy*.65);
      let moss=noise2(hit.xy*2.3);
      color=mix(vec3f(.075,.085,.065),vec3f(.21,.24,.17),stone)*(.7+.3*moss);
      color=mix(color,vec3f(.06,.14,.075),smoothstep(.55,.8,moss)*smoothstep(1.,7.,hit.y));
    }
  }
  if(p.mode==3u || p.mode==4u) {
    for(var i=0u;i<4u;i++) {
      var r=rock(i);
      if(p.mode==4u) {
        r=vec4f((f32(i)-1.5)*3,-.1,-9,2.2);
      }
      let oc=eye-r.xyz;
      let b=dot(oc,dir);
      let h=b*b-dot(oc,oc)+r.w*r.w;
      if(h>0) {
        let t=-b-sqrt(h);
        if(t>0 && t<distance) {
          distance=t;
          let n=normalize(eye+dir*t-r.xyz);
          color=vec3f(.15,.18,.14)*(.45+.55*max(0.,dot(n,sun())));
          color*=.7+.3*noise2((eye+dir*t).xz*5);
        }
      }
    }
  }
  // Two floating buoys: spheres and vertical signal masts, queried from three wave probes each.
  if(p.buoyancy!=0u && p.mode<2u) {
    for(var i=0u;i<2u;i++) {
      let pos=p.buoys[i*3u].xyz+vec3f(0,.32,0);
      let r=.65;
      let oc=eye-pos;
      let b=dot(oc,dir);
      let h=b*b-dot(oc,oc)+r*r;
      if(h>0) {
        let t=-b-sqrt(h);
        if(t>0 && t<distance) {
          distance=t;
          let n=normalize(eye+dir*t-pos);
          let c=select(vec3f(.85,.22,.035),vec3f(.9,.83,.47),i==1u);
          color=c*(.35+.65*max(0.,dot(n,sun())));
        }
      }
      // Mast follows the surface normal computed from the same 3 samples (tilt visible).
      let a=p.buoys[i*3u].xyz;
      let tx=p.buoys[i*3u+1u].xyz-a;
      let tz=p.buoys[i*3u+2u].xyz-a;
      let normal=normalize(cross(tz,tx));
      let top=pos+normal*1.1;
      let oc2=eye-top;
      let b2=dot(oc2,dir);
      let h2=b2*b2-dot(oc2,oc2)+.13*.13;
      if(h2>0) {
        let t=-b2-sqrt(h2);
        if(t>0 && t<distance) {
          distance=t;
          color=vec3f(.95,.82,.18);
        }
      }
    }
  }
  if(distance<10000) {
    color=mix(color,sky_horizon_color(sun(),.3),1-exp(-distance*p.fog));
  }
  return Env(color,distance);
}
struct Background {
  @location(0) color:vec4f, @builtin(frag_depth) depth:f32
}
@fragment fn background(v:FullscreenVarying)->Background {
  let dir=ray(v.uv,p.inv_vp,p.eye);
  let env=environment(p.eye,dir);
  var depth=.999999;
  if(env.distance<9999) {
    let clip=p.vp*vec4f(p.eye+dir*env.distance,1);
    depth=clamp(clip.z/clip.w,0.,.999998);
  }
  return Background(vec4f(linear_to_srgb(tonemap_aces(env.color*p.exposure)),1),depth);
}
@fragment fn reflection(v:FullscreenVarying)->@location(0) vec4f {
  let eye=vec3f(p.eye.x,-p.eye.y,p.eye.z);
  return vec4f(environment(eye,ray(v.uv,p.mirror_inv,eye)).color,1);
}
struct Water {
  @builtin(position) clip:vec4f, @location(0) world:vec3f, @location(1) normal:vec3f, @location(2) foam:f32, @location(3) mesh_level:f32
}
fn spectral(q:vec2f)->vec4f {
  var d=vec3f(0);
  var jac=1.;
  for(var i=0;i<3;i++) {
    let lengths=vec3f(192,48,12);
    let s=textureSampleLevel(disp,linear_sampler,q/lengths[i],i,0);
    d+=s.xyz;
    jac=min(jac,s.w);
  }
  return vec4f(d,jac);
}
fn spectral_normal(q:vec2f)->vec4f {
  var slope=vec2f(0);
  var foam=0.;
  for(var i=0;i<3;i++) {
    let lengths=vec3f(192,48,12);
    let s=textureSampleLevel(deriv,linear_sampler,q/lengths[i],i,0);
    slope+=s.xz;
    foam=max(foam,s.w);
  }
  return vec4f(normalize(vec3f(slope.x,1,slope.y)),foam);
}
@vertex fn water_vs(@location(0) grid:vec4f)->Water {
  var q=grid.xy;
  if(p.strategy==0u) {
    // Project only the lower screen region onto y=0; cap near-horizontal rays at the fog range.
    let horizon=(p.vp*vec4f(0,0,-10000,1));
    let hy=clamp(horizon.y/horizon.w,-.95,.95);
    let above=p.eye.y>=0.;
    let ndcy=select(mix(3.,hy+.006,grid.y),mix(-3.,hy-.006,grid.y),above);
    let uv=vec2f(grid.x*.75+.5,.5-.5*ndcy);
    let dir=ray(uv,p.inv_vp,p.eye);
    let denom=select(max(dir.y,.001),min(dir.y,-.001),above);
    let reference_height=select(-max(abs(p.eye.y),1.),max(abs(p.eye.y),1.),above);
    let distance=clamp(-reference_height/denom,.5,600.);
    q=(p.eye+dir*distance).xz;
  }
  else {
    q+=floor(p.eye.xz/8)*8;
  }
  var w=ocean_wave(q,p.time,p.strength,p.chop);
  var foam=(1-smoothstep(.45,.85,w.jacobian))*p.foam;
  if(p.mode==1u) {
    let d=spectral(q);
    let n=spectral_normal(q);
    w.position=vec3f(q.x,0,q.y)+d.xyz;
    w.normal=n.xyz;
    foam=n.w*p.foam;
  }
  if(p.mode==2u) {
    let depth=max(.02,.5-.15*q.x);
    let envelope=smoothstep(.03,.7,depth);
    let phase=q.x*1.5+.3*sin(q.y*.4)-p.time*1.4;
    w.position=vec3f(q.x+sin(phase)*.18*envelope,p.strength*.25*sin(phase)*envelope*min(1.4,1/sqrt(depth+.15)),q.y);
    w.normal=normalize(vec3f(-.38*p.strength*cos(phase)*envelope,1,0));
    foam=max(foam,(1-smoothstep(.08,.6,depth))*smoothstep(-.2,.5,sin(phase)))*p.foam;
  }
  if(p.mode==3u) {
    let flow=textureSampleLevel(flow_map,linear_sampler,(q+vec2f(16,40))/vec2f(32,64),0).xy;
    let phase=fract(p.time*.15);
    let blend=abs(phase*2-1);
    let n=mix(noise2(q*3-flow*phase*8),noise2(q*3-flow*fract(phase+.5)*8),blend);
    w.position=vec3f(q.x,(n-.5)*.10*p.strength,q.y);
    w.normal=normalize(vec3f((n-.5)*.4,1,(noise2(q*3+1)-.5)*.4));
    foam=0.;
    for(var i=0u;i<4u;i++) {
      let r=rock(i);
      let dist=length(q-r.xz)-r.w;
      foam=max(foam,(1-smoothstep(.05,.8,abs(dist)))*(.5+.5*n));
      let wake=q-r.xz;
      foam=max(foam,exp(-abs(wake.x)*2.5)*smoothstep(0.,1.,-wake.y)*(1-smoothstep(1.,5.,-wake.y))*n*.6);
    }
    foam*=p.foam;
  }
  if(p.mode==4u) {
    w.position=vec3f(q.x,(noise2(q*2-vec2f(0,p.time))-.5)*.06*p.strength,q.y);
    w.normal=normalize(vec3f((noise2(q*4)-.5)*.1,1,(noise2(q*4+p.time)-.5)*.1));
    foam=(1-smoothstep(0.,3.,abs(q.y+8)))*(1-smoothstep(2.,6.,abs(q.x)))*p.foam;
  }
  if(p.mode==6u) {
    w=ocean_wave(q,p.time,p.strength*.045,.1);
    foam=0.;
  }
  return Water(p.vp*vec4f(w.position,1),w.position,w.normal,foam,grid.z);
}
@fragment fn water_fs(v:Water,@builtin(front_facing) front:bool)->@location(0) vec4f {
  if(p.mode==2u && v.world.y<.15*v.world.x-.5) {
    discard;
  }
  if(p.mode==3u && abs(v.world.x-channel(v.world.z))>4.3) {
    discard;
  }
  var n=normalize(v.normal);
  let underwater=!front;
  if(underwater) {
    n=-n;
  }
  let view=normalize(p.eye-v.world);
  let reflect_dir=reflect(-view,n);
  let f=fresnel_water(dot(n,view));
  var reflection_color=sky_radiance_ex(reflect_dir,sun(),.3,0);
  if(p.mode==6u) {
    let clip=p.mirror_vp*vec4f(v.world,1);
    let uv=vec2f(clip.x/clip.w*.5+.5,.5-clip.y/clip.w*.5)+n.xz*.015;
    reflection_color=textureSample(planar,linear_sampler,clamp(uv,vec2f(.001),vec2f(.999))).rgb;
  }
  let crest=sss_crest(view,sun(),n,saturate(v.world.y*.7+.3),.4,3.);
  var body=mix(vec3f(.006,.055,.08),vec3f(.014,.21,.19),saturate(v.world.y*.5+.35));
  if(p.mode==2u) {
    let depth=max(0.,v.world.y-(.15*v.world.x-.5));
    body=mix(vec3f(.27,.29,.12),body,1-exp(-depth*2.5));
  }
  if(p.mode==3u) {
    body=vec3f(.02,.10,.075);
  }
  if(p.mode==6u) {
    body=vec3f(.012,.055,.045);
  }
  body+=vec3f(.015,.19,.13)*crest*p.scattering;
  let glint=spec_sun(n,view,sun(),.07,SKY_SUN_ANGULAR_RADIUS)*sky_sun_color(sun())*SKY_SUN_RADIANCE;
  var col=mix(body,reflection_color,f)+glint;
  let flecks=smoothstep(.30,.66,noise2(v.world.xz*7+vec2f(p.time*.12,0)));
  col=mix(col,vec3f(.64,.8,.79)*(.65+.35*max(dot(n,sun()),0.)),saturate(v.foam*flecks));
  col=mix(col,sky_horizon_color(sun(),.3),1-exp(-length(p.eye-v.world)*p.fog));
  if(p.mode==5u && underwater) {
    col=mix(col,vec3f(.01,.18,.19),.65);
  }
  if(p.view==1u) {
    col=n*.5+.5;
  }
  if(p.view==2u) {
    col=vec3f(v.foam);
  }
  if(p.view==3u) {
    col=mix(vec3f(.04,.2,.55),vec3f(.9,.4,.05),v.mesh_level/4);
    col*=.65+.35*step(.08,fract(v.world.x*2))*step(.08,fract(v.world.z*2));
  }
  return vec4f(linear_to_srgb(tonemap_aces(col*p.exposure)),1);
}
// A vertical, scrolling sheet, supported by the waterfall's mist particles and plunge pool.
@vertex fn falls_vs(@builtin(vertex_index) i:u32)->Water {
  let uv=array<vec2f,6>(vec2f(0,0),vec2f(1,0),vec2f(0,1),vec2f(0,1),vec2f(1,0),vec2f(1,1))[i];
  let world=vec3f((uv.x-.5)*7,uv.y*8,-8+.12*sin(uv.x*30+p.time*2)*p.strength);
  return Water(p.vp*vec4f(world,1),world,vec3f(0,0,1),uv.y,0);
}
@fragment fn falls_fs(v:Water)->@location(0) vec4f {
  let streak=noise2(vec2f(v.world.x*4,v.world.y*.7+p.time*4));
  let detail=noise2(vec2f(v.world.x*11,v.world.y*1.5+p.time*8));
  let edge=1-smoothstep(3.1,3.5,abs(v.world.x)+.12*streak);
  let alpha=(.7+.25*streak)*edge;
  let col=mix(vec3f(.10,.27,.28),vec3f(.85,.95,.94),streak*.5+detail*.5);
  return vec4f(linear_to_srgb(tonemap_aces(col*p.exposure))*alpha,alpha);
}
@group(1) @binding(0) var scene_color:texture_2d<f32>;
@fragment fn waterline(v:FullscreenVarying)->@location(0) vec4f {
  let color=textureSample(scene_color,linear_sampler,v.uv).rgb;
  // Wave height at a short distance along each pixel's ray defines a moving waterline.
  let dir=ray(v.uv,p.inv_vp,p.eye);
  let pos=p.eye+dir*2.;
  let height=ocean_world_wave(pos.xz,p.time,p.strength,p.chop).position.y;
  let signed_distance=pos.y-height;
  let width=max(fwidth(signed_distance)*2,.015);
  let below=1-smoothstep(-width,width,signed_distance);
  let meniscus=(1-smoothstep(width,width*4,abs(signed_distance)))*.65;
  let underwater=mix(color*vec3f(.30,.74,.78),vec3f(.015,.20,.23),.32);
  return vec4f(mix(mix(color,underwater,below),vec3f(.14,.39,.39),meniscus),1);
}
