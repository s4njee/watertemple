#include "fullscreen.wgsl"
#include "color.wgsl"
#include "noise.wgsl"
#include "params.wgsl"
// Word coverage uses explicit texture LOD. Derivatives are only used for AA within a glyph.
@group(0) @binding(0) var<uniform> u: UIParams;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var backdrop: texture_2d<f32>;
@group(0) @binding(3) var next_page: texture_2d<f32>;
@group(0) @binding(4) var atlas: texture_2d<f32>;
@group(0) @binding(5) var wet: texture_2d<f32>;
@group(0) @binding(6) var image: texture_2d<f32>;
#include "text.wgsl"
fn round_rect(p:vec2f,b:vec2f,r:f32)->f32  {
  let q=abs(p)-b+r;
  return length(max(q,vec2f(0.0)))+min(max(q.x,q.y),0.0)-r;
}
fn smooth_min(a:f32,b:f32,k:f32)->f32  {
  let h=max(k-abs(a-b),0.0)/max(k,0.0001);
  return min(a,b)-h*h*k*0.25;
}
fn source(p:vec2f)->vec3f  {
  if(u.has_image>0.5&&p.x>0.48&&p.x<0.94&&p.y>0.14&&p.y<0.9)  {
    let q=((p-vec2f(0.48,0.14))/vec2f(0.46,0.76)-0.5)*u.image_fit+0.5;
    return textureSampleLevel(image,linear_sampler,clamp(q,vec2f(0),vec2f(1)),0.0).rgb;
  }
  return textureSampleLevel(backdrop,linear_sampler,clamp(p,vec2f(0.001),vec2f(0.999)),0.0).rgb;
}
fn lens(p:vec2f,n:vec2f,amount:f32,blur:f32)->vec3f  {
  let q=p+n*amount;
  let px=blur/u.resolution;
  var c=source(q)*0.4;
  c+=(source(q+px)+source(q-px)+source(q+vec2f(px.x,-px.y))+source(q+vec2f(-px.x,px.y)))*0.15;
  let fringe=n*u.dispersion;
  c.r=source(q+fringe).r*0.7+c.r*0.3;
  c.b=source(q-fringe).b*0.7+c.b*0.3;
  return c;
}
fn water_surface(x:f32,level:f32)->f32  {
  return 1.0-level+sin(level*3.14159265)*u.strength*(0.012*sin(x*23.0-u.clock*2.2)+0.006*sin(x*37.0+u.clock*1.3));
}
fn blob_field(p:vec3f)->f32  {
  let a=length(p-vec3f(-0.28,0.12*sin(u.clock),0.0))-0.68;
  let b=length(p-vec3f(0.43,0.19*cos(u.clock*0.8),0.13*sin(u.clock*0.5)))-0.51;
  let c=length(p-vec3f(0.1,0.62*sin(u.clock*0.6),-0.15))-0.44;
  return smooth_min(smooth_min(a,b,u.extra),c,u.extra);
}
fn blob_normal(p:vec3f)->vec3f  {
  let e=0.002;
  return normalize(vec3f(blob_field(p+vec3f(e,0,0))-blob_field(p-vec3f(e,0,0)),blob_field(p+vec3f(0,e,0))-blob_field(p-vec3f(0,e,0)),blob_field(p+vec3f(0,0,e))-blob_field(p-vec3f(0,0,e))));
}
fn environment(d:vec3f)->vec3f  {
  let horizon=mix(vec3f(0.05,0.13,0.18),vec3f(0.62,0.81,0.76),smoothstep(-0.5,0.7,d.y));
  return horizon+vec3f(1.1,0.98,0.76)*pow(max(dot(d,normalize(vec3f(-0.6,0.9,0.8))),0.0),25.0);
}
@fragment fn fs(in:FullscreenVarying)->@location(0) vec4f  {
  let uv=in.uv;
  let aspect=u.resolution.x/u.resolution.y;
  let pixel=1.5/u.resolution.y;
  let p=(uv-0.5)*vec2f(aspect,1.0);
  var c=source(uv);
  let ink=vec3f(0.025,0.065,0.075);
  if(MODE==0u)  {
    // Liquid glass rounded panel.
    let q=p-vec2f(0.02,0.02);
    let d=round_rect(q,vec2f(min(aspect*0.35,0.6),0.29),u.radius);
    let mask=1.0-smoothstep(-pixel,pixel,d);
    let normal=normalize(vec2f(q.x/max(0.02,abs(q.x)),q.y/max(0.02,abs(q.y)))+q*3.0+vec2f(0.0001));
    let edge=exp(-abs(d)*55.0);
    let convex=q*(0.3+edge*0.7);
    var glass=lens(uv,convex,u.strength*0.055,u.blur);
    glass=mix(glass,u.tint,0.1)+edge*(0.06+0.3*pow(max(dot(normal,normalize(vec2f(-0.5,-0.8))),0.0),8.0));
    c=mix(c,glass,mask);
    let label=word(uv,vec2f(0.51,0.49),vec2f(0.043,0.135),0u);
    c=mix(c,vec3f(0.84,0.93,0.87),label*mask);
    let line=step(abs(uv.y-0.62),0.001)*step(abs(uv.x-0.5),0.15)*mask;
    c=mix(c,vec3f(0.85),line*0.65);
  }
  if(MODE==1u)  {
    // Smooth union of four navigation pills and a cursor follower.
    var d=10.0;
    for(var i=0u;i<4u;i++)  {
      let center=vec2f((f32(i)-1.5)*0.185,0.035*sin(u.clock+f32(i)));
      d=smooth_min(d,round_rect(p-center,vec2f(0.076,0.078),0.075),u.extra);
    }
    let cursor=mix(vec2f(0.21*sin(u.clock),0.13*cos(u.clock*1.1)),(u.pointer-0.5)*vec2f(aspect,1.0),u.interaction);
    d=smooth_min(d,length(p-cursor)-u.radius*u.strength,u.extra);
    let mask=1.0-smoothstep(-pixel,pixel,d);
    let edge=exp(-abs(d)*95.0);
    c=mix(c,ink+u.tint*0.08+edge*0.23,mask);
    c=mix(c,vec3f(0.84,0.90,0.79),word(uv,vec2f(0.5),vec2f(0.024,0.09),2u)*mask);
  }
  if(MODE==2u)  {
    // CPU spring drives the boundary, press launches a travelling ring.
    let q=p;
    let wobble=(sin(atan2(q.y,q.x)*5.0+u.clock*5.0)*0.009)*u.spring*u.strength;
    let d=round_rect(q,vec2f(0.23,0.095),0.065)-wobble-u.spring*0.009;
    let mask=1.0-smoothstep(-pixel,pixel,d);
    let radial=length((uv-u.press)*vec2f(aspect,1.0));
    let wave=sin(radial*85.0-u.ripple*16.0)*exp(-pow((radial-u.ripple*0.35)*13.0,2.0))*exp(-u.ripple*2.0)*u.pressed;
    let edge=exp(-abs(d)*130.0);
    var button=vec3f(0.018,0.19,0.22)+u.tint*(0.1+wave*0.09)+edge*0.19;
    c=mix(c,button,mask);
    c=mix(c,vec3f(0.9,0.97,0.86),word(uv,vec2f(0.5),vec2f(0.035,0.112),2u)*mask);
  }
  if(MODE==3u)  {
    let t=u.level;
    let w=0.1*sin(uv.x*11.0+u.clock*1.7)+0.04*sin(uv.x*31.0-u.clock);
    var boundary=uv.y-(t*1.3-0.15+w*u.strength);
    if(u.style==0u)  {
      let drip=pow(0.5+0.5*sin(uv.x*19.0+sin(uv.x*31.0)),3.0);
      boundary=uv.y-(t*1.8-0.4+drip*0.3*u.strength);
    }
    if(u.style==2u)  {
      boundary=gnoise_2d(uv*7.0+vec2f(u.clock*0.1))*(0.25*u.strength)+0.5-t;
    }
    let blend=select(select(1.0-smoothstep(-0.02,0.02,boundary),0.0,t<=0.0),1.0,t>=1.0);
    let edge=exp(-abs(boundary)*90.0)*select(1.0,0.0,t<=0.0||t>=1.0);
    let offset=vec2f(0.012*sin(uv.y*20.0),0.035)*edge*u.strength;
    let next=textureSampleLevel(next_page,linear_sampler,clamp(uv-offset,vec2f(0),vec2f(1)),0.0).rgb;
    c=mix(source(uv+offset),next,blend)+edge*vec3f(0.16,0.3,0.28);
  }
  if(MODE==4u)  {
    let q=(uv-u.pointer)*vec2f(aspect,1.0);
    let r=length(q);
    let wave=sin(r*55.0-u.clock*5.0)*exp(-r*4.0)*u.interaction;
    let n=q/max(r,0.001)*wave;
    let image=step(0.48,uv.x)*step(uv.x,0.94)*step(0.14,uv.y)*step(uv.y,0.9);
    c=mix(c,lens(uv,n,u.strength*0.027,0.0),image);
  }
  if(MODE==5u)  {
    c=mix(ink,vec3f(0.07,0.19,0.22),uv.y);
    var q=uv;
    if(u.style==0u)  {
      q+=vec2f(sin(uv.y*17.0+u.clock*2.0),cos(uv.x*22.0-u.clock*2.0))*u.strength*0.011;
    }
    if(u.style==1u)  {
      q.y-=u.strength*0.07*pow(0.5+0.5*sin(uv.x*35.0+u.clock),3.0)*smoothstep(0.35,0.85,uv.y);
    }
    let coverage_type=word(q,vec2f(0.5),vec2f(0.105,0.43),0u);
    let surface=water_surface(uv.x,0.30+u.level*0.40);
    let fill=smoothstep(surface-pixel,surface+pixel,uv.y);
    let water=mix(vec3f(0.035,0.40,0.45),vec3f(0.17,0.76,0.71),exp(-abs(uv.y-surface)*18.0));
    var color=vec3f(0.86,0.88,0.75);
    if(u.style==2u)  {
      color=mix(color*0.28,water,fill);
    }
    c=mix(c,color,coverage_type);
    let small=word(uv,vec2f(0.5,0.79),vec2f(0.022,0.06),1u);
    c=mix(c,vec3f(0.47,0.65,0.61),small);
  }
  if(MODE==6u)  {
    var d=round_rect(p,vec2f(0.32,0.24),0.065);
    if(u.style==0u)  {
      d=round_rect(p,vec2f(min(0.6,aspect*0.42),0.075),0.07);
    }
    if(u.style==1u)  {
      d=length(p)-0.23;
    }
    let mask=1.0-smoothstep(-pixel,pixel,d);
    let surface=select(0.74-u.level*0.48,0.575-u.level*0.15,u.style==0u)+u.strength*(0.013*sin(uv.x*24.0-u.clock*2.0)+0.004*sin(uv.x*43.0+u.clock));
    let fill=smoothstep(surface-pixel,surface+pixel,uv.y);
    let empty=source(uv)*0.8+vec3f(0.035);
    let liquid=lens(uv,vec2f(0.01*sin(uv.y*28.0-u.clock),0),u.strength,0.0)*0.2+u.tint*vec3f(0.025,0.40,0.42);
    c=mix(c,mix(empty,liquid,fill)+exp(-abs(d)*110.0)*0.22,mask);
    c+=exp(-abs(uv.y-surface)*220.0)*mask*0.2;
    // Keep exact empty/full end points; waves stop exposing a gap at 100%.
    if(u.level<=0.0)  {
      c=mix(c,empty,mask);
    }
    if(u.level>=1.0)  {
      c=mix(c,liquid,mask);
    }
  }
  if(MODE==7u)  {
    let ro=vec3f(0,0,3.2);
    let rd=normalize(vec3f(p.x*2.0,-p.y*2.0,-2.4));
    var distance=0.0;
    var hit=false;
    var point=ro;
    for(var i=0u;i<72u;i++)  {
      point=ro+rd*distance;
      let d=blob_field(point);
      if(d<0.0015)  {
        hit=true;
        break;
      }
      distance+=max(d*0.8,0.001);
      if(distance>6.0)  {
        break;
      }
    }
    if(hit)  {
      let n=blob_normal(point);
      let reflected=environment(reflect(rd,n));
      let refracted=refract(rd,n,1.0/1.333);
      // Exit is marched inside the SDF; thickness drives absorption and the second refraction.
      var travel=0.025;
      var exit=point;
      for(var i=0u;i<40u;i++)  {
        exit=point+refracted*travel;
        let d=blob_field(exit);
        if(d>0.0)  {
          break;
        }
        travel+=max(-d*0.7,0.006);
      }
      let exit_n=-blob_normal(exit);
      var through=refract(refracted,exit_n,1.333);
      if(dot(through,through)<0.001)  {
        through=reflect(refracted,exit_n);
      }
      let backdrop_uv=uv+through.xy*0.12*u.strength;
      let trans=mix(source(backdrop_uv),environment(through),0.28)*exp(-travel*vec3f(0.65,0.11,0.08)*u.strength);
      let fresnel=0.02+0.98*pow(1.0-max(dot(-rd,n),0.0),5.0);
      c=mix(trans,reflected,fresnel)+pow(max(dot(reflect(rd,n),normalize(vec3f(-0.5,0.8,1))),0.0),64.0)*0.8;
    }
  }
  if(MODE==8u)  {
    let px=1.0/vec2f(textureDimensions(wet));
    let w=textureSampleLevel(wet,linear_sampler,uv,0.0).r;
    let grad=vec2f(textureSampleLevel(wet,linear_sampler,uv+vec2f(px.x,0),0.0).r-textureSampleLevel(wet,linear_sampler,uv-vec2f(px.x,0),0.0).r,textureSampleLevel(wet,linear_sampler,uv+vec2f(0,px.y),0.0).r-textureSampleLevel(wet,linear_sampler,uv-vec2f(0,px.y),0.0).r);
    c=mix(c,lens(uv,grad,0.085,0.0)*0.78,w*0.75)+pow(clamp(dot(grad,vec2f(-0.7,-0.7))*4.0,0.0,1.0),2.0)*0.2;
  }
  if(MODE==9u)  {
    let surface=water_surface(uv.x,u.level);
    let submerged=smoothstep(surface-pixel,surface+pixel,uv.y);
    let depth=max(uv.y-surface,0.0);
    let n=vec2f(sin(uv.y*24.0+u.clock*2),cos(uv.x*19.0-u.clock))*0.4;
    let water=lens(uv,n,0.015*u.strength,1.0)*exp(-depth*vec3f(1.3,0.4,0.2))+vec3f(0.01,0.07,0.08)*depth;
    c=mix(c,water,submerged)+exp(-abs(uv.y-surface)*180.0)*0.35;
  }
  if(MODE==10u)  {
    let d=round_rect(p,vec2f(min(aspect*0.38,0.62),0.34),0.035);
    let card=1.0-smoothstep(-pixel,pixel,d);
    c=mix(c,lens(uv,vec2f(0),0.0,1.5)*0.85,card);
    var coverage=0.0;
    var normal=vec2f(0);
    for(var i=0u;i<48u;i++)  {
      let drop=u.drops[i];
      let q=(uv-drop.xy)*vec2f(aspect,1.0);
      let r=max(drop.z,0.0001);
      let ratio=length(q)/r;
      let mask=(1.0-smoothstep(0.92,1.05,ratio))*select(0.0,1.0,drop.z>0.0);
      if(mask>coverage)  {
        coverage=mask;
        normal=q/r*0.55;
      }
    }
    let refracted=lens(uv,normal,u.strength*0.05,0.0);
    let shine=pow(max(dot(normal,normalize(vec2f(-0.6,-0.8))),0.0),12.0)*0.4;
    c=mix(c,refracted+shine,coverage*card);
  }
  return vec4f(linear_to_srgb(max(c,vec3f(0))),1.0);
}
