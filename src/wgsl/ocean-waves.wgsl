// ocean-waves.wgsl — analytic six-wave Gerstner surface, y up, meters / seconds.
// ocean_wave(q,time,strength,chop) -> OceanWave (position, tangents, normal, Jacobian).
// ocean_world_wave(xz,...) inverts horizontal displacement for buoyancy (bounded Newton steps).
// Wave constants and phase offsets mirror src/gpu/ocean/waves.ts. No bindings or entrypoints.
struct OceanWave {
  position: vec3f, dx: vec3f, dz: vec3f, normal: vec3f, jacobian: f32
}
fn ocean_wave(q: vec2f, time: f32, strength: f32, chop: f32) -> OceanWave {
  let waves = array<vec4f, 6>(vec4f(.94,.342,.52,.24),vec4f(.77,-.638,.28,.43),vec4f(-.24,.971,.15,.78),vec4f(.61,.792,.085,1.31),vec4f(-.88,.475,.042,2.37),vec4f(.36,-.933,.024,4.17));
  var p = vec3f(q.x, 0, q.y);
  var dx = vec3f(1,0,0);
  var dz = vec3f(0,0,1);
  for(var i=0u; i<6u; i++) {
    let w=waves[i];
    let d=normalize(w.xy);
    let a=w.z*strength;
    let k=w.w;
    let phase=k*dot(d,q)-sqrt(9.81*k)*time+f32(i*4u)*.31;
    let s=sin(phase);
    let c=cos(phase);
    p+=vec3f(chop*a*d.x*c,a*s,chop*a*d.y*c);
    dx+=vec3f(-chop*a*k*d.x*d.x*s,a*k*d.x*c,-chop*a*k*d.x*d.y*s);
    dz+=vec3f(-chop*a*k*d.x*d.y*s,a*k*d.y*c,-chop*a*k*d.y*d.y*s);
  }
  return OceanWave(p,dx,dz,normalize(cross(dz,dx)),dx.x*dz.z-dx.z*dz.x);
}
fn ocean_world_wave(xz:vec2f,time:f32,strength:f32,chop:f32)->OceanWave {
  var q=xz;
  for(var i=0u;i<8u;i++) {
    let w=ocean_wave(q,time,strength,chop);
    let e=w.position.xz-xz;
    if(abs(w.jacobian)<.1) {
      break;
    }
    q-=clamp(vec2f(w.dz.z*e.x-w.dz.x*e.y,-w.dx.z*e.x+w.dx.x*e.y)/w.jacobian,vec2f(-1),vec2f(1));
  }
  return ocean_wave(q,time,strength,chop);
}
