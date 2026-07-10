// Visualization shader with smooth analytical mask and vorticity calculation.

struct CardState {
  cx     : f32,
  cy     : f32,
  theta  : f32,
  vx     : f32,
  vy     : f32,
  omega  : f32,
  fx     : f32,
  fy     : f32,
  tz     : f32,
  mass   : f32,
  i_body : f32,
  g_eff  : f32,
  a      : f32,
  b      : f32,
  v_max  : f32,
  o_max  : f32,
  cx_old : f32,
  cy_old : f32,
  th_old : f32,
  tau    : f32,
  y_total: f32,
  x_total: f32,
  off_x  : f32,
  off_y  : f32,
  off_x_old : f32,
  off_y_old : f32,
}

@group(0) @binding(0) var<storage, read> vel   : array<f32>;
@group(0) @binding(1) var<storage, read> state : CardState;

override W : u32;
override H : u32;

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv : vec2<f32>,
}

const p = array<vec2<f32>,6>(
  vec2(-1f,-1f), vec2( 1f,-1f), vec2(-1f, 1f),
  vec2(-1f, 1f), vec2( 1f,-1f), vec2( 1f, 1f)
);

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VSOut {
  var out: VSOut;
  out.pos = vec4(p[vi], 0f, 1f);
  out.uv  = p[vi] * 0.5f + 0.5f;
  return out;
}

fn get_phi(p: vec2<f32>, state: CardState) -> f32 {
    let ca = cos(state.theta);
    let sa = sin(state.theta);
    var dx = p.x - state.cx;
    var dy = p.y - state.cy;
    dx -= f32(W) * round(dx / f32(W));
    dy -= f32(H) * round(dy / f32(H));
    let lx = dx * ca + dy * sa;
    let ly = -dx * sa + dy * ca;
    let d = sqrt((lx*lx)/(state.a*state.a) + (ly*ly)/(state.b*state.b)) - 1.0;
    return d * state.b; 
}

fn get_chi(phi: f32) -> f32 {
    let epsilon = 1.5f;
    // [intel-xe fix] clamp tanh arg: Intel Gen12LP tanh() overflows to NaN for
    // large |arg| (exp overflow -> Inf/Inf); the sigmoid is fully saturated by
    // +/-10 so this is numerically identical in the valid regime but never NaNs.
    return 0.5f * (1.0f - tanh(clamp(phi / epsilon, -20.0f, 20.0f)));
}

fn get_uy(x: i32, y: i32) -> f32 {
    let wx = (u32(x) + W) % W;
    let wy = (u32(y) + H) % H;
    let bx = (wx + u32(state.off_x)) % W;
    let by = (wy + u32(state.off_y)) % H;
    return vel[(by * W + bx) * 2u + 1u];
}

fn get_ux(x: i32, y: i32) -> f32 {
    let wx = (u32(x) + W) % W;
    let wy = (u32(y) + H) % H;
    let bx = (wx + u32(state.off_x)) % W;
    let by = (wy + u32(state.off_y)) % H;
    return vel[(by * W + bx) * 2u];
}

@fragment
fn fs_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
  let fx = uv.x * f32(W); let fy = (1.0 - uv.y) * f32(H);
  let ix = i32(fx); let iy = i32(fy);
  
  let chi = get_chi(get_phi(vec2(fx, fy), state));

  // Discrete vorticity: du_y/dx - du_x/dy
  let omega = (get_uy(ix + 1, iy) - get_uy(ix - 1, iy)) * 0.5f
            - (get_ux(ix, iy + 1) - get_ux(ix, iy - 1)) * 0.5f;

  // Blue for clockwise (negative), red for counter-clockwise (positive)
  let val = clamp(omega * 80.0f, -1.0f, 1.0f);
  var c: vec3<f32>;
  if (val > 0.0) {
    c = mix(vec3(0.05, 0.05, 0.1), vec3(1.0, 0.3, 0.2), val);
  } else {
    c = mix(vec3(0.05, 0.05, 0.1), vec3(0.2, 0.5, 1.0), -val);
  }
  
  // Blend with solid color
  let solid_color = vec3(1.0, 0.8, 0.4);
  c = mix(c, solid_color, chi);
  
  return vec4(c, 1.0);
}
