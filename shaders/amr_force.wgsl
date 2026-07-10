// Force and torque on the solid body via integration of the penalty force.
//
// Milestone 1 (plans/AMR.md): block-major buffer layout, same rationale and
// derivation as amr_step.wgsl's file header -- dispatch over buffer-space
// coordinates, derive window coordinates per-thread for the card SDF.

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

@group(0) @binding(0) var<storage, read>       state  : CardState;
@group(0) @binding(1) var<storage, read>       f_in   : array<f32>;
@group(0) @binding(2) var<storage, read_write> forces : array<atomic<i32>, 4>;

override W : u32;
override H : u32;
const FSCALE = 10000f;
const BLOCK = 8u;

const ex = array<i32,9>( 0, 1, 0,-1, 0, 1,-1,-1, 1);
const ey = array<i32,9>( 0, 0, 1, 0,-1, 1, 1,-1,-1);

// Block-major linear index for a cell at BUFFER coordinates (cx, cy).
// See amr_step.wgsl for the full derivation.
fn cellIndex(cx: u32, cy: u32) -> u32 {
  let nbx = W / BLOCK;
  let bx = cx / BLOCK; let by = cy / BLOCK;
  let lx = cx % BLOCK; let ly = cy % BLOCK;
  let blockID = by * nbx + bx;
  return blockID * (BLOCK * BLOCK) + ly * BLOCK + lx;
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

var<workgroup> wg_fx : array<f32, 64>;
var<workgroup> wg_fy : array<f32, 64>;
var<workgroup> wg_tz : array<f32, 64>;

@compute @workgroup_size(8, 8)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32
) {
  let cx = gid.x; let cy = gid.y;

  var fx_body = 0.0f;
  var fy_body = 0.0f;
  var tz_body = 0.0f;

  if (cx < W && cy < H) {
    let wx   = (cx + W - u32(state.off_x)) % W;
    let wy   = (cy + H - u32(state.off_y)) % H;
    let cell = cellIndex(cx, cy);
    let p    = vec2<f32>(f32(wx), f32(wy));

    let phi = get_phi(p, state);
    let chi = get_chi(phi);

    if (chi >= 1e-6) {
      var rho = 0f; var ux_star = 0f; var uy_star = 0f;
      for (var i = 0u; i < 9u; i++) {
        let fi = f_in[i * (W * H) + cell];
        rho     += fi;
        ux_star += fi * f32(ex[i]);
        uy_star += fi * f32(ey[i]);
      }
      ux_star /= rho; uy_star /= rho;

      // Local solid velocity Us
      var rx = p.x - state.cx;
      var ry = p.y - state.cy;
      rx -= f32(W) * round(rx / f32(W));
      ry -= f32(H) * round(ry / f32(H));
      let usx = state.vx - state.omega * ry;
      let usy = state.vy + state.omega * rx;

      // Penalty Force F = rho * chi * (Us - u*)
      let Fx = rho * chi * (usx - ux_star);
      let Fy = rho * chi * (usy - uy_star);

      // Integrate NEGATIVE of penalty force onto body
      fx_body = -Fx;
      fy_body = -Fy;
      tz_body = rx * fy_body - ry * fx_body;
    }
  }

  // Workgroup reduction
  wg_fx[lid] = fx_body;
  wg_fy[lid] = fy_body;
  wg_tz[lid] = tz_body;
  workgroupBarrier();

  // Simple reduction tree or linear sum for 64 elements
  if (lid == 0u) {
    var sum_fx = 0.0f;
    var sum_fy = 0.0f;
    var sum_tz = 0.0f;
    for (var i = 0u; i < 64u; i++) {
      sum_fx += wg_fx[i];
      sum_fy += wg_fy[i];
      sum_tz += wg_tz[i];
    }
    atomicAdd(&forces[0], i32(sum_fx * FSCALE));
    atomicAdd(&forces[1], i32(sum_fy * FSCALE));
    atomicAdd(&forces[2], i32(sum_tz * FSCALE));
  }
}
