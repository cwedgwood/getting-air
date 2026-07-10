// Milestone 4 (plans/AMR.md): fine-level (L=1) LBM step, POOL-AWARE.
// Supersedes Milestone 2's single-fixed-region version -- see
// amr_interp_c2f.wgsl's file header for the pool addressing scheme this
// shares (dispatch over (tile, tile, slot), slotToBlock indirection,
// buffer-space-native coarse addressing).
//
// Unlike the interpolation pass, this kernel DOES need window coordinates
// (for the card SDF and penalization physics, both physically anchored),
// derived by inverting the moving-window off_x/off_y mapping -- the same
// derivation amr_step.wgsl's coarse kernel uses, just applied to a
// continuous fine-cell position instead of an integer coarse-cell one.
//
// Streaming clamps at the slot's own buffer edge (2-cell ghost border
// degrades over the 2 fine substeps by design -- see
// amr_interp_c2f.wgsl's header); force integration stays coarse-only this
// milestone (same scope cut as Milestone 2).

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

@group(0) @binding(0) var<storage, read>       state       : CardState;
@group(0) @binding(1) var<storage, read>       f_in        : array<f32>;
@group(0) @binding(2) var<storage, read_write> f_out       : array<f32>;
@group(0) @binding(3) var<storage, read_write> vel_pool    : array<f32>;
@group(0) @binding(4) var<storage, read>       slotToBlock : array<i32>;

override W : u32; // coarse grid dims, needed for the off_x/off_y window wrap
override H : u32;
override RB : u32;
const BLOCK = 8u;
const GHOST = 2u;

const ex = array<i32,9>( 0, 1, 0,-1, 0, 1,-1,-1, 1);
const ey = array<i32,9>( 0, 0, 1, 0,-1, 1, 1,-1,-1);
const wt = array<f32,9>(
  0.44444444f,
  0.11111111f, 0.11111111f, 0.11111111f, 0.11111111f,
  0.02777778f, 0.02777778f, 0.02777778f, 0.02777778f
);

fn fineToCoarseUnit(fCoord: u32, origin: u32) -> f32 {
  let j = f32(i32(fCoord) - i32(GHOST));
  return f32(origin) - 0.25 + 0.5 * j;
}

fn wrapf(v: f32, n: f32) -> f32 {
  var r = v % n;
  if (r < 0.0) { r += n; }
  return r;
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

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let fx = gid.x; let fy = gid.y;
  let slot = gid.z;
  let FB = RB * 2u + 2u * GHOST;
  if (fx >= FB || fy >= FB) { return; }

  let blockID = slotToBlock[slot];
  if (blockID < 0) { return; }

  let nbx = W / BLOCK;
  let originX = (u32(blockID) % nbx) * RB;
  let originY = (u32(blockID) / nbx) * RB;

  let poolPlaneStride = arrayLength(&f_in) / 9u;
  let cell = slot * (FB * FB) + fy * FB + fx;

  // 1. Pull Streaming: clamp at the slot's own buffer edge.
  var f: array<f32,9>;
  for (var i = 0u; i < 9u; i++) {
    let srcX = clamp(i32(fx) - ex[i], 0, i32(FB) - 1);
    let srcY = clamp(i32(fy) - ey[i], 0, i32(FB) - 1);
    let srcCell = slot * (FB * FB) + u32(srcY) * FB + u32(srcX);
    f[i] = f_in[i * poolPlaneStride + srcCell];
  }

  // 2. Local Macroscopic Variables
  var rho = 0f; var ux_star = 0f; var uy_star = 0f;
  for (var i = 0u; i < 9u; i++) {
    rho     += f[i];
    ux_star += f[i] * f32(ex[i]);
    uy_star += f[i] * f32(ey[i]);
  }
  ux_star /= rho; uy_star /= rho;

  // 3. Penalty Force and Solid Coupling. Buffer-space fine position ->
  // window position by inverting off_x/off_y (see file header).
  let bufX = fineToCoarseUnit(fx, originX);
  let bufY = fineToCoarseUnit(fy, originY);
  let wx = wrapf(bufX - state.off_x, f32(W));
  let wy = wrapf(bufY - state.off_y, f32(H));
  let p = vec2<f32>(wx, wy);
  let rx = p.x - state.cx;
  let ry = p.y - state.cy;

  let usx = state.vx - state.omega * ry;
  let usy = state.vy + state.omega * rx;

  let phi = get_phi(p, state);
  let chi = get_chi(phi);

  let Fx = rho * chi * (usx - ux_star);
  let Fy = rho * chi * (usy - uy_star);

  let ux = ux_star + Fx / (2.0f * rho);
  let uy = uy_star + Fy / (2.0f * rho);
  let u_sq = ux*ux + uy*uy;

  vel_pool[cell * 2u] = ux; vel_pool[cell * 2u + 1u] = uy;

  // 4. Collision and ALBC sponge. Milestone 4b fix: this used to skip the
  // sponge entirely on the (then-true) assumption that the fine region
  // never reaches the window edge -- valid when M2 hand-placed a single
  // static box, but false once refinement is criterion-driven and can
  // trigger anywhere, including near the sponge band where the coarse step
  // (amr_step.wgsl) DOES damp toward equilibrium. A refined block there
  // with no sponge of its own diverges from its damped coarse neighbors,
  // and the average pass then writes that undamped state back onto them --
  // exactly the boundary artifact this was fixed in response to. Same
  // formula as amr_step.wgsl's sponge, reusing the wx/wy already computed
  // above for the card SDF.
  let SPONGE_W = 4.0f;
  let dist_x = min(wx, f32(W) - 1.0f - wx);
  let dist_y = min(wy, f32(H) - 1.0f - wy);
  var sponge_weight = clamp(1.0f - min(dist_x, dist_y) / SPONGE_W, 0.0f, 1.0f);
  sponge_weight = sponge_weight * sponge_weight * (3.0f - 2.0f * sponge_weight);

  let tau_fine = 2.0f * state.tau - 0.5f;
  let omg = 1.0f / tau_fine;
  for (var i = 0u; i < 9u; i++) {
    let exf = f32(ex[i]); let eyf = f32(ey[i]);
    let eu  = exf*ux + eyf*uy;
    let feq = wt[i] * rho * (1f + 3f*eu + 4.5f*eu*eu - 1.5f*u_sq);

    let term1x = (exf - ux) * 3.0f;
    let term1y = (eyf - uy) * 3.0f;
    let term2  = (exf*ux + eyf*uy) * 9.0f;
    let Si = (1.0f - 0.5f * omg) * wt[i] * ( (term1x + term2*exf)*Fx + (term1y + term2*eyf)*Fy );

    let f_collide = f[i] - omg * (f[i] - feq) + Si;
    let f_target  = wt[i] * 1.0f;
    f_out[i * poolPlaneStride + cell] = mix(f_collide, f_target, sponge_weight);
  }
}
