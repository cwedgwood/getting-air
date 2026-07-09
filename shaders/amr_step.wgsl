// Fused LBM Kernel: Pull-Streaming + Collision + Source Term
//
// Milestone 1 (plans/AMR.md): block-major buffer layout. Buffer cells are
// grouped into fixed BLOCK x BLOCK tiles laid out contiguously per tile --
// this is the addressable "cell-block" unit later milestones pool/refine/
// coarsen (AGAL section 3.2's cell-block decomposition), though at this
// milestone the block <-> position mapping is still a straight identity (no
// pool indirection yet -- that's Milestone 4).
//
// Dispatch is now over BUFFER coordinates (not window coordinates): gid.x/
// gid.y address a fixed memory location that never moves as the moving
// window (off_x/off_y) pans. Window/physical coordinates (needed only for
// the card SDF and the ALBC sponge, both physically anchored, not buffer-
// anchored) are derived per-thread by inverting the moving-window mapping.
// Streaming between buffer-adjacent cells is equivalent to streaming
// between window-adjacent cells because off_x/off_y is a single shift
// applied uniformly to every cell: if bx = (wx + off_x) % W for every
// cell, then the neighbor at window offset -ex lands at buffer offset -ex
// too, regardless of off_x's actual value.

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

@group(0) @binding(0) var<storage, read>       state : CardState;
@group(0) @binding(1) var<storage, read>       f_in  : array<f32>;
@group(0) @binding(2) var<storage, read_write> f_out : array<f32>;
@group(0) @binding(3) var<storage, read_write> vel   : array<f32>;

override W : u32;
override H : u32;
// [intel-xe spike] robustness net (?safeCollide=1). Off => byte-identical to
// before. On => floor rho before dividing and cap |u| below the LBM Mach
// limit, so a single fp-tipped cell (Gen12LP rounding differs from NVIDIA at
// this near-omega=2 relaxation) cannot cascade to a full blow-up. Only engages
// on already-pathological cells, so the stable regime is unaffected.
override SAFE_COLLIDE : u32 = 0u;
const BLOCK = 8u;

const ex = array<i32,9>( 0, 1, 0,-1, 0, 1,-1,-1, 1);
const ey = array<i32,9>( 0, 0, 1, 0,-1, 1, 1,-1,-1);
const wt = array<f32,9>(
  0.44444444f,
  0.11111111f, 0.11111111f, 0.11111111f, 0.11111111f,
  0.02777778f, 0.02777778f, 0.02777778f, 0.02777778f
);

// Block-major linear index for a cell at BUFFER coordinates (cx, cy).
// W and H are always exact multiples of BLOCK (resLog2 clamps W,H to
// powers of two >= 64), so this partitions the buffer exactly.
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

    // Algebraic distance approximation for ellipse
    let d = sqrt((lx*lx)/(state.a*state.a) + (ly*ly)/(state.b*state.b)) - 1.0;
    return d * state.b;
}

fn get_chi(phi: f32) -> f32 {
    let epsilon = 1.5f;
    return 0.5f * (1.0f - tanh(phi / epsilon));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let cx = gid.x; let cy = gid.y;
  if (cx >= W || cy >= H) { return; }

  // Window/physical coordinates: invert the moving-window shift.
  let wx = (cx + W - u32(state.off_x)) % W;
  let wy = (cy + H - u32(state.off_y)) % H;

  // 1. Pull Streaming: buffer-space neighbor (see file header derivation).
  var f: array<f32,9>;
  for (var i = 0u; i < 9u; i++) {
    let bx_src = (cx + W - u32(ex[i])) % W;
    let by_src = (cy + H - u32(ey[i])) % H;
    f[i] = f_in[i * (W * H) + cellIndex(bx_src, by_src)];
  }

  // 2. Local Macroscopic Variables
  var rho = 0f; var ux_star = 0f; var uy_star = 0f;
  for (var i = 0u; i < 9u; i++) {
    rho     += f[i];
    ux_star += f[i] * f32(ex[i]);
    uy_star += f[i] * f32(ey[i]);
  }
  ux_star /= rho; uy_star /= rho;

  var rhoDiv = rho;
  if (SAFE_COLLIDE != 0u) {
    // Floor density before it is used as a divisor below; recompute u* on the
    // floored value so a near-zero/negative rho can't produce Inf velocities.
    rhoDiv = max(rho, 1e-3f);
    ux_star = (ux_star * rho) / rhoDiv;
    uy_star = (uy_star * rho) / rhoDiv;
  }

  // 3. Penalty Force and Solid Coupling (window-space)
  let p = vec2<f32>(f32(wx), f32(wy));
  var rx = p.x - state.cx;
  var ry = p.y - state.cy;
  rx -= f32(W) * round(rx / f32(W));
  ry -= f32(H) * round(ry / f32(H));

  let usx = state.vx - state.omega * ry;
  let usy = state.vy + state.omega * rx;

  let phi = get_phi(p, state);
  let chi = get_chi(phi);

  // Penalty Force F = rho * chi * (Us - u*)
  let Fx = rho * chi * (usx - ux_star);
  let Fy = rho * chi * (usy - uy_star);

  // Actual fluid velocity u = u* + F/(2rho)
  var ux = ux_star + Fx / (2.0f * rhoDiv);
  var uy = uy_star + Fy / (2.0f * rhoDiv);
  if (SAFE_COLLIDE != 0u) {
    // Cap |u| below the LBM Mach limit -- keeps feq's (1 - 1.5*u_sq) term from
    // going negative and driving populations negative. 0.35 is well above any
    // physical velocity in the stable regime (author's run: |vy|~0.02), so
    // this is inert until a cell is already diverging.
    let UMAX = 0.35f;
    let m2 = ux*ux + uy*uy;
    if (m2 > UMAX*UMAX) { let s = UMAX * inverseSqrt(m2); ux = ux * s; uy = uy * s; }
  }
  let u_sq = ux*ux + uy*uy;

  // Store velocity for rendering (block-major buffer cell index)
  let cell = cellIndex(cx, cy);
  vel[cell * 2u] = ux; vel[cell * 2u + 1u] = uy;

  // 4. Collision and ALBC Sponge (window-space distance to window edges)
  let SPONGE_W = 4.0f;
  let dist_x = min(f32(wx), f32(W - 1u - wx));
  let dist_y = min(f32(wy), f32(H - 1u - wy));
  var sponge_weight = clamp(1.0f - min(dist_x, dist_y) / SPONGE_W, 0.0f, 1.0f);
  sponge_weight = sponge_weight * sponge_weight * (3.0f - 2.0f * sponge_weight);

  let omg = 1.0f / state.tau;
  for (var i = 0u; i < 9u; i++) {
    let exf = f32(ex[i]); let eyf = f32(ey[i]);
    let eu  = exf*ux + eyf*uy;
    let feq = wt[i] * rho * (1f + 3f*eu + 4.5f*eu*eu - 1.5f*u_sq);

    // Guo's Source Term Si = (1 - 1/(2tau)) * wi * [ (ei-u)/cs2 + (ei.u)/cs4 * ei ] . F
    let term1x = (exf - ux) * 3.0f;
    let term1y = (eyf - uy) * 3.0f;
    let term2  = (exf*ux + eyf*uy) * 9.0f;
    let Si = (1.0f - 0.5f * omg) * wt[i] * ( (term1x + term2*exf)*Fx + (term1y + term2*eyf)*Fy );

    let f_collide = f[i] - omg * (f[i] - feq) + Si;
    let f_target  = wt[i] * 1.0f; // rho=1.0, u=0 equilibrium

    f_out[i * (W * H) + cell] = mix(f_collide, f_target, sponge_weight);
  }
}
