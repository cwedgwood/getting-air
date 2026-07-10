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

@group(0) @binding(0) var<storage, read> vel         : array<f32>;
@group(0) @binding(1) var<storage, read> state       : CardState;
@group(0) @binding(2) var<storage, read> vel_pool    : array<f32>; // Milestone 4: fine pool
@group(0) @binding(3) var<storage, read> blockSlot   : array<i32>; // Milestone 4: coarse block -> pool slot

override W : u32;
override H : u32;
const BLOCK = 8u;

// Milestone 4 (plans/AMR.md): fine-region visual overlay, so a seam or
// discontinuity at the coarse-fine interface is immediately visible rather
// than only showing up in a numerical diff. Pool-aware -- supersedes
// Milestone 2's single-fixed-region version.
override RB : u32;
const GHOST = 2u;

// Block-major linear index for a cell at BUFFER coordinates (cx, cy).
// See amr_step.wgsl for the full derivation; vel is laid out this way
// (Milestone 1, plans/AMR.md) instead of flat row-major.
fn cellIndex(cx: u32, cy: u32) -> u32 {
  let nbx = W / BLOCK;
  let bx = cx / BLOCK; let by = cy / BLOCK;
  let lx = cx % BLOCK; let ly = cy % BLOCK;
  let blockID = by * nbx + bx;
  return blockID * (BLOCK * BLOCK) + ly * BLOCK + lx;
}

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
    return vel[cellIndex(bx, by) * 2u + 1u];
}

fn get_ux(x: i32, y: i32) -> f32 {
    let wx = (u32(x) + W) % W;
    let wy = (u32(y) + H) % H;
    let bx = (wx + u32(state.off_x)) % W;
    let by = (wy + u32(state.off_y)) % H;
    return vel[cellIndex(bx, by) * 2u];
}

fn wrapf(v: f32, n: f32) -> f32 {
    var r = v % n;
    if (r < 0.0) { r += n; }
    return r;
}

// Continuous window coords -> (fine-local x, fine-local y, pool slot).
// slot = -1 if the coarse block this window position falls in isn't
// currently refined. Buffer-space-native lookup (adds off_x/off_y once,
// same inversion amr_step1.wgsl's kernel uses), matching Milestone 4's
// pool addressing scheme -- see amr_interp_c2f.wgsl's file header.
fn windowToPool(wx: f32, wy: f32) -> vec3<i32> {
    let bufX = wrapf(wx + state.off_x, f32(W));
    let bufY = wrapf(wy + state.off_y, f32(H));
    let nbx = W / BLOCK;
    // Resolve the containing block by NEAREST coarse-cell center (+0.5 before
    // truncation): coarse cells are integer-centered (see amr_step1.wgsl's
    // fineToCoarseUnit -- the two fine cells straddling coarse cell c sit at
    // c-0.25 and c+0.25), so a plain u32(bufX)/RB mis-assigns the outer half of
    // each boundary coarse cell to the wrong block.
    let blockBX = u32(wrapf(bufX + 0.5, f32(W))) / RB;
    let blockBY = u32(wrapf(bufY + 0.5, f32(H))) / RB;
    let blockID = i32(blockBY * nbx + blockBX);
    let slot = blockSlot[blockID];
    if (slot < 0) { return vec3<i32>(0, 0, -1); }
    let originX = blockBX * RB;
    let originY = blockBY * RB;
    let fxc = f32(GHOST) + 2.0 * (bufX - f32(originX)) + 0.5;
    let fyc = f32(GHOST) + 2.0 * (bufY - f32(originY)) + 0.5;
    return vec3<i32>(i32(round(fxc)), i32(round(fyc)), slot);
}

// Fine-pool velocity at a WINDOW position, resolving which block actually
// contains that position (buffer-space, same off_x/off_y inversion as
// windowToPool). Returns (ux, uy, valid); valid=0 if that position isn't in
// a currently-refined block. Because each lookup is resolved independently,
// a vorticity stencil tap that crosses a seam lands in the NEIGHBOR block's
// REAL interior rather than this block's stale ghost ring -- so a chain of
// refined blocks samples as one contiguous fine region.
fn fineVelAt(wx: f32, wy: f32) -> vec3<f32> {
    let bufX = wrapf(wx + state.off_x, f32(W));
    let bufY = wrapf(wy + state.off_y, f32(H));
    let nbx = W / BLOCK;
    // Nearest coarse-cell-center block resolution (+0.5 before truncation) --
    // see windowToPool. Without it the outer half-coarse-cell ring of a block
    // resolves to the wrong block and samples its own stale ghost cell instead
    // of the neighbor's real interior, leaving a residual half-cell seam and
    // defeating the coarse fallback at a true fine/coarse perimeter. With it,
    // every position (and each +/-0.5 stencil tap) maps to a REAL cell [2,17]
    // of the correct block.
    let bBX = u32(wrapf(bufX + 0.5, f32(W))) / RB;
    let bBY = u32(wrapf(bufY + 0.5, f32(H))) / RB;
    let slot = blockSlot[i32(bBY * nbx + bBX)];
    if (slot < 0) { return vec3<f32>(0.0, 0.0, 0.0); }
    let originX = bBX * RB;
    let originY = bBY * RB;
    let fxc = f32(GHOST) + 2.0 * (bufX - f32(originX)) + 0.5;
    let fyc = f32(GHOST) + 2.0 * (bufY - f32(originY)) + 0.5;
    let FB = RB * 2u + 2u * GHOST;
    let cx = u32(clamp(i32(round(fxc)), 0, i32(FB) - 1));
    let cy = u32(clamp(i32(round(fyc)), 0, i32(FB) - 1));
    let cell = u32(slot) * (FB * FB) + cy * FB + cx;
    return vec3<f32>(vel_pool[cell * 2u], vel_pool[cell * 2u + 1u], 1.0);
}

@fragment
fn fs_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
  let fx = uv.x * f32(W); let fy = (1.0 - uv.y) * f32(H);
  let ix = i32(fx); let iy = i32(fy);

  let chi = get_chi(get_phi(vec2(fx, fy), state));

  // Discrete vorticity: du_y/dx - du_x/dy, coarse (always computed as the
  // fallback -- see below).
  var omega = (get_uy(ix + 1, iy) - get_uy(ix - 1, iy)) * 0.5f
            - (get_ux(ix, iy + 1) - get_ux(ix, iy - 1)) * 0.5f;

  // Milestone 4 (plans/AMR.md): sample the fine pool instead of the coarse
  // grid wherever this window position falls in a currently-refined block.
  // Each vorticity stencil tap is resolved to whichever block contains it
  // (see fineVelAt), so a tap at a seam crosses into the neighbor's REAL
  // interior and a chain of refined blocks renders as one contiguous fine
  // region -- no coarse-sampled stripe (periodic block-pitch seam) at every
  // internal block edge. Only where a tap lands in a genuinely COARSE
  // neighbor (the true fine/coarse perimeter) is any tap invalid, in which
  // case we keep the coarse omega computed above.
  let pool = windowToPool(fx, fy);
  let c0 = fineVelAt(fx, fy);
  if (c0.z > 0.5) {
    // Fine cells are 0.5 coarse units apart, so a +/-0.5 window step is
    // exactly one fine cell; central difference over +/-1 fine cell (factor
    // 1/(2*0.5)=1, matching the pre-existing fine path).
    let xp = fineVelAt(fx + 0.5, fy);
    let xm = fineVelAt(fx - 0.5, fy);
    let yp = fineVelAt(fx, fy + 0.5);
    let ym = fineVelAt(fx, fy - 0.5);
    if (xp.z > 0.5 && xm.z > 0.5 && yp.z > 0.5 && ym.z > 0.5) {
      omega = (xp.y - xm.y) - (yp.x - ym.x);
    }
  }

  // Blue for clockwise (negative), red for counter-clockwise (positive)
  let val = clamp(omega * 80.0f, -1.0f, 1.0f);
  var c: vec3<f32>;
  if (val > 0.0) {
    c = mix(vec3(0.05, 0.05, 0.1), vec3(1.0, 0.3, 0.2), val);
  } else {
    c = mix(vec3(0.05, 0.05, 0.1), vec3(0.2, 0.5, 1.0), -val);
  }

  // Refined-block coverage overlay: additive green (not a mix toward gray --
  // a mix is barely visible against the near-black low-vorticity
  // background where coverage most needs to be legible) over the whole
  // coarse block footprint currently holding a pool slot (pool.z>=0), not
  // just its sampled interior. Reuses pool.z already computed above at no
  // extra cost. Canvas output is unorm, so this saturates harmlessly in
  // already-bright (high-vorticity or solid-body) regions.
  if (pool.z >= 0) {
    c += vec3(0.0, 0.22, 0.0);
  }

  // Blend with solid color
  let solid_color = vec3(1.0, 0.8, 0.4);
  c = mix(c, solid_color, chi);

  return vec4(c, 1.0);
}
