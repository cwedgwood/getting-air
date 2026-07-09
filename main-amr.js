// AMR dev build. Starts as a literal fork of main.js (same physics, same
// shaders content under new amr_*.wgsl filenames) so it can diverge without
// ever touching the reference sim in main.js/shaders/lbm_*.wgsl. See
// plans/AMR.md Milestone 0. Adds: a pause/resume + snapshot/restore debug
// API (window.__AMR) for CDP-driven verification tooling (tools/amr-*.js),
// modeled on the vpm branch's window.__VPM / debugSnapshotSave/Load, and a
// per-frame GPU validation error scope -- the vpm branch hit a real silent
// CPU<->GPU transfer failure from a buffer declared with the wrong usage
// flags (commit 83d3c8c), so this build checks eagerly rather than
// discovering that kind of bug from wrong-looking output.

const canvas   = document.getElementById('c');
const statusEl = document.getElementById('status');

const urlParams = new URLSearchParams(window.location.search);
let resLog2 = parseInt(urlParams.get('res')) || 8;
if (resLog2 < 6) resLog2 = 6;
if (resLog2 > 11) resLog2 = 11;

let W = 1 << resLog2;
let H = W;
let NCELLS = W * H;

// ── Milestone 4 (plans/AMR.md): dynamic refinement via a fixed-capacity ───
// fine-block pool. Supersedes Milestone 2's single hardcoded fine region:
// refinement now happens at M1's own 8x8 coarse-block granularity, and any
// of MAX_FINE_BLOCKS pool slots can be assigned to any coarse block via
// blockSlot[]/slotToBlock[] indirection. Buffer-space-native throughout
// (unlike M2, which was window-anchored) -- M1's coarse blocks are already
// buffer-space, so this stays consistent; only the fine-level step kernel
// needs window coordinates, for the card SDF specifically.
const GHOST = 2;       // ghost layers per side, matches the 2-fine-substeps requirement
const BLOCK = 8;       // coarse block size (matches M1's cellIndex)
const RB = BLOCK;      // refine block size in coarse cells -- refine at block granularity
const FB = RB * 2 + 2 * GHOST; // per-slot fine buffer side length (20 for RB=8,GHOST=2)
const NCELLS1 = FB * FB; // cells per pool slot
const MAX_FINE_BLOCKS = urlParams.has('maxFineBlocks') ? parseInt(urlParams.get('maxFineBlocks')) : 64;
const NBX = W / BLOCK, NBY = H / BLOCK, NBLOCKS = NBX * NBY; // coarse block grid

// ── Milestone 4b (plans/AMR.md): automatic vorticity-driven refinement ────
// Simplified AGAL Algorithm 3 for our 2-level case (see amr_criterion.wgsl/
// amr_manage.wgsl headers): a single refine threshold plus a lower coarsen
// threshold for hysteresis, both in log2|omega| units. Calibrated against
// an actual live run, not guessed: at step ~4096 (default IC, card still
// accelerating from rest) the true domain-wide max|omega| was only 0.0202
// (log2 ~= -5.63), measured directly from a debugSnapshotSave readback --
// the original guess of -5 never triggered any refinement at that stage.
// -6/-7 reliably triggers refinement tracking the wake. Still expect to
// retune as later milestones (larger domains, different A/B/tau) shift the
// sim's operating range.
const REFINE_EVERY = urlParams.has('refineEvery') ? parseInt(urlParams.get('refineEvery')) : 16;
const REFINE_THRESH = urlParams.has('refineThresh') ? parseFloat(urlParams.get('refineThresh')) : -6;
const COARSEN_THRESH = urlParams.has('coarsenThresh') ? parseFloat(urlParams.get('coarsenThresh')) : -7;
// [intel-xe spike] E1 discriminator: ?autoRefine=0 starts with refinement OFF
// so the AMR pool/management path is bypassed, isolating the coarse solver.
const INITIAL_AUTOREFINE = urlParams.get('autoRefine') !== '0';

const resSlider = document.getElementById('slider-RES');
const resVal    = document.getElementById('val-RES');
resSlider.value = resLog2;
resVal.textContent = W;
resSlider.onchange = () => {
  const url = new URL(window.location);
  url.searchParams.set('res', resSlider.value);
  window.location.href = url.href;
};
resSlider.oninput = () => {
  resVal.textContent = 1 << parseInt(resSlider.value);
};

// ── Pesavento & Wang (2004) physical parameters ───────────────────────────────
// These constants define the "regime" of the simulation (Falling Paper).

let A = 64, B = 8;
let I_STAR = 0.34;
let TAU = 0.509;
let U_T = 0.05;

let RHO_B, MASS, I_BODY, G_LU, G_EFF;

function recalculate() {
  RHO_B  = I_STAR * 2 * A**3 / (B * (A**2 + B**2));
  RHO_B  = Math.max(1.05, RHO_B);
  MASS   = RHO_B * Math.PI * A * B;
  I_BODY = RHO_B * Math.PI * A * B * (A**2 + B**2) / 4;
  G_LU   = U_T**2 / (Math.PI * B * (RHO_B - 1));
  G_EFF  = G_LU * (1 - 1 / RHO_B);
}
recalculate();

const FSCALE  = 1e4;

const EX = [0, 1, 0,-1, 0, 1,-1,-1, 1];
const EY = [0, 0, 1, 0,-1, 1, 1,-1,-1];
const WT = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36];

function feq(rho, ux, uy, i) {
  const eu = EX[i]*ux + EY[i]*uy;
  return WT[i] * rho * (1 + eu*3 + eu*eu*4.5 - (ux*ux+uy*uy)*1.5);
}

function initF() {
  const f = new Float32Array(NCELLS * 9);
  for (let c = 0; c < NCELLS; c++) {
    for (let i = 0; i < 9; i++) {
      f[i * NCELLS + c] = feq(1, 0, 0, i);
    }
  }
  return f;
}

// The IC is spatially uniform (rho=1, u=0 everywhere), so the fine grid's
// t=0 state is trivially also uniform equilibrium -- interpolating a
// uniform coarse field gives back the same uniform field. No need for a
// real GPU interpolation dispatch at init.
// Fills the WHOLE pool (all MAX_FINE_BLOCKS slots), not just currently-
// assigned ones -- harmless since unassigned slots are never read (guarded
// by slotToBlock[slot]<0 in the shaders), and means a slot never holds
// uninitialized GPU memory between being freed and reassigned.
function initFPool() {
  const NPOOL = MAX_FINE_BLOCKS * NCELLS1;
  const f = new Float32Array(NPOOL * 9);
  for (let c = 0; c < NPOOL; c++) {
    for (let i = 0; i < 9; i++) {
      f[i * NPOOL + c] = feq(1, 0, 0, i);
    }
  }
  return f;
}

function initCardState() {
  return new Float32Array([
    W/2, H/2, 0.2,   // cx, cy, theta
    0, 0, 0,         // vx, vy, omega
    0, 0, 0,         // fx, fy, tz
    MASS, I_BODY, G_EFF,
    A, B,
    0.3, 0.025,      // v_max, o_max
    W/2, H/2, 0.2,   // cx_old, cy_old, th_old
    TAU,             // tau
    0, 0,            // y_total, x_total
    0, 0, 0, 0       // off_x, off_y, off_x_old, off_y_old
  ]);
}

async function loadShader(device, path) {
  const r = await fetch(path + '?v=' + Date.now());
  if (!r.ok) throw new Error(`failed to load ${path}`);
  const code = await r.text();
  return device.createShaderModule({ code });
}

function handleErr(e) {
  statusEl.textContent = `error: ${e.message}`;
  statusEl.style.color = '#f77';
  console.error('WebGPU Error:', e);
}

// base64 chunked in 8192-byte pieces -- a single huge String.fromCharCode
// spread risks "Maximum call stack size exceeded" for larger grids.
function bytesToB64(bytes) {
  let binary = '';
  const chunk = 8192;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}
function b64ToFloat32(b64, floatCount) {
  const binary = atob(b64);
  const bytes = new Uint8Array(floatCount * 4);
  for (let i = 0; i < bytes.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

async function init() {
  if (!navigator.gpu) { statusEl.textContent = 'WebGPU not available'; return; }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { statusEl.textContent = 'No adapter'; return; }

  // Milestone 6 needs real per-level GPU timing; leave this on for the AMR
  // dev build from the start (main.js keeps it off with `0 &&` -- don't
  // touch that file, this is deliberately different here).
  const hasTimestamp = 0 && adapter.features.has('timestamp-query');

  // WebGPU devices default to the spec MINIMUM limits (128 MiB storage
  // buffer bindings, 256 MiB total buffer size) regardless of what the
  // adapter can actually do -- f_a/f_b (NCELLS*9*4 bytes) exceeds the
  // default storage-binding limit at any resolution >= ~1536^2 (144 MiB at
  // 2048^2). This is the "WebGPU allocation limit" this project has hit
  // before; it's a device-limit *request* that was never made, unrelated
  // to AMR block size (the coarse grid is still one dense NCELLS-sized
  // buffer through Milestone 2 -- AMR's actual memory-footprint payoff
  // doesn't land until Milestone 4's block pool). Request exactly what the
  // current resolution needs, capped at the adapter's real capability, and
  // fail with a clear message rather than a cryptic validation error if
  // the requested resolution genuinely exceeds this GPU.
  const DEFAULT_MAX_STORAGE_BINDING = 128 * 1024 * 1024;
  const DEFAULT_MAX_BUFFER_SIZE = 256 * 1024 * 1024;
  const neededBufferBytes = NCELLS * 9 * 4; // f_a/f_b: the largest storage-bound buffers
  if (neededBufferBytes > adapter.limits.maxStorageBufferBindingSize) {
    const mib = (b) => (b / 1048576).toFixed(0);
    statusEl.textContent = `error: ${W}x${H} needs a ${mib(neededBufferBytes)} MiB buffer binding, this GPU's max is ${mib(adapter.limits.maxStorageBufferBindingSize)} MiB`;
    return;
  }
  const requiredLimits = {
    maxStorageBufferBindingSize: Math.min(Math.max(neededBufferBytes, DEFAULT_MAX_STORAGE_BINDING), adapter.limits.maxStorageBufferBindingSize),
    maxBufferSize: Math.min(Math.max(neededBufferBytes, DEFAULT_MAX_BUFFER_SIZE), adapter.limits.maxBufferSize),
  };
  const device = await adapter.requestDevice({
    requiredFeatures: hasTimestamp ? ['timestamp-query'] : [],
    requiredLimits,
  });

  // [intel-xe spike] E0 diagnostics: dump adapter identity + the limits that
  // matter for the res>128 investigation, and surface a silent device loss.
  try {
    const info = adapter.info || {};
    console.log('[amr-spike] adapter.info', { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description });
    const limKeys = ['maxStorageBufferBindingSize', 'maxBufferSize', 'maxComputeWorkgroupStorageSize', 'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupsPerDimension', 'maxStorageBuffersPerShaderStage'];
    console.log('[amr-spike] adapter.limits', Object.fromEntries(limKeys.map(k => [k, adapter.limits[k]])));
    console.log('[amr-spike] granted device.limits', Object.fromEntries(['maxStorageBufferBindingSize', 'maxBufferSize'].map(k => [k, device.limits[k]])));
    console.log('[amr-spike] sizing', { W, H, NCELLS, NBX, NBY, NBLOCKS, MAX_FINE_BLOCKS, neededBufferBytes, initialAutoRefine: INITIAL_AUTOREFINE });
  } catch (e) { console.warn('[amr-spike] E0 diagnostics failed', e); }
  device.lost.then(info => console.error('[amr-spike] DEVICE LOST:', info.reason, info.message));
  device.addEventListener('uncapturederror', (e) => console.error('[amr-spike] uncapturederror:', e.error));

  const querySet = hasTimestamp ? device.createQuerySet({
    type: 'timestamp',
    count: 2
  }) : null;
  const queryResolveBuffer = hasTimestamp ? device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
  }) : null;
  const queryReadBuffer = hasTimestamp ? device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  }) : null;

  device.pushErrorScope('validation');

  const ctx = canvas.getContext('webgpu');
  const fmt = navigator.gpu.getPreferredCanvasFormat();

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width  = Math.round(canvas.clientWidth * dpr);
    canvas.height = Math.round(canvas.clientHeight * dpr);
    ctx.configure({ device, format: fmt, alphaMode: 'opaque' });
  }
  window.addEventListener('resize', resize);
  resize();

  const U = GPUBufferUsage;
  const fSize   = NCELLS * 9 * 4;
  // COPY_SRC added on both f buffers (main.js's f_b lacks it) so debug
  // snapshotting can read back whichever buffer is authoritative without
  // needing a bind-group-layout-specific copy path. Flagged explicitly
  // because this exact class of bug (buffer usage flags silently wrong)
  // already bit the vpm branch once (commit 83d3c8c).
  const f_a     = device.createBuffer({ size: fSize, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const f_b     = device.createBuffer({ size: fSize, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const velBuf  = device.createBuffer({ size: NCELLS * 2 * 4, usage: U.STORAGE | U.COPY_SRC });
  const forceBuf = device.createBuffer({ size: 16, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });

  // CardState: 26 floats = 104 bytes
  const cardStateBuf = device.createBuffer({ size: 104, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });

  // Fine-block pool (Milestone 4): MAX_FINE_BLOCKS slots of NCELLS1 cells
  // each, plain flat layout within a slot (block-major-of-slots overall,
  // matching amr_step1.wgsl's `slot*(FB*FB) + local` indexing). Size is
  // independent of coarse domain size -- this is the actual memory-
  // footprint payoff (see plans/AMR.md's Milestone 4 design note).
  const fSizePool = MAX_FINE_BLOCKS * NCELLS1 * 9 * 4;
  const finePoolF_a   = device.createBuffer({ size: fSizePool, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const finePoolF_b   = device.createBuffer({ size: fSizePool, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const finePoolVel   = device.createBuffer({ size: MAX_FINE_BLOCKS * NCELLS1 * 2 * 4, usage: U.STORAGE | U.COPY_SRC });
  // blockSlot[NBLOCKS]: coarse block -> pool slot, or -1. slotToBlock is
  // its inverse (pool slot -> coarse block, or -1 if free) -- both are
  // needed since the coarse-side management pass indexes by block and the
  // fine-side interp/step/average passes are dispatched per pool slot (see
  // plans/AMR.md's Milestone 4 design note on why).
  const blockSlotBuf   = device.createBuffer({ size: NBLOCKS * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const slotToBlockBuf = device.createBuffer({ size: MAX_FINE_BLOCKS * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });

  // Milestone 4b: automatic refinement bookkeeping.
  const blockCriterionBuf  = device.createBuffer({ size: NBLOCKS * 4, usage: U.STORAGE | U.COPY_DST });
  const freeListBuf        = device.createBuffer({ size: MAX_FINE_BLOCKS * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  // Single atomic<i32> counter -- how many slots are free (top-of-stack
  // index into freeList). Separate 4-byte buffer since WGSL atomics need
  // their own binding, matching shaders/lbm_force.wgsl's forces buffer.
  const freeCountBuf       = device.createBuffer({ size: 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
  const newlyActivatedBuf  = device.createBuffer({ size: MAX_FINE_BLOCKS * 4, usage: U.STORAGE | U.COPY_DST });

  device.queue.writeBuffer(cardStateBuf, 0, initCardState());
  device.queue.writeBuffer(f_a, 0, initF());
  device.queue.writeBuffer(finePoolF_a, 0, initFPool());
  device.queue.writeBuffer(blockSlotBuf, 0, new Int32Array(NBLOCKS).fill(-1));
  device.queue.writeBuffer(slotToBlockBuf, 0, new Int32Array(MAX_FINE_BLOCKS).fill(-1));
  device.queue.writeBuffer(freeListBuf, 0, new Int32Array(MAX_FINE_BLOCKS).map((_, i) => i));
  device.queue.writeBuffer(freeCountBuf, 0, new Int32Array([MAX_FINE_BLOCKS]));

  let paramsDirty = false;
  const updateGPUParams = () => {
    const data = new Float32Array([MASS, I_BODY, G_EFF, A, B]);
    device.queue.writeBuffer(cardStateBuf, 9 * 4, data);
    device.queue.writeBuffer(cardStateBuf, 19 * 4, new Float32Array([TAU]));
  };

  const sliders = [
    { id: 'A', setter: v => A = v, dp: 0 },
    { id: 'B', setter: v => B = v, dp: 0 },
    { id: 'I_STAR', setter: v => I_STAR = v, dp: 2 },
    { id: 'TAU', setter: v => TAU = v, dp: 3 },
    { id: 'U_T', setter: v => U_T = v, dp: 3 },
  ];
  sliders.forEach(s => {
    const el = document.getElementById(`slider-${s.id}`);
    const valEl = document.getElementById(`val-${s.id}`);
    el.oninput = () => {
      s.setter(parseFloat(el.value));
      valEl.textContent = el.value;
      recalculate();
      paramsDirty = true;
    };
  });

  const [stepSM, frcSM, phySM, renSM, interpSM, step1SM, avgSM, criterionSM, manageSM] = await Promise.all([
    loadShader(device, 'shaders/amr_step.wgsl'),
    loadShader(device, 'shaders/amr_force.wgsl'),
    loadShader(device, 'shaders/amr_physics.wgsl'),
    loadShader(device, 'shaders/amr_render.wgsl'),
    loadShader(device, 'shaders/amr_interp_c2f.wgsl'),
    loadShader(device, 'shaders/amr_step1.wgsl'),
    loadShader(device, 'shaders/amr_average_f2c.wgsl'),
    loadShader(device, 'shaders/amr_criterion.wgsl'),
    loadShader(device, 'shaders/amr_manage.wgsl'),
  ]);

  const stepBGL = device.createBindGroupLayout({ label: 'stepBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  const frcBGL = device.createBindGroupLayout({ label: 'frcBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  const phyBGL = device.createBindGroupLayout({ label: 'phyBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  const renBGL = device.createBindGroupLayout({ label: 'renBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }
  ]});

  // Milestone 4: interp (coarse->fine ghosts), fine step, average (fine->coarse),
  // all pool-aware (an extra read-only slotToBlock/blockSlot binding vs. M2).
  // Binding 4 (newlyActivated) is Milestone 4b: only read by the GHOST_ONLY=0
  // init pipeline, but must still be present in the layout both pipelines share.
  // Milestone 4c: binding 5 (blockSlot) added so a ghost cell can check
  // whether its edge-adjacent neighbor block is also currently refined (see
  // amr_interp_c2f.wgsl's file header on fine-fine ghost consultation).
  const interpBGL = device.createBindGroupLayout({ label: 'interpBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
  ]});
  // Milestone 4b: criterion (per-block vorticity max) and manage (refine/coarsen decision).
  const criterionBGL = device.createBindGroupLayout({ label: 'criterionBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  const manageBGL = device.createBindGroupLayout({ label: 'manageBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
  ]});
  const step1BGL = device.createBindGroupLayout({ label: 'step1BGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
  ]});
  const avgBGL = device.createBindGroupLayout({ label: 'avgBGL', entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
  ]});

  const constants = { W, H };
  const fineConstants = { W, H, RB };
  // GHOST_ONLY=1: steady-state ghost-only reinterpolation (every macro-step).
  // GHOST_ONLY=0: full-slot fill, used once on block activation (see debugActivateBlock).
  const interpConstants = { W, H, RB, GHOST_ONLY: 1 };
  const interpInitConstants = { W, H, RB, GHOST_ONLY: 0 };
  // Between-substep fine-fine-only ghost re-exchange (see amr_interp_c2f.wgsl's
  // FINE_FINE_ONLY note and the dispatch between f1a/f1b below).
  const interpFFConstants = { W, H, RB, GHOST_ONLY: 1, FINE_FINE_ONLY: 1 };
  const step1Constants = { W, H, RB };
  const criterionConstants = { W, H };
  const manageConstants = { W, H, REFINE_THRESH, COARSEN_THRESH };

  const stepPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [stepBGL] }),
    compute: { module: stepSM, entryPoint: 'main', constants }
  });
  const frcPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [frcBGL] }),
    compute: { module: frcSM, entryPoint: 'main', constants }
  });
  const phyPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [phyBGL] }),
    compute: { module: phySM, entryPoint: 'main', constants }
  });
  const renPL = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [renBGL] }),
    vertex: { module: renSM, entryPoint: 'vs_main', constants },
    fragment: { module: renSM, entryPoint: 'fs_main', targets: [{ format: fmt }], constants: fineConstants },
    primitive: { topology: 'triangle-list' },
  });
  const interpPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [interpBGL] }),
    compute: { module: interpSM, entryPoint: 'main', constants: interpConstants }
  });
  // Same module/entry point as interpPL, different override constant --
  // WGSL/WebGPU compiles this as a separate pipeline. Used once per newly-
  // activated slot to fill the whole region (no prior fine-level state to
  // evolve from), vs. interpPL's steady-state ghost-only reinterpolation.
  const interpInitPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [interpBGL] }),
    compute: { module: interpSM, entryPoint: 'main', constants: interpInitConstants }
  });
  // Fine-fine-only ghost re-exchange pipeline (same module, FINE_FINE_ONLY=1).
  const interpFFPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [interpBGL] }),
    compute: { module: interpSM, entryPoint: 'main', constants: interpFFConstants }
  });
  const step1PL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [step1BGL] }),
    compute: { module: step1SM, entryPoint: 'main', constants: step1Constants }
  });
  const avgPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [avgBGL] }),
    compute: { module: avgSM, entryPoint: 'main', constants: fineConstants }
  });
  const criterionPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [criterionBGL] }),
    compute: { module: criterionSM, entryPoint: 'main', constants: criterionConstants }
  });
  // Two pipelines, same module, different entry points -- dispatched as two
  // SEPARATE passes (coarsen fully completing before refine starts) to
  // avoid a same-dispatch free-list race. See amr_manage.wgsl's header for
  // the bug this fixes (found by this milestone's own validation).
  const manageCoarsenPL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [manageBGL] }),
    compute: { module: manageSM, entryPoint: 'coarsen', constants: manageConstants }
  });
  const manageRefinePL = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [manageBGL] }),
    compute: { module: manageSM, entryPoint: 'refine', constants: manageConstants }
  });

  const stepBG_ab = device.createBindGroup({ layout: stepBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: f_b } }, { binding: 3, resource: { buffer: velBuf } }]});
  const stepBG_ba = device.createBindGroup({ layout: stepBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: f_a } }, { binding: 3, resource: { buffer: velBuf } }]});

  const frcBG_a = device.createBindGroup({ layout: frcBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: forceBuf } }]});
  const frcBG_b = device.createBindGroup({ layout: frcBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: forceBuf } }]});

  const phyBG = device.createBindGroup({ layout: phyBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: forceBuf } }]});
  const renBG = device.createBindGroup({ layout: renBGL, entries: [{ binding: 0, resource: { buffer: velBuf } }, { binding: 1, resource: { buffer: cardStateBuf } }, { binding: 2, resource: { buffer: finePoolVel } }, { binding: 3, resource: { buffer: blockSlotBuf } }]});

  // Milestone 4 bind groups (pool-aware, superseding M2's single-region ones).
  // interp always WRITES finePoolF_a (the pool's current-at-macro-step-
  // boundary buffer, mirroring f_a's own invariant -- 2 fine substeps per
  // macro-step is even), but READS whichever coarse buffer is "current"
  // this macro-step (same source the force pass reads).
  const interpBG_readA = device.createBindGroup({ layout: interpBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: finePoolF_a } }, { binding: 3, resource: { buffer: slotToBlockBuf } }, { binding: 4, resource: { buffer: newlyActivatedBuf } }, { binding: 5, resource: { buffer: blockSlotBuf } }]});
  const interpBG_readB = device.createBindGroup({ layout: interpBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: finePoolF_a } }, { binding: 3, resource: { buffer: slotToBlockBuf } }, { binding: 4, resource: { buffer: newlyActivatedBuf } }, { binding: 5, resource: { buffer: blockSlotBuf } }]});
  // Fine ping-pong within a macro-step is a fixed 2-call sequence (ab then
  // ba), not a persistent toggle like the coarse useB -- always call both,
  // in order, every macro-step.
  const step1BG_ab = device.createBindGroup({ layout: step1BGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: finePoolF_a } }, { binding: 2, resource: { buffer: finePoolF_b } }, { binding: 3, resource: { buffer: finePoolVel } }, { binding: 4, resource: { buffer: slotToBlockBuf } }]});
  const step1BG_ba = device.createBindGroup({ layout: step1BGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: finePoolF_b } }, { binding: 2, resource: { buffer: finePoolF_a } }, { binding: 3, resource: { buffer: finePoolVel } }, { binding: 4, resource: { buffer: slotToBlockBuf } }]});
  // Fine-fine-only ghost re-exchange, run BETWEEN f1a and f1b. f1a writes the
  // post-substep-1 pool into finePoolF_b (the buffer f1b then reads), so this
  // refreshes each block's fine-fine seam ghosts IN PLACE in finePoolF_b from
  // the neighbor's just-updated interior. binding 1 (f_coarse) is unused in
  // FINE_FINE_ONLY mode; f_a is bound only to satisfy the shared layout.
  const interpFFBG_b = device.createBindGroup({ layout: interpBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: finePoolF_b } }, { binding: 3, resource: { buffer: slotToBlockBuf } }, { binding: 4, resource: { buffer: newlyActivatedBuf } }, { binding: 5, resource: { buffer: blockSlotBuf } }]});
  // average always READS finePoolF_a (pool is current again after 2
  // substeps) but WRITES whichever coarse buffer the coarse step just
  // wrote this macro-step -- named by target, matching stepBG_ba being the
  // one that writes f_a.
  const avgBG_targetA = device.createBindGroup({ layout: avgBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: finePoolF_a } }, { binding: 2, resource: { buffer: f_a } }, { binding: 3, resource: { buffer: slotToBlockBuf } }]});
  const avgBG_targetB = device.createBindGroup({ layout: avgBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: finePoolF_a } }, { binding: 2, resource: { buffer: f_b } }, { binding: 3, resource: { buffer: slotToBlockBuf } }]});
  // Init variant (GHOST_ONLY=0, fills the whole slot): only ever called on
  // a just-activated slot immediately after coarse->fine interpolation
  // logically depends on the CURRENT coarse state, i.e. same source
  // selection as the steady-state interp bind groups above.
  const interpInitBG_readA = device.createBindGroup({ layout: interpBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_a } }, { binding: 2, resource: { buffer: finePoolF_a } }, { binding: 3, resource: { buffer: slotToBlockBuf } }, { binding: 4, resource: { buffer: newlyActivatedBuf } }, { binding: 5, resource: { buffer: blockSlotBuf } }]});
  const interpInitBG_readB = device.createBindGroup({ layout: interpBGL, entries: [{ binding: 0, resource: { buffer: cardStateBuf } }, { binding: 1, resource: { buffer: f_b } }, { binding: 2, resource: { buffer: finePoolF_a } }, { binding: 3, resource: { buffer: slotToBlockBuf } }, { binding: 4, resource: { buffer: newlyActivatedBuf } }, { binding: 5, resource: { buffer: blockSlotBuf } }]});

  // Milestone 4b bind groups.
  const criterionBG = device.createBindGroup({ layout: criterionBGL, entries: [{ binding: 0, resource: { buffer: velBuf } }, { binding: 1, resource: { buffer: blockCriterionBuf } }]});
  const manageBG = device.createBindGroup({ layout: manageBGL, entries: [{ binding: 0, resource: { buffer: blockCriterionBuf } }, { binding: 1, resource: { buffer: blockSlotBuf } }, { binding: 2, resource: { buffer: slotToBlockBuf } }, { binding: 3, resource: { buffer: freeListBuf } }, { binding: 4, resource: { buffer: freeCountBuf } }, { binding: 5, resource: { buffer: newlyActivatedBuf } }]});

  const error = await device.popErrorScope();
  if (error) { handleErr(error); return; }

  const WGX = Math.ceil(W / 8), WGY = Math.ceil(H / 8);
  // Milestone 4: interp/fine-step dispatch over (tile, tile, pool slot) --
  // cost scales with MAX_FINE_BLOCKS, not domain size (see plans/AMR.md's
  // Milestone 4 design note). average dispatches one workgroup per slot
  // exactly (RB*RB=8*8=64 cells = 1 workgroup, see amr_average_f2c.wgsl).
  const WGX1 = Math.ceil(FB / 8), WGY1 = Math.ceil(FB / 8);
  // Milestone 4b: manage dispatches one thread per coarse block.
  const WG_MANAGE = Math.ceil(NBLOCKS / 64);
  const STEPS_PER_FRAME = 64;
  let step = 0, lastT = performance.now();
  let useB = false;
  let liveMode = true;
  let autoRefine = INITIAL_AUTOREFINE; // Milestone 4b: on by default so refinement (and its coverage overlay) is visible without a console command; setAutoRefine(false) to disable for manual debugActivateBlock/debugDeactivateBlock testing. [intel-xe spike] initial value honors ?autoRefine=0.
  let macroStepCounter = 0;

  const trajectory = [];

  document.getElementById('download').onclick = () => {
    const header = "step,cx,cy_total,cx_total,theta,vx,vy,omega,fx,fy,tz\n";
    const rows = trajectory.map(r => r.map(v => v.toFixed(6)).join(",")).join("\n");
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trajectory_amr_${W}x${H}.csv`;
    a.click();
  };

  // Triple-buffering for readbacks to avoid CPU-GPU stalls
  const STAGES = 3;
  const stages = Array.from({ length: STAGES }, () => ({
    card: device.createBuffer({ size: 104, usage: U.MAP_READ | U.COPY_DST }),
    query: hasTimestamp ? device.createBuffer({ size: 16, usage: U.MAP_READ | U.COPY_DST }) : null,
    inFlight: false,
    step: 0
  }));
  let currentStageIdx = 0;

  const mlupsEl = document.getElementById('val-mlups');
  const gpuMsEl = document.getElementById('val-gpu-ms');
  const syncMsEl = document.getElementById('val-sync-ms');

  // ── Debug/verification support (window.__AMR) ────────────────────────────
  // Dedicated staging buffers, separate from the triple-buffered readback
  // stages above, so debug reads can't race frame()'s own in-flight readback.
  const stagingF     = device.createBuffer({ size: fSize, usage: U.MAP_READ | U.COPY_DST });
  const stagingVel   = device.createBuffer({ size: NCELLS * 2 * 4, usage: U.MAP_READ | U.COPY_DST });
  const stagingCard  = device.createBuffer({ size: 104, usage: U.MAP_READ | U.COPY_DST });
  const stagingFPool   = device.createBuffer({ size: fSizePool, usage: U.MAP_READ | U.COPY_DST });
  const stagingVelPool = device.createBuffer({ size: MAX_FINE_BLOCKS * NCELLS1 * 2 * 4, usage: U.MAP_READ | U.COPY_DST });
  const stagingBlockSlot   = device.createBuffer({ size: NBLOCKS * 4, usage: U.MAP_READ | U.COPY_DST });
  const stagingSlotToBlock = device.createBuffer({ size: MAX_FINE_BLOCKS * 4, usage: U.MAP_READ | U.COPY_DST });

  // Invariant this relies on: STEPS_PER_FRAME is even, so useB always
  // returns to its initial value (false) at a frame boundary, meaning f_a
  // (not f_b) is always the authoritative/current buffer whenever no frame
  // is mid-flight. Only call snapshot save/load while liveMode is false.
  async function debugSnapshotSave() {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(f_a, 0, stagingF, 0, fSize);
    enc.copyBufferToBuffer(velBuf, 0, stagingVel, 0, NCELLS * 2 * 4);
    enc.copyBufferToBuffer(cardStateBuf, 0, stagingCard, 0, 104);
    enc.copyBufferToBuffer(finePoolF_a, 0, stagingFPool, 0, fSizePool);
    enc.copyBufferToBuffer(finePoolVel, 0, stagingVelPool, 0, MAX_FINE_BLOCKS * NCELLS1 * 2 * 4);
    enc.copyBufferToBuffer(blockSlotBuf, 0, stagingBlockSlot, 0, NBLOCKS * 4);
    enc.copyBufferToBuffer(slotToBlockBuf, 0, stagingSlotToBlock, 0, MAX_FINE_BLOCKS * 4);
    device.queue.submit([enc.finish()]);
    await Promise.all([
      stagingF.mapAsync(GPUMapMode.READ),
      stagingVel.mapAsync(GPUMapMode.READ),
      stagingCard.mapAsync(GPUMapMode.READ),
      stagingFPool.mapAsync(GPUMapMode.READ),
      stagingVelPool.mapAsync(GPUMapMode.READ),
      stagingBlockSlot.mapAsync(GPUMapMode.READ),
      stagingSlotToBlock.mapAsync(GPUMapMode.READ),
    ]);
    const f = new Float32Array(stagingF.getMappedRange()).slice();
    const vel = new Float32Array(stagingVel.getMappedRange()).slice();
    const card = Array.from(new Float32Array(stagingCard.getMappedRange()).slice());
    const fPool = new Float32Array(stagingFPool.getMappedRange()).slice();
    const velPool = new Float32Array(stagingVelPool.getMappedRange()).slice();
    const blockSlotArr = Array.from(new Int32Array(stagingBlockSlot.getMappedRange()).slice());
    const slotToBlockArr = Array.from(new Int32Array(stagingSlotToBlock.getMappedRange()).slice());
    stagingF.unmap();
    stagingVel.unmap();
    stagingCard.unmap();
    stagingFPool.unmap();
    stagingVelPool.unmap();
    stagingBlockSlot.unmap();
    stagingSlotToBlock.unmap();

    const snapshot = {
      formatVersion: 4,
      // 'block8': f/vel are laid out in fixed 8x8 buffer-space cell-blocks
      // (see shaders/amr_step.wgsl's cellIndex, Milestone 1 of
      // plans/AMR.md), not flat row-major -- tools/amr-diff.js needs this
      // tag to decode snapshots correctly.
      layout: 'block8',
      W, H, step,
      cardState: card,
      fB64: bytesToB64(new Uint8Array(f.buffer, f.byteOffset, f.byteLength)),
      velB64: bytesToB64(new Uint8Array(vel.buffer, vel.byteOffset, vel.byteLength)),
      params: { A, B, I_STAR, TAU, U_T, resLog2 },
      // Milestone 4: fine-block pool (supersedes M2's single fine region).
      // pool.f/vel are indexed slot*(FB*FB)+local (see amr_step1.wgsl).
      pool: {
        RB, GHOST, FB, MAX_FINE_BLOCKS, NBLOCKS,
        blockSlot: blockSlotArr, slotToBlock: slotToBlockArr,
        fB64: bytesToB64(new Uint8Array(fPool.buffer, fPool.byteOffset, fPool.byteLength)),
        velB64: bytesToB64(new Uint8Array(velPool.buffer, velPool.byteOffset, velPool.byteLength)),
      },
    };
    console.log('[AMR snapshot] saved', { W, H, step });
    return snapshot;
  }

  async function debugSnapshotLoad(snapshot) {
    if (snapshot.W !== W || snapshot.H !== H) {
      throw new Error(`snapshot is ${snapshot.W}x${snapshot.H}, page is ${W}x${H} -- reload with ?res=${Math.log2(snapshot.W)}`);
    }
    // Raw f_a/velBuf bytes are only meaningful under the layout they were
    // captured with (see debugSnapshotSave's 'layout' field) -- loading a
    // pre-Milestone-1 flat-row-major snapshot here would silently
    // reinterpret it as block-major and corrupt state with no thrown error,
    // exactly the class of silent-failure this project has learned to
    // guard against explicitly rather than discover from wrong output.
    if (snapshot.layout !== 'block8') {
      throw new Error(`snapshot layout is '${snapshot.layout}', this build expects 'block8'`);
    }
    const f = b64ToFloat32(snapshot.fB64, NCELLS * 9);
    const vel = b64ToFloat32(snapshot.velB64, NCELLS * 2);
    device.queue.writeBuffer(f_a, 0, f.buffer, f.byteOffset, fSize);
    // velBuf is a separate GPU buffer, not derived from f_a by anything
    // debugSnapshotLoad itself runs -- omitting this write left it holding
    // whatever was there before the load (stale ux/uy from a prior run)
    // until the next real step overwrote it. Caught by amr-diff.js: rho
    // (derived from f in the diff tool) round-tripped exactly, but ux/uy
    // (read from velBuf) didn't -- the asymmetry was the tell.
    device.queue.writeBuffer(velBuf, 0, vel.buffer, vel.byteOffset, NCELLS * 2 * 4);
    device.queue.writeBuffer(cardStateBuf, 0, new Float32Array(snapshot.cardState));
    if (snapshot.pool) {
      if (snapshot.pool.RB !== RB || snapshot.pool.MAX_FINE_BLOCKS !== MAX_FINE_BLOCKS || snapshot.pool.NBLOCKS !== NBLOCKS) {
        throw new Error(`snapshot pool (RB=${snapshot.pool.RB},MAX_FINE_BLOCKS=${snapshot.pool.MAX_FINE_BLOCKS},NBLOCKS=${snapshot.pool.NBLOCKS}) doesn't match this page's (RB=${RB},MAX_FINE_BLOCKS=${MAX_FINE_BLOCKS},NBLOCKS=${NBLOCKS})`);
      }
      const fPool = b64ToFloat32(snapshot.pool.fB64, MAX_FINE_BLOCKS * NCELLS1 * 9);
      const velPool = b64ToFloat32(snapshot.pool.velB64, MAX_FINE_BLOCKS * NCELLS1 * 2);
      device.queue.writeBuffer(finePoolF_a, 0, fPool.buffer, fPool.byteOffset, fSizePool);
      device.queue.writeBuffer(finePoolVel, 0, velPool.buffer, velPool.byteOffset, MAX_FINE_BLOCKS * NCELLS1 * 2 * 4);
      device.queue.writeBuffer(blockSlotBuf, 0, new Int32Array(snapshot.pool.blockSlot));
      device.queue.writeBuffer(slotToBlockBuf, 0, new Int32Array(snapshot.pool.slotToBlock));
      // Sync the CPU-side mirrors debugActivateBlock/debugDeactivateBlock
      // rely on -- omitting this would leave them reflecting whatever was
      // active before the load, not what the loaded snapshot actually has,
      // exactly the class of GPU/CPU-state desync bug this project has
      // already been bitten by once (see debugSnapshotSave's velBuf note).
      blockSlotCPU.set(snapshot.pool.blockSlot);
      slotToBlockCPU.set(snapshot.pool.slotToBlock);
      freeSlots = [];
      for (let slot = 0; slot < MAX_FINE_BLOCKS; slot++) {
        if (slotToBlockCPU[slot] === -1) freeSlots.push(slot);
      }
      // Milestone 4b: the GPU-side freeList/freeCount (which the automatic
      // management pass owns) aren't part of the snapshot -- rebuild them
      // from the loaded slotToBlock instead of restoring a captured copy.
      // Free-list ORDER doesn't affect correctness (any permutation of the
      // free slots works equally as a stack), so this is exact, not an
      // approximation, and avoids growing the snapshot format for state
      // that's fully redundant with slotToBlock.
      device.queue.writeBuffer(freeListBuf, 0, new Int32Array(freeSlots));
      device.queue.writeBuffer(freeCountBuf, 0, new Int32Array([freeSlots.length]));
    }
    useB = false;
    step = snapshot.step;
    console.log('[AMR snapshot] loaded', { W, H, step });
    return { step };
  }

  // Milestone 2 macro-step (plans/AMR.md): 1 coarse step + 2 fine substeps,
  // ordered per AGAL's Fig. 13 recursive routine -- interpolate ghosts from
  // the CURRENT (pre-step) coarse state, then coarse-step and fine-step-x2
  // independently (both read only pre-step data, so their relative order
  // doesn't matter), then average the now-twice-advanced fine interior back
  // onto the coarse cells the coarse step just (less accurately) computed.
  // Factored out of frame()'s loop so debugStepSync can reuse it exactly --
  // duplicating this 7-pass sequence would risk the two silently drifting
  // apart.
  function dispatchMacroStep(enc) {
    const stepBG       = useB ? stepBG_ba          : stepBG_ab;
    const frcBG        = useB ? frcBG_b            : frcBG_a;
    const interpBG     = useB ? interpBG_readB     : interpBG_readA;
    const interpInitBG = useB ? interpInitBG_readB : interpInitBG_readA;
    const avgBG        = useB ? avgBG_targetA      : avgBG_targetB;

    // Milestone 4b: re-evaluate refinement every REFINE_EVERY macro-steps.
    // Runs BEFORE the steady-state interp pass below so a block refined
    // this round gets its one-time full-slot fill (interpInitPL, gated on
    // newlyActivated -- see amr_interp_c2f.wgsl) before anything else this
    // macro-step reads its pool slot. Reads velBuf as populated by the
    // PREVIOUS macro-step's coarse step, i.e. the same "current, pre-step"
    // data the force pass also reads.
    if (autoRefine && macroStepCounter % REFINE_EVERY === 0) {
      enc.clearBuffer(newlyActivatedBuf); // GPU-recorded, not queue.writeBuffer --
      // see plans/AMR.md's Milestone 4b note on why a JS-side writeBuffer
      // wouldn't interleave correctly with commands already recorded into
      // this same not-yet-submitted encoder.
      const crit = enc.beginComputePass(); crit.setPipeline(criterionPL); crit.setBindGroup(0, criterionBG); crit.dispatchWorkgroups(WGX, WGY); crit.end();
      // Coarsen MUST fully complete before refine starts -- see amr_manage.wgsl's header.
      const coarsenP = enc.beginComputePass(); coarsenP.setPipeline(manageCoarsenPL); coarsenP.setBindGroup(0, manageBG); coarsenP.dispatchWorkgroups(WG_MANAGE); coarsenP.end();
      const refineP = enc.beginComputePass(); refineP.setPipeline(manageRefinePL); refineP.setBindGroup(0, manageBG); refineP.dispatchWorkgroups(WG_MANAGE); refineP.end();
      const init = enc.beginComputePass(); init.setPipeline(interpInitPL); init.setBindGroup(0, interpInitBG); init.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); init.end();
    }
    macroStepCounter++;

    const frc = enc.beginComputePass(); frc.setPipeline(frcPL); frc.setBindGroup(0, frcBG); frc.dispatchWorkgroups(WGX, WGY); frc.end();
    const phy = enc.beginComputePass(); phy.setPipeline(phyPL); phy.setBindGroup(0, phyBG); phy.dispatchWorkgroups(1); phy.end();
    // Z dimension selects pool slot -- cost scales with MAX_FINE_BLOCKS, not domain size.
    const ipl = enc.beginComputePass(); ipl.setPipeline(interpPL); ipl.setBindGroup(0, interpBG); ipl.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); ipl.end();
    const stp = enc.beginComputePass(); stp.setPipeline(stepPL); stp.setBindGroup(0, stepBG); stp.dispatchWorkgroups(WGX, WGY); stp.end();
    const f1a = enc.beginComputePass(); f1a.setPipeline(step1PL); f1a.setBindGroup(0, step1BG_ab); f1a.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); f1a.end();
    // Refresh fine-fine seam ghosts from neighbors' post-substep-1 interiors so
    // the second fine substep couples to CURRENT neighbor state (a chain of
    // fine blocks then behaves like one contiguous fine region). Separate pass
    // => WebGPU barrier after f1a's writes, before f1b reads finePoolF_b.
    const ff = enc.beginComputePass(); ff.setPipeline(interpFFPL); ff.setBindGroup(0, interpFFBG_b); ff.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); ff.end();
    const f1b = enc.beginComputePass(); f1b.setPipeline(step1PL); f1b.setBindGroup(0, step1BG_ba); f1b.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS); f1b.end();
    // Exactly one workgroup per slot (RB*RB=64 cells = 1 workgroup, see amr_average_f2c.wgsl).
    const avg = enc.beginComputePass(); avg.setPipeline(avgPL); avg.setBindGroup(0, avgBG); avg.dispatchWorkgroups(1, 1, MAX_FINE_BLOCKS); avg.end();

    useB = !useB;
  }

  // CPU-side mirror of blockSlot/slotToBlock, kept in sync with the GPU
  // buffers via small writeBuffer calls on every activate/deactivate.
  // Sub-step A (plans/AMR.md's Milestone 4 "staged landing" note): manual
  // CPU-orchestrated activation, proving the pool addressing mechanism
  // works, before wiring up the automatic vorticity criterion.
  let blockSlotCPU = new Int32Array(NBLOCKS).fill(-1);
  let slotToBlockCPU = new Int32Array(MAX_FINE_BLOCKS).fill(-1);
  let freeSlots = Array.from({ length: MAX_FINE_BLOCKS }, (_, i) => i);

  function resetSim() {
    device.queue.writeBuffer(f_a, 0, initF());
    device.queue.writeBuffer(finePoolF_a, 0, initFPool());
    device.queue.writeBuffer(cardStateBuf, 0, initCardState());
    device.queue.writeBuffer(forceBuf, 0, new Int32Array([0, 0, 0, 0]));
    blockSlotCPU.fill(-1);
    slotToBlockCPU.fill(-1);
    device.queue.writeBuffer(blockSlotBuf, 0, blockSlotCPU);
    device.queue.writeBuffer(slotToBlockBuf, 0, slotToBlockCPU);
    freeSlots = Array.from({ length: MAX_FINE_BLOCKS }, (_, i) => i);
    device.queue.writeBuffer(freeListBuf, 0, new Int32Array(MAX_FINE_BLOCKS).map((_, i) => i));
    device.queue.writeBuffer(freeCountBuf, 0, new Int32Array([MAX_FINE_BLOCKS]));
    autoRefine = INITIAL_AUTOREFINE; // matches the initial state -- reset shouldn't silently flip it ([intel-xe spike] honors ?autoRefine=0)
    macroStepCounter = 0;
    useB = false;
    step = 0;
    trajectory.length = 0;
  }

  // Activates coarse block (bx,by) [0<=bx<NBX, 0<=by<NBY, buffer-space --
  // see plans/AMR.md's Milestone 4 design note on why block IDs are
  // buffer-space-native] against a free pool slot, filling the whole new
  // slot from the CURRENT coarse state (GHOST_ONLY=0 pipeline) since there
  // is no prior fine-level state for it to evolve from. Only valid while
  // liveMode is false, matching the debugSnapshotSave/Load convention --
  // dispatchMacroStep's useB toggling and this function's direct queue
  // writes would otherwise race the frame() loop's own encoder.
  // Reads blockSlot/slotToBlock directly from GPU -- the authoritative
  // source once Milestone 4b's automatic management can mutate pool state
  // without going through the CPU mirror at all. Reuses the same staging
  // buffers debugSnapshotSave uses; not safe to call concurrently with
  // another in-flight readback through those buffers, which is fine for an
  // interactive debug tool but worth noting if this ever needs to run on a
  // hot path.
  async function readPoolIndirection() {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(blockSlotBuf, 0, stagingBlockSlot, 0, NBLOCKS * 4);
    enc.copyBufferToBuffer(slotToBlockBuf, 0, stagingSlotToBlock, 0, MAX_FINE_BLOCKS * 4);
    device.queue.submit([enc.finish()]);
    await Promise.all([
      stagingBlockSlot.mapAsync(GPUMapMode.READ),
      stagingSlotToBlock.mapAsync(GPUMapMode.READ),
    ]);
    const blockSlot = new Int32Array(stagingBlockSlot.getMappedRange()).slice();
    const slotToBlock = new Int32Array(stagingSlotToBlock.getMappedRange()).slice();
    stagingBlockSlot.unmap();
    stagingSlotToBlock.unmap();
    return { blockSlot, slotToBlock };
  }

  // Milestone 4b: toggles automatic vorticity-driven refinement. Manual
  // debugActivateBlock/debugDeactivateBlock are guarded against running
  // while this is on (see below) -- both mutate blockSlotCPU/slotToBlockCPU/
  // freeSlots directly, which would race the GPU-side free-list the
  // automatic management pass owns while enabled. Turning it off resyncs
  // those CPU mirrors from a fresh GPU readback, since automatic management
  // may have changed pool state the CPU mirror never saw.
  async function setAutoRefine(v) {
    autoRefine = !!v;
    if (!autoRefine) {
      const { blockSlot, slotToBlock } = await readPoolIndirection();
      blockSlotCPU.set(blockSlot);
      slotToBlockCPU.set(slotToBlock);
      freeSlots = [];
      for (let slot = 0; slot < MAX_FINE_BLOCKS; slot++) {
        if (slotToBlockCPU[slot] === -1) freeSlots.push(slot);
      }
    }
  }

  async function debugActivateBlock(bx, by) {
    if (autoRefine) throw new Error('debugActivateBlock: disable autoRefine first (setAutoRefine(false)) -- manual activation would race the GPU-side free-list');
    if (bx < 0 || bx >= NBX || by < 0 || by >= NBY) {
      throw new Error(`block (${bx},${by}) out of range [0,${NBX})x[0,${NBY})`);
    }
    const blockID = by * NBX + bx;
    if (blockSlotCPU[blockID] !== -1) return { slot: blockSlotCPU[blockID], alreadyActive: true };
    if (freeSlots.length === 0) throw new Error(`pool exhausted (MAX_FINE_BLOCKS=${MAX_FINE_BLOCKS})`);
    const slot = freeSlots.pop();
    blockSlotCPU[blockID] = slot;
    slotToBlockCPU[slot] = blockID;
    device.queue.writeBuffer(blockSlotBuf, blockID * 4, new Int32Array([slot]));
    device.queue.writeBuffer(slotToBlockBuf, slot * 4, new Int32Array([blockID]));
    // BUGFIX: the GHOST_ONLY=0 pipeline's own guard (see amr_interp_c2f.wgsl)
    // is `if (GHOST_ONLY==0u && newlyActivated[slot]==0u) { return; }` --
    // without this write, every thread hits that guard and the dispatch
    // below silently does nothing, leaving the slot's fine pool at whatever
    // uniform-rest state initFPool() set it to. The automatic refine() path
    // in amr_manage.wgsl sets this correctly; this manual CPU-driven path
    // had never set it, meaning this debug function has been silently
    // non-functional (activating a slot without ever actually initializing
    // its fine data) since it was written. Reset back to 0 after dispatch,
    // matching the automatic path's per-round clearBuffer lifecycle.
    device.queue.writeBuffer(newlyActivatedBuf, slot * 4, new Uint32Array([1]));

    const interpInitBG = useB ? interpInitBG_readB : interpInitBG_readA;
    const enc = device.createCommandEncoder();
    const ipl = enc.beginComputePass();
    ipl.setPipeline(interpInitPL);
    ipl.setBindGroup(0, interpInitBG);
    ipl.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS);
    ipl.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    device.queue.writeBuffer(newlyActivatedBuf, slot * 4, new Uint32Array([0]));
    return { slot, alreadyActive: false };
  }

  // Deactivates coarse block (bx,by). No explicit "final average" needed:
  // the average pass already runs every macro-step while the block is
  // active, so the coarse cells already reflect the latest fine-derived
  // state as of the most recent macro-step -- deactivation just stops
  // future fine-level evolution and frees the slot for reuse.
  function debugDeactivateBlock(bx, by) {
    if (autoRefine) throw new Error('debugDeactivateBlock: disable autoRefine first (setAutoRefine(false)) -- manual deactivation would race the GPU-side free-list');
    if (bx < 0 || bx >= NBX || by < 0 || by >= NBY) {
      throw new Error(`block (${bx},${by}) out of range [0,${NBX})x[0,${NBY})`);
    }
    const blockID = by * NBX + bx;
    const slot = blockSlotCPU[blockID];
    if (slot === -1) return { wasActive: false };
    blockSlotCPU[blockID] = -1;
    slotToBlockCPU[slot] = -1;
    device.queue.writeBuffer(blockSlotBuf, blockID * 4, new Int32Array([-1]));
    device.queue.writeBuffer(slotToBlockBuf, slot * 4, new Int32Array([-1]));
    freeSlots.push(slot);
    return { wasActive: true, slot };
  }

  // TEMPORARY diagnostic (Milestone 4c investigation): writes a synthetic
  // f[0]=fx*100+fy marker into every pool cell, dispatches ONLY the
  // steady-state ghost-fill pass once (bypassing coarse step / fine step1 /
  // average entirely), and returns the resulting f[0] plane. Since the
  // marker survives untouched in every INTERIOR cell (this pass never
  // writes interior cells) and ghost cells get overwritten by whatever the
  // shader's neighbor-consultation logic picks, this directly reveals which
  // cell a ghost cell actually read from, with zero confounding from
  // streaming/collision. Remove once the fine-fine indexing bug is found.
  async function debugProbeGhostFill() {
    const NPOOL = MAX_FINE_BLOCKS * NCELLS1;
    const marker = new Float32Array(NPOOL * 9);
    for (let s = 0; s < MAX_FINE_BLOCKS; s++) {
      for (let fy = 0; fy < FB; fy++) {
        for (let fx = 0; fx < FB; fx++) {
          const cell = s * (FB * FB) + fy * FB + fx;
          marker[0 * NPOOL + cell] = fx * 100 + fy;
        }
      }
    }
    device.queue.writeBuffer(finePoolF_a, 0, marker);

    const enc = device.createCommandEncoder();
    const ipl = enc.beginComputePass();
    ipl.setPipeline(interpPL);
    ipl.setBindGroup(0, interpBG_readA);
    ipl.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS);
    ipl.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    const enc2 = device.createCommandEncoder();
    enc2.copyBufferToBuffer(finePoolF_a, 0, stagingFPool, 0, fSizePool);
    device.queue.submit([enc2.finish()]);
    await stagingFPool.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(stagingFPool.getMappedRange()).slice();
    stagingFPool.unmap();
    return Array.from(result.subarray(0, NPOOL));
  }

  // TEMPORARY diagnostic: dispatches ONLY the steady-state (GHOST_ONLY=1)
  // ghost-fill pass, in isolation, WITHOUT first overwriting finePoolF_a --
  // unlike debugProbeGhostFill (which stomps the pool with a marker
  // pattern), this preserves whatever real interior data debugActivateBlock
  // already seeded, so it can be used to test the fine-fine consultation
  // path (which only runs in GHOST_ONLY=1, never in debugActivateBlock's own
  // GHOST_ONLY=0 init dispatch) against a known synthetic field's already-
  // correctly-interpolated interior, isolating exactly the mechanism the
  // Phase 4c ghost-consultation code exercises in real macro-steps.
  async function debugRunSteadyGhostFill() {
    const enc = device.createCommandEncoder();
    const ipl = enc.beginComputePass();
    ipl.setPipeline(interpPL);
    ipl.setBindGroup(0, useB ? interpBG_readB : interpBG_readA);
    ipl.dispatchWorkgroups(WGX1, WGY1, MAX_FINE_BLOCKS);
    ipl.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  // TEMPORARY diagnostic (root-cause investigation of the pre-existing
  // coarse<->fine interface artifact): overwrites f_a with a Taylor-Green-
  // like analytic vortex field (ux=-A*sin(2*pi*y/L), uy=A*sin(2*pi*x/L),
  // rho=1) instead of the usual uniform rest state. Unlike a linear ramp,
  // this has genuine curvature AND nonzero, smoothly-varying vorticity
  // (omega = A*(2*pi/L)*(cos(2*pi*x/L)+cos(2*pi*y/L))), so any error the
  // coarse->fine interpolation introduces at a block boundary shows up
  // against a known analytic ground truth, not against chaotic real flow
  // structure that's hard to reason about. Buffer-space coordinates (no
  // window conversion -- off_x/off_y are 0 right after reset() anyway).
  function debugInjectSyntheticField(A, L) {
    const f = new Float32Array(NCELLS * 9);
    for (let by = 0; by < NBY; by++) {
      for (let bx = 0; bx < NBX; bx++) {
        for (let ly = 0; ly < BLOCK; ly++) {
          for (let lx = 0; lx < BLOCK; lx++) {
            const x = bx * BLOCK + lx, y = by * BLOCK + ly;
            const blockID = by * NBX + bx;
            const cell = blockID * (BLOCK * BLOCK) + ly * BLOCK + lx;
            const ux = -A * Math.sin(2 * Math.PI * y / L);
            const uy = A * Math.sin(2 * Math.PI * x / L);
            for (let i = 0; i < 9; i++) f[i * NCELLS + cell] = feq(1, ux, uy, i);
          }
        }
      }
    }
    device.queue.writeBuffer(f_a, 0, f);
  }

  // Always reads GPU state directly (not the CPU mirror, which goes stale
  // the instant autoRefine's automatic management mutates pool state
  // without the CPU ever seeing it) -- see readPoolIndirection.
  async function debugListActiveBlocks() {
    const { blockSlot } = await readPoolIndirection();
    const active = [];
    for (let blockID = 0; blockID < NBLOCKS; blockID++) {
      if (blockSlot[blockID] !== -1) {
        active.push({ bx: blockID % NBX, by: Math.floor(blockID / NBX), slot: blockSlot[blockID] });
      }
    }
    return active;
  }

  // Deterministic synchronous stepping, bypassing rAF entirely -- lets two
  // separate builds be driven to an EXACT matching step count for a fair
  // diff. Wall-clock polling of the normal rAF-driven `liveMode` loop can't
  // guarantee this: STEPS_PER_FRAME-sized jumps land unpredictably relative
  // to any external poll interval (confirmed directly while re-validating
  // Milestones 1 and 2 at 256x256 -- see plans/AMR.md).
  // TEMPORARY diagnostic: single-macro-step granularity (debugStepSync is
  // locked to STEPS_PER_FRAME=64-step batches), for bisecting exactly which
  // macro-step a divergence first appears on.
  async function debugStepOne() {
    liveMode = false;
    const enc = device.createCommandEncoder();
    dispatchMacroStep(enc);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    step += 1;
    return { step };
  }

  async function debugStepSync(n) {
    liveMode = false;
    for (let k = 0; k < n; k += STEPS_PER_FRAME) {
      const enc = device.createCommandEncoder();
      for (let s = 0; s < STEPS_PER_FRAME; s++) dispatchMacroStep(enc);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      step += STEPS_PER_FRAME;
    }
    return { step };
  }

  // [intel-xe spike] E2 smoking-gun test for the pool free-list race (H1):
  // read back blockSlot + slotToBlock and verify they are consistent mutual
  // inverses. Any duplicate-slot or inverse-mismatch => the fine-block pool
  // allocator (amr_manage.wgsl) corrupted the mapping, which is the "allocation
  // problem" candidate. Run after letting a res=8 sim churn on the failing GPU:
  //   await window.__AMR.checkPool()
  async function checkPool() {
    const { blockSlot, slotToBlock } = await readPoolIndirection();
    const conflicts = [];
    const slotOwner = new Map();
    let activeBlocks = 0;
    for (let b = 0; b < blockSlot.length; b++) {
      const s = blockSlot[b];
      if (s < 0) continue;
      activeBlocks++;
      if (s >= slotToBlock.length) { conflicts.push({ type: 'slot-out-of-range', block: b, slot: s }); continue; }
      if (slotToBlock[s] !== b) conflicts.push({ type: 'inverse-mismatch', block: b, slot: s, slotToBlock_says: slotToBlock[s] });
      if (slotOwner.has(s)) conflicts.push({ type: 'duplicate-slot', slot: s, blocks: [slotOwner.get(s), b] });
      else slotOwner.set(s, b);
    }
    let activeSlots = 0;
    for (let s = 0; s < slotToBlock.length; s++) {
      const b = slotToBlock[s];
      if (b < 0) continue;
      activeSlots++;
      if (b >= blockSlot.length || blockSlot[b] !== s) conflicts.push({ type: 'reverse-mismatch', slot: s, block: b, blockSlot_says: (b < blockSlot.length ? blockSlot[b] : 'oob') });
    }
    const report = { ok: conflicts.length === 0, step, activeBlocks, activeSlots, conflicts };
    console[report.ok ? 'log' : 'error']('[amr-spike] checkPool', report);
    return report;
  }

  // [intel-xe spike] instability quantifier: scan the coarse velocity buffer
  // for NaN/Inf and report magnitude extremes, so "behaves differently /
  // instability" becomes a number and E3 can locate the first bad step.
  //   await window.__AMR.health()
  let _healthStaging = null;
  async function health() {
    if (!_healthStaging) _healthStaging = device.createBuffer({ size: NCELLS * 2 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(velBuf, 0, _healthStaging, 0, NCELLS * 2 * 4);
    device.queue.submit([enc.finish()]);
    await _healthStaging.mapAsync(GPUMapMode.READ);
    const v = new Float32Array(_healthStaging.getMappedRange()).slice();
    _healthStaging.unmap();
    let nan = 0, inf = 0, maxAbs = 0;
    for (let i = 0; i < v.length; i++) {
      const x = v[i];
      if (Number.isNaN(x)) { nan++; continue; }
      if (!Number.isFinite(x)) { inf++; continue; }
      const a = Math.abs(x); if (a > maxAbs) maxAbs = a;
    }
    const report = { step, nan, inf, maxAbsVel: maxAbs, healthy: nan === 0 && inf === 0 };
    console[report.healthy ? 'log' : 'error']('[amr-spike] health', report);
    return report;
  }

  window.__AMR = {
    setLive: (v) => { liveMode = !!v; },
    isLive: () => liveMode,
    reset: resetSim,
    checkPool,
    health,
    getStep: () => step,
    getDims: () => ({ W, H }),
    debugSnapshotSave,
    debugSnapshotLoad,
    debugStepSync,
    debugStepOne,
    debugActivateBlock,
    debugDeactivateBlock,
    debugListActiveBlocks,
    debugProbeGhostFill,
    debugRunSteadyGhostFill,
    debugInjectSyntheticField,
    setAutoRefine,
    isAutoRefine: () => autoRefine,
    getBlockGridDims: () => ({ NBX, NBY, RB, MAX_FINE_BLOCKS }),
    getRefineParams: () => ({ REFINE_EVERY, REFINE_THRESH, COARSEN_THRESH }),
  };

  async function frame() {
    try {
      if (!liveMode) {
        requestAnimationFrame(() => frame().catch(handleErr));
        return;
      }
      if (paramsDirty) {
        updateGPUParams();
        paramsDirty = false;
      }

      const stage = stages[currentStageIdx];
      // Backpressure: if the oldest stage is still in flight, we must wait.
      if (stage.inFlight) {
        requestAnimationFrame(() => frame().catch(handleErr));
        return;
      }

      device.pushErrorScope('validation');
      const enc = device.createCommandEncoder();

      if (hasTimestamp) {
        // enc.writeTimestamp(querySet, 0);
      }

      for (let s = 0; s < STEPS_PER_FRAME; s++) dispatchMacroStep(enc);
      step += STEPS_PER_FRAME;

      if (hasTimestamp) {
        // enc.writeTimestamp(querySet, 1);
        enc.resolveQuerySet(querySet, 0, 2, queryResolveBuffer, 0);
        enc.copyBufferToBuffer(queryResolveBuffer, 0, stage.query, 0, 16);
      }

      const rp = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), clearValue: { r:0.07, g:0.07, b:0.1, a:1 }, loadOp: 'clear', storeOp: 'store' }]});
      rp.setPipeline(renPL); rp.setBindGroup(0, renBG); rp.draw(6); rp.end();

      enc.copyBufferToBuffer(cardStateBuf, 0, stage.card, 0, 104);

      const tSubmit = performance.now();
      device.queue.submit([enc.finish()]);
      device.popErrorScope().then(err => { if (err) handleErr(err); });

      stage.inFlight = true;
      stage.step = step;

      const processReadback = async (st) => {
        const pCard = st.card.mapAsync(GPUMapMode.READ);
        const pQuery = hasTimestamp ? st.query.mapAsync(GPUMapMode.READ) : Promise.resolve();

        await Promise.all([pCard, pQuery]);

        const d = new Float32Array(st.card.getMappedRange());
        let gpuTime = 0;
        if (hasTimestamp) {
          const timestamps = new BigUint64Array(st.query.getMappedRange());
          gpuTime = Number(timestamps[1] - timestamps[0]) / 1e6;
          st.query.unmap();
        } else {
          gpuTime = performance.now() - tSubmit;
        }

        if (st.step < 100000) {
          trajectory.push([st.step, d[0], d[20], d[21], d[2], d[3], d[4], d[5], d[6], d[7], d[8]]);
        }

        if (performance.now() - lastT > 250) {
          const mlups = (NCELLS * STEPS_PER_FRAME) / (gpuTime * 1e3);
          mlupsEl.textContent = mlups.toFixed(1);
          gpuMsEl.textContent = gpuTime.toFixed(2);
          syncMsEl.textContent = (performance.now() - tSubmit).toFixed(2);
          statusEl.textContent = `[AMR-dev] step ${st.step}  y=${d[20].toFixed(1)}  x=${d[21].toFixed(1)}  vy=${d[4].toFixed(4)}  Fy=${d[7].toExponential(2)}  θ=${d[2].toFixed(2)}`;
          lastT = performance.now();
        }

        st.card.unmap();
        st.inFlight = false;
      };

      processReadback(stage);

      currentStageIdx = (currentStageIdx + 1) % STAGES;
      requestAnimationFrame(() => frame().catch(handleErr));
    } catch (e) {
      handleErr(e);
    }
  }
  frame().catch(handleErr);
}
init().catch(handleErr);
