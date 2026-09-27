//! GPU inference backend using wgpu compute shaders.
//! Compiled when the `gpu` feature is enabled (on by default).

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::onnx::{Node, OnnxModel, Tensor};
use crate::shape::*;

// ── GPU Context ──────────────────────────────────────────────────────────────

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub name: String,
}

impl GpuContext {
    pub fn try_new() -> Option<Self> {
        Self::with_buffer_limit(None)
    }

    /// `max_buffer`: cap on buffer size below the adapter's (tests force failures with it).
    fn with_buffer_limit(max_buffer: Option<u64>) -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })) {
            Ok(a) => a,
            Err(e) => { eprintln!("GPU adapter request failed: {e}"); return None; }
        };
        let name = adapter.get_info().name.clone();
        let adapter_limits = adapter.limits();
        let max_buffer = max_buffer.unwrap_or(adapter_limits.max_buffer_size)
            .min(adapter_limits.max_buffer_size);
        let (device, queue) = match pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("nanoimg"),
                required_limits: wgpu::Limits {
                    max_buffer_size: max_buffer,
                    max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size
                        .min(max_buffer),
                    ..wgpu::Limits::downlevel_defaults()
                },
                ..Default::default()
            },
        )) {
            Ok(pair) => pair,
            Err(e) => { eprintln!("GPU device request failed: {e}"); return None; }
        };
        Some(GpuContext { device, queue, name })
    }
}

// ── GPU Tensor ───────────────────────────────────────────────────────────────

struct GpuTensor {
    shape: Vec<usize>,
    buffer: wgpu::Buffer,
}

impl GpuTensor {
    fn numel(&self) -> usize { self.shape.iter().product::<usize>().max(1) }

    /// Create a new GpuTensor wrapping the same buffer with a different shape (reshape/squeeze/unsqueeze)
    fn reshape(&self, new_shape: Vec<usize>) -> Self {
        GpuTensor { shape: new_shape, buffer: self.buffer.clone() }
    }
}

// ── WGSL Shaders ─────────────────────────────────────────────────────────────

const SHADER_MATMUL: &str = "
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct MatMulParams {
    M: u32, N: u32, K: u32, batch: u32,
    a_batch_stride: u32, b_batch_stride: u32, c_batch_stride: u32,
    _pad: u32,
}
@group(0) @binding(3) var<uniform> params: MatMulParams;

// Register-blocked SGEMM: a 16x16 workgroup computes a 64x64 tile of C, each
// thread a 4x4 block. Thread (tx, ty) owns rows ty + 16*i and cols tx + 16*j,
// so shared-memory reads and global stores are contiguous across a warp.
const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;
var<workgroup> tile_a: array<f32, 1024>; // [BK][BM], k-major
var<workgroup> tile_b: array<f32, 1024>; // [BK][BN]

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let row0 = wid.y * BM;
    let col0 = wid.x * BN;
    let a_off = wid.z * params.a_batch_stride;
    let b_off = wid.z * params.b_batch_stride;
    let c_off = wid.z * params.c_batch_stride;
    let M = params.M;
    let N = params.N;
    let K = params.K;

    // Accumulators as named vec4s (row i, cols j): indexed arrays would spill
    // out of registers since naga does not unroll the loops.
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    for (var k0 = 0u; k0 < K; k0 += BK) {
        // Load A tile (BM x BK) and B tile (BK x BN): 4 elements per thread each
        for (var l = 0u; l < 4u; l++) {
            let e = tid + l * 256u;
            // A: e -> (r = e / BK, kk = e % BK); consecutive threads walk k (contiguous in A)
            let ar = e / BK;
            let ak = e % BK;
            let gr = row0 + ar;
            let gk = k0 + ak;
            var av = 0.0;
            if (gr < M && gk < K) { av = a[a_off + gr * K + gk]; }
            tile_a[ak * BM + ar] = av;
            // B: e -> (kk = e / BN, cc = e % BN); consecutive threads walk n (contiguous in B)
            let bk = e / BN;
            let bc = e % BN;
            let gk2 = k0 + bk;
            let gc = col0 + bc;
            var bv = 0.0;
            if (gk2 < K && gc < N) { bv = b[b_off + gk2 * N + gc]; }
            tile_b[bk * BN + bc] = bv;
        }
        workgroupBarrier();

        for (var kk = 0u; kk < BK; kk++) {
            let ab = kk * BM + ty;
            let bb = kk * BN + tx;
            let rb = vec4<f32>(tile_b[bb], tile_b[bb + 16u], tile_b[bb + 32u], tile_b[bb + 48u]);
            acc0 = fma(vec4<f32>(tile_a[ab]), rb, acc0);
            acc1 = fma(vec4<f32>(tile_a[ab + 16u]), rb, acc1);
            acc2 = fma(vec4<f32>(tile_a[ab + 32u]), rb, acc2);
            acc3 = fma(vec4<f32>(tile_a[ab + 48u]), rb, acc3);
        }
        workgroupBarrier();
    }

    store_row(row0 + ty, col0 + tx, c_off, acc0);
    store_row(row0 + ty + 16u, col0 + tx, c_off, acc1);
    store_row(row0 + ty + 32u, col0 + tx, c_off, acc2);
    store_row(row0 + ty + 48u, col0 + tx, c_off, acc3);
}

fn store_row(r: u32, c0: u32, c_off: u32, v: vec4<f32>) {
    if (r >= params.M) { return; }
    let base = c_off + r * params.N;
    if (c0 < params.N) { c[base + c0] = v.x; }
    if (c0 + 16u < params.N) { c[base + c0 + 16u] = v.y; }
    if (c0 + 32u < params.N) { c[base + c0 + 32u] = v.z; }
    if (c0 + 48u < params.N) { c[base + c0 + 48u] = v.w; }
}
";

const SHADER_BINARY: &str = "
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct BinaryParams {
    total: u32,
    ndim: u32,
    opcode: u32, // 0=add 1=mul 2=sub 3=div 4=pow
    // Per-operand index mode: 0 = same layout as out, 1 = scalar,
    // 2 = trailing broadcast (index % len), 3 = general strided
    a_mode: u32,
    b_mode: u32,
    a_len: u32,
    b_len: u32,
    _pad: u32,
}
@group(0) @binding(3) var<uniform> params: BinaryParams;
@group(0) @binding(4) var<storage, read> shape_data: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid2: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // Large dispatches span a 2D grid of 256-wide groups (see GpuExecutor::grid)
    let gid = vec3<u32>(gid2.x + gid2.y * nwg.x * 256u, 0u, 0u);
    let i = gid.x;
    if (i >= params.total) { return; }

    var ai = i;
    var bi = i;
    if (params.a_mode == 1u) { ai = 0u; } else if (params.a_mode == 2u) { ai = i % params.a_len; }
    if (params.b_mode == 1u) { bi = 0u; } else if (params.b_mode == 2u) { bi = i % params.b_len; }
    if (params.a_mode == 3u || params.b_mode == 3u) {
        let ndim = params.ndim;
        // shape_data layout: out_shape[ndim], a_strides[ndim], b_strides[ndim], out_strides[ndim]
        var sa: u32 = 0u;
        var sb: u32 = 0u;
        var rem = i;
        for (var d: u32 = 0u; d < ndim; d = d + 1u) {
            let out_stride = shape_data[3u * ndim + d];
            let coord = rem / out_stride;
            rem = rem % out_stride;
            sa += coord * shape_data[ndim + d];
            sb += coord * shape_data[2u * ndim + d];
        }
        if (params.a_mode == 3u) { ai = sa; }
        if (params.b_mode == 3u) { bi = sb; }
    }

    let va = a[ai];
    let vb = b[bi];
    var result: f32;
    switch params.opcode {
        case 0u: { result = va + vb; }
        case 1u: { result = va * vb; }
        case 2u: { result = va - vb; }
        case 3u: { result = va / vb; }
        // WGSL pow() is undefined for negative x. Fast path for x^2 (LayerNorm);
        // abs() loses sign for other exponents — acceptable for this model's usage.
        case 4u: { if (vb == 2.0) { result = va * va; } else { result = pow(abs(va), vb); } }
        default: { result = va + vb; }
    }
    out[i] = result;
}
";

const SHADER_UNARY: &str = "
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct UnaryParams {
    total: u32,
    opcode: u32, // 0=sqrt 1=tanh
}
@group(0) @binding(2) var<uniform> params: UnaryParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid2: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // Large dispatches span a 2D grid of 256-wide groups (see GpuExecutor::grid)
    let gid = vec3<u32>(gid2.x + gid2.y * nwg.x * 256u, 0u, 0u);
    let i = gid.x;
    if (i >= params.total) { return; }
    let v = input[i];
    switch params.opcode {
        case 0u: { output[i] = sqrt(v); }
        case 1u: { output[i] = tanh(v); }
        default: { output[i] = v; }
    }
}
";

const SHADER_SOFTMAX: &str = "
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct SoftmaxParams {
    outer: u32, dim: u32, inner: u32, _pad: u32,
}
@group(0) @binding(2) var<uniform> params: SoftmaxParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid2: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // Large dispatches span a 2D grid of 256-wide groups (see GpuExecutor::grid)
    let gid = vec3<u32>(gid2.x + gid2.y * nwg.x * 256u, 0u, 0u);
    let idx = gid.x;
    let total_lanes = params.outer * params.inner;
    if (idx >= total_lanes) { return; }

    let o = idx / params.inner;
    let i = idx % params.inner;
    let base = o * params.dim * params.inner + i;

    // Find max
    var max_val: f32 = -3.402823e+38;
    for (var d: u32 = 0u; d < params.dim; d = d + 1u) {
        let v = input[base + d * params.inner];
        max_val = max(max_val, v);
    }

    // Exp and sum
    var sum_val: f32 = 0.0;
    for (var d: u32 = 0u; d < params.dim; d = d + 1u) {
        let v = exp(input[base + d * params.inner] - max_val);
        output[base + d * params.inner] = v;
        sum_val += v;
    }

    // Normalize
    for (var d: u32 = 0u; d < params.dim; d = d + 1u) {
        output[base + d * params.inner] /= sum_val;
    }
}
";

const SHADER_REDUCE_MEAN: &str = "
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct ReduceParams {
    outer: u32, dim: u32, inner: u32, _pad: u32,
}
@group(0) @binding(2) var<uniform> params: ReduceParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid2: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // Large dispatches span a 2D grid of 256-wide groups (see GpuExecutor::grid)
    let gid = vec3<u32>(gid2.x + gid2.y * nwg.x * 256u, 0u, 0u);
    let idx = gid.x;
    let total_lanes = params.outer * params.inner;
    if (idx >= total_lanes) { return; }

    let o = idx / params.inner;
    let i = idx % params.inner;
    let in_base = o * params.dim * params.inner + i;
    let out_idx = o * params.inner + i;

    var sum_val: f32 = 0.0;
    for (var d: u32 = 0u; d < params.dim; d = d + 1u) {
        sum_val += input[in_base + d * params.inner];
    }
    output[out_idx] = sum_val / f32(params.dim);
}
";

const SHADER_TRANSPOSE: &str = "
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct TransposeParams {
    total: u32, ndim: u32, _p0: u32, _p1: u32,
}
@group(0) @binding(2) var<uniform> params: TransposeParams;
// shape_data layout: perm[ndim], in_strides[ndim], out_strides[ndim]
@group(0) @binding(3) var<storage, read> shape_data: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid2: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // Large dispatches span a 2D grid of 256-wide groups (see GpuExecutor::grid)
    let gid = vec3<u32>(gid2.x + gid2.y * nwg.x * 256u, 0u, 0u);
    let i = gid.x;
    if (i >= params.total) { return; }

    let ndim = params.ndim;
    var in_idx: u32 = 0u;
    var rem = i;
    for (var d: u32 = 0u; d < ndim; d = d + 1u) {
        let out_stride = shape_data[2u * ndim + d];
        let coord = rem / out_stride;
        rem = rem % out_stride;
        let perm_d = shape_data[d];
        in_idx += coord * shape_data[ndim + perm_d];
    }
    output[i] = input[in_idx];
}
";

const SHADER_GATHER: &str = "
@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct GatherParams {
    outer: u32, axis_size: u32, inner: u32, idx_count: u32,
}
@group(0) @binding(3) var<uniform> params: GatherParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid2: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // Large dispatches span a 2D grid of 256-wide groups (see GpuExecutor::grid)
    let gid = vec3<u32>(gid2.x + gid2.y * nwg.x * 256u, 0u, 0u);
    let total = params.outer * params.idx_count * params.inner;
    if (gid.x >= total) { return; }

    let i = gid.x;
    let inner = params.inner;
    let ic = params.idx_count;
    let o = i / (ic * inner);
    let rem = i % (ic * inner);
    let ip = rem / inner;
    let j = rem % inner;

    var idx = indices[ip];
    if (idx < 0) { idx = idx + i32(params.axis_size); }
    let src = (o * params.axis_size + u32(idx)) * inner + j;
    output[i] = data[src];
}
";

const SHADER_CONCAT: &str = "
// Concat is dispatched per-input tensor via separate dispatches with offset params.
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct ConcatParams {
    outer: u32, in_axis: u32, out_axis: u32, inner: u32,
    axis_offset: u32, _p0: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(2) var<uniform> params: ConcatParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid2: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // Large dispatches span a 2D grid of 256-wide groups (see GpuExecutor::grid)
    let gid = vec3<u32>(gid2.x + gid2.y * nwg.x * 256u, 0u, 0u);
    let total = params.outer * params.in_axis * params.inner;
    if (gid.x >= total) { return; }

    let i = gid.x;
    let inner = params.inner;
    let o = i / (params.in_axis * inner);
    let rem = i % (params.in_axis * inner);
    let a = rem / inner;
    let j = rem % inner;

    let src = i;
    let dst = (o * params.out_axis + params.axis_offset + a) * inner + j;
    output[dst] = input[src];
}
";

const SHADER_SLICE: &str = "
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct SliceParams {
    total: u32, ndim: u32, _p0: u32, _p1: u32,
}
@group(0) @binding(2) var<uniform> params: SliceParams;
// shape_data: slice_start[ndim], slice_step[ndim], in_strides[ndim], out_strides[ndim]
@group(0) @binding(3) var<storage, read> shape_data: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid2: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // Large dispatches span a 2D grid of 256-wide groups (see GpuExecutor::grid)
    let gid = vec3<u32>(gid2.x + gid2.y * nwg.x * 256u, 0u, 0u);
    let i = gid.x;
    if (i >= params.total) { return; }

    let ndim = params.ndim;
    var in_idx: u32 = 0u;
    var rem = i;
    for (var d: u32 = 0u; d < ndim; d = d + 1u) {
        let out_stride = shape_data[3u * ndim + d];
        let coord = rem / out_stride;
        rem = rem % out_stride;
        in_idx += (shape_data[d] + coord * shape_data[ndim + d]) * shape_data[2u * ndim + d];
    }
    output[i] = input[in_idx];
}
";

const SHADER_TILE: &str = "
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct TileParams {
    total: u32, ndim: u32, _p0: u32, _p1: u32,
}
@group(0) @binding(2) var<uniform> params: TileParams;
// shape_data: in_shape[ndim], in_strides[ndim], out_strides[ndim]
@group(0) @binding(3) var<storage, read> shape_data: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid2: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // Large dispatches span a 2D grid of 256-wide groups (see GpuExecutor::grid)
    let gid = vec3<u32>(gid2.x + gid2.y * nwg.x * 256u, 0u, 0u);
    let i = gid.x;
    if (i >= params.total) { return; }

    let ndim = params.ndim;
    var in_idx: u32 = 0u;
    var rem = i;
    for (var d: u32 = 0u; d < ndim; d = d + 1u) {
        let out_stride = shape_data[2u * ndim + d];
        let coord = rem / out_stride;
        rem = rem % out_stride;
        in_idx += (coord % shape_data[d]) * shape_data[ndim + d];
    }
    output[i] = input[in_idx];
}
";

const SHADER_CONV_IM2COL: &str = "
// Phase 1: im2col transform
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> col: array<f32>;

struct ConvParams {
    c_in: u32, h: u32, w: u32,
    kh: u32, kw: u32,
    sh: u32, sw: u32,
    h_out: u32, w_out: u32,
    patch_sz: u32,
    n_patches: u32,
    n_batch: u32,
}
@group(0) @binding(2) var<uniform> params: ConvParams;

// Output layout: [n_batch * n_patches, patch_sz]
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid2: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // Large dispatches span a 2D grid of 256-wide groups (see GpuExecutor::grid)
    let gid = vec3<u32>(gid2.x + gid2.y * nwg.x * 256u, 0u, 0u);
    let total = params.n_batch * params.n_patches * params.patch_sz;
    if (gid.x >= total) { return; }

    let per_img = params.n_patches * params.patch_sz;
    let img = gid.x / per_img;
    let pi = (gid.x % per_img) / params.patch_sz;
    let fi = gid.x % params.patch_sz;
    let py = pi / params.w_out;
    let px = pi % params.w_out;
    let c = fi / (params.kh * params.kw);
    let rem = fi % (params.kh * params.kw);
    let ky = rem / params.kw;
    let kx = rem % params.kw;

    let ys = py * params.sh + ky;
    let xs = px * params.sw + kx;
    let in_base = img * params.c_in * params.h * params.w;
    col[gid.x] = input[in_base + c * params.h * params.w + ys * params.w + xs];
}
";

// ── Pipelines ────────────────────────────────────────────────────────────────

struct Pipelines {
    matmul: wgpu::ComputePipeline,
    binary: wgpu::ComputePipeline,
    unary: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    reduce_mean: wgpu::ComputePipeline,
    transpose: wgpu::ComputePipeline,
    gather: wgpu::ComputePipeline,
    concat: wgpu::ComputePipeline,
    slice: wgpu::ComputePipeline,
    tile: wgpu::ComputePipeline,
    conv_im2col: wgpu::ComputePipeline,
}

fn create_pipeline(device: &wgpu::Device, source: &str, label: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

impl Pipelines {
    fn new(device: &wgpu::Device) -> Self {
        Pipelines {
            matmul: create_pipeline(device, SHADER_MATMUL, "matmul"),
            binary: create_pipeline(device, SHADER_BINARY, "binary"),
            unary: create_pipeline(device, SHADER_UNARY, "unary"),
            softmax: create_pipeline(device, SHADER_SOFTMAX, "softmax"),
            reduce_mean: create_pipeline(device, SHADER_REDUCE_MEAN, "reduce_mean"),
            transpose: create_pipeline(device, SHADER_TRANSPOSE, "transpose"),
            gather: create_pipeline(device, SHADER_GATHER, "gather"),
            concat: create_pipeline(device, SHADER_CONCAT, "concat"),
            slice: create_pipeline(device, SHADER_SLICE, "slice"),
            tile: create_pipeline(device, SHADER_TILE, "tile"),
            conv_im2col: create_pipeline(device, SHADER_CONV_IM2COL, "conv_im2col"),
        }
    }
}

// ── GPU Executor ─────────────────────────────────────────────────────────────

/// Graph nodes recorded per command-buffer submission.
const FLUSH_EVERY: usize = 32;

pub struct GpuExecutor {
    ctx: GpuContext,
    pipelines: Pipelines,
    weight_cache: HashMap<String, GpuTensor>,
    /// First device error since the last check. wgpu's default handler panics;
    /// ours records the error so `run` can fail cleanly (e.g. out of VRAM).
    error: Arc<Mutex<Option<String>>>,
}

impl GpuExecutor {
    pub fn new(ctx: GpuContext) -> Self {
        let error = Arc::new(Mutex::new(None));
        let slot = error.clone();
        ctx.device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            let mut slot = slot.lock().unwrap_or_else(|p| p.into_inner());
            if slot.is_none() {
                *slot = Some(error_chain(&e));
            }
        }));
        let pipelines = Pipelines::new(&ctx.device);
        GpuExecutor { ctx, pipelines, weight_cache: HashMap::new(), error }
    }

    fn take_error(&self) -> Option<String> {
        self.error.lock().unwrap_or_else(|p| p.into_inner()).take()
    }

    fn has_error(&self) -> bool {
        self.error.lock().unwrap_or_else(|p| p.into_inner()).is_some()
    }

    /// Create a buffer and fill it via the queue. (`create_buffer_init` maps the
    /// buffer at creation, which panics instead of reporting when allocation fails.)
    fn upload_bytes(&self, data: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let size = (data.len() as u64).max(4).next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
        let buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !data.is_empty() {
            self.ctx.queue.write_buffer(&buffer, 0, data);
        }
        buffer
    }

    fn upload_f32(&self, data: &[f32]) -> wgpu::Buffer {
        self.upload_bytes(bytemuck_cast_slice(data),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC)
    }

    fn upload_i32(&self, data: &[i32]) -> wgpu::Buffer {
        self.upload_bytes(bytemuck_cast_slice(data), wgpu::BufferUsages::STORAGE)
    }

    fn upload_u32(&self, data: &[u32]) -> wgpu::Buffer {
        self.upload_bytes(bytemuck_cast_slice(data), wgpu::BufferUsages::STORAGE)
    }

    fn create_uniform(&self, data: &[u8]) -> wgpu::Buffer {
        self.upload_bytes(data, wgpu::BufferUsages::UNIFORM)
    }

    fn create_storage(&self, size: u64) -> wgpu::Buffer {
        self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn download_f32(&self, buffer: &wgpu::Buffer, len: usize) -> Vec<f32> {
        let size = (len * 4) as u64;
        let staging = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.ctx.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| { tx.send(result).ok(); });
        self.ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        // Any earlier failure (e.g. allocating `staging`) makes the data meaningless
        let mapped = match rx.recv() {
            Ok(Ok(())) if !self.has_error() => slice.get_mapped_range().ok(),
            _ => None,
        };
        let Some(data) = mapped else {
            let mut slot = self.error.lock().unwrap_or_else(|p| p.into_inner());
            slot.get_or_insert_with(|| "failed to map result buffer".into());
            return vec![0.0; len];
        };
        let result: Vec<f32> = data.as_chunks::<4>().0.iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        drop(data);
        staging.unmap();
        result
    }

    fn upload_tensor(&self, t: &Tensor) -> GpuTensor {
        GpuTensor { shape: t.shape.clone(), buffer: self.upload_f32(&t.f32_data()) }
    }

    fn ensure_weights_cached(&mut self, model: &OnnxModel) {
        for (name, tensor) in &model.weights {
            if !self.weight_cache.contains_key(name) && tensor.is_f32() {
                self.weight_cache.insert(name.clone(), self.upload_tensor(tensor));
            }
        }
    }

    fn get_encoder<'a>(&self, enc: &'a mut Option<wgpu::CommandEncoder>) -> &'a mut wgpu::CommandEncoder {
        enc.get_or_insert_with(|| self.ctx.device.create_command_encoder(&Default::default()))
    }

    fn flush(&self, enc: &mut Option<wgpu::CommandEncoder>) {
        if let Some(e) = enc.take() {
            self.ctx.queue.submit(std::iter::once(e.finish()));
            self.ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        }
    }

    fn div_ceil(a: u32, b: u32) -> u32 { a.div_ceil(b) }

    /// Workgroup grid for `threads` invocations of a 256-wide 1D shader. Each
    /// dimension is capped at 65535 groups, so large tensors spill into y.
    fn grid(threads: u32) -> (u32, u32, u32) {
        let groups = Self::div_ceil(threads, 256).max(1);
        let x = groups.min(65535);
        (x, Self::div_ceil(groups, x), 1)
    }

    fn dispatch(&self, pipeline: &wgpu::ComputePipeline, buffers: &[&wgpu::Buffer],
                workgroups: (u32, u32, u32), enc: &mut Option<wgpu::CommandEncoder>) {
        let layout = pipeline.get_bind_group_layout(0);
        let entries: Vec<_> = buffers.iter().enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
            .collect();
        let bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &layout, entries: &entries,
        });
        let encoder = self.get_encoder(enc);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }

    // ── Op implementations ───────────────────────────────────────────────

    fn op_matmul(&self, a: &GpuTensor, b: &GpuTensor, enc: &mut Option<wgpu::CommandEncoder>) -> GpuTensor {
        let an = a.shape.len();
        let bn = b.shape.len();
        let m = a.shape[an - 2] as u32;
        let k = a.shape[an - 1] as u32;
        let n = b.shape[bn - 1] as u32;

        let a_batch = &a.shape[..an - 2];
        let b_batch = &b.shape[..bn - 2];
        let out_batch = broadcast_shape(a_batch, b_batch);
        let batch: u32 = out_batch.iter().product::<usize>().max(1) as u32;

        // Batch strides: 0 if that operand has no batch dims (broadcast)
        let a_batch_elems: usize = a_batch.iter().product::<usize>().max(1);
        let b_batch_elems: usize = b_batch.iter().product::<usize>().max(1);
        let a_batch_stride = if a_batch_elems > 1 { m * k } else { 0 };
        let b_batch_stride = if b_batch_elems > 1 { k * n } else { 0 };
        let c_batch_stride = m * n;
        let out_len = (batch * m * n) as usize;
        let out_buf = self.create_storage((out_len * 4) as u64);

        let params: [u32; 8] = [m, n, k, batch, a_batch_stride, b_batch_stride, c_batch_stride, 0];
        let param_buf = self.create_uniform(bytemuck_cast_slice(&params));
        self.dispatch(&self.pipelines.matmul,
            &[&a.buffer, &b.buffer, &out_buf, &param_buf],
            (Self::div_ceil(n, 64), Self::div_ceil(m, 64), batch), enc);

        let mut shape = out_batch;
        shape.push(m as usize);
        shape.push(n as usize);
        GpuTensor { shape, buffer: out_buf }
    }

    fn op_gemm(&self, a: &GpuTensor, b: &GpuTensor, c: Option<&GpuTensor>, node: &Node, enc: &mut Option<wgpu::CommandEncoder>) -> GpuTensor {
        let trans_a = node.attr_i("transA").unwrap_or(0) != 0;
        let trans_b = node.attr_i("transB").unwrap_or(0) != 0;
        let alpha = node.attr_f("alpha").unwrap_or(1.0);
        let beta = node.attr_f("beta").unwrap_or(1.0);

        if alpha != 1.0 || beta != 1.0 {
            // Rare: scaled GEMM — CPU fallback
            self.flush(enc);
            let a_cpu = Tensor::f32(a.shape.clone(), self.download_f32(&a.buffer, a.numel()));
            let b_cpu = Tensor::f32(b.shape.clone(), self.download_f32(&b.buffer, b.numel()));
            let c_cpu = c.map(|ct| Tensor::f32(ct.shape.clone(), self.download_f32(&ct.buffer, ct.numel())));
            let result = crate::onnx::cpu_gemm(&a_cpu, &b_cpu, c_cpu.as_ref(), trans_a, trans_b, alpha, beta);
            return self.upload_tensor(&result);
        }

        let a_t;
        let a = if trans_a { a_t = self.op_transpose(a, &[1, 0], enc); &a_t } else { a };
        let b_t;
        let b = if trans_b { b_t = self.op_transpose(b, &[1, 0], enc); &b_t } else { b };
        let out = self.op_matmul(a, b, enc);
        match c {
            Some(bias) => self.op_binary(&out, bias, 0, enc),
            None => out,
        }
    }

    /// Conv as im2col + GEMM, entirely on the GPU: col [B·P, patch] × Wᵀ [patch, C] → [B, P, C] → [B, C, P].
    fn op_conv(&self, input: &GpuTensor, weight: &GpuTensor, bias: Option<&GpuTensor>, node: &Node, enc: &mut Option<wgpu::CommandEncoder>) -> GpuTensor {
        let n_batch = input.shape[0];
        let c_in = input.shape[1] as u32;
        let h = input.shape[2] as u32;
        let w = input.shape[3] as u32;
        let c_out = weight.shape[0];
        let kh = weight.shape[2] as u32;
        let kw = weight.shape[3] as u32;

        let strides = node.attr_ints("strides").unwrap_or(&[1, 1]);
        let sh = strides[0] as u32;
        let sw = strides[1] as u32;
        let h_out = (h - kh) / sh + 1;
        let w_out = (w - kw) / sw + 1;
        let patch = c_in * kh * kw;
        let n_patches = h_out * w_out;

        let col_len = n_batch * (n_patches * patch) as usize;
        let col_buf = self.create_storage((col_len * 4) as u64);
        let conv_params: [u32; 12] = [c_in, h, w, kh, kw, sh, sw, h_out, w_out, patch, n_patches, n_batch as u32];
        let param_buf = self.create_uniform(bytemuck_cast_slice(&conv_params));
        self.dispatch(&self.pipelines.conv_im2col,
            &[&input.buffer, &col_buf, &param_buf],
            Self::grid(col_len as u32), enc);
        let col = GpuTensor { shape: vec![n_batch * n_patches as usize, patch as usize], buffer: col_buf };

        let w_t = self.op_transpose(&weight.reshape(vec![c_out, patch as usize]), &[1, 0], enc);
        let mut out = self.op_matmul(&col, &w_t, enc);
        if let Some(bias) = bias {
            out = self.op_binary(&out, bias, 0, enc);
        }
        let out = out.reshape(vec![n_batch, n_patches as usize, c_out]);
        let out = self.op_transpose(&out, &[0, 2, 1], enc);
        out.reshape(vec![n_batch, c_out, h_out as usize, w_out as usize])
    }

    fn op_binary(&self, a: &GpuTensor, b: &GpuTensor, opcode: u32, enc: &mut Option<wgpu::CommandEncoder>) -> GpuTensor {
        let out_shape = broadcast_shape(&a.shape, &b.shape);
        let total: u32 = out_shape.iter().product::<usize>().max(1) as u32;
        let ndim = out_shape.len() as u32;

        let a_strides: Vec<u32> = broadcast_strides(&a.shape, &out_shape).iter().map(|&s| s as u32).collect();
        let b_strides: Vec<u32> = broadcast_strides(&b.shape, &out_shape).iter().map(|&s| s as u32).collect();
        let out_strides: Vec<u32> = compute_strides(&out_shape).iter().map(|&s| s as u32).collect();
        let out_shape_u32: Vec<u32> = out_shape.iter().map(|&s| s as u32).collect();

        // Pack shape data: out_shape, a_strides, b_strides, out_strides
        let mut shape_data: Vec<u32> = Vec::new();
        shape_data.extend_from_slice(&out_shape_u32);
        shape_data.extend_from_slice(&a_strides);
        shape_data.extend_from_slice(&b_strides);
        shape_data.extend_from_slice(&out_strides);

        // Cheap indexing when the operand is the full output, a scalar, or a
        // contiguous trailing block (LayerNorm γ/β, biases) — the common cases
        let mode = |shape: &[usize]| -> (u32, u32) {
            let len: usize = shape.iter().product::<usize>().max(1);
            if len == 1 {
                (1, 1)
            } else if len as u32 == total {
                (0, len as u32)
            } else if out_shape.ends_with(&shape[shape.iter().position(|&d| d != 1).unwrap_or(0)..]) {
                (2, len as u32)
            } else {
                (3, len as u32)
            }
        };
        let (a_mode, a_len) = mode(&a.shape);
        let (b_mode, b_len) = mode(&b.shape);
        let params: [u32; 8] = [total, ndim, opcode, a_mode, b_mode, a_len, b_len, 0];
        let param_buf = self.create_uniform(bytemuck_cast_slice(&params));
        let shape_buf = self.upload_u32(&shape_data);
        let out_buf = self.create_storage((total as u64) * 4);
        self.dispatch(&self.pipelines.binary,
            &[&a.buffer, &b.buffer, &out_buf, &param_buf, &shape_buf],
            Self::grid(total), enc);
        GpuTensor { shape: out_shape, buffer: out_buf }
    }

    fn op_unary(&self, a: &GpuTensor, opcode: u32, enc: &mut Option<wgpu::CommandEncoder>) -> GpuTensor {
        let total = a.numel() as u32;
        let out_buf = self.create_storage((total as u64) * 4);
        let params: [u32; 2] = [total, opcode];
        let param_buf = self.create_uniform(bytemuck_cast_slice(&params));
        self.dispatch(&self.pipelines.unary, &[&a.buffer, &out_buf, &param_buf],
            Self::grid(total), enc);
        GpuTensor { shape: a.shape.clone(), buffer: out_buf }
    }

    fn op_softmax(&self, a: &GpuTensor, axis: i64, enc: &mut Option<wgpu::CommandEncoder>) -> GpuTensor {
        let axis = normalize_axis(axis, a.shape.len());
        let outer: u32 = a.shape[..axis].iter().product::<usize>().max(1) as u32;
        let dim: u32 = a.shape[axis] as u32;
        let inner: u32 = a.shape[axis + 1..].iter().product::<usize>().max(1) as u32;
        let out_buf = self.create_storage((a.numel() * 4) as u64);
        let params: [u32; 4] = [outer, dim, inner, 0];
        let param_buf = self.create_uniform(bytemuck_cast_slice(&params));
        self.dispatch(&self.pipelines.softmax, &[&a.buffer, &out_buf, &param_buf],
            Self::grid(outer * inner), enc);
        GpuTensor { shape: a.shape.clone(), buffer: out_buf }
    }

    fn op_reduce_mean(&self, a: &GpuTensor, axes: &[i64], keepdims: bool, enc: &mut Option<wgpu::CommandEncoder>) -> GpuTensor {
        // Single-axis fast path (most common)
        if axes.len() == 1 {
            let axis = normalize_axis(axes[0], a.shape.len());
            let outer: u32 = a.shape[..axis].iter().product::<usize>().max(1) as u32;
            let dim: u32 = a.shape[axis] as u32;
            let inner: u32 = a.shape[axis + 1..].iter().product::<usize>().max(1) as u32;

            let mut out_shape = Vec::new();
            for (d, &s) in a.shape.iter().enumerate() {
                if d == axis {
                    if keepdims { out_shape.push(1); }
                } else {
                    out_shape.push(s);
                }
            }
            if out_shape.is_empty() { out_shape.push(1); }
            let out_len: usize = out_shape.iter().product::<usize>().max(1);

            let out_buf = self.create_storage((out_len * 4) as u64);
            let params: [u32; 4] = [outer, dim, inner, 0];
            let param_buf = self.create_uniform(bytemuck_cast_slice(&params));
            self.dispatch(&self.pipelines.reduce_mean, &[&a.buffer, &out_buf, &param_buf],
                Self::grid(outer * inner), enc);
            return GpuTensor { shape: out_shape, buffer: out_buf };
        }

        // Multi-axis: fallback to CPU — flush pending work first
        self.flush(enc);
        let data = self.download_f32(&a.buffer, a.numel());
        let cpu_t = Tensor::f32(a.shape.clone(), data);
        let result = crate::onnx::cpu_reduce_mean(&cpu_t, axes, keepdims);
        self.upload_tensor(&result)
    }

    fn op_transpose(&self, a: &GpuTensor, perm: &[usize], enc: &mut Option<wgpu::CommandEncoder>) -> GpuTensor {
        let ndim = a.shape.len();
        let out_shape: Vec<usize> = perm.iter().map(|&p| a.shape[p]).collect();
        let total: u32 = out_shape.iter().product::<usize>().max(1) as u32;
        let in_strides = compute_strides(&a.shape);
        let out_strides = compute_strides(&out_shape);

        // Pack: perm, in_strides, out_strides
        let mut shape_data: Vec<u32> = Vec::new();
        for &p in perm { shape_data.push(p as u32); }
        for &s in &in_strides { shape_data.push(s as u32); }
        for &s in &out_strides { shape_data.push(s as u32); }
        let shape_buf = self.upload_u32(&shape_data);

        let out_buf = self.create_storage((total as u64) * 4);
        let params: [u32; 4] = [total, ndim as u32, 0, 0];
        let param_buf = self.create_uniform(bytemuck_cast_slice(&params));
        self.dispatch(&self.pipelines.transpose,
            &[&a.buffer, &out_buf, &param_buf, &shape_buf],
            Self::grid(total), enc);
        GpuTensor { shape: out_shape, buffer: out_buf }
    }

    fn op_gather(&self, data: &GpuTensor, indices: &[i64], indices_shape: &[usize], axis: i64, enc: &mut Option<wgpu::CommandEncoder>) -> GpuTensor {
        let axis = normalize_axis(axis, data.shape.len());
        let axis_size = data.shape[axis] as u32;
        let outer: u32 = data.shape[..axis].iter().product::<usize>().max(1) as u32;
        let inner: u32 = data.shape[axis + 1..].iter().product::<usize>().max(1) as u32;
        let idx_count: u32 = indices_shape.iter().product::<usize>().max(1) as u32;

        let mut out_shape = Vec::new();
        out_shape.extend_from_slice(&data.shape[..axis]);
        out_shape.extend_from_slice(indices_shape);
        out_shape.extend_from_slice(&data.shape[axis + 1..]);

        let total = (outer * idx_count * inner) as usize;
        let out_buf = self.create_storage((total * 4) as u64);

        let indices_i32: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
        let idx_buf = self.upload_i32(&indices_i32);

        let params: [u32; 4] = [outer, axis_size, inner, idx_count];
        let param_buf = self.create_uniform(bytemuck_cast_slice(&params));
        self.dispatch(&self.pipelines.gather,
            &[&data.buffer, &idx_buf, &out_buf, &param_buf],
            Self::grid(total as u32), enc);
        GpuTensor { shape: out_shape, buffer: out_buf }
    }

    fn op_concat(&self, tensors: &[&GpuTensor], axis: i64, enc: &mut Option<wgpu::CommandEncoder>) -> GpuTensor {
        let ndim = tensors[0].shape.len();
        let axis = normalize_axis(axis, ndim);
        let mut out_shape = tensors[0].shape.clone();
        out_shape[axis] = tensors.iter().map(|t| t.shape[axis]).sum();

        let outer: u32 = out_shape[..axis].iter().product::<usize>().max(1) as u32;
        let inner: u32 = out_shape[axis + 1..].iter().product::<usize>().max(1) as u32;
        let out_axis: u32 = out_shape[axis] as u32;
        let out_len: usize = out_shape.iter().product::<usize>().max(1);

        let out_buf = self.create_storage((out_len * 4) as u64);

        let mut axis_offset: u32 = 0;
        for t in tensors {
            let in_axis = t.shape[axis] as u32;
            let total = outer * in_axis * inner;
            let params: [u32; 8] = [outer, in_axis, out_axis, inner, axis_offset, 0, 0, 0];
            let param_buf = self.create_uniform(bytemuck_cast_slice(&params));
            self.dispatch(&self.pipelines.concat, &[&t.buffer, &out_buf, &param_buf],
                Self::grid(total), enc);
            axis_offset += in_axis;
        }

        GpuTensor { shape: out_shape, buffer: out_buf }
    }

    fn op_slice(&self, data: &GpuTensor, starts: &[i64], ends: &[i64],
                axes: &[i64], steps: &[i64], enc: &mut Option<wgpu::CommandEncoder>) -> GpuTensor {
        let ndim = data.shape.len();
        let n = starts.len();

        let mut slice_start = vec![0u32; ndim];
        let mut slice_step = vec![1u32; ndim];
        let mut out_shape = data.shape.clone();

        for i in 0..n {
            let axis = normalize_axis(axes[i], ndim);
            let dim = data.shape[axis] as i64;
            let step = steps[i].max(1) as u32;
            let mut s = starts[i];
            let mut e = ends[i];
            if s < 0 { s += dim; }
            if e < 0 { e += dim; }
            s = s.clamp(0, dim);
            e = e.clamp(0, dim);
            let len = if e > s { ((e - s) as u32).div_ceil(step) } else { 0 };
            slice_start[axis] = s as u32;
            slice_step[axis] = step;
            out_shape[axis] = len as usize;
        }

        let total: u32 = out_shape.iter().product::<usize>().max(1) as u32;
        let in_strides: Vec<u32> = compute_strides(&data.shape).iter().map(|&s| s as u32).collect();
        let out_strides: Vec<u32> = compute_strides(&out_shape).iter().map(|&s| s as u32).collect();

        // Pack: slice_start, slice_step, in_strides, out_strides
        let mut shape_data: Vec<u32> = Vec::new();
        shape_data.extend_from_slice(&slice_start);
        shape_data.extend_from_slice(&slice_step);
        shape_data.extend_from_slice(&in_strides);
        shape_data.extend_from_slice(&out_strides);
        let shape_buf = self.upload_u32(&shape_data);

        let out_buf = self.create_storage((total as u64) * 4);
        let params: [u32; 4] = [total, ndim as u32, 0, 0];
        let param_buf = self.create_uniform(bytemuck_cast_slice(&params));
        self.dispatch(&self.pipelines.slice,
            &[&data.buffer, &out_buf, &param_buf, &shape_buf],
            Self::grid(total), enc);

        GpuTensor { shape: out_shape, buffer: out_buf }
    }

    fn op_tile(&self, data: &GpuTensor, repeats: &[i64], enc: &mut Option<wgpu::CommandEncoder>) -> GpuTensor {
        let ndim = data.shape.len();
        let out_shape: Vec<usize> = data.shape.iter().zip(repeats.iter())
            .map(|(&s, &r)| s * r as usize).collect();
        let total: u32 = out_shape.iter().product::<usize>().max(1) as u32;

        let in_strides = compute_strides(&data.shape);
        let out_strides = compute_strides(&out_shape);

        // Pack: in_shape, in_strides, out_strides
        let mut shape_data: Vec<u32> = Vec::new();
        for &s in &data.shape { shape_data.push(s as u32); }
        for &s in &in_strides { shape_data.push(s as u32); }
        for &s in &out_strides { shape_data.push(s as u32); }
        let shape_buf = self.upload_u32(&shape_data);

        let out_buf = self.create_storage((total as u64) * 4);
        let params: [u32; 4] = [total, ndim as u32, 0, 0];
        let param_buf = self.create_uniform(bytemuck_cast_slice(&params));
        self.dispatch(&self.pipelines.tile,
            &[&data.buffer, &out_buf, &param_buf, &shape_buf],
            Self::grid(total), enc);
        GpuTensor { shape: out_shape, buffer: out_buf }
    }

    // ── Graph executor ───────────────────────────────────────────────────

    /// Run the graph on the GPU. Device errors (out of memory, limits exceeded)
    /// come back as `Err`; cached weights are dropped so a retry starts clean.
    pub fn run(&mut self, model: &OnnxModel, inputs: Vec<(&str, Tensor)>) -> Result<HashMap<String, Tensor>> {
        self.take_error();
        let result = self.run_graph(model, inputs);
        if let Some(e) = self.take_error() {
            // Buffers created during a failed pass may be invalid; don't reuse them.
            // (wgpu keeps the failed pass's VRAM reserved — callers retrying with
            // less memory should recreate the executor, see backends.rs.)
            self.weight_cache.clear();
            bail!("GPU error: {e}");
        }
        result
    }

    fn run_graph(&mut self, model: &OnnxModel, inputs: Vec<(&str, Tensor)>) -> Result<HashMap<String, Tensor>> {
        self.ensure_weights_cached(model);
        if self.has_error() { bail!("uploading weights failed"); }

        let mut gpu_tensors: HashMap<String, GpuTensor> = HashMap::new();
        let mut cpu_tensors: HashMap<String, Tensor> = HashMap::new();
        // Batched command encoder — lazily created, flushed only for CPU fallbacks
        let mut enc: Option<wgpu::CommandEncoder> = None;

        for (name, tensor) in inputs {
            if tensor.is_f32() {
                gpu_tensors.insert(name.to_string(), self.upload_tensor(&tensor));
            } else {
                cpu_tensors.insert(name.to_string(), tensor);
            }
        }

        // Build last-use map
        let mut last_use: HashMap<&str, usize> = HashMap::new();
        for (i, node) in model.nodes.iter().enumerate() {
            for name in &node.inputs {
                if !name.is_empty() {
                    last_use.insert(name.as_str(), i);
                }
            }
        }
        for name in &model.graph_outputs {
            last_use.insert(name.as_str(), usize::MAX);
        }

        for (i, node) in model.nodes.iter().enumerate() {
            let get_gpu = |idx: usize| -> Option<&GpuTensor> {
                let name = node.inputs.get(idx)?;
                if name.is_empty() { return None; }
                gpu_tensors.get(name).or_else(|| self.weight_cache.get(name))
            };
            let get_cpu = |idx: usize| -> Option<&Tensor> {
                let name = node.inputs.get(idx)?;
                if name.is_empty() { return None; }
                cpu_tensors.get(name).or_else(|| model.weights.get(name))
            };

            let out_name = &node.outputs[0];

            match node.op_type.as_str() {
                "MatMul" => {
                    let a = get_gpu(0).with_context(|| format!("missing gpu input 0 for MatMul node {}", i))?;
                    let b = get_gpu(1).with_context(|| format!("missing gpu input 1 for MatMul node {}", i))?;
                    let result = self.op_matmul(a, b, &mut enc);
                    gpu_tensors.insert(out_name.clone(), result);
                }
                "Gemm" => {
                    let a = get_gpu(0).with_context(|| format!("missing input 0 for Gemm node {}", i))?;
                    let b = get_gpu(1).with_context(|| format!("missing input 1 for Gemm node {}", i))?;
                    let c = get_gpu(2);
                    let result = self.op_gemm(a, b, c, node, &mut enc);
                    gpu_tensors.insert(out_name.clone(), result);
                }
                "Conv" => {
                    let input = get_gpu(0).with_context(|| format!("missing input 0 for Conv node {}", i))?;
                    let weight = get_gpu(1).with_context(|| format!("missing input 1 for Conv node {}", i))?;
                    let bias = get_gpu(2);
                    let result = self.op_conv(input, weight, bias, node, &mut enc);
                    gpu_tensors.insert(out_name.clone(), result);
                }
                "Add" | "Mul" | "Sub" | "Div" | "Pow" => {
                    let opcode = match node.op_type.as_str() {
                        "Add" => 0, "Mul" => 1, "Sub" => 2, "Div" => 3, _ => 4,
                    };
                    let a = get_gpu(0).with_context(|| format!("missing input 0 for {} node {}", node.op_type, i))?;
                    let b = get_gpu(1).with_context(|| format!("missing input 1 for {} node {}", node.op_type, i))?;
                    let result = self.op_binary(a, b, opcode, &mut enc);
                    gpu_tensors.insert(out_name.clone(), result);
                }
                "Sqrt" | "Tanh" => {
                    let opcode = if node.op_type == "Tanh" { 1 } else { 0 };
                    let a = get_gpu(0).with_context(|| format!("missing input for {} node {}", node.op_type, i))?;
                    let result = self.op_unary(a, opcode, &mut enc);
                    gpu_tensors.insert(out_name.clone(), result);
                }
                "Softmax" => {
                    let a = get_gpu(0).with_context(|| format!("missing input for Softmax node {}", i))?;
                    let axis = node.attr_i("axis").unwrap_or(-1);
                    let result = self.op_softmax(a, axis, &mut enc);
                    gpu_tensors.insert(out_name.clone(), result);
                }
                "ReduceMean" => {
                    let a = get_gpu(0).with_context(|| format!("missing input for ReduceMean node {}", i))?;
                    let axes = node.attr_ints("axes").unwrap_or(&[-1]);
                    let keepdims = node.attr_i("keepdims").unwrap_or(1) != 0;
                    let result = self.op_reduce_mean(a, axes, keepdims, &mut enc);
                    gpu_tensors.insert(out_name.clone(), result);
                }
                "Reshape" => {
                    let shape_tensor = get_cpu(1)
                        .with_context(|| format!("Reshape needs i64 shape tensor, node {}", i))?;
                    if let Some(a) = get_gpu(0) {
                        let result = a.reshape(crate::onnx::reshape_dims(&a.shape, shape_tensor.as_i64()));
                        gpu_tensors.insert(out_name.clone(), result);
                    } else {
                        let a_cpu = get_cpu(0).with_context(|| format!("Reshape: no input, node {}", i))?;
                        let result = crate::onnx::cpu_reshape(a_cpu, shape_tensor);
                        cpu_tensors.insert(out_name.clone(), result);
                    }
                }
                "Transpose" => {
                    let a = get_gpu(0).with_context(|| format!("missing input for Transpose node {}", i))?;
                    let ndim = a.shape.len();
                    let default_perm: Vec<i64> = (0..ndim as i64).rev().collect();
                    let perm_i64 = node.attr_ints("perm").unwrap_or(&default_perm);
                    let perm: Vec<usize> = perm_i64.iter().map(|&p| p as usize).collect();
                    let result = self.op_transpose(a, &perm, &mut enc);
                    gpu_tensors.insert(out_name.clone(), result);
                }
                "Unsqueeze" => {
                    let axes_tensor = get_cpu(1)
                        .with_context(|| format!("Unsqueeze needs i64 axes, node {}", i))?;
                    let axes_vals = axes_tensor.as_i64();
                    if let Some(a) = get_gpu(0) {
                        let new_ndim = a.shape.len() + axes_vals.len();
                        let mut axes_set: Vec<usize> = axes_vals.iter()
                            .map(|&ax| if ax < 0 { (new_ndim as i64 + ax) as usize } else { ax as usize })
                            .collect();
                        axes_set.sort();
                        let mut new_shape = Vec::with_capacity(new_ndim);
                        let mut old_idx = 0;
                        for j in 0..new_ndim {
                            if axes_set.contains(&j) {
                                new_shape.push(1);
                            } else {
                                new_shape.push(a.shape[old_idx]);
                                old_idx += 1;
                            }
                        }
                        let result = a.reshape(new_shape);
                        gpu_tensors.insert(out_name.clone(), result);
                    } else {
                        let a_cpu = get_cpu(0).with_context(|| format!("Unsqueeze: no input, node {}", i))?;
                        let result = crate::onnx::cpu_unsqueeze(a_cpu, axes_tensor);
                        cpu_tensors.insert(out_name.clone(), result);
                    }
                }
                "Squeeze" => {
                    if let Some(a) = get_gpu(0) {
                        let new_shape = if let Some(axes) = get_cpu(1) {
                            let ax: Vec<usize> = axes.as_i64().iter()
                                .map(|&ax| normalize_axis(ax, a.shape.len()))
                                .collect();
                            a.shape.iter().enumerate()
                                .filter(|(j, _)| !ax.contains(j))
                                .map(|(_, &s)| s)
                                .collect()
                        } else {
                            a.shape.iter().copied().filter(|&s| s != 1).collect()
                        };
                        let result = a.reshape(new_shape);
                        gpu_tensors.insert(out_name.clone(), result);
                    } else {
                        let a_cpu = get_cpu(0).with_context(|| format!("Squeeze: no input, node {}", i))?;
                        let result = crate::onnx::cpu_squeeze(a_cpu, get_cpu(1));
                        cpu_tensors.insert(out_name.clone(), result);
                    }
                }
                "Concat" => {
                    let axis = node.attr_i("axis").unwrap_or(0);
                    let all_cpu = (0..node.inputs.len()).all(|j| {
                        let name = &node.inputs[j];
                        !name.is_empty() && (cpu_tensors.contains_key(name) || model.weights.get(name).map(|t| !t.is_f32()).unwrap_or(false))
                    });
                    if all_cpu {
                        let ts: Vec<&Tensor> = (0..node.inputs.len()).filter_map(&get_cpu).collect();
                        let result = crate::onnx::cpu_concat(&ts, axis);
                        cpu_tensors.insert(out_name.clone(), result);
                    } else {
                        let ts: Vec<&GpuTensor> = (0..node.inputs.len())
                            .filter_map(&get_gpu)
                            .collect();
                        let result = self.op_concat(&ts, axis, &mut enc);
                        gpu_tensors.insert(out_name.clone(), result);
                    }
                }
                "Slice" => {
                    if let Some(data) = get_gpu(0) {
                        let starts_t = get_cpu(1).with_context(|| format!("Slice starts, node {}", i))?;
                        let ends_t = get_cpu(2).with_context(|| format!("Slice ends, node {}", i))?;
                        let starts = starts_t.as_i64();
                        let ends = ends_t.as_i64();
                        let n = starts.len();
                        let default_axes: Vec<i64> = (0..n as i64).collect();
                        let axes = get_cpu(3).map(|a| a.as_i64().to_vec()).unwrap_or(default_axes);
                        let default_steps: Vec<i64> = vec![1; n];
                        let steps = get_cpu(4).map(|s| s.as_i64().to_vec()).unwrap_or(default_steps);
                        let result = self.op_slice(data, starts, ends, &axes, &steps, &mut enc);
                        gpu_tensors.insert(out_name.clone(), result);
                    } else {
                        let data_cpu = get_cpu(0).with_context(|| format!("Slice: no input, node {}", i))?;
                        let starts_t = get_cpu(1).with_context(|| format!("Slice starts, node {}", i))?;
                        let ends_t = get_cpu(2).with_context(|| format!("Slice ends, node {}", i))?;
                        let result = crate::onnx::cpu_slice(data_cpu, starts_t, ends_t, get_cpu(3), get_cpu(4));
                        cpu_tensors.insert(out_name.clone(), result);
                    }
                }
                "Gather" => {
                    let axis = node.attr_i("axis").unwrap_or(0);
                    if let Some(data) = get_gpu(0) {
                        let indices_t = get_cpu(1)
                            .with_context(|| format!("Gather indices, node {}", i))?;
                        let result = self.op_gather(data, indices_t.as_i64(), &indices_t.shape, axis, &mut enc);
                        gpu_tensors.insert(out_name.clone(), result);
                    } else {
                        let data_cpu = get_cpu(0)
                            .with_context(|| format!("Gather: missing input 0 on both GPU and CPU, node {}", i))?;
                        let indices_cpu = get_cpu(1)
                            .with_context(|| format!("Gather: missing indices, node {}", i))?;
                        let result = crate::onnx::cpu_gather(data_cpu, indices_cpu, axis);
                        cpu_tensors.insert(out_name.clone(), result);
                    }
                }
                "Shape" => {
                    let shape = if let Some(a) = get_gpu(0) {
                        a.shape.clone()
                    } else if let Some(a_cpu) = get_cpu(0) {
                        a_cpu.shape.clone()
                    } else {
                        bail!("Shape: no input, node {}", i);
                    };
                    let dims: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
                    cpu_tensors.insert(out_name.clone(), Tensor::i64(vec![dims.len()], dims));
                }
                "Cast" => {
                    if let Some(a) = get_gpu(0) {
                        crate::onnx::check_cast(node, true)?;
                        gpu_tensors.insert(out_name.clone(), a.reshape(a.shape.clone()));
                    } else {
                        let a_cpu = get_cpu(0).with_context(|| format!("Cast: no input, node {}", i))?;
                        cpu_tensors.insert(out_name.clone(), crate::onnx::op_cast(a_cpu, node)?);
                    }
                }
                "Tile" => {
                    let data = get_gpu(0).with_context(|| format!("missing input for Tile node {}", i))?;
                    let repeats_t = get_cpu(1)
                        .with_context(|| format!("Tile repeats, node {}", i))?;
                    let result = self.op_tile(data, repeats_t.as_i64(), &mut enc);
                    gpu_tensors.insert(out_name.clone(), result);
                }
                _ => bail!("GPU: unsupported op: {}", node.op_type),
            }

            // Free tensors no longer needed
            for name in &node.inputs {
                if !name.is_empty() && last_use.get(name.as_str()) == Some(&i) {
                    gpu_tensors.remove(name);
                    cpu_tensors.remove(name);
                }
            }

            // Dropped buffers are only reclaimed once the work using them has run,
            // so submit periodically to bound peak VRAM (one transformer layer ≈ 54 nodes)
            if i % FLUSH_EVERY == FLUSH_EVERY - 1 {
                self.flush(&mut enc);
                if self.has_error() { bail!("aborted at node {i}"); }
            }
        }

        // Flush any remaining batched work before downloading
        self.flush(&mut enc);

        // Download output tensors
        let mut out = HashMap::new();
        for name in &model.graph_outputs {
            if let Some(gt) = gpu_tensors.remove(name) {
                let data = self.download_f32(&gt.buffer, gt.numel());
                out.insert(name.clone(), Tensor::f32(gt.shape, data));
            } else if let Some(ct) = cpu_tensors.remove(name) {
                out.insert(name.clone(), ct);
            }
        }
        Ok(out)
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// One line: error kind plus its most specific cause, e.g.
/// "Out of Memory: Not enough memory left." (wgpu's Display spans several lines).
fn error_chain(e: &(dyn std::error::Error + 'static)) -> String {
    let first_line = |s: String| s.lines().next().unwrap_or("").trim().to_string();
    let mut innermost = None;
    let mut src = e.source();
    while let Some(s) = src {
        innermost = Some(s);
        src = s.source();
    }
    match innermost {
        Some(cause) => format!("{}: {}", first_line(e.to_string()), first_line(cause.to_string())),
        None => first_line(e.to_string()),
    }
}

/// Byte casting for &[u32] / &[f32] / &[i32] → &[u8] (avoids direct bytemuck dep)
fn bytemuck_cast_slice<T: Copy>(data: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data),
        )
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn model_dir() -> std::path::PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        std::path::PathBuf::from(home).join(".nanoimg/models")
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    }

    #[test]
    fn gpu_upload_download() {
        let ctx = match GpuContext::try_new() {
            Some(c) => c, None => return,
        };
        let exec = GpuExecutor::new(ctx);
        let data: Vec<f32> = (0..1024).map(|i| i as f32 * 0.1).collect();
        let buf = exec.upload_f32(&data);
        let out = exec.download_f32(&buf, 1024);
        assert_eq!(data, out);
    }

    /// Device errors (here: buffers over the device limit, like running out of
    /// VRAM) must come back as Err instead of panicking, and a fresh executor
    /// must work afterwards.
    #[test]
    #[ignore]
    fn gpu_error_is_returned_not_panicked() {
        let Some(ctx) = GpuContext::with_buffer_limit(Some(1 << 20)) else { return };
        let mut exec = GpuExecutor::new(ctx);
        let model = OnnxModel::load(&model_dir().join("siglip2_image.onnx")).expect("load");
        let input = || Tensor::f32(vec![1, 3, 224, 224], vec![0.1f32; 3 * 224 * 224]);
        let err = exec.run(&model, vec![("pixel_values", input())]).err().expect("should fail");
        eprintln!("{err:#}");
        assert!(format!("{err:#}").contains("GPU error"), "{err:#}");
        // The same executor reports (not panics) on the next try as well
        assert!(exec.run(&model, vec![("pixel_values", input())]).is_err());

        let mut fresh = GpuExecutor::new(GpuContext::try_new().unwrap());
        let out = fresh.run(&model, vec![("pixel_values", input())]).expect("fresh device");
        assert_eq!(out["pooler_output"].shape, vec![1, 768]);
    }

    #[test]
    #[ignore]
    fn gpu_cpu_image_match() {
        let ctx = match GpuContext::try_new() { Some(c) => c, None => return };
        let mut exec = GpuExecutor::new(ctx);
        let model = OnnxModel::load(&model_dir().join("siglip2_image.onnx")).expect("load");
        let px: Vec<f32> = (0..3*224*224).map(|i| ((i as f32) * 0.001) % 1.0 - 0.5).collect();
        let input = Tensor::f32(vec![1, 3, 224, 224], px);
        let gpu = exec.run(&model, vec![("pixel_values", input.clone())]).expect("gpu");
        let cpu = crate::onnx::run(&model, vec![("pixel_values", input)]).expect("cpu");
        let c = cosine(gpu["pooler_output"].as_f32(), cpu["pooler_output"].as_f32());
        eprintln!("image cosine: {c:.6}");
        assert!(c > 0.999, "diverge: {c}");
    }

    /// Batched GPU inference (Conv, Reshape, attention pooling over B > 1, and
    /// matmuls larger than one 64x64 tile) must equal per-image CPU inference —
    /// for every installed model, including the fp16 ones.
    #[test]
    #[ignore]
    fn gpu_batch_matches_cpu_singles() {
        let data_dir = model_dir().parent().unwrap().to_path_buf();
        let mut tested = 0;
        for m in crate::models::MODELS {
            let path = m.image_model(&data_dir);
            if !path.exists() { eprintln!("{}: not installed, skipped", m.name); continue; }
            let ctx = match GpuContext::try_new() { Some(c) => c, None => return };
            let mut exec = GpuExecutor::new(ctx);
            let model = OnnxModel::load(&path).expect("load");
            let n = 3 * m.image_size * m.image_size;
            let imgs: Vec<Vec<f32>> = (0..3).map(|k| {
                (0..n).map(|i| (((i * (k + 1) * 7919) % 1000) as f32 / 1000.0) - 0.5).collect()
            }).collect();
            let batch = Tensor::f32(vec![3, 3, m.image_size, m.image_size], imgs.concat());
            let gpu = exec.run(&model, vec![("pixel_values", batch)]).expect("gpu");
            let pooled = &gpu["pooler_output"];
            assert_eq!(pooled.shape, vec![3, m.dims], "{}", m.name);
            for (k, img) in imgs.into_iter().enumerate() {
                let single = Tensor::f32(vec![1, 3, m.image_size, m.image_size], img);
                let cpu = crate::onnx::run(&model, vec![("pixel_values", single)]).expect("cpu");
                let got = &pooled.as_f32()[k * m.dims..(k + 1) * m.dims];
                let c = cosine(got, cpu["pooler_output"].as_f32());
                eprintln!("{} batch item {k} cosine: {c:.6}", m.name);
                assert!(c > 0.999, "{} item {k} diverges: {c}", m.name);
            }
            tested += 1;
        }
        assert!(tested > 0, "no models installed");
    }

    #[test]
    #[ignore]
    fn gpu_cpu_text_match() {
        let ctx = match GpuContext::try_new() { Some(c) => c, None => return };
        let mut exec = GpuExecutor::new(ctx);
        let dir = model_dir();
        let model = OnnxModel::load(&dir.join("siglip2_text.onnx")).expect("load");
        let tok = crate::tokenizer::BpeTokenizer::load(&dir.join("tokenizer.json")).expect("tok");
        let input = Tensor::i64(vec![1, 64], tok.encode("a photo of a cat on a windowsill", 64));
        let gpu = exec.run(&model, vec![("input_ids", input.clone())]).expect("gpu");
        let cpu = crate::onnx::run(&model, vec![("input_ids", input)]).expect("cpu");
        let c = cosine(gpu["pooler_output"].as_f32(), cpu["pooler_output"].as_f32());
        eprintln!("text cosine: {c:.6}");
        assert!(c > 0.999, "diverge: {c}");
    }

    #[test]
    #[ignore]
    fn bench_image_gpu() {
        let ctx = match GpuContext::try_new() { Some(c) => c, None => return };
        eprintln!("GPU: {}", ctx.name);
        let mut exec = GpuExecutor::new(ctx);
        let model = OnnxModel::load(&model_dir().join("siglip2_image.onnx")).expect("load");
        let input = Tensor::f32(vec![1, 3, 224, 224], vec![0.0f32; 3 * 224 * 224]);
        let _ = exec.run(&model, vec![("pixel_values", input.clone())]);
        let n = 10;
        let mut t: Vec<f64> = (0..n).map(|_| {
            let s = std::time::Instant::now();
            let _ = exec.run(&model, vec![("pixel_values", input.clone())]);
            s.elapsed().as_secs_f64() * 1000.0
        }).collect();
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!("GPU image ({n}): min={:.1}ms mean={:.1}ms max={:.1}ms",
            t[0], t.iter().sum::<f64>() / n as f64, t[n - 1]);
    }

    #[test]
    #[ignore]
    fn bench_text_gpu() {
        let ctx = match GpuContext::try_new() { Some(c) => c, None => return };
        eprintln!("GPU: {}", ctx.name);
        let mut exec = GpuExecutor::new(ctx);
        let dir = model_dir();
        let model = OnnxModel::load(&dir.join("siglip2_text.onnx")).expect("load");
        let tok = crate::tokenizer::BpeTokenizer::load(&dir.join("tokenizer.json")).expect("tok");
        let input = Tensor::i64(vec![1, 64], tok.encode("a photo of a cat", 64));
        let _ = exec.run(&model, vec![("input_ids", input.clone())]);
        let n = 10;
        let mut t: Vec<f64> = (0..n).map(|_| {
            let s = std::time::Instant::now();
            let _ = exec.run(&model, vec![("input_ids", input.clone())]);
            s.elapsed().as_secs_f64() * 1000.0
        }).collect();
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!("GPU text ({n}): min={:.1}ms mean={:.1}ms max={:.1}ms",
            t[0], t.iter().sum::<f64>() / n as f64, t[n - 1]);
    }
}
