//! GPU inference backend using wgpu compute shaders.
//! Compiled only with `--features gpu`.

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

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
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
            ..Default::default()
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
        let (device, queue) = match pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("nanoimg"),
                required_limits: wgpu::Limits {
                    max_buffer_size: adapter_limits.max_buffer_size,
                    max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
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

const TILE: u32 = 16u;
var<workgroup> tile_a: array<f32, 256>; // 16*16
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    let batch_idx = wid.z;
    let a_off = batch_idx * params.a_batch_stride;
    let b_off = batch_idx * params.b_batch_stride;
    let c_off = batch_idx * params.c_batch_stride;

    var acc: f32 = 0.0;
    let num_tiles = (params.K + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE + lid.x;
        let b_row = t * TILE + lid.y;

        if (row < params.M && a_col < params.K) {
            tile_a[lid.y * TILE + lid.x] = a[a_off + row * params.K + a_col];
        } else {
            tile_a[lid.y * TILE + lid.x] = 0.0;
        }

        if (b_row < params.K && col < params.N) {
            tile_b[lid.y * TILE + lid.x] = b[b_off + b_row * params.N + col];
        } else {
            tile_b[lid.y * TILE + lid.x] = 0.0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE; k = k + 1u) {
            acc += tile_a[lid.y * TILE + k] * tile_b[k * TILE + lid.x];
        }

        workgroupBarrier();
    }

    if (row < params.M && col < params.N) {
        c[c_off + row * params.N + col] = acc;
    }
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
    _pad: u32,
    // packed: out_shape[8], a_strides[8], b_strides[8], out_strides[8]
}
@group(0) @binding(3) var<uniform> params: BinaryParams;
@group(0) @binding(4) var<storage, read> shape_data: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.total) { return; }

    let ndim = params.ndim;
    // shape_data layout: out_shape[ndim], a_strides[ndim], b_strides[ndim], out_strides[ndim]
    var ai: u32 = 0u;
    var bi: u32 = 0u;
    var rem = i;
    for (var d: u32 = 0u; d < ndim; d = d + 1u) {
        let out_stride = shape_data[3u * ndim + d];
        let coord = rem / out_stride;
        rem = rem % out_stride;
        ai += coord * shape_data[ndim + d];
        bi += coord * shape_data[2u * ndim + d];
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
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
    _pad: u32,
}
@group(0) @binding(2) var<uniform> params: ConvParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.n_patches * params.patch_sz;
    if (gid.x >= total) { return; }

    let pi = gid.x / params.patch_sz;
    let fi = gid.x % params.patch_sz;
    let py = pi / params.w_out;
    let px = pi % params.w_out;
    let c = fi / (params.kh * params.kw);
    let rem = fi % (params.kh * params.kw);
    let ky = rem / params.kw;
    let kx = rem % params.kw;

    let ys = py * params.sh + ky;
    let xs = px * params.sw + kx;
    col[gid.x] = input[c * params.h * params.w + ys * params.w + xs];
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

pub struct GpuExecutor {
    ctx: GpuContext,
    pipelines: Pipelines,
    weight_cache: HashMap<String, GpuTensor>,
}

impl GpuExecutor {
    pub fn new(ctx: GpuContext) -> Self {
        let pipelines = Pipelines::new(&ctx.device);
        GpuExecutor { ctx, pipelines, weight_cache: HashMap::new() }
    }

    #[allow(dead_code)]
    pub fn name(&self) -> &str { &self.ctx.name }

    fn upload_f32(&self, data: &[f32]) -> wgpu::Buffer {
        self.ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck_cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn upload_i32(&self, data: &[i32]) -> wgpu::Buffer {
        self.ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck_cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        })
    }

    fn upload_u32(&self, data: &[u32]) -> wgpu::Buffer {
        self.ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck_cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        })
    }

    fn create_uniform(&self, data: &[u8]) -> wgpu::Buffer {
        self.ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: data,
            usage: wgpu::BufferUsages::UNIFORM,
        })
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
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<f32> = data.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        drop(data);
        staging.unmap();
        result
    }

    fn upload_tensor(&self, t: &Tensor) -> GpuTensor {
        GpuTensor { shape: t.shape.clone(), buffer: self.upload_f32(t.as_f32()) }
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

    fn div_ceil(a: u32, b: u32) -> u32 { (a + b - 1) / b }

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
        let a_batch_stride = if a_batch_elems > 1 { (m * k) as u32 } else { 0 };
        let b_batch_stride = if b_batch_elems > 1 { (k * n) as u32 } else { 0 };
        let c_batch_stride = (m * n) as u32;
        let out_len = (batch * m * n) as usize;
        let out_buf = self.create_storage((out_len * 4) as u64);

        let params: [u32; 8] = [m, n, k, batch, a_batch_stride, b_batch_stride, c_batch_stride, 0];
        let param_buf = self.create_uniform(bytemuck_cast_slice(&params));
        self.dispatch(&self.pipelines.matmul,
            &[&a.buffer, &b.buffer, &out_buf, &param_buf],
            (Self::div_ceil(n, 16), Self::div_ceil(m, 16), batch), enc);

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

        // Flush pending GPU work before downloading
        self.flush(enc);

        let a_data = self.download_f32(&a.buffer, a.numel());
        let b_data = self.download_f32(&b.buffer, b.numel());
        let a_cpu = Tensor::f32(a.shape.clone(), a_data);
        let b_cpu = Tensor::f32(b.shape.clone(), b_data);
        let c_cpu = c.map(|ct| {
            let cd = self.download_f32(&ct.buffer, ct.numel());
            Tensor::f32(ct.shape.clone(), cd)
        });

        // Build a fake node with the attrs
        let result = crate::onnx::cpu_gemm(&a_cpu, &b_cpu, c_cpu.as_ref(), trans_a, trans_b, alpha, beta);
        self.upload_tensor(&result)
    }

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

        // Phase 1: im2col on GPU
        let col_len = (n_patches * patch) as usize;
        let col_buf = self.create_storage((col_len * 4) as u64);

        let conv_params: [u32; 12] = [c_in, h, w, kh, kw, sh, sw, h_out, w_out, patch, n_patches, 0];
        let param_buf = self.create_uniform(bytemuck_cast_slice(&conv_params));
        self.dispatch(&self.pipelines.conv_im2col,
            &[&input.buffer, &col_buf, &param_buf],
            (Self::div_ceil(n_patches * patch, 256), 1, 1), enc);

        // Flush so im2col result is available for CPU BLAS
        self.flush(enc);

        let col_data = self.download_f32(&col_buf, col_len);
        let w_data = self.download_f32(&weight.buffer, weight.numel());

        // weight [c_out, patch] × col^T [patch, n_patches]
        let w_tensor = Tensor::f32(vec![c_out, patch as usize], w_data);
        let col_tensor = Tensor::f32(vec![n_patches as usize, patch as usize], col_data);
        let result_cpu = crate::onnx::cpu_gemm(&w_tensor, &col_tensor, None, false, true, 1.0, 0.0);
        let mut out = result_cpu.as_f32().to_vec();

        if let Some(bias) = bias {
            let b = self.download_f32(&bias.buffer, bias.numel());
            for j in 0..c_out {
                let row = &mut out[j * n_patches as usize..(j + 1) * n_patches as usize];
                let bj = b[j];
                for v in row { *v += bj; }
            }
        }

        let result = Tensor::f32(vec![n_batch, c_out, h_out as usize, w_out as usize], out);
        self.upload_tensor(&result)
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

        let params: [u32; 4] = [total, ndim, opcode, 0];
        let param_buf = self.create_uniform(bytemuck_cast_slice(&params));
        let shape_buf = self.upload_u32(&shape_data);
        let out_buf = self.create_storage((total as u64) * 4);
        self.dispatch(&self.pipelines.binary,
            &[&a.buffer, &b.buffer, &out_buf, &param_buf, &shape_buf],
            (Self::div_ceil(total, 256), 1, 1), enc);
        GpuTensor { shape: out_shape, buffer: out_buf }
    }

    fn op_unary(&self, a: &GpuTensor, opcode: u32, enc: &mut Option<wgpu::CommandEncoder>) -> GpuTensor {
        let total = a.numel() as u32;
        let out_buf = self.create_storage((total as u64) * 4);
        let params: [u32; 2] = [total, opcode];
        let param_buf = self.create_uniform(bytemuck_cast_slice(&params));
        self.dispatch(&self.pipelines.unary, &[&a.buffer, &out_buf, &param_buf],
            (Self::div_ceil(total, 256), 1, 1), enc);
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
            (Self::div_ceil(outer * inner, 256), 1, 1), enc);
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
                (Self::div_ceil(outer * inner, 256), 1, 1), enc);
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
            (Self::div_ceil(total, 256), 1, 1), enc);
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
            (Self::div_ceil(total as u32, 256), 1, 1), enc);
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
                (Self::div_ceil(total, 256), 1, 1), enc);
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
            let len = if e > s { ((e - s) as u32 + step - 1) / step } else { 0 };
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
            (Self::div_ceil(total, 256), 1, 1), enc);

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
            (Self::div_ceil(total, 256), 1, 1), enc);
        GpuTensor { shape: out_shape, buffer: out_buf }
    }

    // ── Graph executor ───────────────────────────────────────────────────

    pub fn run(&mut self, model: &OnnxModel, inputs: Vec<(&str, Tensor)>) -> Result<HashMap<String, Tensor>> {
        self.ensure_weights_cached(model);

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
                        let shape_vals = shape_tensor.as_i64();
                        let total: usize = a.numel();
                        let mut new_shape: Vec<usize> = shape_vals.iter().map(|&v| {
                            if v == 0 { 1 } else if v == -1 { 0 } else { v as usize }
                        }).collect();
                        let known: usize = new_shape.iter().filter(|&&v| v != 0).product::<usize>().max(1);
                        for v in &mut new_shape {
                            if *v == 0 { *v = total / known; }
                        }
                        let result = a.reshape(new_shape);
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
                        let ts: Vec<&Tensor> = (0..node.inputs.len()).filter_map(|j| get_cpu(j)).collect();
                        let result = crate::onnx::cpu_concat(&ts, axis);
                        cpu_tensors.insert(out_name.clone(), result);
                    } else {
                        let ts: Vec<&GpuTensor> = (0..node.inputs.len())
                            .filter_map(|j| get_gpu(j))
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

/// Byte casting for &[u32] / &[f32] / &[i32] → &[u8] (avoids direct bytemuck dep)
fn bytemuck_cast_slice<T: Copy>(data: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<T>(),
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
