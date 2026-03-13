//! Minimal ONNX runtime tailored to SigLIP2 models.
//! Supports the 21 operators used by the image and text encoders.
//! Weight data is loaded from protobuf into owned tensors at model load time.

use anyhow::{bail, Context, Result};
use memmap2::MmapOptions;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ── BLAS FFI ─────────────────────────────────────────────────────────────────

const ROW_MAJOR: i32 = 101;
const NO_TRANS: i32 = 111;
const TRANS: i32 = 112;

extern "C" {
    fn cblas_sgemm(
        order: i32, transa: i32, transb: i32,
        m: i32, n: i32, k: i32,
        alpha: f32, a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32, c: *mut f32, ldc: i32,
    );
}

// ── Tensor ───────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: TData,
}

#[derive(Clone)]
pub enum TData {
    F32(Arc<Vec<f32>>),
    I64(Arc<Vec<i64>>),
}

impl Tensor {
    pub fn f32(shape: Vec<usize>, data: Vec<f32>) -> Self {
        Self { shape, data: TData::F32(Arc::new(data)) }
    }
    pub fn i64(shape: Vec<usize>, data: Vec<i64>) -> Self {
        Self { shape, data: TData::I64(Arc::new(data)) }
    }
    pub fn as_f32(&self) -> &[f32] {
        match &self.data { TData::F32(v) => v, _ => panic!("expected f32") }
    }
    pub fn as_i64(&self) -> &[i64] {
        match &self.data { TData::I64(v) => v, _ => panic!("expected i64") }
    }
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }
    pub fn is_f32(&self) -> bool {
        matches!(&self.data, TData::F32(_))
    }
}

// ── Model ────────────────────────────────────────────────────────────────────

pub struct OnnxModel {
    pub nodes: Vec<Node>,
    pub weights: HashMap<String, Tensor>,
    #[allow(dead_code)]
    pub graph_inputs: Vec<String>,
    pub graph_outputs: Vec<String>,
}

pub struct Node {
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: HashMap<String, AttrVal>,
}

#[derive(Clone)]
pub enum AttrVal {
    F(f32),
    I(i64),
    Ints(Vec<i64>),
}

impl Node {
    fn attr_i(&self, name: &str) -> Option<i64> {
        match self.attrs.get(name)? { AttrVal::I(v) => Some(*v), _ => None }
    }
    fn attr_ints(&self, name: &str) -> Option<&[i64]> {
        match self.attrs.get(name)? { AttrVal::Ints(v) => Some(v), _ => None }
    }
    fn attr_f(&self, name: &str) -> Option<f32> {
        match self.attrs.get(name)? { AttrVal::F(v) => Some(*v), _ => None }
    }
}

// ── Protobuf parser ──────────────────────────────────────────────────────────

struct PbReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> PbReader<'a> {
    fn new(data: &'a [u8]) -> Self { Self { data, pos: 0 } }
    fn remaining(&self) -> bool { self.pos < self.data.len() }

    fn read_varint(&mut self) -> u64 {
        let mut val = 0u64;
        let mut shift = 0;
        loop {
            let b = self.data[self.pos];
            self.pos += 1;
            val |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 { return val; }
            shift += 7;
            if shift >= 64 { return val; }
        }
    }

    fn read_tag(&mut self) -> (u32, u32) {
        let v = self.read_varint();
        ((v >> 3) as u32, (v & 7) as u32)
    }

    fn read_bytes(&mut self, len: usize) -> &'a [u8] {
        let s = &self.data[self.pos..self.pos + len];
        self.pos += len;
        s
    }

    fn read_len_bytes(&mut self) -> &'a [u8] {
        let len = self.read_varint() as usize;
        self.read_bytes(len)
    }

    fn read_string(&mut self) -> String {
        String::from_utf8_lossy(self.read_len_bytes()).into_owned()
    }

    fn read_fixed32(&mut self) -> u32 {
        let b = self.read_bytes(4);
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }

    fn skip_field(&mut self, wire: u32) {
        match wire {
            0 => { self.read_varint(); }
            1 => { self.pos += 8; }
            2 => { let len = self.read_varint() as usize; self.pos += len; }
            5 => { self.pos += 4; }
            _ => {}
        }
    }
}

// Raw tensor parsed from protobuf (before conversion to Tensor)
struct RawTensor {
    name: String,
    dims: Vec<usize>,
    data_type: i32,
    float_data: Vec<f32>,
    int64_data: Vec<i64>,
    raw_data_offset: usize,
    raw_data_len: usize,
}

fn parse_tensor(data: &[u8], base_offset: usize) -> RawTensor {
    let mut r = PbReader::new(data);
    let mut t = RawTensor {
        name: String::new(), dims: vec![], data_type: 0,
        float_data: vec![], int64_data: vec![],
        raw_data_offset: 0, raw_data_len: 0,
    };
    while r.remaining() {
        let (field, wire) = r.read_tag();
        match (field, wire) {
            (1, 0) => t.dims.push(r.read_varint() as usize),
            (1, 2) => {
                let b = r.read_len_bytes();
                let mut sr = PbReader::new(b);
                while sr.remaining() { t.dims.push(sr.read_varint() as usize); }
            }
            (2, 0) => t.data_type = r.read_varint() as i32,
            (4, 2) => {
                let b = r.read_len_bytes();
                for c in b.chunks_exact(4) {
                    t.float_data.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                }
            }
            (4, 5) => t.float_data.push(f32::from_bits(r.read_fixed32())),
            (7, 2) => {
                let b = r.read_len_bytes();
                let mut sr = PbReader::new(b);
                while sr.remaining() { t.int64_data.push(sr.read_varint() as i64); }
            }
            (7, 0) => t.int64_data.push(r.read_varint() as i64),
            (8, 2) => t.name = r.read_string(),
            (9, 2) => {
                let len = r.read_varint() as usize;
                t.raw_data_offset = base_offset + r.pos;
                t.raw_data_len = len;
                r.pos += len;
            }
            _ => r.skip_field(wire),
        }
    }
    t
}

fn parse_attribute(data: &[u8]) -> (String, AttrVal) {
    let mut r = PbReader::new(data);
    let mut name = String::new();
    let mut f_val: Option<f32> = None;
    let mut i_val: Option<i64> = None;
    let mut ints = Vec::new();
    while r.remaining() {
        let (field, wire) = r.read_tag();
        match (field, wire) {
            (1, 2) => name = r.read_string(),
            (2, 5) => f_val = Some(f32::from_bits(r.read_fixed32())),
            (3, 0) => i_val = Some(r.read_varint() as i64),
            (8, 2) => {
                let b = r.read_len_bytes();
                let mut sr = PbReader::new(b);
                while sr.remaining() { ints.push(sr.read_varint() as i64); }
            }
            (8, 0) => ints.push(r.read_varint() as i64),
            _ => r.skip_field(wire),
        }
    }
    let val = if !ints.is_empty() {
        AttrVal::Ints(ints)
    } else if let Some(f) = f_val {
        AttrVal::F(f)
    } else {
        AttrVal::I(i_val.unwrap_or(0))
    };
    (name, val)
}

fn parse_node(data: &[u8]) -> Node {
    let mut r = PbReader::new(data);
    let mut node = Node {
        op_type: String::new(), inputs: vec![], outputs: vec![],
        attrs: HashMap::new(),
    };
    while r.remaining() {
        let (field, wire) = r.read_tag();
        match (field, wire) {
            (1, 2) => node.inputs.push(r.read_string()),
            (2, 2) => node.outputs.push(r.read_string()),
            (3, 2) => { r.read_len_bytes(); } // skip name
            (4, 2) => node.op_type = r.read_string(),
            (5, 2) => {
                let b = r.read_len_bytes();
                let (name, val) = parse_attribute(b);
                node.attrs.insert(name, val);
            }
            _ => r.skip_field(wire),
        }
    }
    node
}

fn parse_value_info_name(data: &[u8]) -> String {
    let mut r = PbReader::new(data);
    while r.remaining() {
        let (field, wire) = r.read_tag();
        if field == 1 && wire == 2 { return r.read_string(); }
        r.skip_field(wire);
    }
    String::new()
}

impl OnnxModel {
    pub fn load(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("open {}", path.display()))?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let mmap_data = &mmap[..];

        // Parse ModelProto → find GraphProto (field 7)
        let mut r = PbReader::new(mmap_data);
        let mut graph_data: &[u8] = &[];
        let mut graph_offset: usize = 0;
        while r.remaining() {
            let (field, wire) = r.read_tag();
            if field == 7 && wire == 2 {
                let len = r.read_varint() as usize;
                graph_offset = r.pos;
                graph_data = &mmap_data[r.pos..r.pos + len];
                r.pos += len;
            } else {
                r.skip_field(wire);
            }
        }
        if graph_data.is_empty() { bail!("no graph in ONNX model"); }

        // Parse GraphProto
        let mut nodes = Vec::new();
        let mut raw_inits = Vec::new();
        let mut graph_inputs = Vec::new();
        let mut graph_outputs = Vec::new();

        let mut r = PbReader::new(graph_data);
        while r.remaining() {
            let (field, wire) = r.read_tag();
            match (field, wire) {
                (1, 2) => nodes.push(parse_node(r.read_len_bytes())),
                (5, 2) => {
                    let len = r.read_varint() as usize;
                    let abs = graph_offset + r.pos;
                    let bytes = &graph_data[r.pos..r.pos + len];
                    r.pos += len;
                    raw_inits.push(parse_tensor(bytes, abs));
                }
                (11, 2) => {
                    let name = parse_value_info_name(r.read_len_bytes());
                    if !name.is_empty() { graph_inputs.push(name); }
                }
                (12, 2) => {
                    let name = parse_value_info_name(r.read_len_bytes());
                    if !name.is_empty() { graph_outputs.push(name); }
                }
                _ => r.skip_field(wire),
            }
        }

        // Convert raw initializers to owned Tensors
        let mut weights = HashMap::new();
        for raw in raw_inits {
            let tensor = if raw.raw_data_len > 0 {
                let bytes = &mmap_data[raw.raw_data_offset..raw.raw_data_offset + raw.raw_data_len];
                match raw.data_type {
                    1 => {
                        let data: Vec<f32> = bytes.chunks_exact(4)
                            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                            .collect();
                        Tensor::f32(raw.dims, data)
                    }
                    7 => {
                        let data: Vec<i64> = bytes.chunks_exact(8)
                            .map(|b| i64::from_le_bytes([b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7]]))
                            .collect();
                        Tensor::i64(raw.dims, data)
                    }
                    dt => bail!("unsupported data type {} in initializer {}", dt, raw.name),
                }
            } else if !raw.float_data.is_empty() {
                Tensor::f32(raw.dims, raw.float_data)
            } else if !raw.int64_data.is_empty() {
                Tensor::i64(raw.dims, raw.int64_data)
            } else {
                Tensor::f32(raw.dims, vec![])
            };
            weights.insert(raw.name, tensor);
        }

        // Filter graph_inputs to only actual runtime inputs (not initializers)
        graph_inputs.retain(|name| !weights.contains_key(name));

        // mmap is dropped here — all weight data is in owned Vecs
        Ok(OnnxModel { nodes, weights, graph_inputs, graph_outputs })
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    if n == 0 { return vec![1]; }
    let mut s = vec![1usize; n];
    for i in (0..n - 1).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let n = a.len().max(b.len());
    let mut out = vec![0; n];
    for i in 0..n {
        let da = if i < n - a.len() { 1 } else { a[i - (n - a.len())] };
        let db = if i < n - b.len() { 1 } else { b[i - (n - b.len())] };
        assert!(da == db || da == 1 || db == 1,
            "broadcast: incompatible dims {} vs {}", da, db);
        out[i] = da.max(db);
    }
    out
}

/// Compute strides for an input tensor broadcast to `out_shape`.
/// Dimensions of size 1 get stride 0 (repeated).
fn broadcast_strides(in_shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
    let n = out_shape.len();
    let in_n = in_shape.len();
    let off = n - in_n;
    let in_s = compute_strides(in_shape);
    let mut result = vec![0usize; n];
    for d in 0..n {
        if d >= off && in_shape[d - off] != 1 {
            result[d] = in_s[d - off];
        }
    }
    result
}

fn normalize_axis(axis: i64, ndim: usize) -> usize {
    if axis < 0 { (ndim as i64 + axis) as usize } else { axis as usize }
}

fn broadcast_batch_idx(flat: usize, out_batch: &[usize], in_batch: &[usize]) -> usize {
    let n = out_batch.len();
    let in_n = in_batch.len();
    if in_n == 0 { return 0; }
    let off = n - in_n;
    let mut result = 0;
    let mut remaining = flat;
    let mut stride = 1;
    for d in (0..n).rev() {
        let coord = remaining % out_batch[d];
        remaining /= out_batch[d];
        if d >= off {
            let id = d - off;
            let c = if in_batch[id] == 1 { 0 } else { coord };
            result += c * stride;
            stride *= in_batch[id];
        }
    }
    result
}

// ── Operators ────────────────────────────────────────────────────────────────

fn binary_op(a: &Tensor, b: &Tensor, f: fn(f32, f32) -> f32) -> Tensor {
    let ad = a.as_f32();
    let bd = b.as_f32();

    // Fast path: same shape (residual adds — most common)
    if a.shape == b.shape {
        return Tensor::f32(a.shape.clone(),
            ad.iter().zip(bd).map(|(&x, &y)| f(x, y)).collect());
    }
    // Fast path: b is scalar (attention scaling)
    if bd.len() == 1 {
        let s = bd[0];
        return Tensor::f32(a.shape.clone(), ad.iter().map(|&x| f(x, s)).collect());
    }
    // Fast path: a is scalar
    if ad.len() == 1 {
        let s = ad[0];
        return Tensor::f32(b.shape.clone(), bd.iter().map(|&y| f(s, y)).collect());
    }
    // Fast path: inner broadcast (LayerNorm γ/β, e.g. [1,197,768] op [768])
    let inner = *a.shape.last().unwrap_or(&0);
    if bd.len() == inner && inner > 0 && ad.len() % inner == 0 {
        let out_shape = broadcast_shape(&a.shape, &b.shape);
        return Tensor::f32(out_shape,
            ad.chunks_exact(inner)
                .flat_map(|c| c.iter().zip(bd).map(|(&x, &y)| f(x, y)))
                .collect());
    }

    // Generic broadcast path
    let out_shape = broadcast_shape(&a.shape, &b.shape);
    let n = out_shape.len();
    let out_size: usize = out_shape.iter().product::<usize>().max(1);
    let a_s = broadcast_strides(&a.shape, &out_shape);
    let b_s = broadcast_strides(&b.shape, &out_shape);
    let out_s = compute_strides(&out_shape);
    let mut out = vec![0f32; out_size];
    for i in 0..out_size {
        let mut ai = 0;
        let mut bi = 0;
        let mut rem = i;
        for d in 0..n {
            let coord = rem / out_s[d];
            rem %= out_s[d];
            ai += coord * a_s[d];
            bi += coord * b_s[d];
        }
        out[i] = f(ad[ai], bd[bi]);
    }
    Tensor::f32(out_shape, out)
}

fn unary_op(a: &Tensor, f: fn(f32) -> f32) -> Tensor {
    let d = a.as_f32();
    Tensor::f32(a.shape.clone(), d.iter().map(|&x| f(x)).collect())
}

fn op_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let an = a.shape.len();
    let bn = b.shape.len();
    assert!(an >= 2 && bn >= 2, "matmul: need 2-D+, got a={:?} b={:?}", a.shape, b.shape);
    let m = a.shape[an - 2];
    let k = a.shape[an - 1];
    let n = b.shape[bn - 1];
    assert_eq!(k, b.shape[bn - 2], "matmul: inner dims mismatch");

    let a_batch = &a.shape[..an - 2];
    let b_batch = &b.shape[..bn - 2];
    let out_batch = broadcast_shape(a_batch, b_batch);
    let batch_size: usize = out_batch.iter().product::<usize>().max(1);
    let a_mat = m * k;
    let b_mat = k * n;
    let c_mat = m * n;
    let ad = a.as_f32();
    let bd = b.as_f32();
    let mut out = vec![0f32; batch_size * c_mat];

    for bi in 0..batch_size {
        let ai = broadcast_batch_idx(bi, &out_batch, a_batch) * a_mat;
        let bj = broadcast_batch_idx(bi, &out_batch, b_batch) * b_mat;
        let ci = bi * c_mat;
        unsafe {
            cblas_sgemm(
                ROW_MAJOR, NO_TRANS, NO_TRANS,
                m as i32, n as i32, k as i32,
                1.0, ad[ai..].as_ptr(), k as i32,
                bd[bj..].as_ptr(), n as i32,
                0.0, out[ci..].as_mut_ptr(), n as i32,
            );
        }
    }

    let mut shape = out_batch;
    shape.push(m);
    shape.push(n);
    Tensor::f32(shape, out)
}

fn op_gemm(a: &Tensor, b: &Tensor, c: Option<&Tensor>, node: &Node) -> Tensor {
    let trans_a = node.attr_i("transA").unwrap_or(0) != 0;
    let trans_b = node.attr_i("transB").unwrap_or(0) != 0;
    let alpha = node.attr_f("alpha").unwrap_or(1.0);
    let beta = node.attr_f("beta").unwrap_or(1.0);

    let (m, k_a) = if trans_a { (a.shape[1], a.shape[0]) } else { (a.shape[0], a.shape[1]) };
    let (k_b, n) = if trans_b { (b.shape[1], b.shape[0]) } else { (b.shape[0], b.shape[1]) };
    assert_eq!(k_a, k_b, "gemm: inner dims mismatch");

    let ad = a.as_f32();
    let bd = b.as_f32();
    let mut out = vec![0f32; m * n];

    // Initialize with bias if present
    if let Some(bias) = c {
        let bdata = bias.as_f32();
        if bdata.len() == n {
            for i in 0..m {
                out[i * n..(i + 1) * n].copy_from_slice(bdata);
            }
        } else {
            out[..bdata.len().min(m * n)].copy_from_slice(&bdata[..bdata.len().min(m * n)]);
        }
    }

    let lda = if trans_a { m } else { k_a };
    let ldb = if trans_b { k_b } else { n };

    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            if trans_a { TRANS } else { NO_TRANS },
            if trans_b { TRANS } else { NO_TRANS },
            m as i32, n as i32, k_a as i32,
            alpha, ad.as_ptr(), lda as i32,
            bd.as_ptr(), ldb as i32,
            beta, out.as_mut_ptr(), n as i32,
        );
    }

    Tensor::f32(vec![m, n], out)
}

fn op_conv(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>, node: &Node) -> Tensor {
    let n_batch = input.shape[0];
    let c_in = input.shape[1];
    let h = input.shape[2];
    let w = input.shape[3];
    let c_out = weight.shape[0];
    let kh = weight.shape[2];
    let kw = weight.shape[3];

    let strides = node.attr_ints("strides").unwrap_or(&[1, 1]);
    let (sh, sw) = (strides[0] as usize, strides[1] as usize);
    let h_out = (h - kh) / sh + 1;
    let w_out = (w - kw) / sw + 1;
    let patch = c_in * kh * kw;
    let n_patches = h_out * w_out;

    let in_data = input.as_f32();
    let w_data = weight.as_f32();

    // im2col
    let mut col = vec![0f32; n_patches * patch];
    for py in 0..h_out {
        for px in 0..w_out {
            let (ys, xs) = (py * sh, px * sw);
            let pi = py * w_out + px;
            for c in 0..c_in {
                for ky in 0..kh {
                    for kx in 0..kw {
                        col[pi * patch + c * kh * kw + ky * kw + kx] =
                            in_data[c * h * w + (ys + ky) * w + (xs + kx)];
                    }
                }
            }
        }
    }

    // weight [c_out, patch] × col^T [patch, n_patches] → [c_out, n_patches] directly
    let mut out = vec![0f32; c_out * n_patches];
    unsafe {
        cblas_sgemm(
            ROW_MAJOR, NO_TRANS, TRANS,
            c_out as i32, n_patches as i32, patch as i32,
            1.0, w_data.as_ptr(), patch as i32,
            col.as_ptr(), patch as i32,
            0.0, out.as_mut_ptr(), n_patches as i32,
        );
    }

    // Add bias ([c_out, n_patches] layout)
    if let Some(bias) = bias {
        let b = bias.as_f32();
        for j in 0..c_out {
            let row = &mut out[j * n_patches..(j + 1) * n_patches];
            let bj = b[j];
            for v in row { *v += bj; }
        }
    }

    Tensor::f32(vec![n_batch, c_out, h_out, w_out], out)
}

fn op_softmax(data: &Tensor, axis: i64) -> Tensor {
    let axis = normalize_axis(axis, data.shape.len());
    let d = data.as_f32();
    let mut out = vec![0f32; d.len()];
    let outer: usize = data.shape[..axis].iter().product();
    let dim = data.shape[axis];
    let inner: usize = data.shape[axis + 1..].iter().product::<usize>().max(1);

    for o in 0..outer {
        for i in 0..inner {
            let base = o * dim * inner + i;
            let mut max = f32::NEG_INFINITY;
            for dd in 0..dim { max = max.max(d[base + dd * inner]); }
            let mut sum = 0f32;
            for dd in 0..dim {
                let idx = base + dd * inner;
                let v = (d[idx] - max).exp();
                out[idx] = v;
                sum += v;
            }
            for dd in 0..dim { out[base + dd * inner] /= sum; }
        }
    }
    Tensor::f32(data.shape.clone(), out)
}

fn op_reduce_mean(data: &Tensor, axes: &[i64], keepdims: bool) -> Tensor {
    let ndim = data.shape.len();
    let axes: Vec<usize> = axes.iter().map(|&a| normalize_axis(a, ndim)).collect();

    let mut out_shape = Vec::new();
    for (d, &s) in data.shape.iter().enumerate() {
        if axes.contains(&d) {
            if keepdims { out_shape.push(1); }
        } else {
            out_shape.push(s);
        }
    }
    if out_shape.is_empty() { out_shape.push(1); }

    let in_data = data.as_f32();
    let reduce_count: usize = axes.iter().map(|&a| data.shape[a]).product();
    let rf = reduce_count as f32;

    // Fast path: single-axis reduction (common case)
    if axes.len() == 1 {
        let axis = axes[0];
        let outer: usize = data.shape[..axis].iter().product::<usize>().max(1);
        let dim = data.shape[axis];
        let inner: usize = data.shape[axis + 1..].iter().product::<usize>().max(1);
        let mut out = vec![0f32; outer * inner];
        for o in 0..outer {
            for d in 0..dim {
                let src = (o * dim + d) * inner;
                let dst = o * inner;
                for i in 0..inner {
                    out[dst + i] += in_data[src + i];
                }
            }
        }
        for v in &mut out { *v /= rf; }
        return Tensor::f32(out_shape, out);
    }

    // Generic path
    let out_size: usize = out_shape.iter().product::<usize>().max(1);
    let in_strides = compute_strides(&data.shape);
    let out_strides = compute_strides(&out_shape);
    let mut out = vec![0f32; out_size];

    let in_size: usize = data.shape.iter().product::<usize>().max(1);
    for in_flat in 0..in_size {
        let mut out_flat = 0;
        let mut rem = in_flat;
        let mut od = 0;
        for d in 0..ndim {
            let coord = rem / in_strides[d];
            rem %= in_strides[d];
            if axes.contains(&d) {
                if keepdims { od += 1; }
            } else {
                if od < out_strides.len() {
                    out_flat += coord * out_strides[od];
                }
                od += 1;
            }
        }
        out[out_flat] += in_data[in_flat];
    }
    for v in &mut out { *v /= rf; }
    Tensor::f32(out_shape, out)
}

fn op_reshape(data: &Tensor, shape: &Tensor) -> Tensor {
    let shape_vals = shape.as_i64();
    let total: usize = data.numel();
    let mut new_shape: Vec<usize> = shape_vals.iter().map(|&v| {
        if v == 0 { 1 } else if v == -1 { 0 } else { v as usize }
    }).collect();

    // Resolve -1
    let known: usize = new_shape.iter().filter(|&&v| v != 0).product::<usize>().max(1);
    for v in &mut new_shape {
        if *v == 0 { *v = total / known; }
    }

    Tensor { shape: new_shape, data: data.data.clone() }
}

fn op_transpose(data: &Tensor, perm: Option<&[i64]>) -> Tensor {
    let ndim = data.shape.len();
    let default_perm: Vec<i64> = (0..ndim as i64).rev().collect();
    let perm = perm.unwrap_or(&default_perm);
    let perm_u: Vec<usize> = perm.iter().map(|&p| p as usize).collect();

    let out_shape: Vec<usize> = perm_u.iter().map(|&p| data.shape[p]).collect();
    let out_size: usize = out_shape.iter().product::<usize>().max(1);
    let in_strides = compute_strides(&data.shape);
    let out_strides = compute_strides(&out_shape);
    let in_data = data.as_f32();
    let mut out = vec![0f32; out_size];

    for i in 0..out_size {
        let mut in_idx = 0;
        let mut rem = i;
        for d in 0..ndim {
            let coord = rem / out_strides[d];
            rem %= out_strides[d];
            in_idx += coord * in_strides[perm_u[d]];
        }
        out[i] = in_data[in_idx];
    }
    Tensor::f32(out_shape, out)
}

fn op_unsqueeze(data: &Tensor, axes: &Tensor) -> Tensor {
    let axes_vals = axes.as_i64();
    let new_ndim = data.shape.len() + axes_vals.len();
    let mut axes_set: Vec<usize> = axes_vals.iter()
        .map(|&a| if a < 0 { (new_ndim as i64 + a) as usize } else { a as usize })
        .collect();
    axes_set.sort();

    let mut new_shape = Vec::with_capacity(new_ndim);
    let mut old_idx = 0;
    for i in 0..new_ndim {
        if axes_set.contains(&i) {
            new_shape.push(1);
        } else {
            new_shape.push(data.shape[old_idx]);
            old_idx += 1;
        }
    }

    Tensor { shape: new_shape, data: data.data.clone() }
}

fn op_squeeze(data: &Tensor, axes: Option<&Tensor>) -> Tensor {
    let new_shape: Vec<usize> = if let Some(axes) = axes {
        let ax: Vec<usize> = axes.as_i64().iter()
            .map(|&a| normalize_axis(a, data.shape.len()))
            .collect();
        data.shape.iter().enumerate()
            .filter(|(i, _)| !ax.contains(i))
            .map(|(_, &s)| s)
            .collect()
    } else {
        data.shape.iter().copied().filter(|&s| s != 1).collect()
    };

    Tensor { shape: new_shape, data: data.data.clone() }
}

fn op_concat(tensors: &[&Tensor], axis: i64) -> Tensor {
    let ndim = tensors[0].shape.len();
    let axis = normalize_axis(axis, ndim);
    let mut out_shape = tensors[0].shape.clone();
    out_shape[axis] = tensors.iter().map(|t| t.shape[axis]).sum();

    let outer: usize = out_shape[..axis].iter().product::<usize>().max(1);
    let inner: usize = out_shape[axis + 1..].iter().product::<usize>().max(1);

    let is_i64 = matches!(&tensors[0].data, TData::I64(_));
    if is_i64 {
        let out_size: usize = out_shape.iter().product();
        let mut out = vec![0i64; out_size];
        for o in 0..outer {
            let mut off = 0;
            for t in tensors {
                let td = t.as_i64();
                let ta = t.shape[axis];
                for a in 0..ta {
                    let src = (o * ta + a) * inner;
                    let dst = (o * out_shape[axis] + off + a) * inner;
                    out[dst..dst + inner].copy_from_slice(&td[src..src + inner]);
                }
                off += ta;
            }
        }
        return Tensor::i64(out_shape, out);
    }

    let out_size: usize = out_shape.iter().product();
    let mut out = vec![0f32; out_size];
    for o in 0..outer {
        let mut off = 0;
        for t in tensors {
            let td = t.as_f32();
            let ta = t.shape[axis];
            for a in 0..ta {
                let src = (o * ta + a) * inner;
                let dst = (o * out_shape[axis] + off + a) * inner;
                out[dst..dst + inner].copy_from_slice(&td[src..src + inner]);
            }
            off += ta;
        }
    }
    Tensor::f32(out_shape, out)
}

fn op_slice(data: &Tensor, starts: &Tensor, ends: &Tensor,
            axes: Option<&Tensor>, steps: Option<&Tensor>) -> Tensor {
    let ndim = data.shape.len();
    let starts_v = starts.as_i64();
    let ends_v = ends.as_i64();
    let n = starts_v.len();
    let default_axes: Vec<i64> = (0..n as i64).collect();
    let axes_v = axes.map(|a| a.as_i64()).unwrap_or(&default_axes);
    let default_steps: Vec<i64> = vec![1; n];
    let steps_v = steps.map(|s| s.as_i64()).unwrap_or(&default_steps);

    // Compute slice params per axis
    let mut slice_start = vec![0usize; ndim];
    let mut slice_step = vec![1usize; ndim];
    let mut out_shape = data.shape.clone();

    for i in 0..n {
        let axis = normalize_axis(axes_v[i], ndim);
        let dim = data.shape[axis] as i64;
        let step = steps_v[i].max(1) as usize;
        let mut s = starts_v[i];
        let mut e = ends_v[i];
        if s < 0 { s += dim; }
        if e < 0 { e += dim; }
        s = s.clamp(0, dim);
        e = e.clamp(0, dim);
        if e > dim { e = dim; }
        let len = if e > s { ((e - s) as usize + step - 1) / step } else { 0 };
        slice_start[axis] = s as usize;
        slice_step[axis] = step;
        out_shape[axis] = len;
    }

    let out_size: usize = out_shape.iter().product::<usize>().max(1);
    let in_strides = compute_strides(&data.shape);
    let out_strides = compute_strides(&out_shape);

    macro_rules! do_slice {
        ($src:expr, $zero:expr) => {{
            let mut out = vec![$zero; out_size];
            for oi in 0..out_size {
                let mut in_idx = 0;
                let mut rem = oi;
                for d in 0..ndim {
                    let coord = rem / out_strides[d];
                    rem %= out_strides[d];
                    in_idx += (slice_start[d] + coord * slice_step[d]) * in_strides[d];
                }
                out[oi] = $src[in_idx];
            }
            out
        }};
    }

    if data.is_f32() {
        Tensor::f32(out_shape, do_slice!(data.as_f32(), 0f32))
    } else {
        Tensor::i64(out_shape, do_slice!(data.as_i64(), 0i64))
    }
}

fn op_gather(data: &Tensor, indices: &Tensor, axis: i64) -> Tensor {
    let axis = normalize_axis(axis, data.shape.len());
    let idx = indices.as_i64();
    let axis_size = data.shape[axis];
    let mut out_shape = Vec::new();
    out_shape.extend_from_slice(&data.shape[..axis]);
    out_shape.extend_from_slice(&indices.shape);
    out_shape.extend_from_slice(&data.shape[axis + 1..]);

    let outer: usize = data.shape[..axis].iter().product::<usize>().max(1);
    let inner: usize = data.shape[axis + 1..].iter().product::<usize>().max(1);
    let idx_count: usize = indices.shape.iter().product::<usize>().max(1);

    match &data.data {
        TData::F32(dv) => {
            let mut out = vec![0f32; out_shape.iter().product::<usize>().max(1)];
            for o in 0..outer {
                for (ip, &iv) in idx.iter().enumerate() {
                    let i = if iv < 0 { axis_size as i64 + iv } else { iv } as usize;
                    let src = (o * axis_size + i) * inner;
                    let dst = (o * idx_count + ip) * inner;
                    out[dst..dst + inner].copy_from_slice(&dv[src..src + inner]);
                }
            }
            Tensor::f32(out_shape, out)
        }
        TData::I64(dv) => {
            let mut out = vec![0i64; out_shape.iter().product::<usize>().max(1)];
            for o in 0..outer {
                for (ip, &iv) in idx.iter().enumerate() {
                    let i = if iv < 0 { axis_size as i64 + iv } else { iv } as usize;
                    let src = (o * axis_size + i) * inner;
                    let dst = (o * idx_count + ip) * inner;
                    out[dst..dst + inner].copy_from_slice(&dv[src..src + inner]);
                }
            }
            Tensor::i64(out_shape, out)
        }
    }
}

fn op_shape(data: &Tensor) -> Tensor {
    let dims: Vec<i64> = data.shape.iter().map(|&s| s as i64).collect();
    Tensor::i64(vec![dims.len()], dims)
}

fn op_tile(data: &Tensor, repeats: &Tensor) -> Tensor {
    let reps = repeats.as_i64();
    let ndim = data.shape.len();
    let out_shape: Vec<usize> = data.shape.iter().zip(reps.iter())
        .map(|(&s, &r)| s * r as usize).collect();
    let out_size: usize = out_shape.iter().product::<usize>().max(1);
    let in_data = data.as_f32();
    let in_strides = compute_strides(&data.shape);
    let out_strides = compute_strides(&out_shape);
    let mut out = vec![0f32; out_size];
    for i in 0..out_size {
        let mut in_idx = 0;
        let mut rem = i;
        for d in 0..ndim {
            let coord = rem / out_strides[d];
            rem %= out_strides[d];
            in_idx += (coord % data.shape[d]) * in_strides[d];
        }
        out[i] = in_data[in_idx];
    }
    Tensor::f32(out_shape, out)
}

// ── Graph executor ───────────────────────────────────────────────────────────

pub fn run(model: &OnnxModel, inputs: Vec<(&str, Tensor)>) -> Result<HashMap<String, Tensor>> {
    let mut computed: HashMap<String, Tensor> = HashMap::new();
    for (name, tensor) in inputs {
        computed.insert(name.to_string(), tensor);
    }

    // Build last-use map: tensor name → last node index that reads it
    let mut last_use: HashMap<&str, usize> = HashMap::new();
    for (i, node) in model.nodes.iter().enumerate() {
        for name in &node.inputs {
            if !name.is_empty() {
                last_use.insert(name.as_str(), i);
            }
        }
    }
    // Ensure graph outputs are never evicted
    for name in &model.graph_outputs {
        last_use.insert(name.as_str(), usize::MAX);
    }

    for (i, node) in model.nodes.iter().enumerate() {
        let result = exec_node(node, &computed, &model.weights)
            .with_context(|| {
                let input_shapes: Vec<_> = node.inputs.iter().map(|n| {
                    if n.is_empty() { return "(empty)".to_string(); }
                    computed.get(n).or_else(|| model.weights.get(n))
                        .map(|t| format!("{:?}", t.shape))
                        .unwrap_or_else(|| "MISSING".to_string())
                }).collect();
                format!("node {} op={} inputs={:?} input_shapes={:?} outputs={:?}",
                    i, node.op_type, node.inputs, input_shapes, node.outputs)
            })?;
        for (name, tensor) in result {
            computed.insert(name, tensor);
        }
        // Free tensors no longer needed — reduces peak memory and cache pressure
        for name in &node.inputs {
            if !name.is_empty() && last_use.get(name.as_str()) == Some(&i) {
                computed.remove(name);
            }
        }
    }

    let mut out = HashMap::new();
    for name in &model.graph_outputs {
        if let Some(t) = computed.remove(name) {
            out.insert(name.clone(), t);
        }
    }
    Ok(out)
}

fn exec_node(
    node: &Node,
    computed: &HashMap<String, Tensor>,
    weights: &HashMap<String, Tensor>,
) -> Result<Vec<(String, Tensor)>> {
    let get = |idx: usize| -> Option<&Tensor> {
        let name = node.inputs.get(idx)?;
        if name.is_empty() { return None; }
        computed.get(name).or_else(|| weights.get(name))
    };
    let inp = |idx: usize| -> &Tensor {
        get(idx).unwrap_or_else(|| panic!("missing input {} ({}) for {}",
            idx, node.inputs.get(idx).unwrap_or(&String::new()), node.op_type))
    };

    let result = match node.op_type.as_str() {
        "MatMul" => op_matmul(inp(0), inp(1)),
        "Gemm" => op_gemm(inp(0), inp(1), get(2), node),
        "Conv" => op_conv(inp(0), inp(1), get(2), node),
        "Add" => binary_op(inp(0), inp(1), |a, b| a + b),
        "Mul" => binary_op(inp(0), inp(1), |a, b| a * b),
        "Sub" => binary_op(inp(0), inp(1), |a, b| a - b),
        "Div" => binary_op(inp(0), inp(1), |a, b| a / b),
        "Sqrt" => unary_op(inp(0), f32::sqrt),
        "Pow" => binary_op(inp(0), inp(1), f32::powf),
        "Tanh" => unary_op(inp(0), f32::tanh),
        "Softmax" => op_softmax(inp(0), node.attr_i("axis").unwrap_or(-1)),
        "ReduceMean" => {
            let axes = node.attr_ints("axes").unwrap_or(&[-1]);
            let keepdims = node.attr_i("keepdims").unwrap_or(1) != 0;
            op_reduce_mean(inp(0), axes, keepdims)
        }
        "Reshape" => op_reshape(inp(0), inp(1)),
        "Transpose" => op_transpose(inp(0), node.attr_ints("perm")),
        "Unsqueeze" => op_unsqueeze(inp(0), inp(1)),
        "Squeeze" => op_squeeze(inp(0), get(1)),
        "Concat" => {
            let ts: Vec<&Tensor> = (0..node.inputs.len()).filter_map(|i| get(i)).collect();
            op_concat(&ts, node.attr_i("axis").unwrap_or(0))
        }
        "Slice" => op_slice(inp(0), inp(1), inp(2), get(3), get(4)),
        "Gather" => op_gather(inp(0), inp(1), node.attr_i("axis").unwrap_or(0)),
        "Shape" => op_shape(inp(0)),
        "Tile" => op_tile(inp(0), inp(1)),
        _ => bail!("unsupported op: {}", node.op_type),
    };

    let out_name = &node.outputs[0];
    Ok(vec![(out_name.clone(), result)])
}
