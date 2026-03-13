//! Shape utilities shared by CPU and GPU executors.

pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    if n == 0 { return vec![1]; }
    let mut s = vec![1usize; n];
    for i in (0..n - 1).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
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
pub fn broadcast_strides(in_shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
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

pub fn normalize_axis(axis: i64, ndim: usize) -> usize {
    if axis < 0 { (ndim as i64 + axis) as usize } else { axis as usize }
}

pub fn broadcast_batch_idx(flat: usize, out_batch: &[usize], in_batch: &[usize]) -> usize {
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
