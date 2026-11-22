//! The multiscale "ladder" ℓ_1 < ℓ_2 < … < ℓ_K = L and the fixed resize
//! operators between length L and length ℓ_k.
//!
//! Down_k averages (area-interpolates) a length-L sequence to ℓ_k vectors and
//! Up_k linearly interpolates ℓ_k vectors back to L positions. Both are dense
//! matrices so they can be applied with a single batched matmul.

use candle_core::{Device, Result, Tensor};

/// Linear ladder from 1 to `l` with `k` levels (Appendix F.1 of the paper),
/// strictly increasing.
pub fn linear_ladder(l: usize, k: usize) -> Vec<usize> {
    assert!(k >= 1 && k <= l, "need 1 <= K <= L");
    if k == 1 {
        return vec![l];
    }
    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        let v = 1.0 + (l as f64 - 1.0) * i as f64 / (k as f64 - 1.0);
        let mut v = v.round() as usize;
        if let Some(&prev) = out.last() {
            v = v.max(prev + 1);
        }
        out.push(v);
    }
    *out.last_mut().unwrap() = l;
    out
}

/// Area-average matrix of shape [m, l] mapping length l -> m (m <= l).
pub fn down_matrix(l: usize, m: usize) -> Vec<f32> {
    let mut w = vec![0f32; m * l];
    let ratio = l as f64 / m as f64;
    for i in 0..m {
        let (a, b) = (i as f64 * ratio, (i + 1) as f64 * ratio);
        let mut total = 0.0;
        for j in 0..l {
            let (ja, jb) = (j as f64, (j + 1) as f64);
            let overlap = (b.min(jb) - a.max(ja)).max(0.0);
            if overlap > 0.0 {
                w[i * l + j] = overlap as f32;
                total += overlap;
            }
        }
        for j in 0..l {
            w[i * l + j] /= total as f32;
        }
    }
    w
}

/// Linear-interpolation matrix of shape [l, m] mapping length m -> l
/// (align_corners = false, as in torch's `F.interpolate(mode="linear")`).
pub fn up_matrix(l: usize, m: usize) -> Vec<f32> {
    let mut w = vec![0f32; l * m];
    let scale = m as f64 / l as f64;
    for i in 0..l {
        let src = ((i as f64 + 0.5) * scale - 0.5).clamp(0.0, (m - 1) as f64);
        let lo = src.floor() as usize;
        let hi = (lo + 1).min(m - 1);
        let t = (src - lo as f64) as f32;
        w[i * m + lo] += 1.0 - t;
        w[i * m + hi] += t;
    }
    w
}

/// All resize operators for a ladder.
pub struct Resizers {
    /// down[k]: [ℓ_k, L]
    pub down: Vec<Tensor>,
    /// up[k]: [L, ℓ_k]
    pub up: Vec<Tensor>,
}

impl Resizers {
    pub fn new(ladder: &[usize], dev: &Device) -> Result<Self> {
        let l = *ladder.last().unwrap();
        let mut down = Vec::new();
        let mut up = Vec::new();
        for &m in ladder {
            down.push(Tensor::from_vec(down_matrix(l, m), (m, l), dev)?);
            up.push(Tensor::from_vec(up_matrix(l, m), (l, m), dev)?);
        }
        Ok(Self { down, up })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ladder_is_strict_and_ends_at_l() {
        for (l, k) in [(81, 8), (81, 12), (64, 8), (128, 32), (10, 10)] {
            let lad = linear_ladder(l, k);
            assert_eq!(lad.len(), k);
            assert_eq!(lad[0], 1);
            assert_eq!(*lad.last().unwrap(), l);
            assert!(lad.windows(2).all(|w| w[0] < w[1]), "{lad:?}");
        }
    }

    #[test]
    fn resize_rows_sum_to_one_and_identity_at_full_length() {
        let d = down_matrix(10, 3);
        for i in 0..3 {
            let s: f32 = d[i * 10..(i + 1) * 10].iter().sum();
            assert!((s - 1.0).abs() < 1e-5);
        }
        let u = up_matrix(10, 3);
        for i in 0..10 {
            let s: f32 = u[i * 3..(i + 1) * 3].iter().sum();
            assert!((s - 1.0).abs() < 1e-5);
        }
        let id = up_matrix(5, 5);
        for i in 0..5 {
            for j in 0..5 {
                assert!((id[i * 5 + j] - if i == j { 1.0 } else { 0.0 }).abs() < 1e-6);
            }
        }
    }
}
