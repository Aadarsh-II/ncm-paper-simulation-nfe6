//! Small transformer building blocks shared by every model in the simulation
//! (VQ-VAE decoder, NCM generator, AR and masked-diffusion baselines).

use candle_core::{D, Device, Module, Result, Tensor};
use candle_nn::{Init, Linear, VarBuilder, linear, ops};

pub const NEG_INF: f32 = -1e9;

/// Multi-head self-attention with an additive mask.
pub struct Attention {
    qkv: Linear,
    out: Linear,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    pub fn new(dim: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        assert!(dim % heads == 0, "hidden width must be divisible by heads");
        Ok(Self {
            qkv: linear(dim, 3 * dim, vb.pp("qkv"))?,
            out: linear(dim, dim, vb.pp("out"))?,
            heads,
            head_dim: dim / heads,
        })
    }

    /// `x`: [B, N, dim]; `mask`: additive [N, N] (0 = attend, NEG_INF = blocked).
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, n, dim) = x.dims3()?;
        let qkv = self
            .qkv
            .forward(x)?
            .reshape((b, n, 3, self.heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?; // [3, B, H, N, hd]
        let q = qkv.get(0)?.contiguous()?;
        let k = qkv.get(1)?.contiguous()?;
        let v = qkv.get(2)?.contiguous()?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut att = (q.matmul(&k.t()?)? * scale)?; // [B, H, N, N]
        if let Some(m) = mask {
            att = att.broadcast_add(m)?;
        }
        // Not `ops::softmax_last_dim`: candle applies that kernel with
        // `apply_op1_no_bwd`, so no gradient would reach the queries and keys
        // and every attention pattern would stay frozen at initialization.
        let att = ops::softmax(&att, D::Minus1)?;
        let y = att.matmul(&v)?; // [B, H, N, hd]
        let y = y.transpose(1, 2)?.contiguous()?.reshape((b, n, dim))?;
        self.out.forward(&y)
    }
}

/// Layer normalization that supports backprop.
///
/// `candle_nn::LayerNorm` dispatches contiguous inputs to a fused kernel
/// applied with `apply_op3_no_bwd`, which silently cuts the gradient: every
/// parameter below the norm would get no gradient and never train. This uses
/// candle's composite (differentiable) `layer_norm_slow` instead.
pub struct Norm {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl Norm {
    pub fn new(dim: usize, eps: f32, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get_with_hints(dim, "weight", Init::Const(1.0))?,
            bias: vb.get_with_hints(dim, "bias", Init::Const(0.0))?,
            eps,
        })
    }
}

impl Module for Norm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        ops::layer_norm_slow(x, &self.weight, &self.bias, self.eps)
    }
}

/// Pre-LN transformer block.
pub struct Block {
    ln1: Norm,
    attn: Attention,
    ln2: Norm,
    fc1: Linear,
    fc2: Linear,
    dropout: f32,
}

impl Block {
    pub fn new(dim: usize, heads: usize, dropout: f32, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln1: Norm::new(dim, 1e-5, vb.pp("ln1"))?,
            attn: Attention::new(dim, heads, vb.pp("attn"))?,
            ln2: Norm::new(dim, 1e-5, vb.pp("ln2"))?,
            fc1: linear(dim, 4 * dim, vb.pp("fc1"))?,
            fc2: linear(4 * dim, dim, vb.pp("fc2"))?,
            dropout,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>, train: bool) -> Result<Tensor> {
        let h = self.attn.forward(&self.ln1.forward(x)?, mask)?;
        let h = drop(&h, self.dropout, train)?;
        let x = (x + h)?;
        let h = self.fc2.forward(&self.fc1.forward(&self.ln2.forward(&x)?)?.gelu()?)?;
        let h = drop(&h, self.dropout, train)?;
        x + h
    }
}

pub fn drop(x: &Tensor, p: f32, train: bool) -> Result<Tensor> {
    if train && p > 0.0 { ops::dropout(x, p) } else { Ok(x.clone()) }
}

/// A stack of blocks followed by a final layer norm.
pub struct Transformer {
    blocks: Vec<Block>,
    ln_f: Norm,
}

impl Transformer {
    pub fn new(dim: usize, depth: usize, heads: usize, dropout: f32, vb: VarBuilder) -> Result<Self> {
        let blocks = (0..depth)
            .map(|i| Block::new(dim, heads, dropout, vb.pp(format!("block{i}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { blocks, ln_f: Norm::new(dim, 1e-5, vb.pp("ln_f"))? })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>, train: bool) -> Result<Tensor> {
        let mut h = x.clone();
        for b in &self.blocks {
            h = b.forward(&h, mask, train)?;
        }
        self.ln_f.forward(&h)
    }
}

/// Build an additive [N, N] mask from a boolean "may attend" predicate.
pub fn mask_from_fn(n: usize, dev: &Device, allow: impl Fn(usize, usize) -> bool) -> Result<Tensor> {
    let mut m = vec![NEG_INF; n * n];
    for i in 0..n {
        for j in 0..n {
            if allow(i, j) {
                m[i * n + j] = 0.0;
            }
        }
    }
    Tensor::from_vec(m, (n, n), dev)
}

/// Per-row log-probability of `targets` under `logits`.
/// `logits`: [M, C]; `targets`: [M] (u32). Returns [M] of -log p.
pub fn nll_per_row(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let logp = ops::log_softmax(logits, D::Minus1)?;
    logp.gather(&targets.unsqueeze(1)?, 1)?.squeeze(1)?.neg()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::VarMap;

    /// Every parameter of a masked transformer must receive a gradient.
    /// (Regression test: candle's fused layer-norm and softmax kernels have
    /// no backward pass, which once froze everything below the output head.)
    #[test]
    fn gradients_reach_every_parameter() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let t = Transformer::new(16, 2, 2, 0.0, vb.pp("t")).unwrap();
        let x = Tensor::randn(0f32, 1.0, (2, 5, 16), &dev).unwrap();
        let mask = mask_from_fn(5, &dev, |i, j| j <= i).unwrap();
        let loss = t.forward(&x, Some(&mask), true).unwrap().sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        for (name, v) in varmap.data().lock().unwrap().iter() {
            let g = grads.get(v.as_tensor()).unwrap_or_else(|| panic!("no gradient for {name}"));
            let n = g.abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            assert!(n > 0.0, "zero gradient for {name}");
        }
    }
}
