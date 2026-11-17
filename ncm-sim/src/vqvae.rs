//! Stage 1 of NCM: the multiscale residual VQ-VAE tokenizer (paper §3.2).
//!
//! Encoder E: token embedding (the paper uses 0 encoder blocks).
//! For k = 1..K:
//!     u_k   = Down_k(z^(k-1))                 ∈ R^{ℓ_k × d}
//!     r_k,i = argmin_j ‖u_k,i − e_k,j‖²
//!     ẑ^(k) = ẑ^(k-1) + R_k(e_k(r_k))
//!     z^(k) = z^(k-1) − R_k(e_k(r_k))
//! with R_k(e) = h + ½·φ_k(h), h = Up_k(e)  (an interpolation followed by a
//! learned per-level refinement, as in next-scale prediction for images).
//! Decoder D: bidirectional transformer mapping ẑ^(k) to token logits.
//!
//! Training uses nested dropout over the cutoff level, a straight-through
//! estimator, EMA codebooks with dead-code restarts and the loss
//!     CE(x | ẑ^(k)) + β·MSE(z, sg[ẑ^(k)]) + MSE(sg[z], ẑ^(k)).

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder, VarMap, embedding, linear};
use rand::Rng;
use rand::rngs::StdRng;

use crate::nn::{Transformer, nll_per_row};
use crate::scales::Resizers;
use crate::train::Trainer;

#[derive(Clone, Debug)]
pub struct VqConfig {
    pub vocab: usize,
    pub ladder: Vec<usize>,
    pub codebook: usize,
    pub dim: usize,
    pub dec_depth: usize,
    pub dec_heads: usize,
    pub dropout: f32,
    pub beta: f64,
    pub ema_decay: f32,
    /// Probability of truncating the hierarchy at a random level (nested dropout).
    pub nested_dropout: f64,
    /// Evaluate the VQ (commitment + quantization) terms on the full hierarchy
    /// instead of at the nested-dropout cutoff.
    pub vq_loss_full: bool,
}

pub struct VqVae {
    pub cfg: VqConfig,
    pub varmap: VarMap,
    pub rs: Resizers,
    tok_emb: Embedding,
    phi: Vec<Linear>,
    dec_pos: Embedding,
    decoder: Transformer,
    head: Linear,
    /// EMA codebooks, one [V, d] tensor per level (not trained by gradients).
    pub codebooks: Vec<Tensor>,
    ema_count: Vec<Vec<f32>>,
    ema_sum: Vec<Vec<f32>>,
    initialized: bool,
    dev: Device,
}

/// Result of quantizing a batch.
pub struct Quantized {
    /// codes[k]: [B, ℓ_k] u32
    pub codes: Vec<Tensor>,
    /// Downsampled residuals u_k (detached), for the EMA update.
    us: Vec<Tensor>,
}

impl VqVae {
    pub fn new(cfg: VqConfig, dev: &Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);
        let d = cfg.dim;
        let l = *cfg.ladder.last().unwrap();
        let tok_emb = embedding(cfg.vocab, d, vb.pp("tok_emb"))?;
        let phi = (0..cfg.ladder.len())
            .map(|k| linear(d, d, vb.pp(format!("phi{k}"))))
            .collect::<Result<Vec<_>>>()?;
        let dec_pos = embedding(l, d, vb.pp("dec_pos"))?;
        let decoder = Transformer::new(d, cfg.dec_depth, cfg.dec_heads, cfg.dropout, vb.pp("dec"))?;
        let head = linear(d, cfg.vocab, vb.pp("head"))?;
        let rs = Resizers::new(&cfg.ladder, dev)?;
        let k = cfg.ladder.len();
        let codebooks = (0..k)
            .map(|_| Tensor::randn(0f32, 1.0, (cfg.codebook, d), dev))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            ema_count: vec![vec![1.0; cfg.codebook]; k],
            ema_sum: vec![vec![0.0; cfg.codebook * d]; k],
            cfg,
            varmap,
            rs,
            tok_emb,
            phi,
            dec_pos,
            decoder,
            head,
            codebooks,
            initialized: false,
            dev: dev.clone(),
        })
    }

    pub fn k(&self) -> usize {
        self.cfg.ladder.len()
    }

    pub fn l(&self) -> usize {
        *self.cfg.ladder.last().unwrap()
    }

    /// Token embedding table (reused by the NCM generator to embed prompts).
    pub fn embed_tokens(&self, x: &Tensor) -> Result<Tensor> {
        self.tok_emb.forward(x)
    }

    /// Encoder: x [B, L] u32 -> z [B, L, d].
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        self.tok_emb.forward(x)
    }

    /// R_k: [B, ℓ_k, d] -> [B, L, d].
    pub fn reconstruct_level(&self, k: usize, e: &Tensor) -> Result<Tensor> {
        let h = self.rs.up[k].broadcast_matmul(e)?;
        h.clone() + (self.phi[k].forward(&h)? * 0.5)?
    }

    /// Codebook lookup e_k(r_k): codes [B, ℓ] -> [B, ℓ, d].
    pub fn lookup(&self, k: usize, codes: &Tensor) -> Result<Tensor> {
        let (b, n) = codes.dims2()?;
        self.codebooks[k]
            .index_select(&codes.flatten_all()?, 0)?
            .reshape((b, n, self.cfg.dim))
    }

    fn nearest(&self, k: usize, u: &Tensor) -> Result<Tensor> {
        let (b, n, d) = u.dims3()?;
        let flat = u.reshape((b * n, d))?;
        let cb = &self.codebooks[k];
        let dist = flat
            .sqr()?
            .sum_keepdim(1)?
            .broadcast_add(&cb.sqr()?.sum_keepdim(1)?.t()?)?
            .sub(&(flat.matmul(&cb.t()?)? * 2.0)?)?;
        dist.argmin(D::Minus1)?.reshape((b, n))
    }

    /// Residual quantization of z (no gradients; codes only).
    pub fn quantize(&self, z: &Tensor) -> Result<Quantized> {
        let z = z.detach();
        let mut resid = z.clone();
        let mut codes = Vec::new();
        let mut us = Vec::new();
        for k in 0..self.k() {
            let u = self.rs.down[k].broadcast_matmul(&resid)?;
            let r = self.nearest(k, &u)?;
            let e = self.lookup(k, &r)?;
            resid = (resid - self.reconstruct_level(k, &e)?.detach())?;
            codes.push(r);
            us.push(u);
        }
        Ok(Quantized { codes, us })
    }

    /// Cumulative reconstructions ẑ^(0..=K) from (possibly corrupted) codes.
    /// Returns K+1 tensors of shape [B, L, d]; entry 0 is all zeros.
    pub fn cumulative(&self, codes: &[Tensor]) -> Result<Vec<Tensor>> {
        let b = codes[0].dim(0)?;
        let mut acc = Tensor::zeros((b, self.l(), self.cfg.dim), DType::F32, &self.dev)?;
        let mut out = vec![acc.clone()];
        for (k, r) in codes.iter().enumerate() {
            acc = (acc + self.reconstruct_level(k, &self.lookup(k, r)?)?)?;
            out.push(acc.clone());
        }
        Ok(out)
    }

    /// Decoder: ẑ [B, L, d] -> logits [B, L, vocab].
    pub fn decode_logits(&self, zhat: &Tensor, train: bool) -> Result<Tensor> {
        let (b, l, _) = zhat.dims3()?;
        let pos = self.dec_pos.forward(&Tensor::arange(0u32, l as u32, &self.dev)?)?;
        let h = zhat.broadcast_add(&pos)?;
        let h = self.decoder.forward(&h, None, train)?;
        let _ = b;
        self.head.forward(&h)
    }

    /// Greedy decode: ẑ -> tokens [B, L] (Vec of rows).
    pub fn decode_tokens(&self, zhat: &Tensor) -> Result<Vec<Vec<u32>>> {
        self.decode_logits(zhat, false)?.argmax(D::Minus1)?.to_vec2::<u32>()
    }

    /// Initialize each level's codebook from random downsampled residuals.
    fn init_codebooks(&mut self, x: &Tensor, rng: &mut StdRng) -> Result<()> {
        let z = self.encode(x)?.detach();
        let mut resid = z;
        let d = self.cfg.dim;
        for k in 0..self.k() {
            let u = self.rs.down[k].broadcast_matmul(&resid)?;
            let flat: Vec<f32> = u.flatten_all()?.to_vec1()?;
            let m = flat.len() / d;
            let mut cb = vec![0f32; self.cfg.codebook * d];
            for c in 0..self.cfg.codebook {
                let src = rng.gen_range(0..m);
                for j in 0..d {
                    cb[c * d + j] = flat[src * d + j] + rng.gen_range(-0.01..0.01);
                }
            }
            self.ema_sum[k] = cb.clone();
            self.ema_count[k] = vec![1.0; self.cfg.codebook];
            self.codebooks[k] = Tensor::from_vec(cb, (self.cfg.codebook, d), &self.dev)?;
            let r = self.nearest(k, &u)?;
            let e = self.lookup(k, &r)?;
            resid = (resid - self.reconstruct_level(k, &e)?.detach())?;
        }
        self.initialized = true;
        Ok(())
    }

    /// EMA codebook update with dead-code restarts.
    fn ema_update(&mut self, q: &Quantized, rng: &mut StdRng) -> Result<()> {
        let d = self.cfg.dim;
        let v = self.cfg.codebook;
        let g = self.cfg.ema_decay;
        for k in 0..self.k() {
            let codes: Vec<u32> = q.codes[k].flatten_all()?.to_vec1()?;
            let us: Vec<f32> = q.us[k].flatten_all()?.to_vec1()?;
            let mut cnt = vec![0f32; v];
            let mut sum = vec![0f32; v * d];
            for (i, &c) in codes.iter().enumerate() {
                let c = c as usize;
                cnt[c] += 1.0;
                for j in 0..d {
                    sum[c * d + j] += us[i * d + j];
                }
            }
            let ec = &mut self.ema_count[k];
            let es = &mut self.ema_sum[k];
            for c in 0..v {
                ec[c] = g * ec[c] + (1.0 - g) * cnt[c];
                for j in 0..d {
                    es[c * d + j] = g * es[c * d + j] + (1.0 - g) * sum[c * d + j];
                }
            }
            let n: f32 = ec.iter().sum();
            let mut cb = vec![0f32; v * d];
            for c in 0..v {
                // Dead code: re-seed from a random vector of this batch.
                if ec[c] < 1e-3 * n / v as f32 {
                    let src = rng.gen_range(0..codes.len());
                    for j in 0..d {
                        es[c * d + j] = us[src * d + j];
                    }
                    ec[c] = 1.0;
                }
                let smoothed = (ec[c] + 1e-5) / (n + v as f32 * 1e-5) * n;
                for j in 0..d {
                    cb[c * d + j] = es[c * d + j] / smoothed;
                }
            }
            self.codebooks[k] = Tensor::from_vec(cb, (v, d), &self.dev)?;
        }
        Ok(())
    }

    /// One tokenizer training step on a batch of token sequences x [B, L].
    pub fn train_step(&mut self, x: &Tensor, trainer: &mut Trainer, rng: &mut StdRng) -> Result<(f32, f32)> {
        if !self.initialized {
            self.init_codebooks(x, rng)?;
        }
        let (b, l) = x.dims2()?;
        let k = self.k();
        let z = self.encode(x)?;
        let q = self.quantize(&z)?;

        // Per-sample nested-dropout cutoff: keep levels 1..=cut.
        let cuts: Vec<usize> = (0..b)
            .map(|_| if rng.gen_bool(self.cfg.nested_dropout) { rng.gen_range(1..=k) } else { k })
            .collect();
        let mut zhat = Tensor::zeros((b, l, self.cfg.dim), DType::F32, &self.dev)?;
        let mut zfull = zhat.clone();
        for lvl in 0..k {
            let keep: Vec<f32> = cuts.iter().map(|&c| if lvl < c { 1.0 } else { 0.0 }).collect();
            let keep = Tensor::from_vec(keep, (b, 1, 1), &self.dev)?;
            let contrib = self.reconstruct_level(lvl, &self.lookup(lvl, &q.codes[lvl])?)?;
            zhat = (zhat + contrib.broadcast_mul(&keep)?)?;
            zfull = (zfull + contrib)?;
        }
        // Straight-through: forward value ẑ, gradient also flows to z.
        let dec_in = ((&zhat + &z)? - z.detach())?;
        let logits = self.decode_logits(&dec_in, true)?;
        let ce = nll_per_row(&logits.reshape((b * l, self.cfg.vocab))?, &x.flatten_all()?)?.mean_all()?;
        let vq_target = if self.cfg.vq_loss_full { &zfull } else { &zhat };
        let commit = (&z - vq_target.detach())?.sqr()?.mean_all()?;
        let qloss = (z.detach() - vq_target)?.sqr()?.mean_all()?;
        let loss = ((&ce + (commit * self.cfg.beta)?)? + qloss)?;
        trainer.step(&loss)?;
        self.ema_update(&q, rng)?;
        Ok((loss.to_scalar::<f32>()?, ce.to_scalar::<f32>()?))
    }

    /// Encode tokens into codes (inference).
    pub fn tokenize(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        Ok(self.quantize(&self.encode(x)?)?.codes)
    }
}

/// Tokenizer evaluation: exact token reconstruction, exact sequence
/// reconstruction, and mean codebook utilization across levels (Table 1).
pub struct TokenizerReport {
    pub token_acc: f64,
    pub seq_acc: f64,
    pub utilization: Vec<f64>,
    /// Token accuracy when decoding from the first k levels only.
    pub partial_acc: Vec<f64>,
}

pub fn evaluate_tokenizer(vq: &VqVae, data: &[Vec<u32>], batch: usize, dev: &Device) -> Result<TokenizerReport> {
    let k = vq.k();
    let mut used = vec![vec![false; vq.cfg.codebook]; k];
    let (mut tok_ok, mut tok_n, mut seq_ok) = (0usize, 0usize, 0usize);
    let mut part_ok = vec![0usize; k];
    for chunk in data.chunks(batch) {
        let x = batch_tensor(chunk, dev)?;
        let codes = vq.tokenize(&x)?;
        for (lvl, c) in codes.iter().enumerate() {
            for id in c.flatten_all()?.to_vec1::<u32>()? {
                used[lvl][id as usize] = true;
            }
        }
        let cum = vq.cumulative(&codes)?;
        for lvl in 0..k {
            let out = vq.decode_tokens(&cum[lvl + 1])?;
            for (row, tgt) in out.iter().zip(chunk) {
                let ok = row.iter().zip(tgt).filter(|(a, b)| a == b).count();
                part_ok[lvl] += ok;
                if lvl == k - 1 {
                    tok_ok += ok;
                    tok_n += tgt.len();
                    if ok == tgt.len() {
                        seq_ok += 1;
                    }
                }
            }
        }
    }
    Ok(TokenizerReport {
        token_acc: tok_ok as f64 / tok_n as f64,
        seq_acc: seq_ok as f64 / data.len() as f64,
        utilization: used.iter().map(|u| u.iter().filter(|&&b| b).count() as f64 / u.len() as f64).collect(),
        partial_acc: part_ok.iter().map(|&o| o as f64 / tok_n as f64).collect(),
    })
}

pub fn batch_tensor(rows: &[Vec<u32>], dev: &Device) -> Result<Tensor> {
    let l = rows[0].len();
    let flat: Vec<u32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Tensor::from_vec(flat, (rows.len(), l), dev)
}
