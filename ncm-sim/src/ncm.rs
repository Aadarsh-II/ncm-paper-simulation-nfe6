//! Stage 2 of NCM: the next-concept transformer (paper §3.3, Algorithms 1–2).
//!
//! Input sequence = [prefix (P positions) | level-1 block | … | level-K block].
//! Block k has ℓ_k positions whose inputs are Down_k(z̃^(k-1)) projected to the
//! hidden width, plus level and position embeddings; it predicts the ℓ_k codes
//! r_k in parallel. Attention is prefix-LM / block-diagonal: prefix positions
//! attend to the prefix; each block attends to the prefix and to itself.
//!
//! Training predicts all K levels in one forward pass with a level-weighted
//! cross-entropy (w_k ∝ ℓ_k^{-α}) and conditioning corruption. Generation runs
//! K forward passes (NFE = K), coarse to fine.

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder, VarMap, embedding, linear, ops};
use rand::Rng;
use rand::rngs::StdRng;

use crate::nn::{Transformer, mask_from_fn, nll_per_row};
use crate::train::Trainer;
use crate::vqvae::VqVae;

#[derive(Clone, Debug)]
pub struct NcmConfig {
    pub hidden: usize,
    pub depth: usize,
    pub heads: usize,
    pub dropout: f32,
    /// Prefix length (1 = BOS for unconditional generation).
    pub prefix: usize,
    /// Loss weighting exponent α (w_k ∝ ℓ_k^{-α}).
    pub alpha: f64,
    /// Corruption rate for the coarsest conditioning level; decreases linearly
    /// to `corrupt_min` at level K-1 (level K is never used as conditioning).
    pub corrupt_max: f64,
    pub corrupt_min: f64,
    /// The prompt is aligned cell by cell with the output (prefix == L, e.g.
    /// Sudoku). Prefix cell i then gets cell embedding E_i, and level k's
    /// positions get Down_k(E), the cell embeddings area-resized to ℓ_k with
    /// the tokenizer's own resize operator, so every level knows which prompt
    /// cells it covers. Otherwise positions are learned per sequence slot.
    pub aligned_prefix: bool,
}

pub struct Ncm {
    pub cfg: NcmConfig,
    pub varmap: VarMap,
    ladder: Vec<usize>,
    offsets: Vec<usize>,
    codebook: usize,
    in_proj: Linear,
    prefix_proj: Linear,
    bos: Embedding,
    level_emb: Embedding,
    pos_emb: Embedding,
    body: Transformer,
    head: Linear,
    train_mask: Tensor,
    level_masks: Vec<Tensor>,
    weights: Vec<f32>,
    dev: Device,
}

impl Ncm {
    pub fn new(cfg: NcmConfig, vq: &VqVae, dev: &Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);
        let ladder = vq.cfg.ladder.clone();
        let k = ladder.len();
        let p = cfg.prefix;
        let mut offsets = vec![p];
        for &l in &ladder {
            offsets.push(offsets.last().unwrap() + l);
        }
        let n = *offsets.last().unwrap();
        let h = cfg.hidden;
        assert!(!cfg.aligned_prefix || p == *ladder.last().unwrap(), "aligned_prefix needs prefix == L");

        // Block id per position: 0 = prefix, k+1 = level k.
        let block_of: Vec<usize> = (0..n)
            .map(|i| if i < p { 0 } else { (0..k).find(|&b| i < offsets[b + 1]).unwrap() + 1 })
            .collect();
        let train_mask = mask_from_fn(n, dev, |i, j| {
            let (bi, bj) = (block_of[i], block_of[j]);
            bj == 0 || bi == bj
        })?;
        // For level-k generation the sequence is [prefix | block k].
        let level_masks = ladder
            .iter()
            .map(|&l| mask_from_fn(p + l, dev, |i, j| j < p || i >= p))
            .collect::<Result<Vec<_>>>()?;

        // w_k = ℓ_k^{-α} / Σ_j ℓ_j^{1-α}
        let norm: f64 = ladder.iter().map(|&l| (l as f64).powf(1.0 - cfg.alpha)).sum();
        let weights = ladder.iter().map(|&l| ((l as f64).powf(-cfg.alpha) / norm) as f32).collect();

        Ok(Self {
            in_proj: linear(vq.cfg.dim, h, vb.pp("in_proj"))?,
            prefix_proj: linear(vq.cfg.dim, h, vb.pp("prefix_proj"))?,
            bos: embedding(1, h, vb.pp("bos"))?,
            level_emb: embedding(k, h, vb.pp("level_emb"))?,
            pos_emb: embedding(n, h, vb.pp("pos_emb"))?,
            body: Transformer::new(h, cfg.depth, cfg.heads, cfg.dropout, vb.pp("body"))?,
            head: linear(h, vq.cfg.codebook, vb.pp("head"))?,
            codebook: vq.cfg.codebook,
            cfg,
            varmap,
            ladder,
            offsets,
            train_mask,
            level_masks,
            weights,
            dev: dev.clone(),
        })
    }

    pub fn k(&self) -> usize {
        self.ladder.len()
    }

    /// Position embeddings of level k (ℓ_k rows).
    fn level_pos(&self, vq: &VqVae, k: usize) -> Result<Tensor> {
        if self.cfg.aligned_prefix {
            let cells = self.pos_emb.embeddings().narrow(0, 0, self.cfg.prefix)?;
            vq.rs.down[k].matmul(&cells)
        } else {
            let l = self.ladder[k];
            self.pos_emb.forward(&Tensor::arange(self.offsets[k] as u32, (self.offsets[k] + l) as u32, &self.dev)?)
        }
    }

    /// Prefix states: prompt tokens embedded with the frozen VQ-VAE table, or BOS.
    fn prefix_states(&self, vq: &VqVae, prompt: Option<&Tensor>, b: usize) -> Result<Tensor> {
        let p = self.cfg.prefix;
        let pos = self.pos_emb.forward(&Tensor::arange(0u32, p as u32, &self.dev)?)?;
        let x = match prompt {
            Some(t) => self.prefix_proj.forward(&vq.embed_tokens(t)?.detach())?,
            None => self
                .bos
                .forward(&Tensor::zeros((b, p), DType::U32, &self.dev)?)?,
        };
        x.broadcast_add(&pos)
    }

    /// Inputs of level k from the cumulative reconstruction of levels < k.
    fn level_inputs(&self, vq: &VqVae, k: usize, zprev: &Tensor) -> Result<Tensor> {
        let x = vq.rs.down[k].broadcast_matmul(&zprev.detach())?;
        let x = self.in_proj.forward(&x)?;
        let pos = self.level_pos(vq, k)?;
        let lvl = self.level_emb.forward(&Tensor::new(&[k as u32], &self.dev)?)?;
        x.broadcast_add(&pos)?.broadcast_add(&lvl)
    }

    /// Conditioning corruption rate for level k (0-based; last level unused).
    pub fn corrupt_rate(&self, k: usize) -> f64 {
        let km = self.k().saturating_sub(1).max(1);
        if k + 1 >= self.k() {
            return 0.0;
        }
        let t = if km <= 1 { 0.0 } else { k as f64 / (km - 1) as f64 };
        self.cfg.corrupt_max + (self.cfg.corrupt_min - self.cfg.corrupt_max) * t
    }

    fn corrupt(&self, codes: &[Tensor], rng: &mut StdRng) -> Result<Vec<Tensor>> {
        codes
            .iter()
            .enumerate()
            .map(|(k, c)| {
                let rate = self.corrupt_rate(k);
                if rate <= 0.0 {
                    return Ok(c.clone());
                }
                let (b, l) = c.dims2()?;
                let mut v: Vec<u32> = c.flatten_all()?.to_vec1()?;
                for x in v.iter_mut() {
                    if rng.gen_bool(rate) {
                        *x = rng.gen_range(0..self.codebook as u32);
                    }
                }
                Tensor::from_vec(v, (b, l), &self.dev)
            })
            .collect()
    }

    /// Algorithm 1: one training step. `x`: target tokens [B, L];
    /// `prompt`: optional prompt tokens [B, P].
    pub fn train_step(
        &self,
        vq: &VqVae,
        x: &Tensor,
        prompt: Option<&Tensor>,
        trainer: &mut Trainer,
        rng: &mut StdRng,
    ) -> Result<f32> {
        let b = x.dim(0)?;
        let codes = vq.tokenize(x)?;
        let noisy = self.corrupt(&codes, rng)?;
        let cum = vq.cumulative(&noisy)?;
        let mut parts = vec![self.prefix_states(vq, prompt, b)?];
        for k in 0..self.k() {
            parts.push(self.level_inputs(vq, k, &cum[k])?);
        }
        let h = Tensor::cat(&parts, 1)?;
        let h = self.body.forward(&h, Some(&self.train_mask), true)?;
        let p = self.cfg.prefix;
        let n_tgt = self.offsets[self.k()] - p;
        let logits = self.head.forward(&h.narrow(1, p, n_tgt)?)?;
        let targets = Tensor::cat(&codes, 1)?;
        let nll = nll_per_row(&logits.reshape((b * n_tgt, self.codebook))?, &targets.flatten_all()?)?
            .reshape((b, n_tgt))?;
        let w: Vec<f32> = self
            .ladder
            .iter()
            .zip(&self.weights)
            .flat_map(|(&l, &w)| std::iter::repeat_n(w, l))
            .collect();
        let w = Tensor::from_vec(w, (1, n_tgt), &self.dev)?;
        let loss = (nll.broadcast_mul(&w)?.sum_all()? / b as f64)?;
        trainer.step(&loss)?;
        loss.to_scalar::<f32>()
    }

    /// Algorithm 2: coarse-to-fine generation in K forward passes.
    /// Returns generated tokens and, if `trace`, the decoded output after every level.
    pub fn generate(
        &self,
        vq: &VqVae,
        b: usize,
        prompt: Option<&Tensor>,
        temperature: f64,
        rng: &mut StdRng,
        trace: bool,
    ) -> Result<(Vec<Vec<u32>>, Vec<Vec<Vec<u32>>>)> {
        let prefix = self.prefix_states(vq, prompt, b)?;
        let mut zhat = Tensor::zeros((b, vq.l(), vq.cfg.dim), DType::F32, &self.dev)?;
        let mut traj = Vec::new();
        for k in 0..self.k() {
            let inp = Tensor::cat(&[prefix.clone(), self.level_inputs(vq, k, &zhat)?], 1)?;
            let h = self.body.forward(&inp, Some(&self.level_masks[k]), false)?;
            let logits = self.head.forward(&h.narrow(1, self.cfg.prefix, self.ladder[k])?)?;
            let codes = sample(&logits, temperature, rng)?;
            zhat = (zhat + vq.reconstruct_level(k, &vq.lookup(k, &codes)?)?)?;
            if trace {
                traj.push(vq.decode_tokens(&zhat)?);
            }
        }
        Ok((vq.decode_tokens(&zhat)?, traj))
    }
}

/// Sample (or argmax at temperature 0) from logits [B, N, V] -> [B, N] u32.
pub fn sample(logits: &Tensor, temperature: f64, rng: &mut StdRng) -> Result<Tensor> {
    if temperature <= 0.0 {
        return logits.argmax(D::Minus1);
    }
    let (b, n, v) = logits.dims3()?;
    let probs: Vec<f32> = ops::softmax_last_dim(&(logits / temperature)?)?.flatten_all()?.to_vec1()?;
    let mut out = Vec::with_capacity(b * n);
    for row in probs.chunks(v) {
        let mut u: f32 = rng.r#gen();
        let mut pick = v - 1;
        for (i, &p) in row.iter().enumerate() {
            if u < p {
                pick = i;
                break;
            }
            u -= p;
        }
        out.push(pick as u32);
    }
    Tensor::from_vec(out, (b, n), logits.device())
}
