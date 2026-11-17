//! Baselines the paper compares against, at small scale:
//!  * `Ar`   – autoregressive prefix-LM transformer (NFE = L).
//!  * `Mdlm` – absorbing-state masked diffusion (MDLM-style ELBO training,
//!    confidence-ordered unmasking; NFE = number of sampling steps T).
//!
//! The AR model is also used, trained separately, as the *reference model*
//! that scores generative perplexity (the paper uses GPT-2 Large for this).

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder, VarMap, embedding, linear, ops};
use rand::Rng;
use rand::rngs::StdRng;

use crate::ncm::sample;
use crate::nn::{Transformer, mask_from_fn, nll_per_row};
use crate::train::Trainer;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Ar,
    Mdlm,
}

#[derive(Clone, Debug)]
pub struct SeqConfig {
    pub kind: Kind,
    pub vocab: usize,
    pub prompt_vocab: usize,
    pub prefix: usize,
    pub len: usize,
    pub hidden: usize,
    pub depth: usize,
    pub heads: usize,
    pub dropout: f32,
}

pub struct SeqModel {
    pub cfg: SeqConfig,
    pub varmap: VarMap,
    tok_emb: Embedding,
    prompt_emb: Embedding,
    pos_emb: Embedding,
    body: Transformer,
    head: Linear,
    mask: Tensor,
    dev: Device,
}

impl SeqModel {
    pub fn new(cfg: SeqConfig, dev: &Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);
        let (p, n) = (cfg.prefix, cfg.prefix + cfg.len);
        let mask = match cfg.kind {
            Kind::Ar => mask_from_fn(n, dev, |i, j| j < p || (i >= p && j <= i))?,
            Kind::Mdlm => mask_from_fn(n, dev, |_, _| true)?,
        };
        Ok(Self {
            // Extra id = BOS (AR) or MASK (MDLM).
            tok_emb: embedding(cfg.vocab + 1, cfg.hidden, vb.pp("tok_emb"))?,
            prompt_emb: embedding(cfg.prompt_vocab.max(1), cfg.hidden, vb.pp("prompt_emb"))?,
            pos_emb: embedding(n, cfg.hidden, vb.pp("pos_emb"))?,
            body: Transformer::new(cfg.hidden, cfg.depth, cfg.heads, cfg.dropout, vb.pp("body"))?,
            head: linear(cfg.hidden, cfg.vocab, vb.pp("head"))?,
            mask,
            cfg,
            varmap,
            dev: dev.clone(),
        })
    }

    fn special(&self) -> u32 {
        self.cfg.vocab as u32
    }

    /// Logits at the L target positions. `tgt_in`: [B, L] u32.
    fn forward(&self, prompt: Option<&Tensor>, tgt_in: &Tensor, train: bool) -> Result<Tensor> {
        let b = tgt_in.dim(0)?;
        let p = self.cfg.prefix;
        let pre = match prompt {
            Some(t) => self.prompt_emb.forward(t)?,
            None => self.prompt_emb.forward(&Tensor::zeros((b, p), DType::U32, &self.dev)?)?,
        };
        let x = Tensor::cat(&[pre, self.tok_emb.forward(tgt_in)?], 1)?;
        let n = p + self.cfg.len;
        let pos = self.pos_emb.forward(&Tensor::arange(0u32, n as u32, &self.dev)?)?;
        let h = self.body.forward(&x.broadcast_add(&pos)?, Some(&self.mask), train)?;
        self.head.forward(&h.narrow(1, p, self.cfg.len)?)
    }

    /// AR teacher-forcing input: [BOS, y_1 .. y_{L-1}].
    fn shift_right(&self, x: &Tensor) -> Result<Tensor> {
        let b = x.dim(0)?;
        let bos = Tensor::full(self.special(), (b, 1), &self.dev)?;
        Tensor::cat(&[bos, x.narrow(1, 0, self.cfg.len - 1)?], 1)
    }

    pub fn train_step(&self, x: &Tensor, prompt: Option<&Tensor>, trainer: &mut Trainer, rng: &mut StdRng) -> Result<f32> {
        let (b, l) = x.dims2()?;
        let v = self.cfg.vocab;
        let loss = match self.cfg.kind {
            Kind::Ar => {
                let logits = self.forward(prompt, &self.shift_right(x)?, true)?;
                nll_per_row(&logits.reshape((b * l, v))?, &x.flatten_all()?)?.mean_all()?
            }
            Kind::Mdlm => {
                // t ~ U(ε, 1); mask each token w.p. t; ELBO weight 1/t on masked tokens.
                let xs: Vec<u32> = x.flatten_all()?.to_vec1()?;
                let mut inp = xs.clone();
                let mut w = vec![0f32; b * l];
                for i in 0..b {
                    let t: f64 = rng.gen_range(1e-3..1.0);
                    for j in 0..l {
                        if rng.gen_bool(t) {
                            inp[i * l + j] = self.special();
                            w[i * l + j] = (1.0 / t) as f32;
                        }
                    }
                }
                let inp = Tensor::from_vec(inp, (b, l), &self.dev)?;
                let w = Tensor::from_vec(w, b * l, &self.dev)?;
                let logits = self.forward(prompt, &inp, true)?;
                let nll = nll_per_row(&logits.reshape((b * l, v))?, &x.flatten_all()?)?;
                ((nll * w)?.sum_all()? / (b * l) as f64)?
            }
        };
        trainer.step(&loss)?;
        loss.to_scalar::<f32>()
    }

    /// Generate `b` sequences. For AR, `steps` is ignored (NFE = L).
    /// For MDLM, `steps` = T denoising steps with confidence-ordered unmasking.
    pub fn generate(
        &self,
        b: usize,
        prompt: Option<&Tensor>,
        steps: usize,
        temperature: f64,
        rng: &mut StdRng,
    ) -> Result<Vec<Vec<u32>>> {
        let l = self.cfg.len;
        let mut cur = vec![vec![self.special(); l]; b];
        match self.cfg.kind {
            Kind::Ar => {
                // No KV cache: every step re-runs the full (causally masked) sequence.
                for t in 0..l {
                    let mut inp = vec![self.special(); b * l];
                    for (i, row) in cur.iter().enumerate() {
                        inp[i * l + 1..i * l + 1 + t].copy_from_slice(&row[..t]);
                    }
                    let inp = Tensor::from_vec(inp, (b, l), &self.dev)?;
                    let logits = self.forward(prompt, &inp, false)?.narrow(1, t, 1)?;
                    let next: Vec<u32> = sample(&logits, temperature, rng)?.flatten_all()?.to_vec1()?;
                    for (row, &tok) in cur.iter_mut().zip(&next) {
                        row[t] = tok;
                    }
                }
            }
            Kind::Mdlm => {
                let steps = steps.clamp(1, l);
                for s in 1..=steps {
                    let inp = Tensor::from_vec(cur.concat(), (b, l), &self.dev)?;
                    let logits = self.forward(prompt, &inp, false)?;
                    let pick: Vec<u32> = sample(&logits, temperature, rng)?.flatten_all()?.to_vec1()?;
                    let probs = ops::softmax_last_dim(&logits)?;
                    let conf: Vec<f32> = probs
                        .gather(&Tensor::from_vec(pick.clone(), (b, l, 1), &self.dev)?, D::Minus1)?
                        .flatten_all()?
                        .to_vec1()?;
                    let target = (l * s).div_ceil(steps);
                    for i in 0..b {
                        let row = &mut cur[i];
                        let done = row.iter().filter(|&&t| t != self.special()).count();
                        let mut masked: Vec<usize> = (0..l).filter(|&j| row[j] == self.special()).collect();
                        masked.sort_by(|&a, &c| conf[i * l + c].total_cmp(&conf[i * l + a]));
                        for &j in masked.iter().take(target.saturating_sub(done)) {
                            row[j] = pick[i * l + j];
                        }
                    }
                }
            }
        }
        Ok(cur)
    }

    /// Per-token negative log-likelihood [B][L] (AR only; used for Gen. PPL).
    pub fn token_nll(&self, x: &Tensor, prompt: Option<&Tensor>) -> Result<Vec<Vec<f32>>> {
        assert_eq!(self.cfg.kind, Kind::Ar);
        let (b, l) = x.dims2()?;
        let logits = self.forward(prompt, &self.shift_right(x)?, false)?;
        nll_per_row(&logits.reshape((b * l, self.cfg.vocab))?, &x.flatten_all()?)?
            .reshape((b, l))?
            .to_vec2()
    }
}
