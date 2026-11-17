//! Training utilities: AdamW with linear warmup + cosine decay to a floor,
//! and global-norm gradient clipping (paper Tables 5 & 6).

use candle_core::{Result, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};

pub struct Trainer {
    opt: AdamW,
    vars: Vec<Var>,
    pub peak_lr: f64,
    pub floor_lr: f64,
    pub warmup: usize,
    pub total: usize,
    pub clip: f64,
    pub step: usize,
}

impl Trainer {
    pub fn new(varmap: &VarMap, peak_lr: f64, weight_decay: f64, beta2: f64, warmup: usize, total: usize) -> Result<Self> {
        let vars = varmap.all_vars();
        let opt = AdamW::new(
            vars.clone(),
            ParamsAdamW { lr: peak_lr, beta1: 0.9, beta2, eps: 1e-8, weight_decay },
        )?;
        Ok(Self { opt, vars, peak_lr, floor_lr: peak_lr / 10.0, warmup, total, clip: 1.0, step: 0 })
    }

    fn lr_at(&self, step: usize) -> f64 {
        if step < self.warmup {
            return self.peak_lr * (step + 1) as f64 / self.warmup as f64;
        }
        let t = (step - self.warmup) as f64 / (self.total.saturating_sub(self.warmup).max(1)) as f64;
        let cos = 0.5 * (1.0 + (std::f64::consts::PI * t.min(1.0)).cos());
        self.floor_lr + (self.peak_lr - self.floor_lr) * cos
    }

    /// Backprop `loss`, clip by global norm, and take one AdamW step.
    /// Returns the pre-clip gradient norm.
    pub fn step(&mut self, loss: &Tensor) -> Result<f64> {
        let mut grads = loss.backward()?;
        let mut sq = 0f64;
        for v in &self.vars {
            if let Some(g) = grads.get(v.as_tensor()) {
                sq += g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            }
        }
        let norm = sq.sqrt();
        if norm > self.clip {
            let s = self.clip / (norm + 1e-6);
            for v in &self.vars {
                if let Some(g) = grads.remove(v.as_tensor()) {
                    grads.insert(v.as_tensor(), (g * s)?);
                }
            }
        }
        self.opt.set_learning_rate(self.lr_at(self.step));
        self.opt.step(&grads)?;
        self.step += 1;
        Ok(norm)
    }
}

/// Count trainable parameters.
pub fn param_count(varmap: &VarMap) -> usize {
    varmap.all_vars().iter().map(|v| v.as_tensor().elem_count()).sum()
}
