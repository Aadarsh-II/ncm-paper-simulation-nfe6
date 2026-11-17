//! Evaluation metrics used in the paper: generative perplexity under a
//! reference model, mean per-sample unigram entropy, plus exact checks.

use std::collections::HashMap;

/// Mean per-sample unigram entropy (nats) over non-pad tokens.
pub fn unigram_entropy(samples: &[Vec<u32>], pad: Option<u32>) -> f64 {
    let mut total = 0.0;
    for s in samples {
        let mut counts: HashMap<u32, usize> = HashMap::new();
        let mut n = 0usize;
        for &t in s {
            if Some(t) != pad {
                *counts.entry(t).or_default() += 1;
                n += 1;
            }
        }
        if n == 0 {
            continue;
        }
        total -= counts
            .values()
            .map(|&c| {
                let p = c as f64 / n as f64;
                p * p.ln()
            })
            .sum::<f64>();
    }
    total / samples.len().max(1) as f64
}

/// Fraction of distinct samples.
pub fn distinct(samples: &[Vec<u32>]) -> f64 {
    let set: std::collections::HashSet<_> = samples.iter().collect();
    set.len() as f64 / samples.len().max(1) as f64
}

/// Corpus-level perplexity from per-token NLLs, counting tokens up to and
/// including the first pad (the end-of-story marker).
pub fn gen_ppl(samples: &[Vec<u32>], nll: &[Vec<f32>], pad: u32) -> f64 {
    let (mut sum, mut n) = (0f64, 0usize);
    for (s, l) in samples.iter().zip(nll) {
        let stop = s.iter().position(|&t| t == pad).map(|p| p + 1).unwrap_or(s.len());
        sum += l[..stop].iter().map(|&x| x as f64).sum::<f64>();
        n += stop;
    }
    (sum / n.max(1) as f64).exp()
}
