# ncm-sim: Next-Concept Modeling in Rust

A small, CPU-only Rust simulation of

> **Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation**
> Kumbong, Khare, Mlauzi, Woen, Ding, Buchanan, Arora, Mirhoseini, Ré. arXiv:2610.14162 (Oct 2026).

The project implements the paper's two-stage method from scratch on top of
[candle](https://github.com/huggingface/candle). It compares NCM against
masked-diffusion and autoregressive baselines on two tasks that need no
downloads: a synthetic story language and Sudoku.

## What NCM is

Diffusion LMs sample in parallel, but they process all `L` positions at every
step and degrade with few steps. NCM instead:

1. **Stage 1: multiscale residual VQ-VAE** (`src/vqvae.rs`). This encodes a
   length-`L` token sequence into `K` levels of discrete codes with lengths
   `ℓ_1 < … < ℓ_K = L`, from coarse to fine. Level `k` quantizes the
   *residual* left by earlier levels:

   ```text
   u_k   = Down_k(z^(k-1))                        (area-resize L → ℓ_k)
   r_k,i = argmin_j ‖u_k,i − e_k,j‖²              (codebook V_k)
   ẑ^(k) = ẑ^(k-1) + R_k(e_k(r_k))                (R_k: interpolate ℓ_k → L, then refine with φ_k)
   z^(k) = z^(k-1) − R_k(e_k(r_k))
   ```

   The tokenizer is trained with **nested dropout**: the hierarchy is cut off
   at a random level, so coarse levels have to carry meaning on their own.
   It also uses a straight-through estimator, EMA codebooks with dead-code
   restarts, and `CE + β·MSE(z, sg ẑ) + MSE(sg z, ẑ)`.
2. **Stage 2: the NCM transformer** (`src/ncm.rs`). With the tokenizer frozen,
   one transformer predicts **all codes of level `k` in parallel** from the
   cumulative reconstruction `ẑ^(k-1)` resized to `ℓ_k` positions.
   - *Training (Algorithm 1)*: all levels are predicted in **one forward
     pass**. Prefix-LM, block-diagonal attention lets each level's block see
     the prompt and itself. The loss is a level-weighted cross-entropy with
     `w_k ∝ ℓ_k^(−α)`. **Conditioning corruption** replaces a share of the
     input codes with random codes, at a rate that falls linearly across
     levels, so the model stays robust to its own sampling errors.
   - *Generation (Algorithm 2)*: `K` forward passes, from coarse to fine, so
     **NFE = K ≪ L**. Early passes are cheap because `ℓ_k` is short. The
     tokenizer decoder maps `ẑ^(K)` to tokens.

Baselines (`src/baselines.rs`):
- **MDLM-style masked diffusion**: absorbing-state ELBO training (1/t
  weighting) and confidence-ordered unmasking in `T` steps (NFE = T).
- **Autoregressive**: a prefix-LM transformer (NFE = L). It has no KV cache,
  so its wall-clock time is pessimistic.
- An independently trained AR model serves as the **reference model for
  generative perplexity**. The paper uses GPT-2 Large for this.

## Tasks

| Task | Paper analogue | Data | Metrics |
|---|---|---|---|
| `stories` | TinyStories / LM1B unconditional generation | A synthetic grammar (`src/data/stories.rs`) with long-range constraints: the hero's name is repeated, pronoun and noun agree with gender, and a later sentence must reuse the object found earlier. L = 48. | Gen. PPL (reference AR), **story%** (whole story exactly grammatical), **sent%** (share of well-formed sentences consistent with the hero and object), unigram entropy H, distinct %, NFE, ms/sample |
| `sudoku` | Sudoku (S-FLM setup) | Generated in Rust (`src/data/sudoku.rs`): random full grids, puzzles carved down to a target clue count with a unique solution. Easy, medium and hard puzzles have 40, 35 and 30 clues. | Completely solved boards %, blank-cell accuracy %, NFE, ms/puzzle |

The tokenizer report mirrors Table 1 of the paper: held-out token and
sequence reconstruction, per-level codebook utilization, and accuracy when
decoding from only the first `k` levels.

## Usage

```bash
cargo build --release
cargo test --release

# Print a multiscale ladder
./target/release/ncm-sim ladder --len 81 --levels 8

# Unconditional generation: tokenizer → NCM → MDLM / AR baselines → reference model → table
./target/release/ncm-sim stories

# Sudoku reasoning
./target/release/ncm-sim sudoku

# Useful knobs (see --help): --levels K, --codebook, --latent, --hidden, --depth,
# --steps, --vq-steps, --batch, --alpha, --corrupt-max/--corrupt-min,
# --nested-dropout, --vq-loss-full, --ncm-only, --seed
./target/release/ncm-sim stories --levels 8 --steps 4000 --alpha 1.0
```

### GPU

GPU support uses candle's CUDA or Metal backend, chosen at build time:

```bash
cargo build --release --features cuda    # NVIDIA (needs the CUDA toolkit / nvcc)
cargo build --release --features cudnn   # NVIDIA with cuDNN
cargo build --release --features metal   # Apple silicon
./target/release/ncm-sim stories                 # --device auto: GPU if present, else CPU
./target/release/ncm-sim sudoku --device cuda --gpu 1
```

The default build is CPU-only. `--device auto` falls back to the CPU when no
GPU backend is compiled in or no GPU is found; `--device cuda` or `metal`
fails with an error instead. At these model sizes a GPU mainly pays off with
a larger `--batch`, `--hidden` and `--depth`.

All three generators (NCM, MDLM and AR) share the hidden width, depth,
batch size, step count and learning-rate schedule (AdamW, warmup, cosine to
a 10% floor, clipping at 1.0).

## Results

See `results/` for full logs of the default runs.

### Stories (default run: 4,000 steps, K = 6, 4 CPU cores)

Tokenizer (held-out): token reconstruction 99.17%, exact-sequence 71.8%, mean
codebook utilization 82%. Decoding from only the first k levels gives
52.8 / 72.9 / 84.0 / 93.1 / 95.9 / 99.2% token accuracy for k = 1…6, so coarse
levels carry most of the content, as nested dropout intends.

| method | NFE | Gen.PPL ↓ | story% ↑ | sent% ↑ | H | ms/sample ↓ |
|---|---|---|---|---|---|---|
| data | – | 2.00 | 100.0 | 100.0 | 3.00 | – |
| **NCM** | **6** | 22.27 | 0.0 | 8.1 | 2.82 | **10.6** |
| MDLM T=6 | 6 | 15.22 | 0.0 | 8.1 | 2.78 | 15.8 |
| MDLM T=16 | 16 | 10.39 | 0.0 | 6.3 | 2.72 | 42.8 |
| MDLM T=48 | 48 | 8.06 | 0.0 | 4.3 | 2.68 | 126.5 |
| AR (no KV cache) | 48 | 3.27 | 0.0 | 27.8 | 3.05 | 123.7 |

What reproduces at this scale:
- **Speed.** NCM is the fastest sampler: about 12× faster than AR or MDLM
  with T = L, and 1.5× faster than MDLM at the same NFE, because its early
  passes only process `ℓ_k` positions.
- **Coarse-to-fine behavior.** The logged trajectory shows level 1 fixing the
  story skeleton ("once upon a time there was a … named …") and later levels
  filling in detail.
- **Matched-NFE comparison.** At NFE = 6, NCM matches MDLM on sentence
  validity (8.1%) and keeps higher unigram entropy (2.82 vs 2.78).

What does not reproduce at this scale:
- NCM has the worst Gen. PPL, and no method produces a fully valid story.
  AR is clearly best on quality. With about 1M parameters and 4k steps every
  model is badly undertrained; the paper's quality gains appear at about 100M
  parameters and 50k+ steps.
- NCM training is about 4.6× slower per step on CPU (3,900 s vs about 830 s),
  because its training sequence has Σℓ = 147 positions against 48.

An earlier 2,000-step run (`results/stories_2k_steps.txt`) showed the same
ranking with worse numbers (NCM Gen. PPL 23.8). Sudoku has not been run at
full length; it takes several hours on CPU.


## Differences from the paper (scale and simplifications)

- **Scale.** Models have about 0.5–1M parameters and train for a few
  thousand steps on 4 CPU cores. The paper trains about 100M-parameter
  models on GPUs for 50k–350k steps. Absolute numbers are not comparable;
  the simulation reproduces the *mechanism* and the *trends* (NFE, cost per
  step, quality versus steps).
- **Data.** Synthetic stories stand in for TinyStories, LM1B and OWT, so
  validity can be checked exactly. A small AR model replaces GPT-2 Large as
  the Gen. PPL judge. XSum and WMT14 are not simulated.
- **Encoder.** It is a token-embedding table, as in the paper (0 encoder
  blocks). `R_k` is linear interpolation followed by a learned per-level
  linear refinement `h + ½φ_k(h)`. The paper does not fully specify this, so
  the form follows next-scale prediction for images.
- **Corruption.** Rates fall linearly from `--corrupt-max` (0.6) to
  `--corrupt-min` (0.1); the paper's selected recipe has a mean of about 0.5.
- **Determinism.** Data and sampling are seeded. Candle's CPU backend does not
  seed weight initialization, so runs vary slightly.
