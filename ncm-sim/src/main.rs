//! ncm-sim: a small-scale, CPU-only Rust simulation of
//! "Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation"
//! (Kumbong, Khare et al., arXiv:2610.14162).
//!
//! Experiments:
//!   stories  – unconditional generation on a synthetic story language
//!              (NCM vs. masked diffusion vs. autoregressive).
//!   sudoku   – conditional reasoning on Sudoku (solved % by difficulty).
//!   ladder   – print a multiscale ladder ℓ_1..ℓ_K.

mod baselines;
mod data;
mod metrics;
mod ncm;
mod nn;
mod scales;
mod train;
mod vqvae;

use std::time::Instant;

use candle_core::{Device, Result, Tensor};
use clap::{Args, Parser, Subcommand, ValueEnum};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use baselines::{Kind, SeqConfig, SeqModel};
use data::{stories, sudoku};
use ncm::{Ncm, NcmConfig};
use train::{Trainer, param_count};
use vqvae::{VqConfig, VqVae, batch_tensor, evaluate_tokenizer};

#[derive(Parser)]
#[command(version, about = "Rust simulation of Next-Concept Modeling (NCM, arXiv:2610.14162)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Unconditional generation on a synthetic story language.
    Stories(Common),
    /// Conditional generation / reasoning on Sudoku.
    Sudoku(Common),
    /// Print the linear multiscale ladder for a target length.
    Ladder {
        #[arg(long, default_value_t = 81)]
        len: usize,
        #[arg(long, default_value_t = 8)]
        levels: usize,
    },
}

#[derive(Args, Clone, Debug)]
struct Common {
    /// Number of levels K (= NCM sampling steps / NFE).
    #[arg(long)]
    levels: Option<usize>,
    /// Codes per level codebook |V_k|.
    #[arg(long)]
    codebook: Option<usize>,
    /// Tokenizer latent width d.
    #[arg(long)]
    latent: Option<usize>,
    /// Hidden width of the generators (NCM and baselines).
    #[arg(long, default_value_t = 128)]
    hidden: usize,
    /// Transformer depth of the generators.
    #[arg(long, default_value_t = 4)]
    depth: usize,
    #[arg(long, default_value_t = 4)]
    heads: usize,
    /// Tokenizer training steps.
    #[arg(long)]
    vq_steps: Option<usize>,
    /// Generator training steps (shared by NCM and baselines).
    #[arg(long)]
    steps: Option<usize>,
    #[arg(long, default_value_t = 64)]
    batch: usize,
    #[arg(long, default_value_t = 1e-3)]
    lr: f64,
    /// Training-set size.
    #[arg(long)]
    train_size: Option<usize>,
    /// Evaluation samples (stories) or puzzles per difficulty (sudoku).
    #[arg(long)]
    eval: Option<usize>,
    /// Loss weighting exponent α.
    #[arg(long, default_value_t = 0.5)]
    alpha: f64,
    /// Conditioning-corruption rate at the coarsest level (linearly decreasing).
    #[arg(long, default_value_t = 0.6)]
    corrupt_max: f64,
    #[arg(long, default_value_t = 0.1)]
    corrupt_min: f64,
    /// Nested-dropout probability (truncate the hierarchy at a random level).
    #[arg(long, default_value_t = 0.5)]
    nested_dropout: f64,
    /// Compute VQ commitment/quantization losses on the full hierarchy
    /// (the paper evaluates them at the nested-dropout cutoff).
    #[arg(long)]
    vq_loss_full: bool,
    /// Skip the baselines (train/evaluate NCM only).
    #[arg(long)]
    ncm_only: bool,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Log every N steps.
    #[arg(long, default_value_t = 100)]
    log_every: usize,
    /// Compute device: auto (GPU if built with one and available, else CPU), cpu, cuda, metal.
    #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,
    /// GPU ordinal for cuda/metal.
    #[arg(long, default_value_t = 0)]
    gpu: usize,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DeviceArg {
    Auto,
    Cpu,
    Cuda,
    Metal,
}

/// Pick the compute device. GPU backends exist only when the binary was
/// built with `--features cuda` (or `cudnn`) or `--features metal`.
fn pick_device(c: &Common) -> Result<Device> {
    let dev = match c.device {
        DeviceArg::Cpu => Device::Cpu,
        DeviceArg::Cuda => Device::new_cuda(c.gpu)?,
        DeviceArg::Metal => Device::new_metal(c.gpu)?,
        DeviceArg::Auto => {
            if candle_core::utils::cuda_is_available() {
                Device::new_cuda(c.gpu)?
            } else if candle_core::utils::metal_is_available() {
                Device::new_metal(c.gpu)?
            } else {
                Device::Cpu
            }
        }
    };
    let name = match &dev {
        Device::Cpu => format!("CPU ({} threads)", candle_core::utils::get_num_threads()),
        Device::Cuda(_) => format!("CUDA GPU {}", c.gpu),
        Device::Metal(_) => format!("Metal GPU {}", c.gpu),
    };
    println!("device: {name}");
    Ok(dev)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Ladder { len, levels } => {
            let lad = scales::linear_ladder(len, levels);
            println!("ladder (K={levels}, L={len}): {lad:?}  Σℓ = {}", lad.iter().sum::<usize>());
            Ok(())
        }
        Cmd::Stories(c) => run_stories(c),
        Cmd::Sudoku(c) => run_sudoku(c),
    }
}

fn header(s: &str) {
    println!("\n=== {s} {}", "=".repeat(70usize.saturating_sub(s.len())));
}

fn sample_batch<'a, T>(data: &'a [T], b: usize, rng: &mut StdRng) -> Vec<&'a T> {
    (0..b).map(|_| &data[rng.gen_range(0..data.len())]).collect()
}

/// Train the multiscale residual VQ-VAE on target sequences and report Table 1-style metrics.
fn train_tokenizer(cfg: VqConfig, data: &[Vec<u32>], test: &[Vec<u32>], c: &Common, steps: usize, dev: &Device, rng: &mut StdRng) -> Result<VqVae> {
    header("Stage 1: multiscale residual VQ-VAE tokenizer");
    let mut vq = VqVae::new(cfg, dev)?;
    println!(
        "ladder ℓ = {:?} (K={}, Σℓ={}), |V_k|={}, d={}, params={}",
        vq.cfg.ladder,
        vq.k(),
        vq.cfg.ladder.iter().sum::<usize>(),
        vq.cfg.codebook,
        vq.cfg.dim,
        param_count(&vq.varmap)
    );
    let mut tr = Trainer::new(&vq.varmap, c.lr, 0.01, 0.99, steps / 20 + 1, steps)?;
    let t0 = Instant::now();
    for step in 0..steps {
        let rows: Vec<Vec<u32>> = sample_batch(data, c.batch, rng).into_iter().cloned().collect();
        let (loss, ce) = vq.train_step(&batch_tensor(&rows, dev)?, &mut tr, rng)?;
        if step % c.log_every == 0 || step + 1 == steps {
            println!("  vq step {step:5}  loss {loss:.4}  recon-CE {ce:.4}  ({:.0}s)", t0.elapsed().as_secs_f64());
        }
    }
    let rep = evaluate_tokenizer(&vq, test, 128, dev)?;
    println!(
        "Tokenizer (held-out): token recon acc {:.2}%  exact-sequence recon {:.2}%  mean utilization {:.1}%",
        100.0 * rep.token_acc,
        100.0 * rep.seq_acc,
        100.0 * rep.utilization.iter().sum::<f64>() / rep.utilization.len() as f64
    );
    println!(
        "  token acc decoding from first k levels: {}",
        rep.partial_acc.iter().enumerate().map(|(k, a)| format!("k={}:{:.1}%", k + 1, 100.0 * a)).collect::<Vec<_>>().join("  ")
    );
    println!(
        "  per-level utilization: {}",
        rep.utilization.iter().map(|u| format!("{:.0}%", 100.0 * u)).collect::<Vec<_>>().join(" ")
    );
    Ok(vq)
}

fn train_loop(name: &str, steps: usize, log_every: usize, mut f: impl FnMut(usize) -> Result<f32>) -> Result<f64> {
    let t0 = Instant::now();
    for step in 0..steps {
        let loss = f(step)?;
        if step % log_every == 0 || step + 1 == steps {
            println!("  {name} step {step:5}  loss {loss:.4}  ({:.0}s)", t0.elapsed().as_secs_f64());
        }
    }
    Ok(t0.elapsed().as_secs_f64())
}

// ---------------------------------------------------------------------------
// Unconditional generation: synthetic stories
// ---------------------------------------------------------------------------

struct GenRow {
    name: String,
    nfe: usize,
    samples: Vec<Vec<u32>>,
    secs_per_seq: f64,
}

fn run_stories(c: Common) -> Result<()> {
    let dev = pick_device(&c)?;
    let mut rng = StdRng::seed_from_u64(c.seed);
    let vocab = stories::Vocab::new();
    let l = stories::LEN;
    let train = stories::dataset(c.train_size.unwrap_or(20_000), &vocab, &mut rng);
    let test = stories::dataset(1_000, &vocab, &mut rng);
    let k = c.levels.unwrap_or(6);
    let steps = c.steps.unwrap_or(4_000);
    let n_eval = c.eval.unwrap_or(256);
    println!("Synthetic stories: vocab {}, L = {l}, train {} / test {}", vocab.len(), train.len(), test.len());
    println!("example: {}", vocab.decode(&train[0]));

    let vq_cfg = VqConfig {
        vocab: vocab.len(),
        ladder: scales::linear_ladder(l, k),
        codebook: c.codebook.unwrap_or(512),
        dim: c.latent.unwrap_or(128),
        dec_depth: 2,
        dec_heads: 4,
        dropout: 0.1,
        beta: 0.25,
        ema_decay: 0.99,
        nested_dropout: c.nested_dropout,
        vq_loss_full: c.vq_loss_full,
    };
    let vq = train_tokenizer(vq_cfg, &train, &test, &c, c.vq_steps.unwrap_or(1_500), &dev, &mut rng)?;

    header("Stage 2: NCM generator");
    let ncm = Ncm::new(ncm_config(&c, 1), &vq, &dev)?;
    println!(
        "NCM params={}  K={}  α={}  corruption {:.2}→{:.2}",
        param_count(&ncm.varmap),
        ncm.k(),
        c.alpha,
        ncm.corrupt_rate(0),
        ncm.corrupt_rate(ncm.k().saturating_sub(2))
    );
    let mut tr = Trainer::new(&ncm.varmap, c.lr, 0.1, 0.95, steps / 20 + 1, steps)?;
    let t_ncm = train_loop("ncm", steps, c.log_every, |_| {
        let rows: Vec<Vec<u32>> = sample_batch(&train, c.batch, &mut rng).into_iter().cloned().collect();
        ncm.train_step(&vq, &batch_tensor(&rows, &dev)?, None, &mut tr, &mut rng)
    })?;
    println!("  NCM training time {t_ncm:.0}s");

    let mk = |kind| SeqConfig {
        kind,
        vocab: vocab.len(),
        prompt_vocab: 1,
        prefix: 1,
        len: l,
        hidden: c.hidden,
        depth: c.depth,
        heads: c.heads,
        dropout: 0.1,
    };
    let train_seq = |m: &SeqModel, name: &str, steps: usize, rng: &mut StdRng| -> Result<()> {
        let mut tr = Trainer::new(&m.varmap, c.lr, 0.1, 0.95, steps / 20 + 1, steps)?;
        println!("{name} params={}", param_count(&m.varmap));
        let t = train_loop(name, steps, c.log_every, |_| {
            let rows: Vec<Vec<u32>> = sample_batch(&train, c.batch, rng).into_iter().cloned().collect();
            m.train_step(&batch_tensor(&rows, &dev)?, None, &mut tr, rng)
        })?;
        println!("  {name} training time {t:.0}s");
        Ok(())
    };

    let mut rows = Vec::new();
    let eval_bs = 64;
    let time_gen = |f: &mut dyn FnMut(usize) -> Result<Vec<Vec<u32>>>| -> Result<(Vec<Vec<u32>>, f64)> {
        let t0 = Instant::now();
        let mut out = Vec::new();
        while out.len() < n_eval {
            out.extend(f(eval_bs.min(n_eval - out.len()))?);
        }
        Ok((out, t0.elapsed().as_secs_f64() / n_eval as f64))
    };

    let (s, t) = time_gen(&mut |b| Ok(ncm.generate(&vq, b, None, 1.0, &mut rng, false)?.0))?;
    rows.push(GenRow { name: "NCM (ours)".into(), nfe: ncm.k(), samples: s, secs_per_seq: t });

    if !c.ncm_only {
        header("Baselines (same width/depth/steps/batch)");
        let mdlm = SeqModel::new(mk(Kind::Mdlm), &dev)?;
        train_seq(&mdlm, "mdlm", steps, &mut rng)?;
        for t_steps in [ncm.k(), 16, l] {
            let (s, t) = time_gen(&mut |b| mdlm.generate(b, None, t_steps, 1.0, &mut rng))?;
            rows.push(GenRow { name: format!("MDLM T={t_steps}"), nfe: t_steps, samples: s, secs_per_seq: t });
        }
        let ar = SeqModel::new(mk(Kind::Ar), &dev)?;
        train_seq(&ar, "ar", steps, &mut rng)?;
        let (s, t) = time_gen(&mut |b| ar.generate(b, None, 0, 1.0, &mut rng))?;
        rows.push(GenRow { name: "AR (no KV cache)".into(), nfe: l, samples: s, secs_per_seq: t });
    }

    header("Reference model for Gen. PPL (independent AR, 2× steps)");
    let judge = SeqModel::new(mk(Kind::Ar), &dev)?;
    train_seq(&judge, "judge", 2 * steps, &mut rng)?;

    header("Results: unconditional generation");
    let data_rows: Vec<Vec<u32>> = test[..n_eval.min(test.len())].to_vec();
    let score = |samples: &[Vec<u32>]| -> Result<f64> {
        let mut nll = Vec::new();
        for ch in samples.chunks(128) {
            nll.extend(judge.token_nll(&batch_tensor(ch, &dev)?, None)?);
        }
        Ok(metrics::gen_ppl(samples, &nll, 0))
    };
    let valid = |samples: &[Vec<u32>]| {
        samples
            .iter()
            .filter(|s| {
                let w: Vec<&str> = s.iter().map(|&i| vocab.words[i as usize].as_str()).collect();
                stories::is_valid(&w)
            })
            .count() as f64
            / samples.len() as f64
    };
    let sent_valid = |samples: &[Vec<u32>]| {
        let (mut ok, mut n) = (0, 0);
        for s in samples {
            let w: Vec<&str> = s.iter().map(|&i| vocab.words[i as usize].as_str()).collect();
            let (a, b) = stories::sentence_validity(&w);
            ok += a;
            n += b;
        }
        ok as f64 / n.max(1) as f64
    };
    println!(
        "{:<18} {:>5} {:>10} {:>9} {:>9} {:>7} {:>9} {:>12}",
        "method", "NFE", "Gen.PPL↓", "story%↑", "sent%↑", "H", "distinct", "ms/sample↓"
    );
    println!(
        "{:<18} {:>5} {:>10.3} {:>9.1} {:>9.1} {:>7.3} {:>9.2} {:>12}",
        "data",
        "-",
        score(&data_rows)?,
        100.0 * valid(&data_rows),
        100.0 * sent_valid(&data_rows),
        metrics::unigram_entropy(&data_rows, Some(0)),
        metrics::distinct(&data_rows),
        "-"
    );
    for r in &rows {
        println!(
            "{:<18} {:>5} {:>10.3} {:>9.1} {:>9.1} {:>7.3} {:>9.2} {:>12.2}",
            r.name,
            r.nfe,
            score(&r.samples)?,
            100.0 * valid(&r.samples),
            100.0 * sent_valid(&r.samples),
            metrics::unigram_entropy(&r.samples, Some(0)),
            metrics::distinct(&r.samples),
            1e3 * r.secs_per_seq
        );
    }
    println!("story% = whole story exactly grammatical; sent% = well-formed, consistent sentences");

    header("NCM generation trajectory (decoded after each level, coarse → fine)");
    let (_, traj) = ncm.generate(&vq, 1, None, 1.0, &mut rng, true)?;
    for (k, out) in traj.iter().enumerate() {
        println!("  level {:2} (ℓ={:2}): {}", k + 1, vq.cfg.ladder[k], vocab.decode(&out[0]));
    }
    for r in &rows {
        println!("\n{} samples:", r.name);
        for s in r.samples.iter().take(3) {
            println!("  - {}", vocab.decode(s));
        }
    }
    Ok(())
}

fn ncm_config(c: &Common, prefix: usize) -> NcmConfig {
    NcmConfig {
        hidden: c.hidden,
        depth: c.depth,
        heads: c.heads,
        dropout: 0.1,
        prefix,
        alpha: c.alpha,
        corrupt_max: c.corrupt_max,
        corrupt_min: c.corrupt_min,
    }
}

// ---------------------------------------------------------------------------
// Reasoning: Sudoku
// ---------------------------------------------------------------------------

fn run_sudoku(c: Common) -> Result<()> {
    let dev = pick_device(&c)?;
    let mut rng = StdRng::seed_from_u64(c.seed);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let n_train = c.train_size.unwrap_or(20_000);
    let n_eval = c.eval.unwrap_or(100);
    let k = c.levels.unwrap_or(8);
    let steps = c.steps.unwrap_or(3_000);

    header("Data");
    let t0 = Instant::now();
    let train = sudoku::dataset(n_train, (30, 45), c.seed, threads);
    let tests: Vec<(&str, usize, Vec<sudoku::Puzzle>)> = [("easy", 40), ("medium", 35), ("hard", 30)]
        .iter()
        .enumerate()
        .map(|(i, &(n, cl))| (n, cl, sudoku::dataset(n_eval, (cl, cl), c.seed + 1000 + i as u64, threads)))
        .collect();
    println!(
        "{} training puzzles (30–45 clues, unique solutions) + {}×3 test puzzles in {:.1}s",
        train.len(),
        n_eval,
        t0.elapsed().as_secs_f64()
    );
    let solutions: Vec<Vec<u32>> = train.iter().map(|p| p.solution.clone()).collect();
    let test_solutions: Vec<Vec<u32>> = tests.iter().flat_map(|t| t.2.iter().map(|p| p.solution.clone())).collect();

    let vq_cfg = VqConfig {
        vocab: sudoku::VOCAB,
        ladder: scales::linear_ladder(sudoku::CELLS, k),
        codebook: c.codebook.unwrap_or(16),
        dim: c.latent.unwrap_or(64),
        dec_depth: 1,
        dec_heads: 4,
        dropout: 0.0,
        beta: 0.25,
        ema_decay: 0.99,
        nested_dropout: c.nested_dropout,
        vq_loss_full: c.vq_loss_full,
    };
    let vq = train_tokenizer(vq_cfg, &solutions, &test_solutions, &c, c.vq_steps.unwrap_or(1_000), &dev, &mut rng)?;

    let batch = |rng: &mut StdRng| -> Result<(Tensor, Tensor)> {
        let ps = sample_batch(&train, c.batch, rng);
        let x: Vec<Vec<u32>> = ps.iter().map(|p| p.solution.clone()).collect();
        let pr: Vec<Vec<u32>> = ps.iter().map(|p| p.clues.clone()).collect();
        Ok((batch_tensor(&x, &dev)?, batch_tensor(&pr, &dev)?))
    };

    header("Stage 2: NCM generator (prefix = puzzle)");
    let ncm = Ncm::new(ncm_config(&c, sudoku::CELLS), &vq, &dev)?;
    println!("NCM params={}  K={}", param_count(&ncm.varmap), ncm.k());
    let mut tr = Trainer::new(&ncm.varmap, c.lr, 0.1, 0.95, steps / 20 + 1, steps)?;
    train_loop("ncm", steps, c.log_every, |_| {
        let (x, p) = batch(&mut rng)?;
        ncm.train_step(&vq, &x, Some(&p), &mut tr, &mut rng)
    })?;

    type Row = (String, usize, Vec<f64>, Vec<f64>, f64);
    let mut results: Vec<Row> = Vec::new();
    let eval = |name: &str, nfe: usize, gen_fn: &mut dyn FnMut(&Tensor, usize) -> Result<Vec<Vec<u32>>>| -> Result<Row> {
        let mut accs = Vec::new();
        let mut cells = Vec::new();
        let t0 = Instant::now();
        let mut total = 0;
        for (_, _, set) in &tests {
            let (mut ok, mut cell_ok, mut cell_n) = (0, 0, 0);
            for ch in set.chunks(50) {
                let clues: Vec<Vec<u32>> = ch.iter().map(|p| p.clues.clone()).collect();
                let out = gen_fn(&batch_tensor(&clues, &dev)?, ch.len())?;
                for (p, o) in ch.iter().zip(&out) {
                    ok += sudoku::is_solved(&p.clues, o) as usize;
                    for i in (0..sudoku::CELLS).filter(|&i| p.clues[i] == 0) {
                        cell_ok += (o[i] == p.solution[i]) as usize;
                        cell_n += 1;
                    }
                }
                total += ch.len();
            }
            accs.push(100.0 * ok as f64 / set.len() as f64);
            cells.push(100.0 * cell_ok as f64 / cell_n as f64);
        }
        Ok((name.to_string(), nfe, accs, cells, 1e3 * t0.elapsed().as_secs_f64() / total as f64))
    };
    results.push(eval("NCM (ours)", ncm.k(), &mut |p, b| Ok(ncm.generate(&vq, b, Some(p), 0.0, &mut StdRng::seed_from_u64(0), false)?.0))?);

    if !c.ncm_only {
        let mk = |kind| SeqConfig {
            kind,
            vocab: sudoku::VOCAB,
            prompt_vocab: sudoku::VOCAB,
            prefix: sudoku::CELLS,
            len: sudoku::CELLS,
            hidden: c.hidden,
            depth: c.depth,
            heads: c.heads,
            dropout: 0.1,
        };
        for kind in [Kind::Mdlm, Kind::Ar] {
            let name = if kind == Kind::Ar { "ar" } else { "mdlm" };
            header(&format!("Baseline: {name}"));
            let m = SeqModel::new(mk(kind), &dev)?;
            println!("{name} params={}", param_count(&m.varmap));
            let mut tr = Trainer::new(&m.varmap, c.lr, 0.1, 0.95, steps / 20 + 1, steps)?;
            train_loop(name, steps, c.log_every, |_| {
                let (x, p) = batch(&mut rng)?;
                m.train_step(&x, Some(&p), &mut tr, &mut rng)
            })?;
            if kind == Kind::Ar {
                results.push(eval("AR (greedy)", sudoku::CELLS, &mut |p, b| m.generate(b, Some(p), 0, 0.0, &mut rng))?);
            } else {
                for t in [ncm.k(), 16, 32] {
                    results.push(eval(&format!("MDLM T={t}"), t, &mut |p, b| m.generate(b, Some(p), t, 0.0, &mut rng))?);
                }
            }
        }
    }

    header("Results: Sudoku — solved boards % (blank-cell accuracy %)");
    println!("{:<14} {:>5} {:>15} {:>15} {:>15} {:>11}", "method", "NFE", "easy (40)", "medium (35)", "hard (30)", "ms/puzzle");
    for (n, nfe, a, cell, ms) in &results {
        let f = |i: usize| format!("{:.1} ({:.1})", a[i], cell[i]);
        println!("{:<14} {:>5} {:>15} {:>15} {:>15} {:>11.2}", n, nfe, f(0), f(1), f(2), ms);
    }

    header("NCM trajectory on a hard puzzle (decoded after each level)");
    let p = &tests[2].2[0];
    println!("puzzle ({} clues):\n{}", tests[2].1, sudoku::render(&p.clues));
    let (out, traj) = ncm.generate(&vq, 1, Some(&batch_tensor(std::slice::from_ref(&p.clues), &dev)?), 0.0, &mut rng, true)?;
    for (k, b) in traj.iter().enumerate() {
        let wrong = b[0].iter().zip(&p.solution).filter(|(a, s)| a != s).count();
        println!("level {:2} (ℓ={:2}): {wrong:2} cells differ from the unique solution", k + 1, vq.cfg.ladder[k]);
    }
    println!("final ({}):\n{}", if sudoku::is_solved(&p.clues, &out[0]) { "solved" } else { "not solved" }, sudoku::render(&out[0]));
    Ok(())
}

