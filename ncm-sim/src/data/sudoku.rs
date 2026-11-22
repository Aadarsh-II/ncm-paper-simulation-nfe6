//! Sudoku data: random complete grids, puzzles with a unique solution and a
//! target number of clues (40 / 35 / 30 = easy / medium / hard, as in the
//! paper's S-FLM setup), and solution checking.
//!
//! Token ids: 0 = blank (prompt only), 1..=9 = digits. Boards are row-major, 81 cells.

use rand::Rng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

pub const CELLS: usize = 81;
pub const VOCAB: usize = 10;

#[derive(Clone, Debug)]
pub struct Puzzle {
    pub clues: Vec<u32>,
    pub solution: Vec<u32>,
}

fn box_of(i: usize) -> usize {
    (i / 27) * 3 + (i % 9) / 3
}

/// Bitmask DFS solver. Counts solutions up to `limit`; stores the first one.
struct Solver {
    rows: [u16; 9],
    cols: [u16; 9],
    boxes: [u16; 9],
    grid: [u8; CELLS],
}

impl Solver {
    fn new(grid: &[u32]) -> Option<Self> {
        let mut s = Solver { rows: [0; 9], cols: [0; 9], boxes: [0; 9], grid: [0; CELLS] };
        for (i, &v) in grid.iter().enumerate() {
            if v != 0 {
                let bit = 1u16 << v;
                let (r, c, b) = (i / 9, i % 9, box_of(i));
                if s.rows[r] & bit != 0 || s.cols[c] & bit != 0 || s.boxes[b] & bit != 0 {
                    return None;
                }
                s.rows[r] |= bit;
                s.cols[c] |= bit;
                s.boxes[b] |= bit;
                s.grid[i] = v as u8;
            }
        }
        Some(s)
    }

    fn candidates(&self, i: usize) -> u16 {
        !(self.rows[i / 9] | self.cols[i % 9] | self.boxes[box_of(i)]) & 0b11_1111_1110
    }

    fn count(&mut self, limit: usize, first: &mut Option<[u8; CELLS]>, rng: Option<&mut StdRng>) -> usize {
        // Most-constrained empty cell.
        let mut best = None;
        let mut best_n = 10;
        for i in 0..CELLS {
            if self.grid[i] == 0 {
                let n = self.candidates(i).count_ones();
                if n < best_n {
                    best_n = n;
                    best = Some(i);
                    if n <= 1 {
                        break;
                    }
                }
            }
        }
        let Some(i) = best else {
            if first.is_none() {
                *first = Some(self.grid);
            }
            return 1;
        };
        let cand = self.candidates(i);
        let mut digits: Vec<u8> = (1..=9).filter(|d| cand & (1 << d) != 0).collect();
        let mut rng = rng;
        if let Some(r) = rng.as_deref_mut() {
            digits.shuffle(r);
        }
        let (r, c, b) = (i / 9, i % 9, box_of(i));
        let mut total = 0;
        for d in digits {
            let bit = 1u16 << d;
            self.rows[r] |= bit;
            self.cols[c] |= bit;
            self.boxes[b] |= bit;
            self.grid[i] = d;
            total += self.count(limit - total, first, rng.as_deref_mut());
            self.rows[r] &= !bit;
            self.cols[c] &= !bit;
            self.boxes[b] &= !bit;
            self.grid[i] = 0;
            if total >= limit {
                break;
            }
        }
        total
    }
}

/// Number of solutions of `grid` (capped at `limit`).
pub fn count_solutions(grid: &[u32], limit: usize) -> usize {
    match Solver::new(grid) {
        Some(mut s) => s.count(limit, &mut None, None),
        None => 0,
    }
}

/// A uniformly shuffled complete valid grid.
pub fn random_grid(rng: &mut StdRng) -> Vec<u32> {
    let mut s = Solver::new(&[0; CELLS]).unwrap();
    let mut first = None;
    s.count(1, &mut first, Some(rng));
    first.unwrap().iter().map(|&v| v as u32).collect()
}

/// Remove clues in random order while the solution stays unique, until
/// `clues` remain (or no further cell can be removed).
pub fn make_puzzle(rng: &mut StdRng, clues: usize) -> Puzzle {
    loop {
        let solution = random_grid(rng);
        let mut grid = solution.clone();
        let mut order: Vec<usize> = (0..CELLS).collect();
        order.shuffle(rng);
        let mut filled = CELLS;
        for i in order {
            if filled == clues {
                break;
            }
            let v = grid[i];
            grid[i] = 0;
            if count_solutions(&grid, 2) == 1 {
                filled -= 1;
            } else {
                grid[i] = v;
            }
        }
        if filled == clues {
            return Puzzle { clues: grid, solution };
        }
    }
}

/// Generate `n` puzzles in parallel with clue counts drawn from `clue_range`.
pub fn dataset(n: usize, clue_range: (usize, usize), seed: u64, threads: usize) -> Vec<Puzzle> {
    use rand::SeedableRng;
    let threads = threads.max(1);
    let per = n.div_ceil(threads);
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            std::thread::spawn(move || {
                let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(1_000_003).wrapping_add(t as u64));
                (0..per)
                    .map(|_| {
                        let c = rng.gen_range(clue_range.0..=clue_range.1);
                        make_puzzle(&mut rng, c)
                    })
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    let mut out: Vec<Puzzle> = handles.into_iter().flat_map(|h| h.join().unwrap()).collect();
    out.truncate(n);
    out
}

/// A board solves a puzzle if it is a valid complete grid that keeps every clue.
pub fn is_solved(clues: &[u32], board: &[u32]) -> bool {
    if board.len() != CELLS || board.iter().any(|&v| !(1..=9).contains(&v)) {
        return false;
    }
    if clues.iter().zip(board).any(|(&c, &b)| c != 0 && c != b) {
        return false;
    }
    Solver::new(board).is_some()
}

/// Pretty-print a board (0 shown as '.').
pub fn render(board: &[u32]) -> String {
    let mut s = String::new();
    for r in 0..9 {
        if r % 3 == 0 && r > 0 {
            s.push_str("------+-------+------\n");
        }
        for c in 0..9 {
            if c % 3 == 0 && c > 0 {
                s.push_str("| ");
            }
            let v = board[r * 9 + c];
            s.push(if v == 0 { '.' } else { char::from_digit(v, 10).unwrap_or('?') });
            s.push(' ');
        }
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn puzzles_are_unique_and_solution_checks() {
        let mut rng = StdRng::seed_from_u64(7);
        for clues in [40, 35, 30] {
            let p = make_puzzle(&mut rng, clues);
            assert_eq!(p.clues.iter().filter(|&&v| v != 0).count(), clues);
            assert_eq!(count_solutions(&p.clues, 2), 1);
            assert!(is_solved(&p.clues, &p.solution));
            let mut bad = p.solution.clone();
            bad.swap(0, 1);
            assert!(!is_solved(&p.clues, &bad) || p.solution[0] == p.solution[1]);
        }
    }
}
