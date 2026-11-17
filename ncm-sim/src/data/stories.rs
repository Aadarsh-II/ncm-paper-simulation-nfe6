//! A synthetic "tiny stories" language with long-range structure, standing in
//! for TinyStories / LM1B. Because the grammar is known, every generated
//! sample can be checked *exactly* for validity, alongside the paper's
//! generative-perplexity and unigram-entropy metrics.
//!
//! A story is
//!   once upon a time there was a ADJ KIND named NAME .
//!   then 2–4 body sentences, each one of
//!     (a) PRON liked to VERB in the PLACE .
//!     (b) one day NAME found a ADJ OBJ .           (at most once)
//!     (c) PRON was very EMO .
//!     (d) NAME played with the OBJ all day .       (only after (b), same OBJ)
//!   the end .
//! padded with <pad> to L tokens. Constraints that span the whole sequence:
//! NAME is repeated consistently, KIND and PRON agree with NAME's gender, and
//! (d) must refer to the object introduced in (b).

use rand::Rng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::collections::HashMap;

pub const LEN: usize = 48;

const GIRLS: &[&str] = &["lily", "mia", "sue", "anna", "zoe", "emma"];
const BOYS: &[&str] = &["tom", "ben", "max", "leo", "sam", "jack"];
const GIRL_KIND: &[&str] = &["girl", "princess"];
const BOY_KIND: &[&str] = &["boy", "prince"];
const ADJ: &[&str] = &["little", "happy", "big", "shiny", "small", "red", "kind", "old"];
const VERB: &[&str] = &["play", "run", "sing", "read", "dance", "swim"];
const PLACE: &[&str] = &["park", "garden", "forest", "house", "river", "school"];
const OBJ: &[&str] = &["ball", "box", "hat", "toy", "kite", "book", "cup", "key"];
const EMO: &[&str] = &["happy", "sad", "tired", "brave", "excited", "scared"];
const FIXED: &[&str] = &[
    "<pad>", "once", "upon", "a", "time", "there", "was", "named", ".", "she", "he", "liked", "to", "in",
    "the", "one", "day", "found", "very", "played", "with", "all", "end",
];

pub struct Vocab {
    pub words: Vec<String>,
    pub ids: HashMap<String, u32>,
}

impl Vocab {
    pub fn new() -> Self {
        let mut words: Vec<String> = Vec::new();
        for list in [FIXED, GIRLS, BOYS, GIRL_KIND, BOY_KIND, ADJ, VERB, PLACE, OBJ, EMO] {
            for w in list {
                if !words.iter().any(|x| x == w) {
                    words.push(w.to_string());
                }
            }
        }
        let ids = words.iter().enumerate().map(|(i, w)| (w.clone(), i as u32)).collect();
        Self { words, ids }
    }

    pub fn len(&self) -> usize {
        self.words.len()
    }

    pub fn encode(&self, words: &[&str]) -> Vec<u32> {
        let mut v: Vec<u32> = words.iter().map(|w| self.ids[*w]).collect();
        v.resize(LEN, 0);
        v
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|&i| self.words.get(i as usize).map(String::as_str).unwrap_or("?"))
            .filter(|w| *w != "<pad>")
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl Default for Vocab {
    fn default() -> Self {
        Self::new()
    }
}

pub fn sample_story(rng: &mut StdRng) -> Vec<&'static str> {
    let girl = rng.gen_bool(0.5);
    let name = *if girl { GIRLS } else { BOYS }.choose(rng).unwrap();
    let kind = *if girl { GIRL_KIND } else { BOY_KIND }.choose(rng).unwrap();
    let pron = if girl { "she" } else { "he" };
    let mut s = vec!["once", "upon", "a", "time", "there", "was", "a", ADJ.choose(rng).unwrap(), kind, "named", name, "."];
    let n_body = rng.gen_range(2..=4);
    let mut obj: Option<&str> = None;
    for _ in 0..n_body {
        let mut choices = vec!['a', 'c'];
        if obj.is_none() {
            choices.push('b');
            choices.push('b');
        } else {
            choices.push('d');
            choices.push('d');
        }
        match *choices.choose(rng).unwrap() {
            'a' => s.extend([pron, "liked", "to", VERB.choose(rng).unwrap(), "in", "the", PLACE.choose(rng).unwrap(), "."]),
            'b' => {
                let o = *OBJ.choose(rng).unwrap();
                obj = Some(o);
                s.extend(["one", "day", name, "found", "a", ADJ.choose(rng).unwrap(), o, "."]);
            }
            'c' => s.extend([pron, "was", "very", EMO.choose(rng).unwrap(), "."]),
            _ => s.extend([name, "played", "with", "the", obj.unwrap(), "all", "day", "."]),
        }
    }
    s.extend(["the", "end", "."]);
    debug_assert!(s.len() <= LEN);
    s
}

/// Exact validity check of a decoded sample against the grammar.
pub fn is_valid(words: &[&str]) -> bool {
    let mut w = words.iter().copied().filter(|&x| x != "<pad>").peekable();
    // Padding must only appear at the end.
    if let Some(first_pad) = words.iter().position(|&x| x == "<pad>")
        && words[first_pad..].iter().any(|&x| x != "<pad>")
    {
        return false;
    }
    let expect = |lit: &str, w: &mut std::iter::Peekable<_>| -> bool { w.next() == Some(lit) };
    for lit in ["once", "upon", "a", "time", "there", "was", "a"] {
        if !expect(lit, &mut w) {
            return false;
        }
    }
    let in_list = |x: Option<&str>, l: &[&str]| x.is_some_and(|x| l.contains(&x));
    if !in_list(w.next(), ADJ) {
        return false;
    }
    let kind = w.next();
    let girl = if in_list(kind, GIRL_KIND) {
        true
    } else if in_list(kind, BOY_KIND) {
        false
    } else {
        return false;
    };
    if !expect("named", &mut w) {
        return false;
    }
    let name = match w.next() {
        Some(n) if (girl && GIRLS.contains(&n)) || (!girl && BOYS.contains(&n)) => n,
        _ => return false,
    };
    if !expect(".", &mut w) {
        return false;
    }
    let pron = if girl { "she" } else { "he" };
    let mut obj: Option<&str> = None;
    let mut body = 0;
    loop {
        let Some(first) = w.next() else { return false };
        if first == "the" {
            return body >= 2 && body <= 4 && expect("end", &mut w) && expect(".", &mut w) && w.next().is_none();
        }
        body += 1;
        let ok = if first == pron {
            match w.next() {
                Some("liked") => {
                    expect("to", &mut w)
                        && in_list(w.next(), VERB)
                        && expect("in", &mut w)
                        && expect("the", &mut w)
                        && in_list(w.next(), PLACE)
                }
                Some("was") => expect("very", &mut w) && in_list(w.next(), EMO),
                _ => false,
            }
        } else if first == "one" {
            let ok = obj.is_none()
                && expect("day", &mut w)
                && expect(name, &mut w)
                && expect("found", &mut w)
                && expect("a", &mut w)
                && in_list(w.next(), ADJ);
            let o = w.next();
            obj = o;
            ok && in_list(o, OBJ)
        } else if first == name {
            obj.is_some()
                && expect("played", &mut w)
                && expect("with", &mut w)
                && expect("the", &mut w)
                && w.next() == obj
                && expect("all", &mut w)
                && expect("day", &mut w)
        } else {
            false
        };
        if !ok || !expect(".", &mut w) {
            return false;
        }
    }
}

pub fn dataset(n: usize, vocab: &Vocab, rng: &mut StdRng) -> Vec<Vec<u32>> {
    (0..n).map(|_| vocab.encode(&sample_story(rng))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn generated_stories_are_valid_and_corruptions_are_not() {
        let mut rng = StdRng::seed_from_u64(1);
        let vocab = Vocab::new();
        for _ in 0..500 {
            let s = sample_story(&mut rng);
            assert!(s.len() <= LEN);
            assert!(is_valid(&s), "{s:?}");
            let ids = vocab.encode(&s);
            let back: Vec<&str> = ids.iter().map(|&i| vocab.words[i as usize].as_str()).collect();
            assert!(is_valid(&back));
        }
        let mut s = sample_story(&mut rng);
        let i = s.iter().position(|w| *w == "named").unwrap() + 1;
        s[i] = if GIRLS.contains(&s[i]) { "tom" } else { "lily" };
        assert!(!is_valid(&s), "gender mismatch must be rejected");
    }
}
