use wasm_bindgen::prelude::*;

use crate::common::{dot_f32, fit_f32_pattern, EPSILON};

#[wasm_bindgen]
pub struct DenseHopfieldCore {
    visible_units: usize,
    memory_count: usize,
    beta: f32,
    memory_patterns: Vec<f32>,
    similarity_matrix: Vec<f32>,
    max_similarity_abs: f32,
    current_state: Vec<f32>,
    current_attention: Vec<f32>,
    current_energy: f32,
    current_entropy: f32,
    current_top_attention: f32,
    current_matched_pattern_index: i32,
    current_delta: f32,
    current_converged: bool,
    step: u32,
}

#[wasm_bindgen]
impl DenseHopfieldCore {
    #[wasm_bindgen(constructor)]
    pub fn new(memory_patterns: Vec<f32>, memory_count: usize, visible_units: usize, beta: f32) -> Self {
        let expected = memory_count * visible_units;
        let mut flat = memory_patterns;
        flat.resize(expected, 0.0);
        flat.truncate(expected);

        let mut similarity_matrix = vec![0.0; memory_count * memory_count];
        let mut max_similarity_abs: f32 = 1.0;
        for row in 0..memory_count {
            let row_slice = &flat[row * visible_units..(row + 1) * visible_units];
            for col in 0..memory_count {
                let col_slice = &flat[col * visible_units..(col + 1) * visible_units];
                let similarity = dot_centered(row_slice, col_slice) / visible_units.max(1) as f32;
                similarity_matrix[row * memory_count + col] = similarity;
                max_similarity_abs = max_similarity_abs.max(similarity.abs());
            }
        }

        let mut core = Self {
            visible_units,
            memory_count,
            beta: beta.max(EPSILON),
            memory_patterns: flat,
            similarity_matrix,
            max_similarity_abs,
            current_state: vec![0.0; visible_units],
            current_attention: vec![0.0; memory_count],
            current_energy: 0.0,
            current_entropy: 0.0,
            current_top_attention: 0.0,
            current_matched_pattern_index: -1,
            current_delta: 0.0,
            current_converged: false,
            step: 0,
        };
        core.inspect();
        core
    }

    pub fn set_beta(&mut self, beta: f32) {
        self.beta = beta.max(EPSILON);
        self.inspect();
    }

    pub fn set_state(&mut self, state: Vec<f32>) {
        self.current_state = fit_f32_pattern(state, self.visible_units);
        self.step = 0;
        self.current_delta = 0.0;
        self.current_converged = false;
        self.inspect();
    }

    pub fn inspect(&mut self) {
        let centered = to_centered_slice(&self.current_state);
        let mut scores = vec![0.0; self.memory_count];
        for memory_index in 0..self.memory_count {
            let pattern =
                &self.memory_patterns[memory_index * self.visible_units..(memory_index + 1) * self.visible_units];
            scores[memory_index] = self.beta * dot_centered(pattern, &centered) / self.visible_units.max(1) as f32;
        }

        self.current_attention = softmax(&scores);
        self.current_energy = compute_energy(&centered, &scores, self.beta, self.visible_units);
        self.current_entropy = 0.0;
        self.current_top_attention = 0.0;
        self.current_matched_pattern_index = -1;
        for (index, weight) in self.current_attention.iter().enumerate() {
            if *weight > self.current_top_attention {
                self.current_top_attention = *weight;
                self.current_matched_pattern_index = index as i32;
            }
            if *weight > 0.0 {
                self.current_entropy -= *weight * weight.ln();
            }
        }
    }

    pub fn step(&mut self, tolerance: f32) {
        let previous = self.current_state.clone();
        let mut next = vec![0.0; self.visible_units];
        for memory_index in 0..self.memory_count {
            let weight = self.current_attention[memory_index];
            let pattern =
                &self.memory_patterns[memory_index * self.visible_units..(memory_index + 1) * self.visible_units];
            for visible_index in 0..self.visible_units {
                next[visible_index] += weight * pattern[visible_index];
            }
        }
        self.current_delta = crate::common::mean_absolute_difference(&previous, &next);
        self.current_state = next;
        self.step += 1;
        self.inspect();
        self.current_converged = self.current_delta <= tolerance.max(EPSILON);
    }

    pub fn visible_units(&self) -> usize {
        self.visible_units
    }

    pub fn memory_count(&self) -> usize {
        self.memory_count
    }

    pub fn state(&self) -> Vec<f32> {
        self.current_state.clone()
    }

    pub fn attention(&self) -> Vec<f32> {
        self.current_attention.clone()
    }

    pub fn energy(&self) -> f32 {
        self.current_energy
    }

    pub fn entropy(&self) -> f32 {
        self.current_entropy
    }

    pub fn top_attention(&self) -> f32 {
        self.current_top_attention
    }

    pub fn matched_pattern_index(&self) -> i32 {
        self.current_matched_pattern_index
    }

    pub fn delta(&self) -> f32 {
        self.current_delta
    }

    pub fn converged(&self) -> bool {
        self.current_converged
    }

    pub fn step_index(&self) -> u32 {
        self.step
    }

    pub fn similarity_matrix(&self) -> Vec<f32> {
        self.similarity_matrix.clone()
    }

    pub fn max_similarity_abs(&self) -> f32 {
        self.max_similarity_abs
    }
}

fn to_centered_slice(state: &[f32]) -> Vec<f32> {
    state.iter().map(|value| value * 2.0 - 1.0).collect()
}

fn dot_centered(pattern: &[f32], centered_state: &[f32]) -> f32 {
    pattern
        .iter()
        .zip(centered_state.iter())
        .map(|(left, right)| (left * 2.0 - 1.0) * right)
        .sum()
}

fn softmax(scores: &[f32]) -> Vec<f32> {
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut values = vec![0.0; scores.len()];
    let mut total = 0.0;
    for (index, score) in scores.iter().enumerate() {
        let value = (*score - max_score).exp();
        values[index] = value;
        total += value;
    }
    if total > EPSILON {
        for value in &mut values {
            *value /= total;
        }
    }
    values
}

fn compute_energy(centered_state: &[f32], scores: &[f32], beta: f32, visible_units: usize) -> f32 {
    let norm = dot_f32(centered_state, centered_state) / (2.0 * visible_units.max(1) as f32);
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total = scores.iter().map(|score| (*score - max_score).exp()).sum::<f32>();
    norm - (max_score + total.max(EPSILON).ln()) / beta.max(EPSILON)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_hopfield_retrieval_exposes_attention() {
        let memories = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let mut core = DenseHopfieldCore::new(memories, 2, 4, 8.0);
        core.set_state(vec![1.0, 0.0, 1.0, 0.0]);
        assert_eq!(core.attention().len(), 2);
        core.step(1e-4);
        assert!(core.energy().is_finite());
    }

    #[test]
    fn dense_hopfield_attention_is_normalized() {
        let memories = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let mut core = DenseHopfieldCore::new(memories, 2, 4, 8.0);
        core.set_state(vec![1.0, 0.0, 1.0, 0.0]);
        let sum: f32 = core.attention().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "attention must sum to 1, got {sum}");
    }

    #[test]
    fn dense_hopfield_higher_beta_sharpens_attention() {
        let memories = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let query = vec![1.0, 0.0, 0.8, 0.0];

        let mut soft_core = DenseHopfieldCore::new(memories.clone(), 2, 4, 2.0);
        soft_core.set_state(query.clone());

        let mut sharp_core = DenseHopfieldCore::new(memories, 2, 4, 16.0);
        sharp_core.set_state(query);

        assert!(
            sharp_core.top_attention() >= soft_core.top_attention(),
            "expected sharper beta to increase dominant attention: {} < {}",
            sharp_core.top_attention(),
            soft_core.top_attention()
        );
    }
}
