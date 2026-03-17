use wasm_bindgen::prelude::*;

use crate::common::{fit_f32_pattern, mean_absolute_difference, XorShift64, EPSILON};

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum DenseAssociativeActivationKind {
    ReluPower = 0,
    SignedPower = 1,
    Softmax = 2,
}

#[wasm_bindgen]
pub struct DenseAssociativeMemoryCore {
    visible_units: usize,
    hidden_units: usize,
    activation_kind: DenseAssociativeActivationKind,
    sharpness: u32,
    sample_count: usize,
    reference_count: usize,
    training_samples: Vec<f32>,
    reference_patterns: Vec<f32>,
    weights: Vec<f32>,
    weight_velocity: Vec<f32>,
    current_visible: Vec<f32>,
    current_reconstruction: Vec<f32>,
    current_hidden_activations: Vec<f32>,
    current_hidden_scores: Vec<f32>,
    current_energy: f32,
    current_reconstruction_error: f32,
    current_top_hidden_index: i32,
    current_top_hidden_activation: f32,
    current_hidden_entropy: f32,
    current_matched_pattern_index: i32,
    current_converged: bool,
    step: u32,
    epoch: u32,
}

#[wasm_bindgen]
impl DenseAssociativeMemoryCore {
    #[wasm_bindgen(constructor)]
    pub fn new(
        visible_units: usize,
        hidden_units: usize,
        activation_kind: DenseAssociativeActivationKind,
        sharpness: u32,
        training_samples: Vec<f32>,
        sample_count: usize,
        reference_patterns: Vec<f32>,
        reference_count: usize,
    ) -> Self {
        let mut rng = XorShift64::new(0x9E3779B97F4A7C15);
        let mut weights = vec![0.0; visible_units * hidden_units];
        for hidden_index in 0..hidden_units {
            let sample_index = if sample_count > 0 {
                let start = hidden_index * sample_count / hidden_units.max(1);
                let end = ((hidden_index + 1) * sample_count / hidden_units.max(1)).max(start + 1);
                let span = end.saturating_sub(start).max(1);
                start + rng.next_usize(span)
            } else {
                0
            };
            let sample_offset = sample_index * visible_units;
            let row_offset = hidden_index * visible_units;
            for visible_index in 0..visible_units {
                let seed = if sample_count > 0 {
                    to_centered(training_samples[sample_offset + visible_index])
                } else {
                    0.0
                };
                let noise = rng.next_signed_f32() * 0.08;
                weights[row_offset + visible_index] = (seed + noise).clamp(-1.0, 1.0);
            }
        }
        normalize_rows(&mut weights, visible_units, hidden_units);

        let mut core = Self {
            visible_units,
            hidden_units,
            activation_kind,
            sharpness: sharpness.max(1),
            sample_count,
            reference_count,
            training_samples,
            reference_patterns,
            weights,
            weight_velocity: vec![0.0; visible_units * hidden_units],
            current_visible: vec![0.0; visible_units],
            current_reconstruction: vec![0.0; visible_units],
            current_hidden_activations: vec![0.0; hidden_units],
            current_hidden_scores: vec![0.0; hidden_units],
            current_energy: 0.0,
            current_reconstruction_error: 0.0,
            current_top_hidden_index: -1,
            current_top_hidden_activation: 0.0,
            current_hidden_entropy: 0.0,
            current_matched_pattern_index: -1,
            current_converged: false,
            step: 0,
            epoch: 0,
        };
        core.inspect();
        core
    }

    pub fn set_query(&mut self, visible: Vec<f32>) {
        self.current_visible = fit_f32_pattern(visible, self.visible_units);
        self.step = 0;
        self.current_converged = false;
        self.inspect();
    }

    pub fn inspect(&mut self) {
        let centered_visible = centered_pattern(&self.current_visible);
        let scores = compute_hidden_scores(
            &centered_visible,
            &self.weights,
            self.visible_units,
            self.hidden_units,
        );
        let activations =
            compute_hidden_activations(&scores, self.activation_kind, self.sharpness);
        let reconstruction =
            reconstruct_visible(&activations, &self.weights, self.visible_units, self.hidden_units);

        self.current_energy =
            compute_energy(&centered_visible, &scores, self.activation_kind, self.sharpness);
        self.current_reconstruction_error =
            mean_absolute_difference(&self.current_visible, &reconstruction);
        self.current_top_hidden_index = top_hidden_index(&activations);
        self.current_top_hidden_activation = top_hidden_activation(&activations);
        self.current_hidden_entropy = hidden_entropy(&activations);
        self.current_matched_pattern_index = find_closest_pattern(
            &reconstruction,
            &self.reference_patterns,
            self.reference_count,
            self.visible_units,
        );
        self.current_hidden_scores = scores;
        self.current_hidden_activations = activations;
        self.current_reconstruction = reconstruction;
    }

    pub fn step(&mut self, tolerance: f32) {
        let previous = self.current_visible.clone();
        self.current_visible = self.current_reconstruction.clone();
        self.step += 1;
        self.inspect();
        self.current_converged =
            mean_absolute_difference(&previous, &self.current_visible) <= tolerance.max(EPSILON);
    }

    pub fn train_epoch(
        &mut self,
        learning_rate: f32,
        batch_size: usize,
        momentum: f32,
        weight_decay: f32,
    ) -> Vec<f32> {
        let batch_size = batch_size.max(1).min(self.sample_count.max(1));
        let mut order: Vec<usize> = (0..self.sample_count).collect();
        let mut rng = XorShift64::new(0xD1B54A32D192ED03 ^ self.epoch as u64);
        shuffle_order(&mut order, &mut rng);
        let anti_hebbian = 0.1f32;

        for batch_start in (0..self.sample_count).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(self.sample_count);
            let mut prototype_sums = vec![0.0; self.weights.len()];
            let mut anti_sums = vec![0.0; self.weights.len()];
            let mut assignment_counts = vec![0usize; self.hidden_units];
            let mut anti_counts = vec![0usize; self.hidden_units];

            for batch_index in batch_start..batch_end {
                let sample_index = order[batch_index];
                let visible = sample_at(&self.training_samples, sample_index, self.visible_units);
                let centered_visible = centered_pattern(&visible);
                let positive_scores = compute_hidden_scores(
                    &centered_visible,
                    &self.weights,
                    self.visible_units,
                    self.hidden_units,
                );
                let (winner_index, runner_up_index, _winner_value, runner_up_value) =
                    top_two_hidden(&positive_scores);

                if winner_index < self.hidden_units {
                    assignment_counts[winner_index] += 1;
                    let row_offset = winner_index * self.visible_units;
                    for visible_index in 0..self.visible_units {
                        prototype_sums[row_offset + visible_index] += centered_visible[visible_index];
                    }
                }

                if let Some(runner_index) = runner_up_index.filter(|_| runner_up_value > 0.0) {
                    anti_counts[runner_index] += 1;
                    let row_offset = runner_index * self.visible_units;
                    for visible_index in 0..self.visible_units {
                        anti_sums[row_offset + visible_index] += centered_visible[visible_index];
                    }
                }
            }

            for hidden_index in 0..self.hidden_units {
                let row_offset = hidden_index * self.visible_units;
                let winner_count = assignment_counts[hidden_index].max(1) as f32;
                let anti_count = anti_counts[hidden_index].max(1) as f32;
                let has_updates = assignment_counts[hidden_index] > 0 || anti_counts[hidden_index] > 0;
                if !has_updates {
                    continue;
                }
                for visible_index in 0..self.visible_units {
                    let winner_mean = prototype_sums[row_offset + visible_index] / winner_count;
                    let anti_mean = anti_sums[row_offset + visible_index] / anti_count;
                    let target = winner_mean - anti_hebbian * anti_mean;
                    let index = row_offset + visible_index;
                    let delta = target - self.weights[index];
                    let decayed = delta - weight_decay * self.weights[index];
                    self.weight_velocity[index] =
                        momentum * self.weight_velocity[index] + learning_rate * decayed;
                    self.weights[index] = (self.weights[index] + self.weight_velocity[index]).clamp(-1.0, 1.0);
                }
            }
            normalize_rows(&mut self.weights, self.visible_units, self.hidden_units);
        }

        let mut reconstruction_error = 0.0;
        let mut energy = 0.0;
        let mut hidden_activation = 0.0;
        let mut winner_share = 0.0;
        let mut winner_margin = 0.0;

        for sample_index in 0..self.sample_count {
            let visible = sample_at(&self.training_samples, sample_index, self.visible_units);
            let centered_visible = centered_pattern(&visible);
            let scores = compute_hidden_scores(
                &centered_visible,
                &self.weights,
                self.visible_units,
                self.hidden_units,
            );
            let hidden =
                compute_hidden_activations(&scores, self.activation_kind, self.sharpness);
            let reconstruction =
                reconstruct_visible(&hidden, &self.weights, self.visible_units, self.hidden_units);
            let (_, _, top_value, second_value) = top_two_hidden(&hidden);

            reconstruction_error += mean_absolute_difference(&visible, &reconstruction);
            energy += compute_energy(&centered_visible, &scores, self.activation_kind, self.sharpness);
            winner_margin += (top_value - second_value).max(0.0);

            let mut activation_mass = 0.0;
            let mut max_activation = 0.0;
            for value in &hidden {
                let magnitude = value.abs();
                activation_mass += magnitude;
                if magnitude > max_activation {
                    max_activation = magnitude;
                }
            }
            hidden_activation += activation_mass / self.hidden_units.max(1) as f32;
            winner_share += if activation_mass > EPSILON {
                max_activation / activation_mass
            } else {
                0.0
            };
        }

        let weight_mean_abs =
            self.weights.iter().map(|value| value.abs()).sum::<f32>() / self.weights.len().max(1) as f32;
        self.epoch += 1;
        self.inspect();

        vec![
            self.epoch as f32,
            reconstruction_error / self.sample_count.max(1) as f32,
            winner_margin / (2 * self.sample_count.max(1)) as f32,
            hidden_activation / self.sample_count.max(1) as f32,
            winner_share / self.sample_count.max(1) as f32,
            weight_mean_abs,
            energy / self.sample_count.max(1) as f32,
        ]
    }

    pub fn epoch(&self) -> u32 {
        self.epoch
    }

    pub fn step_index(&self) -> u32 {
        self.step
    }

    pub fn visible_units(&self) -> usize {
        self.visible_units
    }

    pub fn hidden_units(&self) -> usize {
        self.hidden_units
    }

    pub fn weights(&self) -> Vec<f32> {
        self.weights.clone()
    }

    pub fn visible(&self) -> Vec<f32> {
        self.current_visible.clone()
    }

    pub fn reconstruction(&self) -> Vec<f32> {
        self.current_reconstruction.clone()
    }

    pub fn hidden_activations(&self) -> Vec<f32> {
        self.current_hidden_activations.clone()
    }

    pub fn hidden_scores(&self) -> Vec<f32> {
        self.current_hidden_scores.clone()
    }

    pub fn energy(&self) -> f32 {
        self.current_energy
    }

    pub fn reconstruction_error(&self) -> f32 {
        self.current_reconstruction_error
    }

    pub fn top_hidden_index(&self) -> i32 {
        self.current_top_hidden_index
    }

    pub fn top_hidden_activation(&self) -> f32 {
        self.current_top_hidden_activation
    }

    pub fn hidden_entropy(&self) -> f32 {
        self.current_hidden_entropy
    }

    pub fn matched_pattern_index(&self) -> i32 {
        self.current_matched_pattern_index
    }

    pub fn converged(&self) -> bool {
        self.current_converged
    }
}

fn sample_at(samples: &[f32], sample_index: usize, visible_units: usize) -> Vec<f32> {
    let start = sample_index * visible_units;
    let end = start + visible_units;
    samples[start..end].to_vec()
}

fn to_centered(value: f32) -> f32 {
    value * 2.0 - 1.0
}

fn centered_pattern(values: &[f32]) -> Vec<f32> {
    values.iter().map(|value| to_centered(*value)).collect()
}

fn shuffle_order(order: &mut [usize], rng: &mut XorShift64) {
    for index in (1..order.len()).rev() {
        let swap_index = rng.next_usize(index + 1);
        order.swap(index, swap_index);
    }
}

fn normalize_rows(weights: &mut [f32], visible_units: usize, hidden_units: usize) {
    for hidden_index in 0..hidden_units {
        let row_offset = hidden_index * visible_units;
        let mut norm = 0.0;
        for visible_index in 0..visible_units {
            let value = weights[row_offset + visible_index];
            norm += value * value;
        }
        norm = norm.sqrt();
        if norm <= EPSILON {
            continue;
        }
        for visible_index in 0..visible_units {
            weights[row_offset + visible_index] /= norm;
        }
    }
}

fn top_two_hidden(values: &[f32]) -> (usize, Option<usize>, f32, f32) {
    let mut best_index = 0usize;
    let mut second_index: Option<usize> = None;
    let mut best_value = f32::NEG_INFINITY;
    let mut second_value = f32::NEG_INFINITY;

    for (index, value) in values.iter().enumerate() {
        let magnitude = value.abs();
        if magnitude > best_value {
            second_index = Some(best_index);
            second_value = best_value;
            best_index = index;
            best_value = magnitude;
        } else if magnitude > second_value {
            second_index = Some(index);
            second_value = magnitude;
        }
    }

    (
        best_index,
        second_index.filter(|index| *index != best_index),
        best_value.max(0.0),
        second_value.max(0.0),
    )
}

fn compute_hidden_scores(
    visible: &[f32],
    weights: &[f32],
    visible_units: usize,
    hidden_units: usize,
) -> Vec<f32> {
    let mut scores = vec![0.0; hidden_units];
    let visible_norm = visible.iter().map(|value| value * value).sum::<f32>().sqrt().max(EPSILON);
    for hidden_index in 0..hidden_units {
        let row_offset = hidden_index * visible_units;
        let mut total = 0.0;
        let mut row_norm = 0.0;
        for visible_index in 0..visible_units {
            let weight = weights[row_offset + visible_index];
            total += visible[visible_index] * weight;
            row_norm += weight * weight;
        }
        scores[hidden_index] = total / (visible_norm * row_norm.sqrt().max(EPSILON));
    }
    scores
}

fn compute_hidden_activations(
    scores: &[f32],
    activation_kind: DenseAssociativeActivationKind,
    sharpness: u32,
) -> Vec<f32> {
    match activation_kind {
        DenseAssociativeActivationKind::Softmax => {
            let beta = sharpness.max(1) as f32;
            let max_score = scores
                .iter()
                .map(|value| beta * value)
                .fold(f32::NEG_INFINITY, f32::max);
            let mut values: Vec<f32> = scores
                .iter()
                .map(|value| (beta * value - max_score).exp())
                .collect();
            let total: f32 = values.iter().sum();
            let scale = if total > EPSILON { 1.0 / total } else { 0.0 };
            for value in &mut values {
                *value *= scale;
            }
            values
        }
        DenseAssociativeActivationKind::ReluPower => {
            let exponent = sharpness.max(1) as i32;
            let mut values = vec![0.0; scores.len()];
            let (winner_index, _, winner_value, _) = top_two_hidden(scores);
            if winner_value > 0.0 {
                values[winner_index] = winner_value.powi(exponent);
            }
            values
        }
        DenseAssociativeActivationKind::SignedPower => {
            let exponent = sharpness.max(1) as i32;
            let mut values = vec![0.0; scores.len()];
            let (winner_index, _, winner_value, _) = top_two_hidden(scores);
            if winner_value > 0.0 {
                let score = scores[winner_index];
                values[winner_index] = score.signum() * score.abs().powi(exponent);
            }
            values
        }
    }
}

fn reconstruct_visible(
    hidden: &[f32],
    weights: &[f32],
    visible_units: usize,
    hidden_units: usize,
) -> Vec<f32> {
    let mut reconstruction = vec![0.0; visible_units];
    let mut activation_mass = 0.0;
    for hidden_index in 0..hidden_units {
        activation_mass += hidden[hidden_index].abs();
        let row_offset = hidden_index * visible_units;
        for visible_index in 0..visible_units {
            reconstruction[visible_index] += weights[row_offset + visible_index] * hidden[hidden_index];
        }
    }
    let scale = if activation_mass > EPSILON {
        1.0 / activation_mass
    } else {
        0.0
    };
    reconstruction
        .iter()
        .map(|value| ((value * scale).clamp(-1.0, 1.0) + 1.0) * 0.5)
        .collect()
}

fn compute_energy(
    visible: &[f32],
    scores: &[f32],
    activation_kind: DenseAssociativeActivationKind,
    sharpness: u32,
) -> f32 {
    let visible_norm = visible.iter().map(|value| value * value).sum::<f32>() / visible.len().max(1) as f32 * 0.5;
    visible_norm - compute_potential(scores, activation_kind, sharpness)
}

fn compute_potential(
    scores: &[f32],
    activation_kind: DenseAssociativeActivationKind,
    sharpness: u32,
) -> f32 {
    match activation_kind {
        DenseAssociativeActivationKind::Softmax => {
            let beta = sharpness.max(1) as f32;
            let max_score = scores
                .iter()
                .map(|value| beta * value)
                .fold(f32::NEG_INFINITY, f32::max);
            let total = scores
                .iter()
                .map(|value| (beta * value - max_score).exp())
                .sum::<f32>();
            (max_score + total.max(EPSILON).ln()) / beta
        }
        DenseAssociativeActivationKind::ReluPower => {
            let exponent = sharpness.max(1) as i32 + 1;
            scores.iter().map(|value| value.max(0.0).powi(exponent)).sum::<f32>() / exponent as f32
        }
        DenseAssociativeActivationKind::SignedPower => {
            let exponent = sharpness.max(1) as i32 + 1;
            scores.iter().map(|value| value.abs().powi(exponent)).sum::<f32>() / exponent as f32
        }
    }
}

fn top_hidden_index(values: &[f32]) -> i32 {
    let mut best_index = -1;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in values.iter().enumerate() {
        let magnitude = value.abs();
        if magnitude > best_value {
            best_value = magnitude;
            best_index = index as i32;
        }
    }
    best_index
}

fn top_hidden_activation(values: &[f32]) -> f32 {
    values.iter().map(|value| value.abs()).fold(0.0, f32::max)
}

fn hidden_entropy(values: &[f32]) -> f32 {
    let total: f32 = values.iter().map(|value| value.abs()).sum();
    if total <= EPSILON {
        return 0.0;
    }
    values
        .iter()
        .map(|value| value.abs() / total)
        .filter(|probability| *probability > 0.0)
        .map(|probability| -probability * probability.ln())
        .sum()
}

fn find_closest_pattern(
    pattern: &[f32],
    references: &[f32],
    reference_count: usize,
    visible_units: usize,
) -> i32 {
    let mut best_index = -1;
    let mut best_distance = f32::INFINITY;
    for reference_index in 0..reference_count {
        let start = reference_index * visible_units;
        let end = start + visible_units;
        let distance = mean_absolute_difference(pattern, &references[start..end]);
        if distance < best_distance {
            best_distance = distance;
            best_index = reference_index as i32;
        }
    }
    best_index
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_samples() -> Vec<f32> {
        vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    }

    #[test]
    fn dam_epoch_runs() {
        let mut core = DenseAssociativeMemoryCore::new(
            4,
            3,
            DenseAssociativeActivationKind::ReluPower,
            4,
            make_samples(),
            2,
            make_samples(),
            2,
        );
        let metrics = core.train_epoch(0.05, 1, 0.65, 0.0004);
        assert_eq!(metrics.len(), 7);
    }

    #[test]
    fn dam_higher_sharpness_increases_hidden_concentration() {
        let samples = make_samples();
        let query = vec![1.0, 0.0, 0.8, 0.0];

        let mut soft_core = DenseAssociativeMemoryCore::new(
            4,
            3,
            DenseAssociativeActivationKind::ReluPower,
            2,
            samples.clone(),
            2,
            samples.clone(),
            2,
        );
        soft_core.set_query(query.clone());

        let mut sharp_core = DenseAssociativeMemoryCore::new(
            4,
            3,
            DenseAssociativeActivationKind::ReluPower,
            12,
            samples.clone(),
            2,
            samples,
            2,
        );
        sharp_core.set_query(query);

        let soft_concentration = concentration(&soft_core.hidden_activations());
        let sharp_concentration = concentration(&sharp_core.hidden_activations());
        assert!(
            sharp_concentration >= soft_concentration,
            "expected higher sharpness to increase concentration: {} < {}",
            sharp_concentration,
            soft_concentration
        );
    }

    #[test]
    fn dam_softmax_activation_stays_normalized_and_finite() {
        let samples = make_samples();
        let mut core = DenseAssociativeMemoryCore::new(
            4,
            3,
            DenseAssociativeActivationKind::Softmax,
            6,
            samples.clone(),
            2,
            samples,
            2,
        );
        core.set_query(vec![1.0, 0.0, 1.0, 0.0]);
        let activations = core.hidden_activations();
        let sum: f32 = activations.iter().sum();
        assert!(activations.iter().all(|value| value.is_finite() && *value >= 0.0));
        assert!((sum - 1.0).abs() < 1e-5, "softmax activations must sum to 1, got {sum}");
    }

    #[test]
    fn dam_ordered_dataset_initialization_spreads_hidden_seeds() {
        let mut ordered_samples = Vec::new();
        for _ in 0..8 {
            ordered_samples.extend_from_slice(&[1.0, 1.0, 0.0, 0.0]);
        }
        for _ in 0..8 {
            ordered_samples.extend_from_slice(&[0.0, 0.0, 1.0, 1.0]);
        }

        let core = DenseAssociativeMemoryCore::new(
            4,
            8,
            DenseAssociativeActivationKind::ReluPower,
            4,
            ordered_samples.clone(),
            16,
            ordered_samples,
            16,
        );

        let weights = core.weights();
        let mut positive_left = 0usize;
        let mut positive_right = 0usize;
        for hidden_index in 0..8 {
            let row_offset = hidden_index * 4;
            let left = weights[row_offset] + weights[row_offset + 1];
            let right = weights[row_offset + 2] + weights[row_offset + 3];
            if left > right {
                positive_left += 1;
            } else {
                positive_right += 1;
            }
        }

        assert!(positive_left > 0, "expected some hidden slots seeded toward the first block");
        assert!(positive_right > 0, "expected some hidden slots seeded toward later ordered samples");
    }

    #[test]
    fn dam_training_and_sparse_retrieval_do_not_collapse_all_toy_patterns() {
        let samples = vec![
            1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0,
            1.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 1.0,
        ];

        let mut core = DenseAssociativeMemoryCore::new(
            4,
            6,
            DenseAssociativeActivationKind::ReluPower,
            8,
            samples.clone(),
            4,
            samples.clone(),
            4,
        );

        for _ in 0..12 {
            core.train_epoch(0.05, 2, 0.4, 0.0001);
        }

        let references = [
            vec![1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 1.0],
        ];

        let mut matched_patterns = std::collections::BTreeSet::new();
        let mut hidden_winners = std::collections::BTreeSet::new();

        for reference in references {
            core.set_query(reference);
            for _ in 0..4 {
                if core.converged() {
                    break;
                }
                core.step(1e-3);
            }
            matched_patterns.insert(core.matched_pattern_index());
            hidden_winners.insert(core.top_hidden_index());
        }

        assert!(
            matched_patterns.len() >= 2,
            "expected toy retrieval to preserve more than one attractor, got {matched_patterns:?}",
        );
        assert!(
            hidden_winners.len() >= 2,
            "expected toy retrieval to use more than one winning hidden slot, got {hidden_winners:?}",
        );
    }

    fn concentration(values: &[f32]) -> f32 {
        let total = values.iter().map(|value| value.abs()).sum::<f32>();
        let top = values
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f32::max);
        if total > EPSILON { top / total } else { 0.0 }
    }
}
