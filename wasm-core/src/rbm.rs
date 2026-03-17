use wasm_bindgen::prelude::*;

use crate::common::{
    clamp01, fit_f32_pattern, log1p_exp, mean_absolute_difference, sigmoid, XorShift64, EPSILON,
};

#[wasm_bindgen]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RbmVisibleModelKind {
    Bernoulli = 0,
    Gaussian = 1,
}

#[wasm_bindgen]
pub struct RbmCore {
    visible_units: usize,
    hidden_units: usize,
    visible_model: RbmVisibleModelKind,
    sample_count: usize,
    reference_count: usize,
    training_samples: Vec<f32>,
    reference_patterns: Vec<f32>,
    weights: Vec<f32>,
    visible_bias: Vec<f32>,
    hidden_bias: Vec<f32>,
    weight_velocity: Vec<f32>,
    visible_bias_velocity: Vec<f32>,
    hidden_bias_velocity: Vec<f32>,
    order: Vec<usize>,
    rng: XorShift64,
    current_visible: Vec<f32>,
    current_reconstruction: Vec<f32>,
    current_hidden_probabilities: Vec<f32>,
    current_hidden_state: Vec<u8>,
    current_free_energy: f32,
    current_reconstruction_error: f32,
    current_matched_pattern_index: i32,
    current_converged: bool,
    epoch: u32,
    step: u32,
}

#[wasm_bindgen]
impl RbmCore {
    #[wasm_bindgen(constructor)]
    pub fn new(
        visible_units: usize,
        hidden_units: usize,
        visible_model: RbmVisibleModelKind,
        training_samples: Vec<f32>,
        sample_count: usize,
        reference_patterns: Vec<f32>,
        reference_count: usize,
    ) -> Self {
        let mut rng = XorShift64::new(0xA0761D6478BD642F);
        let mut weights = vec![0.0; visible_units * hidden_units];
        for value in &mut weights {
            *value = rng.next_signed_f32() * 0.08;
        }
        let mut core = Self {
            visible_units,
            hidden_units,
            visible_model,
            sample_count,
            reference_count,
            training_samples,
            reference_patterns,
            weights,
            visible_bias: vec![0.0; visible_units],
            hidden_bias: vec![0.0; hidden_units],
            weight_velocity: vec![0.0; visible_units * hidden_units],
            visible_bias_velocity: vec![0.0; visible_units],
            hidden_bias_velocity: vec![0.0; hidden_units],
            order: (0..sample_count).collect(),
            rng,
            current_visible: vec![0.0; visible_units],
            current_reconstruction: vec![0.0; visible_units],
            current_hidden_probabilities: vec![0.0; hidden_units],
            current_hidden_state: vec![0; hidden_units],
            current_free_energy: 0.0,
            current_reconstruction_error: 0.0,
            current_matched_pattern_index: -1,
            current_converged: false,
            epoch: 0,
            step: 0,
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
        let hidden_probabilities = self.compute_hidden_probabilities(&self.current_visible);
        let hidden_state = self.sample_hidden_state(&hidden_probabilities);
        let reconstruction = self.reconstruct_visible_from_probabilities(&hidden_probabilities);

        self.current_hidden_probabilities = hidden_probabilities;
        self.current_hidden_state = hidden_state;
        self.current_reconstruction = reconstruction;
        self.current_free_energy = self.compute_free_energy(&self.current_visible);
        self.current_reconstruction_error =
            mean_absolute_difference(&self.current_visible, &self.current_reconstruction);
        self.current_matched_pattern_index =
            self.find_closest_pattern(&self.current_reconstruction);
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
        cd_steps: usize,
        momentum: f32,
        weight_decay: f32,
    ) -> Vec<f32> {
        let batch_size = batch_size.max(1).min(self.sample_count.max(1));
        let cd_steps = cd_steps.max(1);
        shuffle_order(&mut self.order, &mut self.rng);

        let mut weight_gradient = vec![0.0; self.weights.len()];
        let mut visible_bias_gradient = vec![0.0; self.visible_units];
        let mut hidden_bias_gradient = vec![0.0; self.hidden_units];
        let mut contrastive_gap = 0.0;

        for batch_start in (0..self.sample_count).step_by(batch_size) {
            weight_gradient.fill(0.0);
            visible_bias_gradient.fill(0.0);
            hidden_bias_gradient.fill(0.0);

            let batch_end = (batch_start + batch_size).min(self.sample_count);
            let batch_count = batch_end - batch_start;

            for batch_index in batch_start..batch_end {
                let visible = self.sample_at(self.order[batch_index]);
                let positive_hidden = self.compute_hidden_probabilities(&visible);
                let mut negative_visible = visible.clone();
                let mut negative_hidden = positive_hidden.clone();

                for _ in 0..cd_steps {
                    let hidden_sample = self.sample_hidden_state(&negative_hidden);
                    negative_visible = self.reconstruct_visible_from_state(&hidden_sample);
                    negative_hidden = self.compute_hidden_probabilities(&negative_visible);
                }

                contrastive_gap += mean_absolute_difference(&visible, &negative_visible);

                for visible_index in 0..self.visible_units {
                    visible_bias_gradient[visible_index] +=
                        visible[visible_index] - negative_visible[visible_index];
                }

                for hidden_index in 0..self.hidden_units {
                    hidden_bias_gradient[hidden_index] +=
                        positive_hidden[hidden_index] - negative_hidden[hidden_index];
                    let row_offset = hidden_index * self.visible_units;
                    for visible_index in 0..self.visible_units {
                        weight_gradient[row_offset + visible_index] += visible[visible_index]
                            * positive_hidden[hidden_index]
                            - negative_visible[visible_index] * negative_hidden[hidden_index];
                    }
                }
            }

            let step_scale = learning_rate / batch_count.max(1) as f32;
            for visible_index in 0..self.visible_units {
                self.visible_bias_velocity[visible_index] =
                    momentum * self.visible_bias_velocity[visible_index]
                        + step_scale * visible_bias_gradient[visible_index];
                self.visible_bias[visible_index] += self.visible_bias_velocity[visible_index];
            }
            for hidden_index in 0..self.hidden_units {
                self.hidden_bias_velocity[hidden_index] =
                    momentum * self.hidden_bias_velocity[hidden_index]
                        + step_scale * hidden_bias_gradient[hidden_index];
                self.hidden_bias[hidden_index] += self.hidden_bias_velocity[hidden_index];
                let row_offset = hidden_index * self.visible_units;
                for visible_index in 0..self.visible_units {
                    let offset = row_offset + visible_index;
                    let decayed_gradient = weight_gradient[offset] - weight_decay * self.weights[offset];
                    self.weight_velocity[offset] =
                        momentum * self.weight_velocity[offset] + step_scale * decayed_gradient;
                    self.weights[offset] += self.weight_velocity[offset];
                }
            }
        }

        let mut reconstruction_error = 0.0;
        let mut free_energy = 0.0;
        let mut hidden_activation = 0.0;
        for sample_index in 0..self.sample_count {
            let sample = self.sample_at(sample_index);
            let hidden_probabilities = self.compute_hidden_probabilities(&sample);
            let reconstruction = self.reconstruct_visible_from_probabilities(&hidden_probabilities);
            reconstruction_error += mean_absolute_difference(&sample, &reconstruction);
            free_energy += self.compute_free_energy(&sample);
            hidden_activation += hidden_probabilities.iter().sum::<f32>() / self.hidden_units.max(1) as f32;
        }

        let weight_mean_abs =
            self.weights.iter().map(|value| value.abs()).sum::<f32>() / self.weights.len().max(1) as f32;
        self.epoch += 1;
        self.inspect();

        vec![
            self.epoch as f32,
            reconstruction_error / self.sample_count.max(1) as f32,
            contrastive_gap / self.sample_count.max(1) as f32,
            free_energy / self.sample_count.max(1) as f32,
            hidden_activation / self.sample_count.max(1) as f32,
            weight_mean_abs,
        ]
    }

    pub fn epoch(&self) -> u32 {
        self.epoch
    }

    pub fn step_index(&self) -> u32 {
        self.step
    }

    pub fn weights(&self) -> Vec<f32> {
        self.weights.clone()
    }

    pub fn visible_bias(&self) -> Vec<f32> {
        self.visible_bias.clone()
    }

    pub fn hidden_bias(&self) -> Vec<f32> {
        self.hidden_bias.clone()
    }

    pub fn visible(&self) -> Vec<f32> {
        self.current_visible.clone()
    }

    pub fn reconstruction(&self) -> Vec<f32> {
        self.current_reconstruction.clone()
    }

    pub fn hidden_probabilities(&self) -> Vec<f32> {
        self.current_hidden_probabilities.clone()
    }

    pub fn hidden_state(&self) -> Vec<u8> {
        self.current_hidden_state.clone()
    }

    pub fn free_energy(&self) -> f32 {
        self.current_free_energy
    }

    pub fn reconstruction_error(&self) -> f32 {
        self.current_reconstruction_error
    }

    pub fn matched_pattern_index(&self) -> i32 {
        self.current_matched_pattern_index
    }

    pub fn converged(&self) -> bool {
        self.current_converged
    }

    fn sample_at(&self, sample_index: usize) -> Vec<f32> {
        let start = sample_index * self.visible_units;
        let end = start + self.visible_units;
        self.training_samples[start..end].to_vec()
    }

    fn compute_hidden_probabilities(&self, visible: &[f32]) -> Vec<f32> {
        let mut probabilities = vec![0.0; self.hidden_units];
        for hidden_index in 0..self.hidden_units {
            let row_offset = hidden_index * self.visible_units;
            let mut activation = self.hidden_bias[hidden_index];
            for visible_index in 0..self.visible_units {
                activation += self.weights[row_offset + visible_index] * visible[visible_index];
            }
            probabilities[hidden_index] = sigmoid(activation);
        }
        probabilities
    }

    fn sample_hidden_state(&mut self, probabilities: &[f32]) -> Vec<u8> {
        probabilities
            .iter()
            .map(|probability| if self.rng.next_bool(*probability) { 1 } else { 0 })
            .collect()
    }

    fn reconstruct_visible_from_probabilities(&self, hidden_probabilities: &[f32]) -> Vec<f32> {
        let mut reconstruction = vec![0.0; self.visible_units];
        for visible_index in 0..self.visible_units {
            let mut activation = self.visible_bias[visible_index];
            for hidden_index in 0..self.hidden_units {
                activation +=
                    self.weights[hidden_index * self.visible_units + visible_index]
                        * hidden_probabilities[hidden_index];
            }
            reconstruction[visible_index] = if self.visible_model == RbmVisibleModelKind::Gaussian {
                clamp01(activation)
            } else {
                sigmoid(activation)
            };
        }
        reconstruction
    }

    fn reconstruct_visible_from_state(&self, hidden_state: &[u8]) -> Vec<f32> {
        let mut reconstruction = vec![0.0; self.visible_units];
        for visible_index in 0..self.visible_units {
            let mut activation = self.visible_bias[visible_index];
            for hidden_index in 0..self.hidden_units {
                activation +=
                    self.weights[hidden_index * self.visible_units + visible_index]
                        * hidden_state[hidden_index] as f32;
            }
            reconstruction[visible_index] = if self.visible_model == RbmVisibleModelKind::Gaussian {
                clamp01(activation)
            } else {
                sigmoid(activation)
            };
        }
        reconstruction
    }

    fn compute_free_energy(&self, visible: &[f32]) -> f32 {
        let mut energy = 0.0;
        if self.visible_model == RbmVisibleModelKind::Gaussian {
            for visible_index in 0..self.visible_units {
                let centered = visible[visible_index] - self.visible_bias[visible_index];
                energy += 0.5 * centered * centered;
            }
        } else {
            for visible_index in 0..self.visible_units {
                energy -= self.visible_bias[visible_index] * visible[visible_index];
            }
        }

        for hidden_index in 0..self.hidden_units {
            let row_offset = hidden_index * self.visible_units;
            let mut activation = self.hidden_bias[hidden_index];
            for visible_index in 0..self.visible_units {
                activation += self.weights[row_offset + visible_index] * visible[visible_index];
            }
            energy -= log1p_exp(activation);
        }
        energy
    }

    fn find_closest_pattern(&self, pattern: &[f32]) -> i32 {
        let mut best_index = -1;
        let mut best_distance = f32::INFINITY;
        for reference_index in 0..self.reference_count {
            let start = reference_index * self.visible_units;
            let end = start + self.visible_units;
            let distance = mean_absolute_difference(pattern, &self.reference_patterns[start..end]);
            if distance < best_distance {
                best_distance = distance;
                best_index = reference_index as i32;
            }
        }
        best_index
    }
}

fn shuffle_order(order: &mut [usize], rng: &mut XorShift64) {
    for index in (1..order.len()).rev() {
        let swap_index = rng.next_usize(index + 1);
        order.swap(index, swap_index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rbm_epoch_runs_and_keeps_finite_metrics() {
        let samples = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let mut core = RbmCore::new(
            4,
            3,
            RbmVisibleModelKind::Bernoulli,
            samples.clone(),
            2,
            samples,
            2,
        );
        let metrics = core.train_epoch(0.05, 1, 2, 0.72, 0.00015);
        assert_eq!(metrics.len(), 6);
        assert!(metrics.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn rbm_epoch_advances_and_preserves_tensor_shapes() {
        let samples = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let mut core = RbmCore::new(
            4,
            3,
            RbmVisibleModelKind::Bernoulli,
            samples.clone(),
            2,
            samples,
            2,
        );

        assert_eq!(core.weights().len(), 12);
        assert_eq!(core.visible_bias().len(), 4);
        assert_eq!(core.hidden_bias().len(), 3);

        let before_epoch = core.epoch();
        core.train_epoch(0.05, 1, 2, 0.72, 0.00015);
        assert_eq!(core.epoch(), before_epoch + 1);
    }

    #[test]
    fn rbm_gaussian_mode_keeps_finite_reconstructions() {
        let samples = vec![1.0, 0.2, 0.8, 0.0, 0.0, 0.9, 0.1, 1.0];
        let mut core = RbmCore::new(
            4,
            3,
            RbmVisibleModelKind::Gaussian,
            samples.clone(),
            2,
            samples,
            2,
        );
        core.set_query(vec![1.0, 0.2, 0.8, 0.0]);
        core.train_epoch(0.03, 1, 2, 0.72, 0.00015);
        core.step(1e-4);

        assert!(core.free_energy().is_finite());
        assert!(core.reconstruction_error().is_finite());
        assert!(core.reconstruction().iter().all(|value| value.is_finite() && *value >= 0.0 && *value <= 1.0));
    }
}
