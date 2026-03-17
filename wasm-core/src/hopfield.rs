use wasm_bindgen::prelude::*;

use crate::common::{fit_i8_pattern, XorShift64, EPSILON};

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum HopfieldLearningRule {
    Hebbian = 0,
    Pseudoinverse = 1,
    Storkey = 2,
    KrauthMezard = 3,
    Unlearning = 4,
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum HopfieldConvergenceRule {
    AsyncRandom = 0,
    Synchronous = 1,
    Stochastic = 2,
}

#[wasm_bindgen]
pub struct HopfieldCore {
    size: usize,
    pattern_count: usize,
    patterns: Vec<i8>,
    weights: Vec<f32>,
    state: Vec<i8>,
    previous_state: Vec<i8>,
    scratch_indices: Vec<usize>,
    rng: XorShift64,
    step: u32,
    changed_count: u32,
    converged: bool,
    matched_pattern_index: i32,
    current_energy: f32,
}

#[wasm_bindgen]
impl HopfieldCore {
    #[wasm_bindgen(constructor)]
    pub fn new(size: usize, patterns: Vec<i8>, pattern_count: usize) -> Self {
        let expected = size * pattern_count;
        let mut flat = patterns;
        flat.resize(expected, -1);
        flat.truncate(expected);
        let state = vec![-1; size];
        let mut core = Self {
            size,
            pattern_count,
            patterns: flat,
            weights: vec![0.0; size * size],
            previous_state: state.clone(),
            state,
            scratch_indices: (0..size).collect(),
            rng: XorShift64::new(0x517CC1B727220A95),
            step: 0,
            changed_count: 0,
            converged: false,
            matched_pattern_index: -1,
            current_energy: 0.0,
        };
        core.refresh_metrics();
        core
    }

    pub fn train(
        &mut self,
        learning_rule: HopfieldLearningRule,
        kappa: f32,
        epsilon: f32,
        max_epochs: usize,
        unlearning_steps: usize,
    ) {
        self.weights.fill(0.0);
        match learning_rule {
            HopfieldLearningRule::Hebbian => self.train_hebbian(),
            HopfieldLearningRule::Pseudoinverse => self.train_pseudoinverse(),
            HopfieldLearningRule::Storkey => self.train_storkey(),
            HopfieldLearningRule::KrauthMezard => {
                self.train_krauth_mezard(kappa, epsilon.max(0.0), max_epochs.max(1))
            }
            HopfieldLearningRule::Unlearning => {
                self.train_unlearning(epsilon.max(0.0), unlearning_steps.max(1))
            }
        }
        self.zero_diagonal();
        self.refresh_metrics();
    }

    pub fn set_state(&mut self, state: Vec<i8>) {
        self.state = fit_i8_pattern(state, self.size);
        self.previous_state = self.state.clone();
        self.step = 0;
        self.changed_count = 0;
        self.refresh_metrics();
    }

    pub fn step(&mut self, convergence_rule: HopfieldConvergenceRule, temperature: f32) {
        self.previous_state = self.state.clone();
        let changed = match convergence_rule {
            HopfieldConvergenceRule::AsyncRandom => self.step_async_random(),
            HopfieldConvergenceRule::Synchronous => self.step_synchronous(),
            HopfieldConvergenceRule::Stochastic => self.step_stochastic(temperature.max(0.01)),
        };
        self.step += 1;
        self.changed_count = changed as u32;
        self.refresh_metrics();
    }

    pub fn weights(&self) -> Vec<f32> {
        self.weights.clone()
    }

    pub fn state(&self) -> Vec<i8> {
        self.state.clone()
    }

    pub fn energy(&self) -> f32 {
        self.current_energy
    }

    pub fn step_index(&self) -> u32 {
        self.step
    }

    pub fn changed_count(&self) -> u32 {
        self.changed_count
    }

    pub fn matched_pattern_index(&self) -> i32 {
        self.matched_pattern_index
    }

    pub fn converged(&self) -> bool {
        self.converged
    }

    pub fn max_weight_abs(&self) -> f32 {
        self.weights
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f32::max)
    }

    fn train_hebbian(&mut self) {
        let scale = 1.0 / self.size.max(1) as f32;
        for pattern_index in 0..self.pattern_count {
            let pattern = self.pattern(pattern_index).to_vec();
            for row in 0..self.size {
                let row_offset = row * self.size;
                for col in row + 1..self.size {
                    let delta = pattern[row] as f32 * pattern[col] as f32 * scale;
                    self.weights[row_offset + col] += delta;
                    self.weights[col * self.size + row] += delta;
                }
            }
        }
    }

    fn train_pseudoinverse(&mut self) {
        if self.pattern_count == 0 {
            return;
        }
        let mut overlap = vec![vec![0.0; self.pattern_count]; self.pattern_count];
        for mu in 0..self.pattern_count {
            for nu in mu..self.pattern_count {
                let mut dot = 0.0;
                for index in 0..self.size {
                    dot += self.pattern(mu)[index] as f32 * self.pattern(nu)[index] as f32;
                }
                overlap[mu][nu] = dot;
                overlap[nu][mu] = dot;
            }
        }
        let inverse = invert_with_regularization(overlap);
        for row in 0..self.size {
            for col in row + 1..self.size {
                let mut value = 0.0;
                for mu in 0..self.pattern_count {
                    for nu in 0..self.pattern_count {
                        value += self.pattern(mu)[row] as f32
                            * inverse[mu][nu]
                            * self.pattern(nu)[col] as f32;
                    }
                }
                self.weights[row * self.size + col] = value;
                self.weights[col * self.size + row] = value;
            }
        }
    }

    fn train_storkey(&mut self) {
        let scale = 1.0 / self.size.max(1) as f32;
        let mut local_fields = vec![0.0; self.size];
        for pattern_index in 0..self.pattern_count {
            let pattern = self.pattern(pattern_index).to_vec();
            for row in 0..self.size {
                local_fields[row] = self.local_field_for_state(row, &pattern);
            }
            for row in 0..self.size {
                for col in row + 1..self.size {
                    let delta = (pattern[row] as f32 * pattern[col] as f32
                        - pattern[row] as f32 * local_fields[col]
                        - local_fields[row] * pattern[col] as f32)
                        * scale;
                    self.weights[row * self.size + col] += delta;
                    self.weights[col * self.size + row] += delta;
                }
            }
            self.zero_diagonal();
        }
    }

    fn train_krauth_mezard(&mut self, kappa: f32, epsilon: f32, max_epochs: usize) {
        let scale = epsilon / self.size.max(1) as f32;
        for _ in 0..max_epochs {
            let mut stable = true;
            for pattern_index in 0..self.pattern_count {
                let pattern = self.pattern(pattern_index).to_vec();
                for row in 0..self.size {
                    let stability = pattern[row] as f32 * self.local_field_for_state(row, &pattern);
                    if stability > kappa {
                        continue;
                    }
                    let factor = scale * pattern[row] as f32;
                    for col in 0..self.size {
                        if col == row {
                            continue;
                        }
                        self.weights[row * self.size + col] += factor * pattern[col] as f32;
                    }
                    stable = false;
                }
            }
            self.symmetrize_weights();
            self.zero_diagonal();
            if stable {
                break;
            }
        }
    }

    fn train_unlearning(&mut self, epsilon: f32, steps: usize) {
        self.train_hebbian();
        let scale = epsilon / self.size.max(1) as f32;
        for _ in 0..steps {
            let mut state = self.random_state();
            for _ in 0..18 {
                if self.step_async_random_for_state(&mut state) == 0 {
                    break;
                }
            }
            if self.is_stored_pattern(&state) {
                continue;
            }
            for row in 0..self.size {
                for col in row + 1..self.size {
                    let delta = scale * state[row] as f32 * state[col] as f32;
                    self.weights[row * self.size + col] -= delta;
                    self.weights[col * self.size + row] -= delta;
                }
            }
            self.zero_diagonal();
        }
    }

    fn step_async_random(&mut self) -> usize {
        for index in (1..self.scratch_indices.len()).rev() {
            let swap_index = self.rng.next_usize(index + 1);
            self.scratch_indices.swap(index, swap_index);
        }
        let mut changed = 0;
        for position in 0..self.scratch_indices.len() {
            let neuron = self.scratch_indices[position];
            let next = if self.local_field_for_state(neuron, &self.state) >= 0.0 { 1 } else { -1 };
            if next != self.state[neuron] {
                self.state[neuron] = next;
                changed += 1;
            }
        }
        changed
    }

    fn step_async_random_for_state(&mut self, state: &mut [i8]) -> usize {
        for index in (1..self.scratch_indices.len()).rev() {
            let swap_index = self.rng.next_usize(index + 1);
            self.scratch_indices.swap(index, swap_index);
        }
        let mut changed = 0;
        for position in 0..self.scratch_indices.len() {
            let neuron = self.scratch_indices[position];
            let next = if self.local_field_for_state(neuron, state) >= 0.0 { 1 } else { -1 };
            if next != state[neuron] {
                state[neuron] = next;
                changed += 1;
            }
        }
        changed
    }

    fn step_synchronous(&mut self) -> usize {
        let mut next_state = vec![-1; self.size];
        let mut changed = 0;
        for neuron in 0..self.size {
            let next = if self.local_field_for_state(neuron, &self.state) >= 0.0 { 1 } else { -1 };
            next_state[neuron] = next;
            if next != self.state[neuron] {
                changed += 1;
            }
        }
        self.state = next_state;
        changed
    }

    fn step_stochastic(&mut self, temperature: f32) -> usize {
        let mut changed = 0;
        for _ in 0..self.size {
            let neuron = self.rng.next_usize(self.size);
            let field = self.local_field_for_state(neuron, &self.state);
            let probability = 1.0 / (1.0 + (-2.0 * field / temperature).exp());
            let next = if self.rng.next_bool(probability) { 1 } else { -1 };
            if next != self.state[neuron] {
                self.state[neuron] = next;
                changed += 1;
            }
        }
        changed
    }

    fn local_field_for_state(&self, neuron_index: usize, state: &[i8]) -> f32 {
        let row_offset = neuron_index * self.size;
        let mut total = 0.0;
        for target_index in 0..self.size {
            total += self.weights[row_offset + target_index] * state[target_index] as f32;
        }
        total
    }

    fn refresh_metrics(&mut self) {
        self.current_energy = self.compute_energy();
        self.matched_pattern_index = self.get_matched_pattern_index();
        self.converged = self.previous_state == self.state;
    }

    fn compute_energy(&self) -> f32 {
        let mut energy = 0.0;
        for row in 0..self.size {
            for col in row + 1..self.size {
                energy -= self.weights[row * self.size + col] * self.state[row] as f32 * self.state[col] as f32;
            }
        }
        energy
    }

    fn get_matched_pattern_index(&self) -> i32 {
        let mut best_index = -1;
        let mut best_distance = usize::MAX;
        for pattern_index in 0..self.pattern_count {
            let mut distance = 0;
            let pattern = self.pattern(pattern_index);
            for index in 0..self.size {
                if pattern[index] != self.state[index] {
                    distance += 1;
                }
            }
            if distance < best_distance {
                best_distance = distance;
                best_index = pattern_index as i32;
            }
        }
        best_index
    }

    fn pattern(&self, index: usize) -> &[i8] {
        let start = index * self.size;
        &self.patterns[start..start + self.size]
    }

    fn symmetrize_weights(&mut self) {
        for row in 0..self.size {
            for col in row + 1..self.size {
                let value =
                    (self.weights[row * self.size + col] + self.weights[col * self.size + row]) * 0.5;
                self.weights[row * self.size + col] = value;
                self.weights[col * self.size + row] = value;
            }
        }
    }

    fn zero_diagonal(&mut self) {
        for row in 0..self.size {
            self.weights[row * self.size + row] = 0.0;
        }
    }

    fn random_state(&mut self) -> Vec<i8> {
        (0..self.size)
            .map(|_| if self.rng.next_bool(0.5) { 1 } else { -1 })
            .collect()
    }

    fn is_stored_pattern(&self, candidate: &[i8]) -> bool {
        (0..self.pattern_count).any(|pattern_index| self.pattern(pattern_index) == candidate)
    }
}

fn invert_with_regularization(matrix: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let size = matrix.len();
    let attempts = [0.0, 1e-8, 1e-6, 1e-4];
    for regularization in attempts {
        let mut augmented = vec![vec![0.0; size * 2]; size];
        for row in 0..size {
            for col in 0..size {
                augmented[row][col] = matrix[row][col] + if row == col { regularization } else { 0.0 };
            }
            augmented[row][size + row] = 1.0;
        }
        if gauss_jordan_invert(&mut augmented) {
            return augmented
                .into_iter()
                .map(|row| row[size..].to_vec())
                .collect();
        }
    }
    panic!("Unable to invert pattern overlap matrix");
}

fn gauss_jordan_invert(augmented: &mut [Vec<f32>]) -> bool {
    let size = augmented.len();
    for pivot in 0..size {
        let mut best_row = pivot;
        let mut best_value = augmented[pivot][pivot].abs();
        for row in pivot + 1..size {
            let value = augmented[row][pivot].abs();
            if value > best_value {
                best_value = value;
                best_row = row;
            }
        }
        if best_value <= EPSILON {
            return false;
        }
        if best_row != pivot {
            augmented.swap(pivot, best_row);
        }
        let pivot_value = augmented[pivot][pivot];
        for col in 0..size * 2 {
            augmented[pivot][col] /= pivot_value;
        }
        for row in 0..size {
            if row == pivot {
                continue;
            }
            let factor = augmented[row][pivot];
            if factor.abs() <= EPSILON {
                continue;
            }
            for col in 0..size * 2 {
                augmented[row][col] -= factor * augmented[pivot][col];
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hopfield_hebbian_retrieves_pattern() {
        let patterns = vec![1, -1, 1, -1, -1, 1, -1, 1];
        let mut core = HopfieldCore::new(4, patterns, 2);
        core.train(HopfieldLearningRule::Hebbian, 0.0, 0.0, 1, 1);
        core.set_state(vec![1, -1, 1, -1]);
        core.step(HopfieldConvergenceRule::Synchronous, 1.0);
        assert!(core.energy().is_finite());
    }

    #[test]
    fn hopfield_hebbian_weights_are_symmetric_with_zero_diagonal() {
      let patterns = vec![1, -1, 1, -1, -1, 1, -1, 1];
      let mut core = HopfieldCore::new(4, patterns, 2);
      core.train(HopfieldLearningRule::Hebbian, 0.0, 0.0, 1, 1);
      let weights = core.weights();

      for row in 0..4 {
          assert_eq!(weights[row * 4 + row], 0.0);
          for col in 0..4 {
              assert!((weights[row * 4 + col] - weights[col * 4 + row]).abs() < 1e-6);
          }
      }
    }

    #[test]
    fn hopfield_async_energy_does_not_increase() {
        let patterns = vec![1, -1, 1, -1, -1, 1, -1, 1];
        let mut core = HopfieldCore::new(4, patterns, 2);
        core.train(HopfieldLearningRule::Hebbian, 0.0, 0.0, 1, 1);
        core.set_state(vec![1, -1, -1, -1]);
        let before = core.energy();
        core.step(HopfieldConvergenceRule::AsyncRandom, 1.0);
        let after = core.energy();
        assert!(after <= before + 1e-6, "async Hopfield step increased energy: {before} -> {after}");
    }
}
