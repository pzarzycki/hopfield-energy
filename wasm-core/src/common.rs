pub const EPSILON: f32 = 1e-6;

#[derive(Clone)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }

    pub fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        let normalized = ((self.state >> 32) as u32) as f32 / (u32::MAX as f32);
        normalized
    }

    pub fn next_signed_f32(&mut self) -> f32 {
        self.next_f32() - 0.5
    }

    pub fn next_bool(&mut self, probability: f32) -> bool {
        self.next_f32() < probability
    }

    pub fn next_usize(&mut self, upper_exclusive: usize) -> usize {
        if upper_exclusive <= 1 {
            0
        } else {
            (self.next_f32() * upper_exclusive as f32).floor() as usize % upper_exclusive
        }
    }
}

pub fn clamp01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

pub fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        let exp = (-value).exp();
        1.0 / (1.0 + exp)
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

pub fn log1p_exp(value: f32) -> f32 {
    if value > 0.0 {
        value + (-value).exp().ln_1p()
    } else {
        value.exp().ln_1p()
    }
}

pub fn mean_absolute_difference(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(left, right)| (left - right).abs())
        .sum::<f32>()
        / (a.len().max(1) as f32)
}

pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(left, right)| left * right).sum()
}

pub fn fit_f32_pattern(mut values: Vec<f32>, size: usize) -> Vec<f32> {
    values.resize(size, 0.0);
    values.truncate(size);
    values
}

pub fn fit_i8_pattern(mut values: Vec<i8>, size: usize) -> Vec<i8> {
    values.resize(size, -1);
    values.truncate(size);
    values
}
