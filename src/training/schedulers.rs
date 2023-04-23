//! Various learning rate schedulers.

/// Cosine annealing with warm restarts, described in paper
/// "SGDR: stochastic gradient descent with warm restarts"
/// https://arxiv.org/abs/1608.03983
///
/// Each eapoch it changes the learning rate, oscillating it between `lr_high` and `lr_low`.
/// It takes `period` epochs for the learning rate to drop to its very minimum,
/// after which it quickly returns back to `lr_high` (resets) and everything
/// starts over again.
///
/// With every reset:
///
/// * the period grows, multiplied by a factor of `period_mult`
/// * maximum learning rate drops proportionally to `high_lr_mult`
///
/// If `warmup_period` is greater than zero, the learning rate will be
/// linearly increased from `lr_low` to `lr_high` during the first
/// `warmup_period` epochs before the warm restarts kick in.
#[derive(Debug, Clone)]
pub struct CosineLRSchedule {
    pub lr_high: f64,
    pub lr_low: f64,
    pub initial_period: usize,
    pub period_mult: f64,
    pub high_lr_mult: f64,
    pub warmup_period: usize,
}

impl Default for CosineLRSchedule {
    fn default() -> Self {
        Self {
            lr_high: 1e-5,
            lr_low: 1e-7,
            initial_period: 50,
            period_mult: 2.0,
            high_lr_mult: 0.97,
            warmup_period: 0,
        }
    }
}

impl CosineLRSchedule {
    pub fn new(lr_high: f64, lr_low: f64) -> Self {
        Self {
            lr_high,
            lr_low,
            initial_period: 50,
            period_mult: 2.0,
            high_lr_mult: 0.97,
            warmup_period: 0,
        }
    }

    /// Calculates a new learning rate for the given `epoch`.
    pub fn get_lr(&self, epoch: usize) -> f64 {
        if epoch < self.warmup_period {
            self.lr_high * epoch as f64 / self.warmup_period as f64
        } else {
            self.get_lr_for_epoch(epoch - self.warmup_period)
        }
    }

    fn get_lr_for_epoch(&self, epoch: usize) -> f64 {
        let mut t_cur = 0f64;
        let mut lr_max = self.lr_high;
        let mut period = self.initial_period as f64;
        let mut result = lr_max;
        for i in 0..epoch + 1 {
            if i == epoch {
                // last iteration
                result = self.lr_low
                    + 0.5
                        * (lr_max - self.lr_low)
                        * (1.0 + (std::f64::consts::PI * t_cur / period).cos());
            } else if t_cur == period {
                period *= self.period_mult;
                lr_max *= self.high_lr_mult;
                t_cur = 0.0;
            } else {
                t_cur += 1.0;
            }
        }
        result
    }
}
