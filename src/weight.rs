use rand::{self, Closed01, Rng};
use std::ops::Deref;

/// LinkGene weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Weight(pub f64);

impl Into<f64> for Weight {
    fn into(self) -> f64 {
        self.0
    }
}

impl Deref for Weight {
    type Target = f64;
    fn deref(&self) -> &f64 {
        &self.0
    }
}

impl Weight {
    pub fn new() -> Weight {
        Weight(rand::random::<Closed01<f64>>().0)
    }
    /// Randomse the weight, TODO possibly gaussian is better here !
    pub fn perturb(&mut self) {
        self.add_weight(Weight::new());
    }

    pub fn add_weight(&mut self, weight: Weight) {
        if weight.0 + self.0 >= 1f64 {
            self.0 = 1f64;
        } else {
            self.0 += weight.0
        }
    }
}

mod tests {
    #[test]
    fn new_in_range() {
        assert!(*weight::new() <= 1f64)
    }

    #[test]
    fn add_in_range() {
        let w = Weight::new();
        for _ in 0..100 {
            assert!(*w.add_weight(Weight::new()) <= 1f64);
        }
    }
    #[test]
    fn randmise() {
        let w = Weight::new();
        for _ in 0..100 {
            assert!(*w.perturb() <= 1);
        }
    }


}
