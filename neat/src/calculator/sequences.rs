//! Mathematical sequence recognition and generation
//!
//! This module enables NEAT to discover patterns in mathematical sequences,
//! from simple arithmetic progressions to complex recursive patterns.

use std::fmt;

/// Types of mathematical sequences
#[derive(Clone, PartialEq)]
pub enum SequenceType {
    /// Arithmetic progression (a, a+d, a+2d, ...)
    Arithmetic { first: f64, difference: f64 },
    /// Geometric progression (a, ar, ar², ...)
    Geometric { first: f64, ratio: f64 },
    /// Fibonacci-like (F(n) = F(n-1) + F(n-2))
    Fibonacci { seed1: f64, seed2: f64 },
    /// Polynomial sequence (an² + bn + c)
    Polynomial { a: f64, b: f64, c: f64 },
    /// Prime numbers
    Primes,
    /// Factorials (1!, 2!, 3!, ...)
    Factorial,
}

impl std::fmt::Debug for SequenceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SequenceType::Arithmetic { first, difference } => 
                write!(f, "Arithmetic {{ first: {}, difference: {} }}", first, difference),
            SequenceType::Geometric { first, ratio } => 
                write!(f, "Geometric {{ first: {}, ratio: {} }}", first, ratio),
            SequenceType::Fibonacci { seed1, seed2 } => 
                write!(f, "Fibonacci {{ seed1: {}, seed2: {} }}", seed1, seed2),
            SequenceType::Polynomial { a, b, c } => 
                write!(f, "Polynomial {{ a: {}, b: {}, c: {} }}", a, b, c),
            SequenceType::Primes => write!(f, "Primes"),
            SequenceType::Factorial => write!(f, "Factorial"),
        }
    }
}

/// A mathematical sequence with its generating rule
#[derive(Debug, Clone)]
pub struct Sequence {
    /// Type of sequence
    pub sequence_type: SequenceType,
    /// Generated terms
    pub terms: Vec<f64>,
    /// Maximum number of terms to generate
    pub max_terms: usize,
}

impl Sequence {
    /// Create a new sequence
    pub fn new(sequence_type: SequenceType, max_terms: usize) -> Self {
        let mut sequence = Self {
            sequence_type,
            terms: Vec::new(),
            max_terms,
        };
        sequence.generate_terms();
        sequence
    }
    
    /// Generate terms based on sequence type
    fn generate_terms(&mut self) {
        self.terms.clear();
        
        match &self.sequence_type {
            SequenceType::Arithmetic { first, difference } => {
                for i in 0..self.max_terms {
                    self.terms.push(first + difference * i as f64);
                }
            }
            SequenceType::Geometric { first, ratio } => {
                let mut current = *first;
                for _ in 0..self.max_terms {
                    self.terms.push(current);
                    current *= ratio;
                }
            }
            SequenceType::Fibonacci { seed1, seed2 } => {
                if self.max_terms > 0 {
                    self.terms.push(*seed1);
                }
                if self.max_terms > 1 {
                    self.terms.push(*seed2);
                }
                for i in 2..self.max_terms {
                    let next = self.terms[i-1] + self.terms[i-2];
                    self.terms.push(next);
                }
            }
            SequenceType::Polynomial { a, b, c } => {
                for n in 0..self.max_terms {
                    let n_f64 = n as f64;
                    let term = a * n_f64 * n_f64 + b * n_f64 + c;
                    self.terms.push(term);
                }
            }
            SequenceType::Primes => {
                self.terms = generate_primes(self.max_terms);
            }
            SequenceType::Factorial => {
                let mut fact = 1.0;
                self.terms.push(fact); // 0! = 1
                for i in 1..self.max_terms {
                    fact *= i as f64;
                    self.terms.push(fact);
                }
            }
        }
    }
    
    /// Get the nth term (0-indexed)
    pub fn get_term(&self, n: usize) -> Option<f64> {
        self.terms.get(n).copied()
    }
    
    /// Get a slice of terms
    pub fn get_terms(&self, start: usize, count: usize) -> Vec<f64> {
        self.terms.iter()
            .skip(start)
            .take(count)
            .copied()
            .collect()
    }
    
    /// Predict the next term based on learned pattern
    pub fn predict_next(&self) -> Option<f64> {
        match &self.sequence_type {
            SequenceType::Arithmetic { difference, .. } => {
                self.terms.last().map(|&last| last + difference)
            }
            SequenceType::Geometric { ratio, .. } => {
                self.terms.last().map(|&last| last * ratio)
            }
            SequenceType::Fibonacci { .. } => {
                if self.terms.len() >= 2 {
                    let n = self.terms.len();
                    Some(self.terms[n-1] + self.terms[n-2])
                } else {
                    None
                }
            }
            SequenceType::Polynomial { a, b, c } => {
                let n = self.terms.len() as f64;
                Some(a * n * n + b * n + c)
            }
            _ => None
        }
    }
    
    /// Check if a value matches the expected next term
    pub fn check_prediction(&self, predicted: f64, tolerance: f64) -> bool {
        if let Some(expected) = self.predict_next() {
            (predicted - expected).abs() < tolerance
        } else {
            false
        }
    }
}

impl fmt::Display for Sequence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let preview: Vec<String> = self.terms.iter()
            .take(5)
            .map(|t| format!("{:.1}", t))
            .collect();
        
        write!(f, "{:?}: {}, ...", self.sequence_type, preview.join(", "))
    }
}

/// Sequence learning problem for NEAT
#[derive(Debug, Clone)]
pub struct SequenceProblem {
    /// The sequence to learn
    pub sequence: Sequence,
    /// Number of terms to provide as input
    pub input_length: usize,
    /// Number of terms to predict
    pub predict_length: usize,
    /// Starting position in sequence
    pub start_index: usize,
}

impl SequenceProblem {
    /// Create a new sequence problem
    pub fn new(
        sequence: Sequence,
        input_length: usize,
        predict_length: usize,
        start_index: usize
    ) -> Self {
        Self {
            sequence,
            input_length,
            predict_length,
            start_index,
        }
    }
    
    /// Get input terms for the problem
    pub fn get_input(&self) -> Vec<f64> {
        self.sequence.get_terms(self.start_index, self.input_length)
    }
    
    /// Get expected output terms
    pub fn get_expected_output(&self) -> Vec<f64> {
        let output_start = self.start_index + self.input_length;
        self.sequence.get_terms(output_start, self.predict_length)
    }
    
    /// Create a next-term prediction problem
    pub fn next_term_problem(sequence: Sequence, context_length: usize) -> Self {
        Self {
            sequence,
            input_length: context_length,
            predict_length: 1,
            start_index: 0,
        }
    }
    
    /// Create a pattern completion problem
    pub fn pattern_completion(sequence: Sequence, show: usize, hide: usize) -> Self {
        Self {
            sequence,
            input_length: show,
            predict_length: hide,
            start_index: 0,
        }
    }
}

/// Pattern discovery in sequences
pub struct SequenceAnalyzer {
    /// Tolerance for floating point comparisons
    tolerance: f64,
}

impl SequenceAnalyzer {
    /// Create a new sequence analyzer
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }
    
    /// Try to identify the type of sequence from samples
    pub fn identify_sequence(&self, samples: &[f64]) -> Option<SequenceType> {
        if samples.len() < 3 {
            return None;
        }
        
        // Check for arithmetic progression
        if self.is_arithmetic(samples) {
            let diff = samples[1] - samples[0];
            return Some(SequenceType::Arithmetic {
                first: samples[0],
                difference: diff,
            });
        }
        
        // Check for geometric progression
        if self.is_geometric(samples) {
            let ratio = samples[1] / samples[0];
            return Some(SequenceType::Geometric {
                first: samples[0],
                ratio,
            });
        }
        
        // Check for Fibonacci-like
        if self.is_fibonacci_like(samples) {
            return Some(SequenceType::Fibonacci {
                seed1: samples[0],
                seed2: samples[1],
            });
        }
        
        // Try to fit polynomial
        if let Some((a, b, c)) = self.fit_polynomial(samples) {
            return Some(SequenceType::Polynomial { a, b, c });
        }
        
        None
    }
    
    /// Check if sequence is arithmetic
    fn is_arithmetic(&self, samples: &[f64]) -> bool {
        if samples.len() < 2 {
            return false;
        }
        
        let diff = samples[1] - samples[0];
        for i in 2..samples.len() {
            let expected_diff = samples[i] - samples[i-1];
            if (expected_diff - diff).abs() > self.tolerance {
                return false;
            }
        }
        true
    }
    
    /// Check if sequence is geometric
    fn is_geometric(&self, samples: &[f64]) -> bool {
        if samples.len() < 2 || samples[0].abs() < self.tolerance {
            return false;
        }
        
        let ratio = samples[1] / samples[0];
        for i in 2..samples.len() {
            if samples[i-1].abs() < self.tolerance {
                return false;
            }
            let expected_ratio = samples[i] / samples[i-1];
            if (expected_ratio - ratio).abs() > self.tolerance {
                return false;
            }
        }
        true
    }
    
    /// Check if sequence follows Fibonacci-like pattern
    fn is_fibonacci_like(&self, samples: &[f64]) -> bool {
        if samples.len() < 3 {
            return false;
        }
        
        for i in 2..samples.len() {
            let expected = samples[i-1] + samples[i-2];
            if (samples[i] - expected).abs() > self.tolerance {
                return false;
            }
        }
        true
    }
    
    /// Try to fit a quadratic polynomial
    fn fit_polynomial(&self, samples: &[f64]) -> Option<(f64, f64, f64)> {
        // Simplified: check if it's a perfect square sequence
        let mut is_squares = true;
        for (i, &val) in samples.iter().enumerate() {
            let expected = (i * i) as f64;
            if (val - expected).abs() > self.tolerance {
                is_squares = false;
                break;
            }
        }
        
        if is_squares {
            Some((1.0, 0.0, 0.0))
        } else {
            None
        }
    }
}

/// Generate first n prime numbers
fn generate_primes(n: usize) -> Vec<f64> {
    let mut primes = Vec::new();
    let mut candidate = 2;
    
    while primes.len() < n {
        if is_prime(candidate) {
            primes.push(candidate as f64);
        }
        candidate += 1;
    }
    
    primes
}

/// Check if a number is prime
fn is_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    for i in 2..=(n as f64).sqrt() as usize {
        if n % i == 0 {
            return false;
        }
    }
    true
}

/// Famous mathematical sequences for testing
pub struct FamousSequences;

impl FamousSequences {
    /// Fibonacci sequence
    pub fn fibonacci() -> Sequence {
        Sequence::new(SequenceType::Fibonacci { seed1: 0.0, seed2: 1.0 }, 20)
    }
    
    /// Prime numbers
    pub fn primes() -> Sequence {
        Sequence::new(SequenceType::Primes, 20)
    }
    
    /// Square numbers
    pub fn squares() -> Sequence {
        Sequence::new(SequenceType::Polynomial { a: 1.0, b: 0.0, c: 0.0 }, 20)
    }
    
    /// Triangular numbers
    pub fn triangular() -> Sequence {
        Sequence::new(SequenceType::Polynomial { a: 0.5, b: 0.5, c: 0.0 }, 20)
    }
    
    /// Powers of 2
    pub fn powers_of_two() -> Sequence {
        Sequence::new(SequenceType::Geometric { first: 1.0, ratio: 2.0 }, 20)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_arithmetic_sequence() {
        let seq = Sequence::new(
            SequenceType::Arithmetic { first: 5.0, difference: 3.0 },
            5
        );
        
        assert_eq!(seq.terms, vec![5.0, 8.0, 11.0, 14.0, 17.0]);
        assert_eq!(seq.predict_next(), Some(20.0));
    }
    
    #[test]
    fn test_fibonacci_sequence() {
        let seq = FamousSequences::fibonacci();
        let first_8: Vec<f64> = seq.terms.iter().take(8).copied().collect();
        assert_eq!(first_8, vec![0.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0]);
    }
    
    #[test]
    fn test_sequence_analyzer() {
        let analyzer = SequenceAnalyzer::new(0.001);
        
        // Test arithmetic sequence
        let arithmetic = vec![2.0, 5.0, 8.0, 11.0, 14.0];
        if let Some(SequenceType::Arithmetic { first, difference }) = 
            analyzer.identify_sequence(&arithmetic) {
            assert_eq!(first, 2.0);
            assert_eq!(difference, 3.0);
        } else {
            panic!("Failed to identify arithmetic sequence");
        }
        
        // Test geometric sequence
        let geometric = vec![3.0, 6.0, 12.0, 24.0, 48.0];
        if let Some(SequenceType::Geometric { first, ratio }) = 
            analyzer.identify_sequence(&geometric) {
            assert_eq!(first, 3.0);
            assert_eq!(ratio, 2.0);
        } else {
            panic!("Failed to identify geometric sequence");
        }
    }
    
    #[test]
    fn test_sequence_problem() {
        let seq = FamousSequences::squares();
        let problem = SequenceProblem::next_term_problem(seq, 3);
        
        let input = problem.get_input();
        let expected = problem.get_expected_output();
        
        assert_eq!(input, vec![0.0, 1.0, 4.0]);
        assert_eq!(expected, vec![9.0]);
    }
}