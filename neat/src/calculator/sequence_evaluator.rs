//! Fitness evaluators for sequence learning tasks
//!
//! This module enables NEAT to discover patterns in mathematical sequences,
//! demonstrating emergent pattern recognition and predictive capabilities.

use super::sequences::{Sequence, SequenceProblem, SequenceType, FamousSequences};
use crate::neat::fitness::FitnessEvaluator;
use crate::neat::genome::Genome;
use crate::neat::network::Network;
use crate::error::Result;
use rand::{Rng, SeedableRng, rngs::SmallRng};

/// Configuration for sequence learning evaluation
#[derive(Debug, Clone)]
pub struct SequenceEvaluatorConfig {
    /// Number of sequences to test per evaluation
    pub sequences_per_evaluation: usize,
    /// Length of input context
    pub context_length: usize,
    /// Number of terms to predict
    pub predict_length: usize,
    /// Types of sequences to include
    pub sequence_types: Vec<SequenceType>,
    /// Weight for exact prediction accuracy
    pub exact_weight: f64,
    /// Weight for relative error
    pub error_weight: f64,
    /// Weight for pattern consistency
    pub pattern_weight: f64,
    /// Complexity penalty
    pub complexity_penalty: f64,
    /// Tolerance for correct predictions
    pub tolerance: f64,
    /// Random seed
    pub random_seed: Option<u64>,
}

impl Default for SequenceEvaluatorConfig {
    fn default() -> Self {
        Self {
            sequences_per_evaluation: 20,
            context_length: 5,
            predict_length: 1,
            sequence_types: vec![
                SequenceType::Arithmetic { first: 1.0, difference: 1.0 },
                SequenceType::Geometric { first: 1.0, ratio: 2.0 },
            ],
            exact_weight: 1.0,
            error_weight: 0.5,
            pattern_weight: 0.3,
            complexity_penalty: 0.001,
            tolerance: 0.1,
            random_seed: Some(42),
        }
    }
}

/// Fitness evaluator for sequence learning
pub struct SequenceEvaluator {
    config: SequenceEvaluatorConfig,
    rng: SmallRng,
}

impl SequenceEvaluator {
    /// Create a new sequence evaluator
    pub fn new(config: SequenceEvaluatorConfig) -> Self {
        let rng = match config.random_seed {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => SmallRng::from_entropy(),
        };
        
        Self { config, rng }
    }
    
    /// Generate a batch of sequence problems
    fn generate_problems(&mut self) -> Vec<SequenceProblem> {
        let mut problems = Vec::new();
        
        for _ in 0..self.config.sequences_per_evaluation {
            let sequence = self.generate_random_sequence();
            let start_index = self.rng.gen_range(0..5);
            
            let problem = SequenceProblem::new(
                sequence,
                self.config.context_length,
                self.config.predict_length,
                start_index
            );
            
            problems.push(problem);
        }
        
        problems
    }
    
    /// Generate a random sequence based on configured types
    fn generate_random_sequence(&mut self) -> Sequence {
        if self.config.sequence_types.is_empty() {
            // Default to arithmetic sequence
            return Sequence::new(
                SequenceType::Arithmetic { first: 1.0, difference: 2.0 },
                20
            );
        }
        
        let idx = self.rng.gen_range(0..self.config.sequence_types.len());
        let base_type = &self.config.sequence_types[idx];
        
        // Randomize parameters slightly
        let sequence_type = match base_type {
            SequenceType::Arithmetic { .. } => {
                let first = self.rng.gen_range(1.0..10.0);
                let difference = self.rng.gen_range(1.0..5.0);
                SequenceType::Arithmetic { first, difference }
            }
            SequenceType::Geometric { .. } => {
                let first = self.rng.gen_range(1.0..5.0);
                let ratio = self.rng.gen_range(1.5..3.0);
                SequenceType::Geometric { first, ratio }
            }
            other => other.clone()
        };
        
        Sequence::new(sequence_type, 20)
    }
    
    /// Evaluate network on sequence problems
    pub fn evaluate_on_problems(
        &self,
        network: &Network,
        problems: &[SequenceProblem]
    ) -> Result<SequenceResults> {
        let mut exact_correct = 0;
        let mut total_error = 0.0;
        let mut pattern_scores = Vec::new();
        let mut problem_results = Vec::new();
        
        for problem in problems {
            let input = self.encode_input(&problem.get_input());
            let output = network.activate(&input)?;
            let predicted = self.decode_output(&output, problem.predict_length);
            let expected = problem.get_expected_output();
            
            // Calculate metrics
            let mut problem_exact = 0;
            let mut problem_error = 0.0;
            
            for (i, (&pred, &exp)) in predicted.iter().zip(expected.iter()).enumerate() {
                let error = (pred - exp).abs();
                if error < self.config.tolerance {
                    problem_exact += 1;
                }
                problem_error += error;
            }
            
            let exact_rate = problem_exact as f64 / expected.len() as f64;
            if exact_rate >= 1.0 {
                exact_correct += 1;
            }
            
            total_error += problem_error / expected.len() as f64;
            
            // Pattern consistency score
            let pattern_score = self.calculate_pattern_score(&predicted, &problem.sequence);
            pattern_scores.push(pattern_score);
            
            problem_results.push(SequenceProblemResult {
                sequence_type: format!("{:?}", problem.sequence.sequence_type),
                input: problem.get_input(),
                predicted: predicted.clone(),
                expected: expected.clone(),
                exact_matches: problem_exact,
                average_error: problem_error / expected.len() as f64,
                pattern_score,
            });
        }
        
        let exact_accuracy = exact_correct as f64 / problems.len() as f64;
        let average_error = total_error / problems.len() as f64;
        let average_pattern_score = pattern_scores.iter().sum::<f64>() / pattern_scores.len() as f64;
        
        Ok(SequenceResults {
            exact_accuracy,
            average_error,
            average_pattern_score,
            exact_correct,
            total_problems: problems.len(),
            problem_results,
        })
    }
    
    /// Encode sequence input for neural network
    fn encode_input(&self, sequence: &[f64]) -> Vec<f64> {
        // Normalize to [-1, 1] range
        let max_val = sequence.iter()
            .map(|&x| x.abs())
            .fold(1.0, f64::max);
        
        sequence.iter()
            .map(|&x| x / max_val)
            .collect()
    }
    
    /// Decode network output to sequence predictions
    fn decode_output(&self, output: &[f64], length: usize) -> Vec<f64> {
        // Take first 'length' outputs and denormalize
        output.iter()
            .take(length)
            .map(|&x| x * 100.0) // Simple denormalization
            .collect()
    }
    
    /// Calculate pattern consistency score
    fn calculate_pattern_score(&self, predicted: &[f64], sequence: &Sequence) -> f64 {
        // Check if predictions follow the same pattern type
        match &sequence.sequence_type {
            SequenceType::Arithmetic { difference, .. } => {
                if predicted.len() < 2 {
                    return 0.0;
                }
                
                // Check if differences are consistent
                let mut score = 0.0;
                for i in 1..predicted.len() {
                    let pred_diff = predicted[i] - predicted[i-1];
                    let error = (pred_diff - difference).abs();
                    score += 1.0 / (1.0 + error);
                }
                score / (predicted.len() - 1) as f64
            }
            SequenceType::Geometric { ratio, .. } => {
                if predicted.len() < 2 || predicted[0].abs() < 1e-10 {
                    return 0.0;
                }
                
                // Check if ratios are consistent
                let mut score = 0.0;
                for i in 1..predicted.len() {
                    if predicted[i-1].abs() > 1e-10 {
                        let pred_ratio = predicted[i] / predicted[i-1];
                        let error = (pred_ratio - ratio).abs();
                        score += 1.0 / (1.0 + error);
                    }
                }
                score / (predicted.len() - 1) as f64
            }
            _ => 0.5 // Default score for other patterns
        }
    }
}

impl FitnessEvaluator for SequenceEvaluator {
    fn evaluate(&self, genome: &Genome) -> Result<f64> {
        let network = Network::from_genome(genome)?;
        
        // Generate problems
        let mut evaluator = self.clone();
        let problems = evaluator.generate_problems();
        
        if problems.is_empty() {
            return Ok(0.0);
        }
        
        // Evaluate on problems
        let results = self.evaluate_on_problems(&network, &problems)?;
        
        // Calculate composite fitness
        let mut fitness = 0.0;
        
        // Exact accuracy component
        fitness += self.config.exact_weight * results.exact_accuracy;
        
        // Error component (inverse)
        let error_score = 1.0 / (1.0 + results.average_error);
        fitness += self.config.error_weight * error_score;
        
        // Pattern consistency component
        fitness += self.config.pattern_weight * results.average_pattern_score;
        
        // Complexity penalty
        let complexity = genome.nodes.len() as f64 + 
                        genome.connections.iter().filter(|c| c.enabled).count() as f64 * 0.5;
        fitness -= self.config.complexity_penalty * complexity;
        
        Ok(fitness.max(0.0))
    }
    
    fn input_size(&self) -> usize {
        self.config.context_length
    }
    
    fn output_size(&self) -> usize {
        self.config.predict_length
    }
    
    fn max_fitness(&self) -> f64 {
        self.config.exact_weight + self.config.error_weight + self.config.pattern_weight
    }
}

impl Clone for SequenceEvaluator {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            rng: SmallRng::seed_from_u64(
                self.config.random_seed.unwrap_or(42)
            ),
        }
    }
}

/// Results from sequence evaluation
#[derive(Debug, Clone)]
pub struct SequenceResults {
    /// Percentage of sequences predicted exactly
    pub exact_accuracy: f64,
    /// Average prediction error
    pub average_error: f64,
    /// Average pattern consistency score
    pub average_pattern_score: f64,
    /// Number of exactly correct sequences
    pub exact_correct: usize,
    /// Total number of sequences tested
    pub total_problems: usize,
    /// Individual problem results
    pub problem_results: Vec<SequenceProblemResult>,
}

impl SequenceResults {
    /// Print detailed results
    pub fn print_detailed(&self) {
        println!("Sequence Learning Results:");
        println!("  Exact Accuracy: {:.1}% ({}/{})",
                self.exact_accuracy * 100.0,
                self.exact_correct,
                self.total_problems);
        println!("  Average Error: {:.3}", self.average_error);
        println!("  Pattern Score: {:.3}", self.average_pattern_score);
        
        println!("\nExample Predictions:");
        for (i, result) in self.problem_results.iter().take(3).enumerate() {
            println!("  Sequence {}: {}", i + 1, result.sequence_type);
            println!("    Input: {:?}", result.input);
            println!("    Expected: {:?}", result.expected);
            println!("    Predicted: {:?}", result.predicted);
            println!("    Error: {:.3}, Pattern: {:.3}",
                    result.average_error,
                    result.pattern_score);
        }
    }
}

/// Result for a single sequence problem
#[derive(Debug, Clone)]
pub struct SequenceProblemResult {
    /// Type of sequence
    pub sequence_type: String,
    /// Input sequence
    pub input: Vec<f64>,
    /// Predicted values
    pub predicted: Vec<f64>,
    /// Expected values
    pub expected: Vec<f64>,
    /// Number of exact matches
    pub exact_matches: usize,
    /// Average prediction error
    pub average_error: f64,
    /// Pattern consistency score
    pub pattern_score: f64,
}

/// Create evaluators for famous sequences
pub struct FamousSequenceEvaluators;

impl FamousSequenceEvaluators {
    /// Fibonacci sequence predictor
    pub fn fibonacci() -> SequenceEvaluator {
        let config = SequenceEvaluatorConfig {
            sequences_per_evaluation: 10,
            context_length: 5,
            predict_length: 1,
            sequence_types: vec![
                SequenceType::Fibonacci { seed1: 0.0, seed2: 1.0 },
                SequenceType::Fibonacci { seed1: 1.0, seed2: 1.0 },
            ],
            ..Default::default()
        };
        SequenceEvaluator::new(config)
    }
    
    /// Prime number predictor
    pub fn primes() -> SequenceEvaluator {
        let config = SequenceEvaluatorConfig {
            sequences_per_evaluation: 10,
            context_length: 6,
            predict_length: 1,
            sequence_types: vec![SequenceType::Primes],
            tolerance: 0.5, // Primes are harder
            ..Default::default()
        };
        SequenceEvaluator::new(config)
    }
    
    /// Mixed sequence types
    pub fn mixed() -> SequenceEvaluator {
        let config = SequenceEvaluatorConfig {
            sequences_per_evaluation: 20,
            context_length: 4,
            predict_length: 2,
            sequence_types: vec![
                SequenceType::Arithmetic { first: 1.0, difference: 1.0 },
                SequenceType::Geometric { first: 1.0, ratio: 2.0 },
                SequenceType::Polynomial { a: 1.0, b: 0.0, c: 0.0 },
            ],
            ..Default::default()
        };
        SequenceEvaluator::new(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sequence_evaluator() {
        let config = SequenceEvaluatorConfig {
            sequences_per_evaluation: 5,
            context_length: 3,
            predict_length: 1,
            ..Default::default()
        };
        
        let evaluator = SequenceEvaluator::new(config);
        assert_eq!(evaluator.input_size(), 3);
        assert_eq!(evaluator.output_size(), 1);
    }
    
    #[test]
    fn test_pattern_scoring() {
        let evaluator = SequenceEvaluator::new(SequenceEvaluatorConfig::default());
        
        let seq = Sequence::new(
            SequenceType::Arithmetic { first: 1.0, difference: 2.0 },
            10
        );
        
        // Perfect prediction
        let perfect = vec![7.0, 9.0, 11.0];
        let score1 = evaluator.calculate_pattern_score(&perfect, &seq);
        assert!(score1 > 0.9);
        
        // Imperfect prediction
        let imperfect = vec![7.0, 8.5, 11.5];
        let score2 = evaluator.calculate_pattern_score(&imperfect, &seq);
        assert!(score2 < score1);
    }
}