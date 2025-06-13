//! Fitness evaluators for algebraic learning tasks
//!
//! This module provides advanced fitness evaluation for NEAT networks
//! learning algebraic concepts, from expression evaluation to equation solving.

use super::algebra::{Expression, AlgebraProblem};
use super::algebra_encoding::{AlgebraEncoder, AlgebraEncodingConfig};
use super::{Operation, EncodingConfig};
use crate::neat::fitness::FitnessEvaluator;
use crate::neat::genome::Genome;
use crate::neat::network::Network;
use crate::error::Result;
use std::collections::HashMap;
use rand::{Rng, SeedableRng, rngs::SmallRng};

/// Configuration for algebraic fitness evaluation
#[derive(Debug, Clone)]
pub struct AlgebraEvaluatorConfig {
    /// Number of problems per evaluation
    pub problems_per_evaluation: usize,
    /// Encoding configuration
    pub encoding_config: AlgebraEncodingConfig,
    /// Problem difficulty level
    pub difficulty: AlgebraicDifficulty,
    /// Weight for exact answer accuracy
    pub exact_weight: f64,
    /// Weight for partial accuracy
    pub partial_weight: f64,
    /// Weight for structural understanding
    pub structure_weight: f64,
    /// Complexity penalty
    pub complexity_penalty: f64,
    /// Random seed
    pub random_seed: Option<u64>,
}

impl Default for AlgebraEvaluatorConfig {
    fn default() -> Self {
        Self {
            problems_per_evaluation: 30,
            encoding_config: AlgebraEncodingConfig::default(),
            difficulty: AlgebraicDifficulty::Basic,
            exact_weight: 1.0,
            partial_weight: 0.3,
            structure_weight: 0.2,
            complexity_penalty: 0.001,
            random_seed: Some(42),
        }
    }
}

/// Difficulty levels for algebraic problems
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgebraicDifficulty {
    /// Simple expression evaluation (x + 2 where x = 3)
    Basic,
    /// Multi-variable expressions (2x + 3y)
    Intermediate,
    /// Linear equation solving (2x + 3 = 7)
    Advanced,
    /// Factoring and simplification
    Expert,
}

/// Fitness evaluator for algebraic learning
pub struct AlgebraEvaluator {
    config: AlgebraEvaluatorConfig,
    encoder: AlgebraEncoder,
    problem_generator: AlgebraProblemGenerator,
    output_decoder: OutputDecoder,
}

impl AlgebraEvaluator {
    /// Create a new algebra evaluator
    pub fn new(config: AlgebraEvaluatorConfig) -> Self {
        let mut encoder = AlgebraEncoder::new(config.encoding_config.clone());
        
        // Register standard variables
        let _ = encoder.register_variables(&["x".to_string(), "y".to_string(), "z".to_string()]);
        
        let problem_generator = AlgebraProblemGenerator::new(
            config.difficulty,
            config.random_seed
        );
        
        let output_decoder = OutputDecoder::new(config.encoding_config.number_config.clone());
        
        Self {
            config,
            encoder,
            problem_generator,
            output_decoder,
        }
    }
    
    /// Evaluate a network on algebraic problems
    pub fn evaluate_on_problems(
        &self, 
        network: &Network, 
        problems: &[AlgebraProblem]
    ) -> Result<AlgebraResults> {
        let mut exact_correct = 0;
        let mut total_error = 0.0;
        let mut problem_results = Vec::new();
        
        for problem in problems {
            let input = self.encoder.encode_problem(problem)?;
            let output = network.activate(&input)?;
            
            let (predicted, expected) = match problem {
                AlgebraProblem::Evaluation { expected, .. } => {
                    let pred = self.output_decoder.decode_number(&output)?;
                    (pred, *expected)
                }
                AlgebraProblem::LinearEquation { expected, .. } => {
                    let pred = self.output_decoder.decode_number(&output)?;
                    (pred, *expected)
                }
                _ => continue, // Skip other problem types for now
            };
            
            let error = (predicted - expected).abs();
            let is_exact = error < 0.1;
            
            if is_exact {
                exact_correct += 1;
            }
            total_error += error;
            
            problem_results.push(AlgebraProblemResult {
                problem: problem.clone(),
                predicted,
                expected,
                error,
                is_correct: is_exact,
            });
        }
        
        let exact_accuracy = exact_correct as f64 / problems.len() as f64;
        let average_error = total_error / problems.len() as f64;
        
        Ok(AlgebraResults {
            exact_accuracy,
            average_error,
            exact_correct,
            total_problems: problems.len(),
            problem_results,
        })
    }
    
    /// Calculate structural understanding score
    fn calculate_structure_score(&self, network: &Network, problems: &[AlgebraProblem]) -> f64 {
        // This would analyze if the network understands expression structure
        // For now, return a placeholder
        0.5
    }
    
    /// Calculate complexity penalty
    fn calculate_complexity_penalty(&self, genome: &Genome) -> f64 {
        let node_count = genome.nodes.len() as f64;
        let connection_count = genome.connections.iter().filter(|c| c.enabled).count() as f64;
        self.config.complexity_penalty * (node_count + connection_count * 0.5)
    }
}

impl FitnessEvaluator for AlgebraEvaluator {
    fn evaluate(&self, genome: &Genome) -> Result<f64> {
        let network = Network::from_genome(genome)?;
        
        // Generate problems for evaluation
        let problems = self.problem_generator.generate_batch(self.config.problems_per_evaluation);
        
        if problems.is_empty() {
            return Ok(0.0);
        }
        
        // Evaluate on problems
        let results = self.evaluate_on_problems(&network, &problems)?;
        
        // Calculate composite fitness
        let mut fitness = 0.0;
        
        // Exact accuracy component
        fitness += self.config.exact_weight * results.exact_accuracy;
        
        // Partial accuracy component (inverse of error)
        let partial_score = 1.0 / (1.0 + results.average_error);
        fitness += self.config.partial_weight * partial_score;
        
        // Structure understanding component
        let structure_score = self.calculate_structure_score(&network, &problems);
        fitness += self.config.structure_weight * structure_score;
        
        // Complexity penalty
        fitness -= self.calculate_complexity_penalty(genome);
        
        Ok(fitness.max(0.0))
    }
    
    fn input_size(&self) -> usize {
        self.encoder.encoding_length()
    }
    
    fn output_size(&self) -> usize {
        self.output_decoder.output_length()
    }
    
    fn max_fitness(&self) -> f64 {
        self.config.exact_weight + self.config.partial_weight + self.config.structure_weight
    }
}

/// Results from algebraic evaluation
#[derive(Debug, Clone)]
pub struct AlgebraResults {
    /// Percentage of exactly correct answers
    pub exact_accuracy: f64,
    /// Average error across all problems
    pub average_error: f64,
    /// Number of exactly correct answers
    pub exact_correct: usize,
    /// Total number of problems
    pub total_problems: usize,
    /// Individual problem results
    pub problem_results: Vec<AlgebraProblemResult>,
}

impl AlgebraResults {
    /// Print detailed results
    pub fn print_detailed(&self) {
        println!("Algebraic Evaluation Results:");
        println!("  Exact Accuracy: {:.1}% ({}/{})", 
                self.exact_accuracy * 100.0, 
                self.exact_correct, 
                self.total_problems);
        println!("  Average Error: {:.3}", self.average_error);
        
        println!("\nExample Problems:");
        for (i, result) in self.problem_results.iter().take(5).enumerate() {
            let status = if result.is_correct { "✓" } else { "✗" };
            println!("  {}: {}", i + 1, result.problem.description());
            println!("     Predicted: {:.2}, Expected: {:.2}, Error: {:.3} {}",
                    result.predicted, result.expected, result.error, status);
        }
    }
}

/// Result for a single algebraic problem
#[derive(Debug, Clone)]
pub struct AlgebraProblemResult {
    /// The original problem
    pub problem: AlgebraProblem,
    /// Network's predicted answer
    pub predicted: f64,
    /// Expected answer
    pub expected: f64,
    /// Absolute error
    pub error: f64,
    /// Whether prediction is correct (within tolerance)
    pub is_correct: bool,
}

/// Generator for algebraic problems
struct AlgebraProblemGenerator {
    difficulty: AlgebraicDifficulty,
    rng: SmallRng,
}

impl AlgebraProblemGenerator {
    fn new(difficulty: AlgebraicDifficulty, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => SmallRng::seed_from_u64(s),
            None => SmallRng::from_entropy(),
        };
        
        Self { difficulty, rng }
    }
    
    fn generate_batch(&self, count: usize) -> Vec<AlgebraProblem> {
        let mut problems = Vec::new();
        let mut rng = self.rng.clone();
        
        for _ in 0..count {
            if let Some(problem) = self.generate_problem(&mut rng) {
                problems.push(problem);
            }
        }
        
        problems
    }
    
    fn generate_problem(&self, rng: &mut SmallRng) -> Option<AlgebraProblem> {
        match self.difficulty {
            AlgebraicDifficulty::Basic => self.generate_basic_problem(rng),
            AlgebraicDifficulty::Intermediate => self.generate_intermediate_problem(rng),
            AlgebraicDifficulty::Advanced => self.generate_advanced_problem(rng),
            AlgebraicDifficulty::Expert => self.generate_expert_problem(rng),
        }
    }
    
    fn generate_basic_problem(&self, rng: &mut SmallRng) -> Option<AlgebraProblem> {
        // Simple expression evaluation: ax + b where x is given
        let a = rng.gen_range(1..10) as f64;
        let b = rng.gen_range(0..10) as f64;
        let x_val = rng.gen_range(1..10) as f64;
        
        let expr = Expression::binary(
            Expression::binary(
                Expression::constant(a),
                Operation::Multiply,
                Expression::variable("x")
            ),
            Operation::Add,
            Expression::constant(b)
        );
        
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), x_val);
        
        AlgebraProblem::evaluation(expr, vars).ok()
    }
    
    fn generate_intermediate_problem(&self, rng: &mut SmallRng) -> Option<AlgebraProblem> {
        // Multi-variable expression: ax + by + c
        let a = rng.gen_range(1..5) as f64;
        let b = rng.gen_range(1..5) as f64;
        let c = rng.gen_range(0..10) as f64;
        let x_val = rng.gen_range(1..10) as f64;
        let y_val = rng.gen_range(1..10) as f64;
        
        let expr = Expression::binary(
            Expression::binary(
                Expression::binary(
                    Expression::constant(a),
                    Operation::Multiply,
                    Expression::variable("x")
                ),
                Operation::Add,
                Expression::binary(
                    Expression::constant(b),
                    Operation::Multiply,
                    Expression::variable("y")
                )
            ),
            Operation::Add,
            Expression::constant(c)
        );
        
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), x_val);
        vars.insert("y".to_string(), y_val);
        
        AlgebraProblem::evaluation(expr, vars).ok()
    }
    
    fn generate_advanced_problem(&self, rng: &mut SmallRng) -> Option<AlgebraProblem> {
        // Linear equation solving: ax + b = c
        let a = rng.gen_range(1..10) as f64;
        let b = rng.gen_range(0..20) as f64;
        let c = rng.gen_range(10..50) as f64;
        
        Some(AlgebraProblem::linear_equation(a, b, c))
    }
    
    fn generate_expert_problem(&self, rng: &mut SmallRng) -> Option<AlgebraProblem> {
        // For now, generate advanced problems
        // Future: factoring, simplification, etc.
        self.generate_advanced_problem(rng)
    }
}

/// Decoder for network outputs to mathematical results
struct OutputDecoder {
    config: EncodingConfig,
}

impl OutputDecoder {
    fn new(config: EncodingConfig) -> Self {
        Self { config }
    }
    
    fn decode_number(&self, output: &[f64]) -> Result<f64> {
        // Simple decoding: take first output as normalized value
        if output.is_empty() {
            return Ok(0.0);
        }
        
        // Denormalize from [0, 1] to reasonable range
        let normalized = output[0].clamp(0.0, 1.0);
        let max_value = 10f64.powi(self.config.max_digits as i32);
        
        Ok(normalized * max_value - max_value / 2.0)
    }
    
    fn output_length(&self) -> usize {
        // For now, single output for numeric results
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_algebra_evaluator_creation() {
        let config = AlgebraEvaluatorConfig::default();
        let evaluator = AlgebraEvaluator::new(config);
        
        assert!(evaluator.input_size() > 0);
        assert!(evaluator.output_size() > 0);
        assert!(evaluator.max_fitness() > 0.0);
    }
    
    #[test]
    fn test_problem_generation() {
        let generator = AlgebraProblemGenerator::new(
            AlgebraicDifficulty::Basic,
            Some(42)
        );
        
        let problems = generator.generate_batch(10);
        assert_eq!(problems.len(), 10);
        
        for problem in &problems {
            match problem {
                AlgebraProblem::Evaluation { .. } => {},
                _ => panic!("Expected evaluation problems for basic difficulty"),
            }
        }
    }
    
    #[test]
    fn test_evaluator_with_genome() -> Result<()> {
        let config = AlgebraEvaluatorConfig {
            problems_per_evaluation: 5,
            ..Default::default()
        };
        
        let evaluator = AlgebraEvaluator::new(config);
        let genome = Genome::new(0, evaluator.input_size(), evaluator.output_size());
        
        let fitness = evaluator.evaluate(&genome)?;
        assert!(fitness >= 0.0);
        assert!(fitness <= evaluator.max_fitness());
        
        Ok(())
    }
}