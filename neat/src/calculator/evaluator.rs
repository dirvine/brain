//! Fitness evaluators for mathematical learning tasks
//!
//! This module provides comprehensive fitness evaluation for NEAT networks
//! learning arithmetic operations, with multiple metrics and progressive
//! difficulty assessment.

use super::{ArithmeticGenerator, ArithmeticConfig, ProblemEncoder, EncodingConfig, MathProblem};
use crate::neat::fitness::FitnessEvaluator;
use crate::neat::genome::Genome;
use crate::neat::network::Network;
use crate::error::Result;
use std::collections::HashMap;

/// Configuration for arithmetic fitness evaluation
#[derive(Debug, Clone)]
pub struct ArithmeticEvaluatorConfig {
    /// Number of problems to test per evaluation
    pub problems_per_evaluation: usize,
    /// Encoding configuration for problems and results
    pub encoding_config: EncodingConfig,
    /// Arithmetic generation configuration  
    pub arithmetic_config: ArithmeticConfig,
    /// Tolerance for considering an answer "close enough"
    pub tolerance: f64,
    /// Weight for exact accuracy vs partial credit
    pub exact_weight: f64,
    /// Weight for partial accuracy (digit-by-digit)
    pub partial_weight: f64,
    /// Penalty for network complexity
    pub complexity_penalty: f64,
    /// Bonus for perfect problems
    pub perfect_bonus: f64,
}

impl Default for ArithmeticEvaluatorConfig {
    fn default() -> Self {
        Self {
            problems_per_evaluation: 50,
            encoding_config: EncodingConfig::default(),
            arithmetic_config: ArithmeticConfig::default(),
            tolerance: 0.1,
            exact_weight: 1.0,
            partial_weight: 0.3,
            complexity_penalty: 0.001,
            perfect_bonus: 0.1,
        }
    }
}

/// Fitness evaluator for arithmetic learning
pub struct ArithmeticEvaluator {
    config: ArithmeticEvaluatorConfig,
    generator: ArithmeticGenerator,
    encoder: ProblemEncoder,
}

impl ArithmeticEvaluator {
    /// Create a new arithmetic evaluator
    pub fn new(config: ArithmeticEvaluatorConfig) -> Self {
        let generator = ArithmeticGenerator::new(config.arithmetic_config.clone());
        let encoder = ProblemEncoder::new(config.encoding_config.clone());
        
        Self {
            config,
            generator,
            encoder,
        }
    }
    
    /// Evaluate a network on a specific set of problems
    pub fn evaluate_on_problems(&self, network: &Network, problems: &[MathProblem]) -> Result<ArithmeticResults> {
        let mut exact_correct = 0;
        let mut total_digit_accuracy = 0.0;
        let mut operation_accuracy: HashMap<String, (usize, usize)> = HashMap::new();
        let mut problem_results = Vec::new();
        
        for problem in problems {
            // Encode the problem
            let input = self.encoder.encode_problem(problem)?;
            
            // Get network output
            let output = network.activate(&input)?;
            
            // Decode the result
            let predicted_result = self.encoder.decode_result(&output)?;
            
            // Calculate accuracies
            let is_exact = predicted_result == problem.result;
            let digit_acc = self.calculate_digit_accuracy(predicted_result, problem.result);
            
            if is_exact {
                exact_correct += 1;
            }
            
            total_digit_accuracy += digit_acc;
            
            // Track per-operation accuracy
            let op_key = problem.operation.symbol().to_string();
            let (correct, total) = operation_accuracy.entry(op_key).or_insert((0, 0));
            if is_exact {
                *correct += 1;
            }
            *total += 1;
            
            problem_results.push(ProblemResult {
                problem: problem.clone(),
                predicted: predicted_result,
                exact_match: is_exact,
                digit_accuracy: digit_acc,
            });
        }
        
        let exact_accuracy = exact_correct as f64 / problems.len() as f64;
        let average_digit_accuracy = total_digit_accuracy / problems.len() as f64;
        
        Ok(ArithmeticResults {
            exact_accuracy,
            average_digit_accuracy,
            exact_correct,
            total_problems: problems.len(),
            operation_accuracy,
            problem_results,
        })
    }
    
    /// Calculate digit-by-digit accuracy between predicted and actual results
    fn calculate_digit_accuracy(&self, predicted: i32, actual: i32) -> f64 {
        let pred_str = predicted.abs().to_string();
        let actual_str = actual.abs().to_string();
        
        let max_len = pred_str.len().max(actual_str.len());
        let mut correct_digits = 0;
        
        for i in 0..max_len {
            let pred_digit = pred_str.chars().nth_back(i).unwrap_or('0');
            let actual_digit = actual_str.chars().nth_back(i).unwrap_or('0');
            
            if pred_digit == actual_digit {
                correct_digits += 1;
            }
        }
        
        // Also check sign
        if (predicted >= 0) == (actual >= 0) {
            correct_digits += 1;
        }
        
        correct_digits as f64 / (max_len + 1) as f64
    }
    
    /// Calculate complexity penalty based on network structure
    fn calculate_complexity_penalty(&self, genome: &Genome) -> f64 {
        let node_count = genome.nodes.len() as f64;
        let connection_count = genome.connections.iter().filter(|c| c.enabled).count() as f64;
        
        self.config.complexity_penalty * (node_count + connection_count * 0.5)
    }
    
    /// Get configuration
    pub fn get_config(&self) -> &ArithmeticEvaluatorConfig {
        &self.config
    }
    
    /// Encode a math problem for network input
    pub fn encode_problem(&self, problem: &MathProblem) -> Result<Vec<f64>> {
        self.encoder.encode_problem(problem)
    }
    
    /// Decode network output to a result
    pub fn decode_result(&self, output: &[f64]) -> Result<i32> {
        self.encoder.decode_result(output)
    }
}

impl FitnessEvaluator for ArithmeticEvaluator {
    fn evaluate(&self, genome: &Genome) -> Result<f64> {
        // Create network from genome
        let network = Network::from_genome(genome)?;
        
        // Generate problems for evaluation
        let mut generator = self.generator.clone();
        let problems = generator.generate_batch(self.config.problems_per_evaluation);
        
        if problems.is_empty() {
            return Ok(0.0);
        }
        
        // Evaluate on the problems
        let results = self.evaluate_on_problems(&network, &problems)?;
        
        // Calculate composite fitness
        let mut fitness = 0.0;
        
        // Exact accuracy component
        fitness += self.config.exact_weight * results.exact_accuracy;
        
        // Partial accuracy component
        fitness += self.config.partial_weight * results.average_digit_accuracy;
        
        // Perfect problems bonus
        if results.exact_accuracy >= 1.0 {
            fitness += self.config.perfect_bonus;
        }
        
        // Complexity penalty
        fitness -= self.calculate_complexity_penalty(genome);
        
        Ok(fitness.max(0.0))
    }
    
    fn input_size(&self) -> usize {
        self.encoder.input_length()
    }
    
    fn output_size(&self) -> usize {
        self.encoder.output_length()
    }
    
    fn max_fitness(&self) -> f64 {
        self.config.exact_weight + self.config.partial_weight + self.config.perfect_bonus
    }
}

/// Results of arithmetic evaluation
#[derive(Debug, Clone)]
pub struct ArithmeticResults {
    /// Percentage of exactly correct answers
    pub exact_accuracy: f64,
    /// Average digit-by-digit accuracy
    pub average_digit_accuracy: f64,
    /// Number of exactly correct answers
    pub exact_correct: usize,
    /// Total number of problems tested
    pub total_problems: usize,
    /// Accuracy per operation type
    pub operation_accuracy: HashMap<String, (usize, usize)>,
    /// Individual problem results
    pub problem_results: Vec<ProblemResult>,
}

impl ArithmeticResults {
    /// Print detailed results
    pub fn print_detailed(&self) {
        println!("Arithmetic Evaluation Results:");
        println!("  Exact Accuracy: {:.1}% ({}/{})", 
                self.exact_accuracy * 100.0, 
                self.exact_correct, 
                self.total_problems);
        println!("  Digit Accuracy: {:.1}%", self.average_digit_accuracy * 100.0);
        
        println!("\nPer-Operation Accuracy:");
        for (op, (correct, total)) in &self.operation_accuracy {
            let accuracy = *correct as f64 / *total as f64;
            println!("  {}: {:.1}% ({}/{})", op, accuracy * 100.0, correct, total);
        }
        
        // Show some example problems
        println!("\nExample Problems:");
        for (i, result) in self.problem_results.iter().take(5).enumerate() {
            let status = if result.exact_match { "✓" } else { "✗" };
            println!("  {}: {} -> {} (expected: {}) {}", 
                    i + 1,
                    result.problem.to_string().split(" = ").next().unwrap(),
                    result.predicted,
                    result.problem.result,
                    status);
        }
    }
    
    /// Get accuracy for a specific operation
    pub fn operation_accuracy(&self, operation: &str) -> Option<f64> {
        self.operation_accuracy.get(operation)
            .map(|(correct, total)| *correct as f64 / *total as f64)
    }
    
    /// Check if all problems were solved correctly
    pub fn is_perfect(&self) -> bool {
        self.exact_accuracy >= 1.0
    }
}

/// Result for a single problem
#[derive(Debug, Clone)]
pub struct ProblemResult {
    /// The original problem
    pub problem: MathProblem,
    /// Network's predicted answer
    pub predicted: i32,
    /// Whether prediction exactly matches expected result
    pub exact_match: bool,
    /// Digit-by-digit accuracy score
    pub digit_accuracy: f64,
}

/// Curriculum-based evaluator that progressively increases difficulty
pub struct CurriculumEvaluator {
    evaluators: Vec<ArithmeticEvaluator>,
    current_level: usize,
    success_threshold: f64,
    problems_per_level: usize,
}

impl CurriculumEvaluator {
    /// Create a new curriculum evaluator
    pub fn new(
        configs: Vec<ArithmeticEvaluatorConfig>,
        success_threshold: f64,
        problems_per_level: usize,
    ) -> Self {
        let evaluators = configs.into_iter()
            .map(ArithmeticEvaluator::new)
            .collect();
        
        Self {
            evaluators,
            current_level: 0,
            success_threshold,
            problems_per_level,
        }
    }
    
    /// Evaluate genome and potentially advance curriculum level
    pub fn evaluate_with_progression(&mut self, genome: &Genome) -> Result<(f64, bool)> {
        if self.current_level >= self.evaluators.len() {
            return Ok((0.0, false));
        }
        
        let evaluator = &self.evaluators[self.current_level];
        let fitness = evaluator.evaluate(genome)?;
        
        // Check if we should advance to next level
        let should_advance = fitness >= self.success_threshold;
        
        if should_advance && self.current_level < self.evaluators.len() - 1 {
            self.current_level += 1;
        }
        
        Ok((fitness, should_advance))
    }
    
    /// Get current curriculum level
    pub fn current_level(&self) -> usize {
        self.current_level
    }
    
    /// Check if curriculum is complete
    pub fn is_complete(&self) -> bool {
        self.current_level >= self.evaluators.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neat::genome::Genome;
    
    #[test]
    fn test_arithmetic_evaluator_creation() {
        let config = ArithmeticEvaluatorConfig::default();
        let evaluator = ArithmeticEvaluator::new(config);
        
        // Check that sizes are reasonable (actual values depend on encoding config)
        assert!(evaluator.input_size() > 0);
        assert!(evaluator.output_size() > 0);
        println!("Input size: {}, Output size: {}", evaluator.input_size(), evaluator.output_size());
    }
    
    #[test]
    fn test_evaluator_with_genome() -> Result<()> {
        let config = ArithmeticEvaluatorConfig {
            problems_per_evaluation: 10,
            ..Default::default()
        };
        
        let evaluator = ArithmeticEvaluator::new(config);
        
        // Create a simple genome
        let genome = Genome::new(0, evaluator.input_size(), evaluator.output_size());
        
        // Evaluate fitness (should be low but not error)
        let fitness = evaluator.evaluate(&genome)?;
        assert!(fitness >= 0.0);
        assert!(fitness <= evaluator.max_fitness());
        
        Ok(())
    }
    
    #[test]
    fn test_digit_accuracy_calculation() {
        let config = ArithmeticEvaluatorConfig::default();
        let evaluator = ArithmeticEvaluator::new(config);
        
        // Perfect match
        assert_eq!(evaluator.calculate_digit_accuracy(123, 123), 1.0);
        
        // One digit off
        let accuracy = evaluator.calculate_digit_accuracy(123, 124);
        assert!(accuracy > 0.5 && accuracy < 1.0);
        
        // Completely wrong
        let accuracy2 = evaluator.calculate_digit_accuracy(123, 999);
        assert!(accuracy2 < 0.5);
    }
    
    #[test] 
    fn test_curriculum_evaluator() -> Result<()> {
        let configs = vec![
            ArithmeticEvaluatorConfig {
                problems_per_evaluation: 5,
                ..Default::default()
            },
            ArithmeticEvaluatorConfig {
                problems_per_evaluation: 10,
                ..Default::default()
            },
        ];
        
        let mut curriculum = CurriculumEvaluator::new(configs, 0.8, 20);
        
        assert_eq!(curriculum.current_level(), 0);
        assert!(!curriculum.is_complete());
        
        let evaluator = ArithmeticEvaluator::new(ArithmeticEvaluatorConfig::default());
        let genome = Genome::new(0, evaluator.input_size(), evaluator.output_size());
        let (fitness, advanced) = curriculum.evaluate_with_progression(&genome)?;
        
        assert!(fitness >= 0.0);
        // Unlikely to advance with random genome, but shouldn't error
        
        Ok(())
    }
}