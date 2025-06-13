//! Arithmetic problem generation and management
//!
//! This module provides comprehensive arithmetic problem generation for
//! training NEAT networks on mathematical reasoning tasks.

use super::{Operation, DifficultyLevel, MathProblem};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use std::collections::HashMap;

/// Configuration for arithmetic problem generation
#[derive(Debug, Clone)]
pub struct ArithmeticConfig {
    /// Operations to include in problems
    pub operations: Vec<Operation>,
    /// Difficulty level for problems
    pub difficulty: DifficultyLevel,
    /// Random seed for reproducible generation
    pub random_seed: Option<u64>,
    /// Maximum result value (to prevent overflow issues)
    pub max_result: i32,
    /// Allow negative results
    pub allow_negative: bool,
    /// Ensure results are within reasonable bounds
    pub bounded_results: bool,
}

impl Default for ArithmeticConfig {
    fn default() -> Self {
        Self {
            operations: vec![Operation::Add, Operation::Subtract],
            difficulty: DifficultyLevel::SingleDigit,
            random_seed: Some(42),
            max_result: 999,
            allow_negative: false,
            bounded_results: true,
        }
    }
}

/// Generator for arithmetic problems
#[derive(Clone)]
pub struct ArithmeticGenerator {
    config: ArithmeticConfig,
    rng: SmallRng,
    generated_count: usize,
}

impl ArithmeticGenerator {
    /// Create a new arithmetic generator
    pub fn new(config: ArithmeticConfig) -> Self {
        let rng = match config.random_seed {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => SmallRng::from_entropy(),
        };
        
        Self {
            config,
            rng,
            generated_count: 0,
        }
    }
    
    /// Generate a single arithmetic problem
    pub fn generate_problem(&mut self) -> Option<MathProblem> {
        for _ in 0..100 { // Try up to 100 times to generate a valid problem
            let operation = self.config.operations[self.rng.gen_range(0..self.config.operations.len())];
            
            let (operand1, operand2) = self.generate_operands(operation);
            
            if let Some(problem) = MathProblem::new(operand1, operand2, operation) {
                // Check if result meets our constraints
                if self.is_valid_result(&problem) {
                    self.generated_count += 1;
                    return Some(problem);
                }
            }
        }
        
        None // Failed to generate valid problem
    }
    
    /// Generate a batch of arithmetic problems
    pub fn generate_batch(&mut self, count: usize) -> Vec<MathProblem> {
        (0..count)
            .filter_map(|_| self.generate_problem())
            .collect()
    }
    
    /// Generate operands based on difficulty level and operation
    fn generate_operands(&mut self, operation: Operation) -> (i32, i32) {
        let min_val = self.config.difficulty.min_value();
        let max_val = self.config.difficulty.max_value();
        
        match operation {
            Operation::Add => {
                let operand1 = self.rng.gen_range(min_val..=max_val);
                let operand2 = self.rng.gen_range(min_val..=max_val);
                (operand1, operand2)
            },
            Operation::Subtract => {
                if self.config.allow_negative {
                    let operand1 = self.rng.gen_range(min_val..=max_val);
                    let operand2 = self.rng.gen_range(min_val..=max_val);
                    (operand1, operand2)
                } else {
                    // Ensure positive result
                    let operand2 = self.rng.gen_range(min_val..=max_val);
                    let operand1 = self.rng.gen_range(operand2..=max_val);
                    (operand1, operand2)
                }
            },
            Operation::Multiply => {
                // For multiplication, use smaller numbers to avoid overflow
                let adjusted_max = (max_val as f64).sqrt() as i32;
                let operand1 = self.rng.gen_range(min_val..=adjusted_max.max(min_val));
                let operand2 = self.rng.gen_range(min_val..=adjusted_max.max(min_val));
                (operand1, operand2)
            },
            Operation::Divide => {
                // Generate divisor first, then dividend
                let divisor = self.rng.gen_range(1..=max_val.min(20)); // Keep divisors reasonable
                let quotient = self.rng.gen_range(min_val..=max_val);
                let dividend = divisor * quotient;
                (dividend, divisor)
            }
        }
    }
    
    /// Check if a problem's result meets our constraints
    fn is_valid_result(&self, problem: &MathProblem) -> bool {
        // Check negative constraint
        if !self.config.allow_negative && problem.result < 0 {
            return false;
        }
        
        // Check bounded results
        if self.config.bounded_results && problem.result.abs() > self.config.max_result {
            return false;
        }
        
        true
    }
    
    /// Get statistics about generated problems
    pub fn get_stats(&self) -> GeneratorStats {
        GeneratorStats {
            problems_generated: self.generated_count,
            config: self.config.clone(),
        }
    }
    
    /// Reset the generator
    pub fn reset(&mut self) {
        self.generated_count = 0;
        if let Some(seed) = self.config.random_seed {
            self.rng = SmallRng::seed_from_u64(seed);
        }
    }
}

/// Statistics about problem generation
#[derive(Debug, Clone)]
pub struct GeneratorStats {
    /// Total problems generated
    pub problems_generated: usize,
    /// Configuration used
    pub config: ArithmeticConfig,
}

/// Curriculum for progressive learning
pub struct ArithmeticCurriculum {
    levels: Vec<ArithmeticConfig>,
    current_level: usize,
}

impl ArithmeticCurriculum {
    /// Create a standard curriculum progressing through difficulty levels
    pub fn standard() -> Self {
        let levels = vec![
            // Level 1: Single-digit addition
            ArithmeticConfig {
                operations: vec![Operation::Add],
                difficulty: DifficultyLevel::SingleDigit,
                allow_negative: false,
                ..Default::default()
            },
            // Level 2: Single-digit addition and subtraction
            ArithmeticConfig {
                operations: vec![Operation::Add, Operation::Subtract],
                difficulty: DifficultyLevel::SingleDigit,
                allow_negative: false,
                ..Default::default()
            },
            // Level 3: Two-digit addition
            ArithmeticConfig {
                operations: vec![Operation::Add],
                difficulty: DifficultyLevel::TwoDigit,
                allow_negative: false,
                max_result: 200,
                ..Default::default()
            },
            // Level 4: Two-digit addition and subtraction
            ArithmeticConfig {
                operations: vec![Operation::Add, Operation::Subtract],
                difficulty: DifficultyLevel::TwoDigit,
                allow_negative: false,
                ..Default::default()
            },
            // Level 5: Single-digit multiplication
            ArithmeticConfig {
                operations: vec![Operation::Multiply],
                difficulty: DifficultyLevel::SingleDigit,
                allow_negative: false,
                ..Default::default()
            },
            // Level 6: All operations on single digits
            ArithmeticConfig {
                operations: Operation::all().to_vec(),
                difficulty: DifficultyLevel::SingleDigit,
                allow_negative: false,
                ..Default::default()
            },
            // Level 7: All operations on mixed difficulty
            ArithmeticConfig {
                operations: Operation::all().to_vec(),
                difficulty: DifficultyLevel::TwoDigit,
                allow_negative: true,
                max_result: 10000,
                ..Default::default()
            },
        ];
        
        Self {
            levels,
            current_level: 0,
        }
    }
    
    /// Get the current level configuration
    pub fn current_config(&self) -> Option<&ArithmeticConfig> {
        self.levels.get(self.current_level)
    }
    
    /// Advance to the next level
    pub fn next_level(&mut self) -> bool {
        if self.current_level < self.levels.len() - 1 {
            self.current_level += 1;
            true
        } else {
            false
        }
    }
    
    /// Reset to first level
    pub fn reset(&mut self) {
        self.current_level = 0;
    }
    
    /// Get current level number (0-based)
    pub fn level(&self) -> usize {
        self.current_level
    }
    
    /// Get total number of levels
    pub fn total_levels(&self) -> usize {
        self.levels.len()
    }
    
    /// Check if at final level
    pub fn is_final_level(&self) -> bool {
        self.current_level >= self.levels.len() - 1
    }
}

/// Collection of standard arithmetic problems for testing
pub struct ArithmeticTestSuite {
    problems: HashMap<String, Vec<MathProblem>>,
}

impl ArithmeticTestSuite {
    /// Create standard test suite
    pub fn standard() -> Self {
        let mut problems = HashMap::new();
        
        // Single-digit addition
        let single_add = vec![
            MathProblem::new(1, 1, Operation::Add).unwrap(),
            MathProblem::new(5, 3, Operation::Add).unwrap(),
            MathProblem::new(9, 9, Operation::Add).unwrap(),
            MathProblem::new(0, 7, Operation::Add).unwrap(),
        ];
        problems.insert("single_digit_addition".to_string(), single_add);
        
        // Carry addition
        let carry_add = vec![
            MathProblem::new(19, 15, Operation::Add).unwrap(),
            MathProblem::new(67, 48, Operation::Add).unwrap(),
            MathProblem::new(99, 99, Operation::Add).unwrap(),
        ];
        problems.insert("carry_addition".to_string(), carry_add);
        
        // Multiplication tables
        let mult_table = vec![
            MathProblem::new(2, 3, Operation::Multiply).unwrap(),
            MathProblem::new(7, 8, Operation::Multiply).unwrap(),
            MathProblem::new(9, 9, Operation::Multiply).unwrap(),
            MathProblem::new(6, 4, Operation::Multiply).unwrap(),
        ];
        problems.insert("multiplication_table".to_string(), mult_table);
        
        Self { problems }
    }
    
    /// Get problems for a specific test category
    pub fn get_problems(&self, category: &str) -> Option<&Vec<MathProblem>> {
        self.problems.get(category)
    }
    
    /// Get all test categories
    pub fn categories(&self) -> Vec<&String> {
        self.problems.keys().collect()
    }
    
    /// Get total number of test problems
    pub fn total_problems(&self) -> usize {
        self.problems.values().map(|v| v.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_arithmetic_generator() {
        let config = ArithmeticConfig::default();
        let mut generator = ArithmeticGenerator::new(config);
        
        let problem = generator.generate_problem().unwrap();
        assert!(problem.operand1 <= 9);
        assert!(problem.operand2 <= 9);
        assert!(matches!(problem.operation, Operation::Add | Operation::Subtract));
        
        let batch = generator.generate_batch(10);
        assert_eq!(batch.len(), 10);
    }
    
    #[test]
    fn test_arithmetic_curriculum() {
        let mut curriculum = ArithmeticCurriculum::standard();
        
        assert_eq!(curriculum.level(), 0);
        assert_eq!(curriculum.total_levels(), 7);
        assert!(!curriculum.is_final_level());
        
        let config = curriculum.current_config().unwrap();
        assert_eq!(config.operations, vec![Operation::Add]);
        assert_eq!(config.difficulty, DifficultyLevel::SingleDigit);
        
        curriculum.next_level();
        assert_eq!(curriculum.level(), 1);
        
        let config2 = curriculum.current_config().unwrap();
        assert_eq!(config2.operations, vec![Operation::Add, Operation::Subtract]);
    }
    
    #[test]
    fn test_bounded_generation() {
        let config = ArithmeticConfig {
            operations: vec![Operation::Multiply],
            difficulty: DifficultyLevel::SingleDigit,
            max_result: 50,
            bounded_results: true,
            ..Default::default()
        };
        
        let mut generator = ArithmeticGenerator::new(config);
        
        for _ in 0..20 {
            if let Some(problem) = generator.generate_problem() {
                assert!(problem.result <= 50);
            }
        }
    }
    
    #[test]
    fn test_test_suite() {
        let suite = ArithmeticTestSuite::standard();
        
        assert!(suite.get_problems("single_digit_addition").is_some());
        assert!(suite.get_problems("nonexistent").is_none());
        assert!(suite.total_problems() > 0);
        
        let categories = suite.categories();
        assert!(categories.contains(&&"single_digit_addition".to_string()));
    }
}