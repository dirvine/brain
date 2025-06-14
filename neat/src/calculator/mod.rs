//! Calculator module for NEAT mathematical reasoning experiments
//!
//! This module provides a comprehensive framework for testing NEAT's ability
//! to learn mathematical operations through evolution. It offers:
//!
//! - Infinite training data generation
//! - Multiple number encoding schemes
//! - Progressive difficulty curricula
//! - Comprehensive evaluation metrics
//! - Mathematical reasoning benchmarks

pub mod arithmetic;
pub mod encoding;
pub mod evaluator;
pub mod algebra;
pub mod algebra_encoding;
pub mod algebra_evaluator;
pub mod sequences;
pub mod sequence_evaluator;
pub mod modules;
pub mod arithmetic_modules;
pub mod algebra_modules;
pub mod discovery;
pub mod conjecture;
pub mod calculus;
pub mod trigonometry;
pub mod statistics;
pub mod discrete_math;

pub use arithmetic::*;
pub use encoding::*;
pub use evaluator::*;
pub use algebra::*;
pub use algebra_encoding::*;
pub use algebra_evaluator::*;
pub use sequences::*;
pub use sequence_evaluator::*;
pub use modules::*;
pub use arithmetic_modules::*;
pub use algebra_modules::*;
pub use discovery::*;
pub use conjecture::*;
pub use calculus::*;
pub use trigonometry::*;
pub use statistics::*;
pub use discrete_math::*;

/// Mathematical operations that can be learned
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    /// Addition: a + b = c
    Add,
    /// Subtraction: a - b = c  
    Subtract,
    /// Multiplication: a * b = c
    Multiply,
    /// Division: a / b = c (integer division)
    Divide,
}

impl Operation {
    /// Get all supported operations
    pub fn all() -> &'static [Operation] {
        &[Operation::Add, Operation::Subtract, Operation::Multiply, Operation::Divide]
    }
    
    /// Get operation symbol
    pub fn symbol(&self) -> char {
        match self {
            Operation::Add => '+',
            Operation::Subtract => '-',
            Operation::Multiply => '*',
            Operation::Divide => '/',
        }
    }
    
    /// Perform the operation
    pub fn apply(&self, a: i32, b: i32) -> Option<i32> {
        match self {
            Operation::Add => a.checked_add(b),
            Operation::Subtract => a.checked_sub(b),
            Operation::Multiply => a.checked_mul(b),
            Operation::Divide => {
                if b != 0 {
                    Some(a / b)
                } else {
                    None
                }
            }
        }
    }
    
    /// Get operation difficulty (for curriculum ordering)
    pub fn difficulty(&self) -> u8 {
        match self {
            Operation::Add => 1,
            Operation::Subtract => 2,
            Operation::Multiply => 3,
            Operation::Divide => 4,
        }
    }
}

/// Difficulty levels for mathematical problems
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifficultyLevel {
    /// Single digits: 0-9
    SingleDigit,
    /// Two digits: 10-99
    TwoDigit,
    /// Three digits: 100-999
    ThreeDigit,
    /// Four digits: 1000-9999
    FourDigit,
    /// Mixed difficulty
    Mixed,
}

impl DifficultyLevel {
    /// Get the maximum value for this difficulty level
    pub fn max_value(&self) -> i32 {
        match self {
            DifficultyLevel::SingleDigit => 9,
            DifficultyLevel::TwoDigit => 99,
            DifficultyLevel::ThreeDigit => 999,
            DifficultyLevel::FourDigit => 9999,
            DifficultyLevel::Mixed => 9999,
        }
    }
    
    /// Get the minimum value for this difficulty level
    pub fn min_value(&self) -> i32 {
        match self {
            DifficultyLevel::SingleDigit => 0,
            DifficultyLevel::TwoDigit => 10,
            DifficultyLevel::ThreeDigit => 100,
            DifficultyLevel::FourDigit => 1000,
            DifficultyLevel::Mixed => 0,
        }
    }
    
    /// Get all difficulty levels in order
    pub fn all() -> &'static [DifficultyLevel] {
        &[
            DifficultyLevel::SingleDigit,
            DifficultyLevel::TwoDigit,
            DifficultyLevel::ThreeDigit,
            DifficultyLevel::FourDigit,
            DifficultyLevel::Mixed,
        ]
    }
}

/// A mathematical problem for NEAT to solve
#[derive(Debug, Clone, PartialEq)]
pub struct MathProblem {
    /// First operand
    pub operand1: i32,
    /// Second operand
    pub operand2: i32,
    /// Mathematical operation
    pub operation: Operation,
    /// Expected result
    pub result: i32,
}

impl MathProblem {
    /// Create a new math problem
    pub fn new(operand1: i32, operand2: i32, operation: Operation) -> Option<Self> {
        if let Some(result) = operation.apply(operand1, operand2) {
            Some(Self {
                operand1,
                operand2,
                operation,
                result,
            })
        } else {
            None
        }
    }
    
    /// Get the problem as a string (e.g., "5 + 3 = 8")
    pub fn to_string(&self) -> String {
        format!("{} {} {} = {}", 
                self.operand1, 
                self.operation.symbol(), 
                self.operand2, 
                self.result)
    }
    
    /// Get the difficulty level of this problem
    pub fn difficulty(&self) -> DifficultyLevel {
        let max_operand = self.operand1.max(self.operand2);
        let result_abs = self.result.abs();
        let max_value = max_operand.max(result_abs);
        
        match max_value {
            0..=9 => DifficultyLevel::SingleDigit,
            10..=99 => DifficultyLevel::TwoDigit,
            100..=999 => DifficultyLevel::ThreeDigit,
            1000..=9999 => DifficultyLevel::FourDigit,
            _ => DifficultyLevel::Mixed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_operations() {
        assert_eq!(Operation::Add.apply(5, 3), Some(8));
        assert_eq!(Operation::Subtract.apply(10, 4), Some(6));
        assert_eq!(Operation::Multiply.apply(7, 6), Some(42));
        assert_eq!(Operation::Divide.apply(15, 3), Some(5));
        assert_eq!(Operation::Divide.apply(10, 0), None); // Division by zero
    }
    
    #[test]
    fn test_difficulty_levels() {
        assert_eq!(DifficultyLevel::SingleDigit.max_value(), 9);
        assert_eq!(DifficultyLevel::TwoDigit.min_value(), 10);
    }
    
    #[test]
    fn test_math_problem() {
        let problem = MathProblem::new(5, 3, Operation::Add).unwrap();
        assert_eq!(problem.result, 8);
        assert_eq!(problem.to_string(), "5 + 3 = 8");
        assert_eq!(problem.difficulty(), DifficultyLevel::SingleDigit);
        
        let problem2 = MathProblem::new(15, 25, Operation::Add).unwrap();
        assert_eq!(problem2.difficulty(), DifficultyLevel::TwoDigit);
    }
    
    #[test]
    fn test_invalid_problems() {
        assert!(MathProblem::new(10, 0, Operation::Divide).is_none());
    }
}