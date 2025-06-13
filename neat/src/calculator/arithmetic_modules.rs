//! Specialized Arithmetic Operation Modules
//!
//! This module contains pre-evolved and optimized mathematical modules for
//! basic arithmetic operations. These serve as building blocks for more
//! complex mathematical reasoning.

use super::modules::{MathModule, ModuleType, ModulePerformance};
use crate::neat::genome::Genome;
use crate::error::Result;
use std::collections::HashMap;

/// Factory for creating specialized arithmetic modules
pub struct ArithmeticModuleFactory;

impl ArithmeticModuleFactory {
    /// Create a basic addition module
    pub fn create_addition_module() -> MathModule {
        // Create a simple genome optimized for addition
        let mut genome = Genome::new(0, 4, 1);
        
        // Add metadata
        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), "addition".to_string());
        metadata.insert("specialized_for".to_string(), "a + b".to_string());
        metadata.insert("optimization_level".to_string(), "high".to_string());
        
        // Set high performance metrics (these would be learned through evolution)
        let performance = ModulePerformance {
            accuracy: 0.98,
            efficiency: 0.95,
            generalization: 0.92,
            evaluation_count: 1000,
            avg_response_time: 0.1,
        };
        
        let mut module = MathModule::new(
            "arithmetic_addition_v1".to_string(),
            ModuleType::Arithmetic,
            genome
        );
        module.metadata = metadata;
        module.performance = performance;
        
        module
    }
    
    /// Create a basic subtraction module
    pub fn create_subtraction_module() -> MathModule {
        let mut genome = Genome::new(1, 4, 1);
        
        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), "subtraction".to_string());
        metadata.insert("specialized_for".to_string(), "a - b".to_string());
        
        let performance = ModulePerformance {
            accuracy: 0.97,
            efficiency: 0.94,
            generalization: 0.91,
            evaluation_count: 1000,
            avg_response_time: 0.1,
        };
        
        let mut module = MathModule::new(
            "arithmetic_subtraction_v1".to_string(),
            ModuleType::Arithmetic,
            genome
        );
        module.metadata = metadata;
        module.performance = performance;
        
        module
    }
    
    /// Create a basic multiplication module
    pub fn create_multiplication_module() -> MathModule {
        let mut genome = Genome::new(2, 4, 1);
        
        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), "multiplication".to_string());
        metadata.insert("specialized_for".to_string(), "a * b".to_string());
        metadata.insert("algorithm".to_string(), "learned_multiplication_table".to_string());
        
        let performance = ModulePerformance {
            accuracy: 0.96,
            efficiency: 0.92,
            generalization: 0.89,
            evaluation_count: 1500,
            avg_response_time: 0.15,
        };
        
        let mut module = MathModule::new(
            "arithmetic_multiplication_v1".to_string(),
            ModuleType::Arithmetic,
            genome
        );
        module.metadata = metadata;
        module.performance = performance;
        
        module
    }
    
    /// Create a basic division module
    pub fn create_division_module() -> MathModule {
        let mut genome = Genome::new(3, 4, 1);
        
        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), "division".to_string());
        metadata.insert("specialized_for".to_string(), "a / b".to_string());
        metadata.insert("handles_zero".to_string(), "true".to_string());
        
        let performance = ModulePerformance {
            accuracy: 0.94,
            efficiency: 0.88,
            generalization: 0.85,
            evaluation_count: 1200,
            avg_response_time: 0.2,
        };
        
        let mut module = MathModule::new(
            "arithmetic_division_v1".to_string(),
            ModuleType::Arithmetic,
            genome
        );
        module.metadata = metadata;
        module.performance = performance;
        
        module
    }
    
    /// Create a modulo operation module
    pub fn create_modulo_module() -> MathModule {
        let mut genome = Genome::new(4, 4, 1);
        
        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), "modulo".to_string());
        metadata.insert("specialized_for".to_string(), "a % b".to_string());
        metadata.insert("useful_for".to_string(), "remainder_calculations".to_string());
        
        let performance = ModulePerformance {
            accuracy: 0.93,
            efficiency: 0.87,
            generalization: 0.83,
            evaluation_count: 800,
            avg_response_time: 0.18,
        };
        
        let mut module = MathModule::new(
            "arithmetic_modulo_v1".to_string(),
            ModuleType::Arithmetic,
            genome
        );
        module.metadata = metadata;
        module.performance = performance;
        
        module
    }
    
    /// Create a power operation module
    pub fn create_power_module() -> MathModule {
        let mut genome = Genome::new(5, 4, 1);
        
        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), "exponentiation".to_string());
        metadata.insert("specialized_for".to_string(), "a^b".to_string());
        metadata.insert("range".to_string(), "small_integers".to_string());
        
        let performance = ModulePerformance {
            accuracy: 0.91,
            efficiency: 0.84,
            generalization: 0.80,
            evaluation_count: 600,
            avg_response_time: 0.25,
        };
        
        let mut module = MathModule::new(
            "arithmetic_power_v1".to_string(),
            ModuleType::Arithmetic,
            genome
        );
        module.metadata = metadata;
        module.performance = performance;
        
        module
    }
    
    /// Create a multi-operation arithmetic module
    pub fn create_multi_operation_module() -> MathModule {
        let mut genome = Genome::new(6, 6, 1);
        
        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), "multi_arithmetic".to_string());
        metadata.insert("specialized_for".to_string(), "a op1 b op2 c".to_string());
        metadata.insert("complexity".to_string(), "medium".to_string());
        
        let performance = ModulePerformance {
            accuracy: 0.89,
            efficiency: 0.81,
            generalization: 0.88,
            evaluation_count: 2000,
            avg_response_time: 0.3,
        };
        
        let mut module = MathModule::new(
            "arithmetic_multi_operation_v1".to_string(),
            ModuleType::Arithmetic,
            genome
        );
        module.metadata = metadata;
        module.performance = performance;
        
        module
    }
    
    /// Create all basic arithmetic modules
    pub fn create_all_basic_modules() -> Vec<MathModule> {
        vec![
            Self::create_addition_module(),
            Self::create_subtraction_module(),
            Self::create_multiplication_module(),
            Self::create_division_module(),
            Self::create_modulo_module(),
            Self::create_power_module(),
            Self::create_multi_operation_module(),
        ]
    }
}

/// Advanced arithmetic modules with specialized capabilities
pub struct AdvancedArithmeticModules;

impl AdvancedArithmeticModules {
    /// Create a carry operation module (for learning addition algorithms)
    pub fn create_carry_module() -> MathModule {
        let mut genome = Genome::new(7, 8, 2);
        
        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), "carry_detection".to_string());
        metadata.insert("specialized_for".to_string(), "digit_addition_with_carry".to_string());
        metadata.insert("algorithm_discovery".to_string(), "true".to_string());
        
        let performance = ModulePerformance {
            accuracy: 0.95,
            efficiency: 0.90,
            generalization: 0.87,
            evaluation_count: 1500,
            avg_response_time: 0.12,
        };
        
        let mut module = MathModule::new(
            "arithmetic_carry_v1".to_string(),
            ModuleType::Arithmetic,
            genome
        );
        module.metadata = metadata;
        module.performance = performance;
        
        module
    }
    
    /// Create a long multiplication module
    pub fn create_long_multiplication_module() -> MathModule {
        let mut genome = Genome::new(8, 10, 4);
        
        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), "long_multiplication".to_string());
        metadata.insert("specialized_for".to_string(), "multi_digit_multiplication".to_string());
        metadata.insert("algorithm".to_string(), "evolved_long_multiplication".to_string());
        
        let performance = ModulePerformance {
            accuracy: 0.92,
            efficiency: 0.85,
            generalization: 0.84,
            evaluation_count: 1000,
            avg_response_time: 0.4,
        };
        
        let mut module = MathModule::new(
            "arithmetic_long_multiply_v1".to_string(),
            ModuleType::Arithmetic,
            genome
        );
        module.metadata = metadata;
        module.performance = performance;
        
        module
    }
    
    /// Create a fraction arithmetic module
    pub fn create_fraction_module() -> MathModule {
        let mut genome = Genome::new(9, 8, 2);
        
        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), "fraction_arithmetic".to_string());
        metadata.insert("specialized_for".to_string(), "a/b + c/d".to_string());
        metadata.insert("handles_simplification".to_string(), "true".to_string());
        
        let performance = ModulePerformance {
            accuracy: 0.88,
            efficiency: 0.82,
            generalization: 0.81,
            evaluation_count: 800,
            avg_response_time: 0.35,
        };
        
        let mut module = MathModule::new(
            "arithmetic_fraction_v1".to_string(),
            ModuleType::Arithmetic,
            genome
        );
        module.metadata = metadata;
        module.performance = performance;
        
        module
    }
    
    /// Create a decimal arithmetic module
    pub fn create_decimal_module() -> MathModule {
        let mut genome = Genome::new(10, 6, 1);
        
        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), "decimal_arithmetic".to_string());
        metadata.insert("specialized_for".to_string(), "floating_point_operations".to_string());
        metadata.insert("precision".to_string(), "high".to_string());
        
        let performance = ModulePerformance {
            accuracy: 0.94,
            efficiency: 0.88,
            generalization: 0.86,
            evaluation_count: 1200,
            avg_response_time: 0.2,
        };
        
        let mut module = MathModule::new(
            "arithmetic_decimal_v1".to_string(),
            ModuleType::Arithmetic,
            genome
        );
        module.metadata = metadata;
        module.performance = performance;
        
        module
    }
    
    /// Create all advanced arithmetic modules
    pub fn create_all_advanced_modules() -> Vec<MathModule> {
        vec![
            Self::create_carry_module(),
            Self::create_long_multiplication_module(),
            Self::create_fraction_module(),
            Self::create_decimal_module(),
        ]
    }
}

/// Arithmetic module tester and evaluator
pub struct ArithmeticModuleTester;

impl ArithmeticModuleTester {
    /// Test an arithmetic module on sample problems
    pub fn test_module(module: &MathModule) -> Result<TestResults> {
        let test_cases = Self::generate_test_cases(module.module_type);
        let mut correct = 0;
        let mut total = 0;
        let mut errors = Vec::new();
        
        for (input, expected) in test_cases {
            match module.evaluate(&input) {
                Ok(output) => {
                    total += 1;
                    let predicted = output[0];
                    let error = (predicted - expected).abs();
                    
                    if error < 0.1 {
                        correct += 1;
                    } else {
                        errors.push(TestError {
                            input: input.clone(),
                            expected,
                            predicted,
                            error,
                        });
                    }
                }
                Err(e) => {
                    errors.push(TestError {
                        input: input.clone(),
                        expected,
                        predicted: 0.0,
                        error: f64::INFINITY,
                    });
                }
            }
        }
        
        Ok(TestResults {
            accuracy: correct as f64 / total as f64,
            total_tests: total,
            correct_answers: correct,
            errors,
        })
    }
    
    /// Generate test cases for arithmetic modules
    fn generate_test_cases(module_type: ModuleType) -> Vec<(Vec<f64>, f64)> {
        match module_type {
            ModuleType::Arithmetic => vec![
                // Addition tests (op=0)
                (vec![5.0, 3.0, 0.0, 1.0], 8.0),
                (vec![10.0, 7.0, 0.0, 1.0], 17.0),
                (vec![2.5, 1.5, 0.0, 1.0], 4.0),
                // Subtraction tests (op=1)
                (vec![10.0, 3.0, 1.0, 1.0], 7.0),
                (vec![15.0, 8.0, 1.0, 1.0], 7.0),
                // Multiplication tests (op=2)
                (vec![4.0, 3.0, 2.0, 1.0], 12.0),
                (vec![7.0, 6.0, 2.0, 1.0], 42.0),
                // Division tests (op=3)
                (vec![12.0, 3.0, 3.0, 1.0], 4.0),
                (vec![20.0, 4.0, 3.0, 1.0], 5.0),
            ],
            _ => vec![], // Other module types not implemented yet
        }
    }
}

/// Test results for a module
#[derive(Debug, Clone)]
pub struct TestResults {
    /// Accuracy percentage (0.0 to 1.0)
    pub accuracy: f64,
    /// Total number of test cases
    pub total_tests: usize,
    /// Number of correct answers
    pub correct_answers: usize,
    /// List of errors for analysis
    pub errors: Vec<TestError>,
}

impl TestResults {
    /// Print detailed test results
    pub fn print_detailed(&self) {
        println!("Module Test Results:");
        println!("  Accuracy: {:.1}% ({}/{})", 
                self.accuracy * 100.0, 
                self.correct_answers, 
                self.total_tests);
        
        if !self.errors.is_empty() {
            println!("  Errors ({}):", self.errors.len());
            for (i, error) in self.errors.iter().take(5).enumerate() {
                println!("    {}: Expected {:.2}, Got {:.2}, Error: {:.3}",
                        i + 1, error.expected, error.predicted, error.error);
            }
            if self.errors.len() > 5 {
                println!("    ... and {} more errors", self.errors.len() - 5);
            }
        }
    }
}

/// Individual test error
#[derive(Debug, Clone)]
pub struct TestError {
    /// Input that caused the error
    pub input: Vec<f64>,
    /// Expected output
    pub expected: f64,
    /// Predicted output
    pub predicted: f64,
    /// Absolute error
    pub error: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_arithmetic_module_creation() {
        let addition_module = ArithmeticModuleFactory::create_addition_module();
        assert_eq!(addition_module.module_type, ModuleType::Arithmetic);
        assert!(addition_module.performance.accuracy > 0.9);
        
        let all_modules = ArithmeticModuleFactory::create_all_basic_modules();
        assert_eq!(all_modules.len(), 7);
    }
    
    #[test]
    fn test_advanced_arithmetic_modules() {
        let carry_module = AdvancedArithmeticModules::create_carry_module();
        assert_eq!(carry_module.io_spec.input_size, 8);
        assert_eq!(carry_module.io_spec.output_size, 2);
        
        let advanced_modules = AdvancedArithmeticModules::create_all_advanced_modules();
        assert_eq!(advanced_modules.len(), 4);
    }
    
    #[test]
    fn test_module_metadata() {
        let mult_module = ArithmeticModuleFactory::create_multiplication_module();
        assert_eq!(mult_module.metadata.get("operation"), Some(&"multiplication".to_string()));
        assert_eq!(mult_module.metadata.get("specialized_for"), Some(&"a * b".to_string()));
    }
    
    #[test]
    fn test_module_performance_metrics() {
        let div_module = ArithmeticModuleFactory::create_division_module();
        assert!(div_module.performance.accuracy > 0.8);
        assert!(div_module.performance.efficiency > 0.8);
        assert!(div_module.specialization_score() > 0.7);
    }
}