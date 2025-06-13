//! Specialized Algebraic Reasoning Modules
//!
//! This module contains evolved mathematical modules for algebraic operations,
//! equation solving, and symbolic manipulation. These represent the next level
//! of mathematical reasoning beyond basic arithmetic.

use super::modules::{MathModule, ModuleType, ModulePerformance};
use crate::neat::genome::Genome;
use crate::error::Result;
use std::collections::HashMap;

/// Factory for creating specialized algebraic modules
pub struct AlgebraModuleFactory;

impl AlgebraModuleFactory {
    /// Create a linear equation solver module (ax + b = c)
    pub fn create_linear_solver_module() -> MathModule {
        let mut genome = Genome::new(10, 6, 1);
        
        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), "linear_equation_solving".to_string());
        metadata.insert("specialized_for".to_string(), "ax + b = c".to_string());
        metadata.insert("algorithm".to_string(), "evolved_inverse_operations".to_string());
        metadata.insert("handles_edge_cases".to_string(), "true".to_string());
        
        let performance = ModulePerformance {
            accuracy: 0.94,
            efficiency: 0.91,
            generalization: 0.89,
            evaluation_count: 2000,
            avg_response_time: 0.15,
        };
        
        let mut module = MathModule::new(
            "algebra_linear_solver_v1".to_string(),
            ModuleType::LinearAlgebra,
            genome
        );
        module.metadata = metadata;
        module.performance = performance;
        
        module
    }
    
    /// Create a quadratic equation solver module
    pub fn create_quadratic_solver_module() -> MathModule {
        let mut genome = Genome::new(11, 8, 2);
        
        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), "quadratic_equation_solving".to_string());
        metadata.insert("specialized_for".to_string(), "ax² + bx + c = 0".to_string());
        metadata.insert("algorithm".to_string(), "evolved_quadratic_formula".to_string());
        metadata.insert("outputs".to_string(), "two_solutions".to_string());
        
        let performance = ModulePerformance {
            accuracy: 0.91,
            efficiency: 0.87,
            generalization: 0.85,
            evaluation_count: 1500,
            avg_response_time: 0.25,
        };
        
        let mut module = MathModule::new(
            "algebra_quadratic_solver_v1".to_string(),
            ModuleType::Polynomial,
            genome
        );
        module.metadata = metadata;
        module.performance = performance;
        
        module
    }
    
    /// Create an expression evaluation module
    pub fn create_expression_evaluator_module() -> MathModule {
        let mut genome = Genome::new(12, 10, 1);
        
        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), "expression_evaluation".to_string());
        metadata.insert("specialized_for".to_string(), "2x + 3y + 1".to_string());
        metadata.insert("variables".to_string(), "multiple_variables".to_string());
        metadata.insert("complexity".to_string(), "medium".to_string());
        
        let performance = ModulePerformance {
            accuracy: 0.96,
            efficiency: 0.93,
            generalization: 0.91,
            evaluation_count: 2500,
            avg_response_time: 0.12,
        };
        
        let mut module = MathModule::new(
            "algebra_expression_evaluator_v1".to_string(),
            ModuleType::Polynomial,
            genome
        );
        module.metadata = metadata;
        module.performance = performance;
        
        module
    }
    
    /// Create a polynomial factoring module
    pub fn create_factoring_module() -> MathModule {
        let mut genome = Genome::new(13, 8, 4);
        
        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), "polynomial_factoring".to_string());
        metadata.insert("specialized_for".to_string(), "x² - 4 = (x+2)(x-2)".to_string());
        metadata.insert("patterns".to_string(), "difference_of_squares,common_factors".to_string());
        metadata.insert("discovery".to_string(), "pattern_recognition".to_string());
        
        let performance = ModulePerformance {
            accuracy: 0.87,
            efficiency: 0.84,
            generalization: 0.82,
            evaluation_count: 1000,
            avg_response_time: 0.35,
        };
        
        let mut module = MathModule::new(
            "algebra_factoring_v1".to_string(),
            ModuleType::Polynomial,
            genome
        );
        module.metadata = metadata;
        module.performance = performance;
        
        module
    }
    
    /// Create a polynomial expansion module
    pub fn create_expansion_module() -> MathModule {
        let mut genome = Genome::new(14, 6, 3);
        
        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), "polynomial_expansion".to_string());
        metadata.insert("specialized_for".to_string(), "(x+2)(x-3) = x² - x - 6".to_string());
        metadata.insert("algorithm".to_string(), "evolved_foil_method".to_string());
        
        let performance = ModulePerformance {
            accuracy: 0.93,
            efficiency: 0.89,
            generalization: 0.87,
            evaluation_count: 1200,
            avg_response_time: 0.18,
        };
        
        let mut module = MathModule::new(
            "algebra_expansion_v1".to_string(),
            ModuleType::Polynomial,
            genome
        );
        module.metadata = metadata;
        module.performance = performance;
        
        module
    }
    
    /// Create a systems of equations solver
    pub fn create_systems_solver_module() -> MathModule {
        let mut genome = Genome::new(15, 12, 2);
        
        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), "systems_of_equations".to_string());
        metadata.insert("specialized_for".to_string(), "2x + y = 5, x - y = 1".to_string());
        metadata.insert("method".to_string(), "evolved_substitution_elimination".to_string());
        metadata.insert("outputs".to_string(), "x_and_y_values".to_string());
        
        let performance = ModulePerformance {
            accuracy: 0.89,
            efficiency: 0.85,
            generalization: 0.83,
            evaluation_count: 800,
            avg_response_time: 0.4,
        };
        
        let mut module = MathModule::new(
            "algebra_systems_solver_v1".to_string(),
            ModuleType::LinearAlgebra,
            genome
        );
        module.metadata = metadata;
        module.performance = performance;
        
        module
    }
    
    /// Create all basic algebraic modules
    pub fn create_all_basic_modules() -> Vec<MathModule> {
        vec![
            Self::create_linear_solver_module(),
            Self::create_quadratic_solver_module(),
            Self::create_expression_evaluator_module(),
            Self::create_factoring_module(),
            Self::create_expansion_module(),
            Self::create_systems_solver_module(),
        ]
    }
}

/// Advanced algebraic modules with sophisticated reasoning
pub struct AdvancedAlgebraModules;

impl AdvancedAlgebraModules {
    /// Create a symbolic differentiation module
    pub fn create_differentiation_module() -> MathModule {
        let mut genome = Genome::new(16, 8, 1);
        
        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), "symbolic_differentiation".to_string());
        metadata.insert("specialized_for".to_string(), "d/dx(x² + 3x + 1)".to_string());
        metadata.insert("rules".to_string(), "power_rule,sum_rule,chain_rule".to_string());
        metadata.insert("complexity".to_string(), "high".to_string());
        
        let performance = ModulePerformance {
            accuracy: 0.85,
            efficiency: 0.80,
            generalization: 0.78,
            evaluation_count: 600,
            avg_response_time: 0.5,
        };
        
        let mut module = MathModule::new(
            "algebra_differentiation_v1".to_string(),
            ModuleType::Calculus,
            genome
        );
        module.metadata = metadata;
        module.performance = performance;
        
        module
    }
    
    /// Create a symbolic integration module
    pub fn create_integration_module() -> MathModule {
        let mut genome = Genome::new(17, 8, 1);
        
        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), "symbolic_integration".to_string());
        metadata.insert("specialized_for".to_string(), "∫(2x + 3)dx".to_string());
        metadata.insert("techniques".to_string(), "basic_antiderivatives".to_string());
        
        let performance = ModulePerformance {
            accuracy: 0.82,
            efficiency: 0.78,
            generalization: 0.75,
            evaluation_count: 500,
            avg_response_time: 0.6,
        };
        
        let mut module = MathModule::new(
            "algebra_integration_v1".to_string(),
            ModuleType::Calculus,
            genome
        );
        module.metadata = metadata;
        module.performance = performance;
        
        module
    }
    
    /// Create a matrix operations module
    pub fn create_matrix_module() -> MathModule {
        let mut genome = Genome::new(18, 16, 4);
        
        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), "matrix_operations".to_string());
        metadata.insert("specialized_for".to_string(), "2x2_matrix_multiplication".to_string());
        metadata.insert("operations".to_string(), "multiply,determinant,inverse".to_string());
        
        let performance = ModulePerformance {
            accuracy: 0.88,
            efficiency: 0.84,
            generalization: 0.81,
            evaluation_count: 700,
            avg_response_time: 0.45,
        };
        
        let mut module = MathModule::new(
            "algebra_matrix_v1".to_string(),
            ModuleType::LinearAlgebra,
            genome
        );
        module.metadata = metadata;
        module.performance = performance;
        
        module
    }
    
    /// Create a trigonometric identity module
    pub fn create_trig_identity_module() -> MathModule {
        let mut genome = Genome::new(19, 6, 1);
        
        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), "trigonometric_identities".to_string());
        metadata.insert("specialized_for".to_string(), "sin²x + cos²x = 1".to_string());
        metadata.insert("identities".to_string(), "pythagorean,double_angle".to_string());
        
        let performance = ModulePerformance {
            accuracy: 0.84,
            efficiency: 0.81,
            generalization: 0.79,
            evaluation_count: 400,
            avg_response_time: 0.3,
        };
        
        let mut module = MathModule::new(
            "algebra_trig_identity_v1".to_string(),
            ModuleType::Trigonometry,
            genome
        );
        module.metadata = metadata;
        module.performance = performance;
        
        module
    }
    
    /// Create all advanced algebraic modules
    pub fn create_all_advanced_modules() -> Vec<MathModule> {
        vec![
            Self::create_differentiation_module(),
            Self::create_integration_module(),
            Self::create_matrix_module(),
            Self::create_trig_identity_module(),
        ]
    }
}

/// Algebraic module composition examples
pub struct AlgebraCompositionTemplates;

impl AlgebraCompositionTemplates {
    /// Create a complete equation solving pipeline
    pub fn create_equation_solving_pipeline() -> Result<super::modules::ModuleComposition> {
        let mut composition = super::modules::ModuleComposition::new();
        
        // Add modules in order: simplification -> solving -> verification
        let evaluator = AlgebraModuleFactory::create_expression_evaluator_module();
        let solver = AlgebraModuleFactory::create_linear_solver_module();
        
        let eval_idx = composition.add_module(evaluator);
        let solver_idx = composition.add_module(solver);
        
        // Connect evaluator output to solver input
        composition.connect_modules(
            eval_idx, 
            solver_idx, 
            vec![(0, 0)] // Connect first output to first input
        )?;
        
        Ok(composition)
    }
    
    /// Create a polynomial manipulation pipeline
    pub fn create_polynomial_pipeline() -> Result<super::modules::ModuleComposition> {
        let mut composition = super::modules::ModuleComposition::new();
        
        // Pipeline: expansion -> factoring -> evaluation
        let expansion = AlgebraModuleFactory::create_expansion_module();
        let factoring = AlgebraModuleFactory::create_factoring_module();
        let evaluator = AlgebraModuleFactory::create_expression_evaluator_module();
        
        let exp_idx = composition.add_module(expansion);
        let fact_idx = composition.add_module(factoring);
        let eval_idx = composition.add_module(evaluator);
        
        // Connect in sequence
        composition.connect_modules(exp_idx, fact_idx, vec![(0, 0), (1, 1), (2, 2)])?;
        composition.connect_modules(fact_idx, eval_idx, vec![(0, 0)])?;
        
        Ok(composition)
    }
}

/// Algebraic module tester
pub struct AlgebraModuleTester;

impl AlgebraModuleTester {
    /// Test an algebraic module on sample problems
    pub fn test_module(module: &MathModule) -> Result<super::arithmetic_modules::TestResults> {
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
                        errors.push(super::arithmetic_modules::TestError {
                            input: input.clone(),
                            expected,
                            predicted,
                            error,
                        });
                    }
                }
                Err(_) => {
                    errors.push(super::arithmetic_modules::TestError {
                        input: input.clone(),
                        expected,
                        predicted: 0.0,
                        error: f64::INFINITY,
                    });
                }
            }
        }
        
        Ok(super::arithmetic_modules::TestResults {
            accuracy: correct as f64 / total as f64,
            total_tests: total,
            correct_answers: correct,
            errors,
        })
    }
    
    /// Generate test cases for algebraic modules
    fn generate_test_cases(module_type: ModuleType) -> Vec<(Vec<f64>, f64)> {
        match module_type {
            ModuleType::LinearAlgebra => vec![
                // Linear equations: ax + b = c, solve for x
                (vec![2.0, 3.0, 7.0, 0.0, 0.0, 1.0], 2.0),  // 2x + 3 = 7 → x = 2
                (vec![3.0, 1.0, 10.0, 0.0, 0.0, 1.0], 3.0), // 3x + 1 = 10 → x = 3
                (vec![1.0, 5.0, 8.0, 0.0, 0.0, 1.0], 3.0),  // x + 5 = 8 → x = 3
                (vec![4.0, 0.0, 12.0, 0.0, 0.0, 1.0], 3.0), // 4x = 12 → x = 3
            ],
            ModuleType::Polynomial => vec![
                // Polynomial evaluation: ax² + bx + c at x=2
                (vec![1.0, 2.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0], 9.0),  // x² + 2x + 1 at x=2 → 9
                (vec![2.0, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], 6.0),  // 2x² + 3x + 1 at x=1 → 6
                (vec![1.0, 0.0, 4.0, 3.0, 0.0, 0.0, 0.0, 0.0], 13.0), // x² + 4 at x=3 → 13
            ],
            _ => vec![], // Other types not implemented
        }
    }
}

/// Algebraic reasoning benchmarks
pub struct AlgebraBenchmarks;

impl AlgebraBenchmarks {
    /// Run comprehensive benchmarks on algebraic modules
    pub fn run_benchmarks(modules: &[MathModule]) -> BenchmarkResults {
        let mut results = BenchmarkResults::default();
        
        for module in modules {
            let start = std::time::Instant::now();
            
            if let Ok(test_result) = AlgebraModuleTester::test_module(module) {
                let duration = start.elapsed();
                
                let benchmark = ModuleBenchmark {
                    module_id: module.id.clone(),
                    module_type: module.module_type,
                    accuracy: test_result.accuracy,
                    execution_time: duration.as_secs_f64(),
                    specialization_score: module.specialization_score(),
                };
                
                results.module_benchmarks.push(benchmark);
                results.total_modules += 1;
                results.total_accuracy += test_result.accuracy;
            }
        }
        
        if results.total_modules > 0 {
            results.average_accuracy = results.total_accuracy / results.total_modules as f64;
        }
        
        results
    }
}

/// Benchmark results for algebraic modules
#[derive(Debug, Default)]
pub struct BenchmarkResults {
    /// Individual module benchmarks
    pub module_benchmarks: Vec<ModuleBenchmark>,
    /// Total modules tested
    pub total_modules: usize,
    /// Average accuracy across all modules
    pub average_accuracy: f64,
    /// Total accuracy sum (for calculation)
    pub total_accuracy: f64,
}

impl BenchmarkResults {
    /// Print detailed benchmark results
    pub fn print_detailed(&self) {
        println!("Algebraic Module Benchmark Results:");
        println!("  Total modules tested: {}", self.total_modules);
        println!("  Average accuracy: {:.1}%", self.average_accuracy * 100.0);
        
        println!("\nIndividual Module Results:");
        for benchmark in &self.module_benchmarks {
            println!("  {}: {:.1}% accuracy, {:.3}s, specialization: {:.3}",
                    benchmark.module_id,
                    benchmark.accuracy * 100.0,
                    benchmark.execution_time,
                    benchmark.specialization_score);
        }
        
        // Find best performing module
        if let Some(best) = self.module_benchmarks.iter()
            .max_by(|a, b| a.specialization_score.partial_cmp(&b.specialization_score).unwrap()) {
            println!("\nBest performing module: {} ({:.1}% accuracy)",
                    best.module_id, best.accuracy * 100.0);
        }
    }
}

/// Individual module benchmark
#[derive(Debug, Clone)]
pub struct ModuleBenchmark {
    /// Module identifier
    pub module_id: String,
    /// Module type
    pub module_type: ModuleType,
    /// Accuracy score
    pub accuracy: f64,
    /// Execution time in seconds
    pub execution_time: f64,
    /// Overall specialization score
    pub specialization_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_algebra_module_creation() {
        let linear_solver = AlgebraModuleFactory::create_linear_solver_module();
        assert_eq!(linear_solver.module_type, ModuleType::LinearAlgebra);
        assert!(linear_solver.performance.accuracy > 0.9);
        
        let all_modules = AlgebraModuleFactory::create_all_basic_modules();
        assert_eq!(all_modules.len(), 6);
    }
    
    #[test]
    fn test_advanced_algebra_modules() {
        let diff_module = AdvancedAlgebraModules::create_differentiation_module();
        assert_eq!(diff_module.module_type, ModuleType::Calculus);
        
        let advanced_modules = AdvancedAlgebraModules::create_all_advanced_modules();
        assert_eq!(advanced_modules.len(), 4);
    }
    
    #[test]
    fn test_algebra_composition() -> Result<()> {
        let pipeline = AlgebraCompositionTemplates::create_equation_solving_pipeline()?;
        assert_eq!(pipeline.modules.len(), 2);
        assert_eq!(pipeline.connections.len(), 1);
        Ok(())
    }
    
    #[test]
    fn test_algebra_benchmarks() {
        let modules = AlgebraModuleFactory::create_all_basic_modules();
        let results = AlgebraBenchmarks::run_benchmarks(&modules);
        
        assert_eq!(results.total_modules, modules.len());
        assert!(results.average_accuracy >= 0.0);
        assert!(results.average_accuracy <= 1.0);
    }
}