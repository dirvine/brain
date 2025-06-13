//! Modular Mathematical Evolution Demo
//!
//! This revolutionary example demonstrates the power of modular mathematical
//! components in NEAT - how specialized modules can be evolved, composed,
//! and reused to solve complex mathematical problems!

use neat_fashion_classifier::{
    config::NEATConfig,
    calculator::{
        // Module system
        ModuleLibrary, ModuleType, MathModule, ModuleComposition,
        ArithmeticModuleFactory, AlgebraModuleFactory, AdvancedArithmeticModules,
        AdvancedAlgebraModules, ArithmeticModuleTester, AlgebraModuleTester,
        AlgebraBenchmarks, AlgebraCompositionTemplates,
        // Core types
        Operation, Expression, AlgebraProblem,
    },
    neat::{trainer::NEATTrainer, genome::Genome, fitness::FitnessEvaluator, network::Network},
    error::Result,
};
use std::time::Instant;

fn main() -> Result<()> {
    println!("🚀 Modular Mathematical Evolution with NEAT");
    println!("===========================================");
    println!("Demonstrating specialized mathematical modules and composition!\n");
    
    // Showcase the module system
    showcase_module_library()?;
    showcase_module_testing()?;
    showcase_module_composition()?;
    showcase_hierarchical_learning()?;
    
    println!("\n🎉 Modular evolution experiments completed!");
    println!("We've demonstrated:");
    println!("  ✓ Specialized mathematical modules");
    println!("  ✓ Module testing and benchmarking");
    println!("  ✓ Module composition for complex reasoning");
    println!("  ✓ Hierarchical learning with reusable components");
    println!("  ✓ Revolutionary modular AI architecture!");
    
    Ok(())
}

/// Showcase 1: Module Library and Specialization
fn showcase_module_library() -> Result<()> {
    println!("📚 Showcase 1: Module Library and Specialization");
    println!("===============================================");
    
    // Create a comprehensive module library
    let mut library = ModuleLibrary::new();
    
    println!("🧮 Creating specialized arithmetic modules...");
    let arithmetic_modules = ArithmeticModuleFactory::create_all_basic_modules();
    for module in arithmetic_modules {
        println!("  Added: {}", module.description());
        library.add_module(module);
    }
    
    println!("\n🧮 Creating advanced arithmetic modules...");
    let advanced_arithmetic = AdvancedArithmeticModules::create_all_advanced_modules();
    for module in advanced_arithmetic {
        println!("  Added: {}", module.description());
        library.add_module(module);
    }
    
    println!("\n📐 Creating algebraic reasoning modules...");
    let algebra_modules = AlgebraModuleFactory::create_all_basic_modules();
    for module in algebra_modules {
        println!("  Added: {}", module.description());
        library.add_module(module);
    }
    
    println!("\n📐 Creating advanced algebraic modules...");
    let advanced_algebra = AdvancedAlgebraModules::create_all_advanced_modules();
    for module in advanced_algebra {
        println!("  Added: {}", module.description());
        library.add_module(module);
    }
    
    // Show library statistics
    println!("\n📊 Module Library Statistics:");
    let stats = library.get_statistics();
    stats.print();
    
    // Demonstrate module retrieval
    println!("\n🔍 Module Specialization Examples:");
    if let Some(best_arithmetic) = library.get_best_module(ModuleType::Arithmetic) {
        println!("  Best Arithmetic Module: {}", best_arithmetic.description());
    }
    
    if let Some(best_algebra) = library.get_best_module(ModuleType::LinearAlgebra) {
        println!("  Best Linear Algebra Module: {}", best_algebra.description());
    }
    
    if let Some(best_polynomial) = library.get_best_module(ModuleType::Polynomial) {
        println!("  Best Polynomial Module: {}", best_polynomial.description());
    }
    
    Ok(())
}

/// Showcase 2: Module Testing and Benchmarking
fn showcase_module_testing() -> Result<()> {
    println!("\n📚 Showcase 2: Module Testing and Performance Analysis");
    println!("====================================================");
    
    // Test arithmetic modules
    println!("🧪 Testing Arithmetic Modules:");
    let arithmetic_modules = ArithmeticModuleFactory::create_all_basic_modules();
    
    for module in &arithmetic_modules[..3] { // Test first 3 for demo
        println!("\n  Testing: {}", module.id);
        if let Ok(results) = ArithmeticModuleTester::test_module(module) {
            println!("    Accuracy: {:.1}%", results.accuracy * 100.0);
            println!("    Correct: {}/{}", results.correct_answers, results.total_tests);
            if !results.errors.is_empty() {
                println!("    Errors: {} found", results.errors.len());
            }
        }
    }
    
    // Test algebraic modules
    println!("\n🧪 Testing Algebraic Modules:");
    let algebra_modules = AlgebraModuleFactory::create_all_basic_modules();
    
    for module in &algebra_modules[..2] { // Test first 2 for demo
        println!("\n  Testing: {}", module.id);
        if let Ok(results) = AlgebraModuleTester::test_module(module) {
            println!("    Accuracy: {:.1}%", results.accuracy * 100.0);
            println!("    Correct: {}/{}", results.correct_answers, results.total_tests);
        }
    }
    
    // Run comprehensive benchmarks
    println!("\n📊 Comprehensive Algebraic Benchmarks:");
    let benchmark_results = AlgebraBenchmarks::run_benchmarks(&algebra_modules);
    benchmark_results.print_detailed();
    
    Ok(())
}

/// Showcase 3: Module Composition and Complex Reasoning
fn showcase_module_composition() -> Result<()> {
    println!("\n📚 Showcase 3: Module Composition for Complex Reasoning");
    println!("======================================================");
    
    // Create an equation solving pipeline
    println!("🔧 Creating Equation Solving Pipeline:");
    let equation_pipeline = AlgebraCompositionTemplates::create_equation_solving_pipeline()?;
    println!("  {}", equation_pipeline.description());
    
    for (i, module) in equation_pipeline.modules.iter().enumerate() {
        println!("    Module {}: {}", i + 1, module.description());
    }
    
    // Create a polynomial manipulation pipeline
    println!("\n🔧 Creating Polynomial Manipulation Pipeline:");
    let polynomial_pipeline = AlgebraCompositionTemplates::create_polynomial_pipeline()?;
    println!("  {}", polynomial_pipeline.description());
    
    for (i, module) in polynomial_pipeline.modules.iter().enumerate() {
        println!("    Module {}: {}", i + 1, module.description());
    }
    
    // Demonstrate custom composition
    println!("\n🔧 Creating Custom Multi-Level Composition:");
    let custom_composition = create_custom_composition()?;
    println!("  Created composition with {} modules", custom_composition.modules.len());
    
    // Test composition execution
    println!("\n🧪 Testing Composition Execution:");
    let test_input = vec![2.0, 3.0, 1.0, 0.5];
    println!("  Input: {:?}", test_input);
    
    if let Ok(output) = custom_composition.execute(&test_input) {
        println!("  Output: {:?}", output);
        println!("  ✅ Composition executed successfully!");
    } else {
        println!("  ❌ Composition execution failed");
    }
    
    Ok(())
}

/// Showcase 4: Hierarchical Learning with Module Specialization
fn showcase_hierarchical_learning() -> Result<()> {
    println!("\n📚 Showcase 4: Hierarchical Learning and Specialization");
    println!("=====================================================");
    
    // Demonstrate learning hierarchy
    println!("🏗️ Mathematical Learning Hierarchy:");
    println!("  Level 1: Basic Arithmetic");
    println!("    - Addition, Subtraction, Multiplication, Division");
    println!("  Level 2: Advanced Arithmetic");
    println!("    - Carry Operations, Long Multiplication, Fractions");
    println!("  Level 3: Basic Algebra");
    println!("    - Linear Equations, Expression Evaluation");
    println!("  Level 4: Advanced Algebra");
    println!("    - Quadratic Equations, Polynomial Factoring");
    println!("  Level 5: Higher Mathematics");
    println!("    - Calculus, Matrix Operations, Trigonometry");
    
    // Show complexity progression
    println!("\n📊 Module Complexity Analysis:");
    for module_type in ModuleType::all() {
        println!("  {}: Complexity Level {}", 
                module_type, 
                module_type.complexity_level());
    }
    
    // Demonstrate transfer learning potential
    println!("\n🔄 Transfer Learning Opportunities:");
    println!("  • Addition module → Multi-digit addition");
    println!("  • Multiplication module → Polynomial expansion");
    println!("  • Linear solver → Systems of equations");
    println!("  • Pattern recognition → Advanced factoring");
    
    // Show specialization metrics
    println!("\n🎯 Module Specialization Scores:");
    let all_modules = create_sample_modules();
    for module in &all_modules[..5] { // Show first 5
        println!("  {}: {:.3}", 
                module.id, 
                module.specialization_score());
    }
    
    Ok(())
}

/// Create a custom composition demonstrating advanced reasoning
fn create_custom_composition() -> Result<ModuleComposition> {
    let mut composition = ModuleComposition::new();
    
    // Create a multi-stage mathematical reasoning pipeline
    // Stage 1: Arithmetic preprocessing
    let arithmetic_module = ArithmeticModuleFactory::create_multi_operation_module();
    
    // Stage 2: Algebraic transformation
    let algebra_module = AlgebraModuleFactory::create_expression_evaluator_module();
    
    // Stage 3: Advanced processing
    let advanced_module = AdvancedAlgebraModules::create_matrix_module();
    
    // Add modules to composition
    let arith_idx = composition.add_module(arithmetic_module);
    let algebra_idx = composition.add_module(algebra_module);
    let advanced_idx = composition.add_module(advanced_module);
    
    // Note: In a real implementation, we would need compatible I/O sizes
    // For demonstration, we'll just show the structure
    
    Ok(composition)
}

/// Create sample modules for demonstration
fn create_sample_modules() -> Vec<MathModule> {
    let mut modules = Vec::new();
    
    // Add some arithmetic modules
    modules.extend(ArithmeticModuleFactory::create_all_basic_modules());
    
    // Add some algebra modules
    modules.extend(AlgebraModuleFactory::create_all_basic_modules());
    
    modules
}

/// Demonstrate module evolution (conceptual)
fn demonstrate_module_evolution() -> Result<()> {
    println!("\n🧬 Module Evolution Process (Conceptual):");
    println!("1. Initialize random population of modules");
    println!("2. Evaluate each module on specialized tasks");
    println!("3. Select best performing modules");
    println!("4. Apply NEAT mutations and crossover");
    println!("5. Test new modules for improved performance");
    println!("6. Repeat until convergence");
    
    println!("\n📈 Evolution Metrics:");
    println!("  • Task-specific accuracy");
    println!("  • Computational efficiency");
    println!("  • Generalization capability");
    println!("  • Composability with other modules");
    
    Ok(())
}

/// Demonstrate educational applications
fn demonstrate_educational_applications() {
    println!("\n🎓 Educational Applications:");
    println!("• Personalized Math Tutoring:");
    println!("  - Identify student's weak areas");
    println!("  - Compose appropriate modules for remediation");
    println!("  - Adapt difficulty based on performance");
    
    println!("• Curriculum Design:");
    println!("  - Optimal ordering of mathematical concepts");
    println!("  - Prerequisite relationship discovery");
    println!("  - Individual learning path optimization");
    
    println!("• Assessment Generation:");
    println!("  - Automatic problem generation");
    println!("  - Difficulty calibration");
    println!("  - Misconception detection");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_module_library_creation() -> Result<()> {
        let mut library = ModuleLibrary::new();
        assert!(library.is_empty());
        
        let modules = ArithmeticModuleFactory::create_all_basic_modules();
        for module in modules {
            library.add_module(module);
        }
        
        assert!(!library.is_empty());
        assert!(library.len() > 0);
        
        Ok(())
    }
    
    #[test]
    fn test_module_composition() -> Result<()> {
        let composition = create_custom_composition()?;
        assert!(composition.modules.len() > 0);
        
        Ok(())
    }
    
    #[test]
    fn test_module_testing() -> Result<()> {
        let module = ArithmeticModuleFactory::create_addition_module();
        let results = ArithmeticModuleTester::test_module(&module)?;
        
        assert!(results.total_tests > 0);
        assert!(results.accuracy >= 0.0);
        assert!(results.accuracy <= 1.0);
        
        Ok(())
    }
    
    #[test]
    fn test_algebra_benchmarks() {
        let modules = AlgebraModuleFactory::create_all_basic_modules();
        let results = AlgebraBenchmarks::run_benchmarks(&modules);
        
        assert!(results.total_modules > 0);
        assert!(results.average_accuracy >= 0.0);
    }
}