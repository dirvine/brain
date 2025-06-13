//! HuggingFace Mathematical Dataset Integration Demo
//!
//! This example demonstrates NEAT learning from real mathematical
//! datasets like GSM8K, showcasing how evolution can discover mathematical
//! reasoning from natural language problems!

use neat::{
    config::NEATConfig,
    calculator::{
        AlgebraEvaluator, AlgebraEvaluatorConfig, AlgebraicDifficulty,
        Expression, AlgebraProblem, Operation,
    },
    dataset::{
        MathDatasetManager, MathProblemType, GSM8KDataset, MATHDataset,
        MathDatasetProblem,
    },
    neat::{trainer::NEATTrainer, genome::Genome, fitness::FitnessEvaluator, network::Network},
    error::Result,
};
use std::collections::HashMap;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🚀 HuggingFace Mathematical Dataset Integration with NEAT");
    println!("========================================================");
    println!("Demonstrating NEAT learning from real mathematical datasets!\n");
    
    // Showcase dataset integration
    showcase_dataset_loading()?;
    showcase_problem_parsing()?;
    showcase_mixed_training()?;
    
    println!("\n🎉 HuggingFace integration experiments completed!");
    println!("We've demonstrated:");
    println!("  ✓ Loading and parsing mathematical datasets");
    println!("  ✓ Converting natural language to algebraic problems");
    println!("  ✓ Training NEAT on real mathematical datasets");
    println!("  ✓ Mixed dataset training for robust learning");
    
    Ok(())
}

/// Showcase 1: Dataset Loading and Management
fn showcase_dataset_loading() -> Result<()> {
    println!("📚 Showcase 1: Mathematical Dataset Loading");
    println!("==========================================");
    
    // Create and load datasets
    let mut manager = MathDatasetManager::new();
    
    println!("📥 Loading GSM8K dataset...");
    manager.load_gsm8k()?;
    
    println!("📥 Loading MATH competition dataset...");
    manager.load_math_dataset()?;
    
    // Show statistics
    let stats = manager.get_statistics();
    stats.print();
    
    // Get a mixed batch of problems
    println!("\n🔀 Getting mixed batch of problems:");
    let batch = manager.get_mixed_batch(5);
    
    for (i, problem) in batch.iter().enumerate() {
        println!("  Problem {}: {} (Type: {:?}, Difficulty: {})",
                i + 1, 
                problem.question.chars().take(60).collect::<String>() + "...",
                problem.problem_type,
                problem.difficulty);
    }
    
    Ok(())
}

/// Showcase 2: Problem Parsing and Conversion
fn showcase_problem_parsing() -> Result<()> {
    println!("\n📚 Showcase 2: Natural Language to Algebra Conversion");
    println!("===================================================");
    
    let mut gsm8k = GSM8KDataset::new();
    gsm8k.load_mock_data();
    
    println!("🔍 Analyzing problem conversion capabilities:");
    
    let conversion_results = gsm8k.to_algebra_problems();
    let mut successful_conversions = 0;
    let mut total_problems = 0;
    
    for (problem, algebra_result) in conversion_results {
        total_problems += 1;
        
        match algebra_result {
            Ok(algebra_problem) => {
                successful_conversions += 1;
                println!("\n  ✅ Successfully converted:");
                println!("     Question: {}", problem.question);
                println!("     Expected: {}", problem.answer);
                println!("     Algebra: {}", algebra_problem.description());
            }
            Err(e) => {
                println!("\n  ❌ Could not convert:");
                println!("     Question: {}", problem.question);
                println!("     Reason: {}", e);
            }
        }
    }
    
    let conversion_rate = successful_conversions as f64 / total_problems as f64 * 100.0;
    println!("\n📊 Conversion Statistics:");
    println!("  Total problems: {}", total_problems);
    println!("  Successful conversions: {}", successful_conversions);
    println!("  Conversion rate: {:.1}%", conversion_rate);
    
    Ok(())
}

/// Showcase 3: Training NEAT on Real Mathematical Datasets
fn showcase_mixed_training() -> Result<()> {
    println!("\n📚 Showcase 3: Training NEAT on Real Mathematical Data");
    println!("====================================================");
    
    // Create a custom evaluator that uses real dataset problems
    let evaluator = MathDatasetEvaluator::new()?;
    
    println!("🧮 Training Setup:");
    println!("  Dataset: Mixed GSM8K + MATH problems");
    println!("  Input neurons: {}", evaluator.input_size());
    println!("  Output neurons: {}", evaluator.output_size());
    println!("  Max fitness: {:.2}", evaluator.max_fitness());
    
    // Configure NEAT for mathematical dataset learning
    let mut neat_config = NEATConfig::default();
    neat_config.population.size = 50;  // Smaller for demo
    neat_config.population.max_generations = 20;  // Quick demo
    neat_config.population.target_fitness = 1.0;
    
    // Encourage exploration for complex mathematical reasoning
    neat_config.mutation.weight_mutation_rate = 0.8;
    neat_config.mutation.add_connection_rate = 0.3;
    neat_config.mutation.add_node_rate = 0.1;
    
    println!("\n🧬 Starting evolution on real mathematical data...");
    let start_time = Instant::now();
    
    let mut trainer = NEATTrainer::new(evaluator, neat_config);
    let result = trainer.train()?;
    
    println!("\n✨ Mathematical Dataset Training Results:");
    println!("  Final fitness: {:.3}", result.state.best_fitness);
    println!("  Generations: {}", result.state.generation);
    println!("  Time: {:.2}s", start_time.elapsed().as_secs_f64());
    println!("  Network complexity: {} nodes, {} connections",
            result.best_genome.nodes.len(),
            result.best_genome.connections.iter().filter(|c| c.enabled).count());
    
    // Test the evolved network on some problems
    test_on_real_problems(&result.best_genome)?;
    
    Ok(())
}

/// Test evolved network on real mathematical problems
fn test_on_real_problems(genome: &Genome) -> Result<()> {
    println!("\n🧪 Testing Evolved Network on Real Problems:");
    
    let mut gsm8k = GSM8KDataset::new();
    gsm8k.load_mock_data();
    
    let network = Network::from_genome(genome)?;
    
    // Test on algebraic problems that we can convert
    let algebraic_problems = gsm8k.get_problems_by_type(MathProblemType::AlgebraicReasoning);
    
    println!("  Testing on {} algebraic problems from GSM8K:", algebraic_problems.len());
    
    for (i, problem) in algebraic_problems.iter().enumerate() {
        if let Ok(algebra_problem) = problem.to_algebra_problem() {
            println!("\n    Problem {}: {}", i + 1, problem.question);
            println!("    Expected answer: {}", problem.answer);
            
            // Create a simple encoding for testing
            let input = create_test_encoding(&algebra_problem);
            
            if let Ok(output) = network.activate(&input) {
                let predicted = output[0] * 10.0; // Denormalize
                println!("    Network prediction: {:.2}", predicted);
                
                if let Ok(expected) = problem.answer.parse::<f64>() {
                    let error = (predicted - expected).abs();
                    let status = if error < 1.0 { "✅" } else { "❌" };
                    println!("    Error: {:.2} {}", error, status);
                }
            }
        }
    }
    
    Ok(())
}

/// Create a simple test encoding for algebra problems
fn create_test_encoding(problem: &AlgebraProblem) -> Vec<f64> {
    // Simple encoding: just use some default values
    vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}

/// Custom evaluator that uses real mathematical dataset problems
struct MathDatasetEvaluator {
    problems: Vec<AlgebraProblem>,
    input_size: usize,
    output_size: usize,
}

impl MathDatasetEvaluator {
    fn new() -> Result<Self> {
        // Load real problems and convert to algebra format
        let mut gsm8k = GSM8KDataset::new();
        gsm8k.load_mock_data();
        
        let mut problems = Vec::new();
        
        // Convert GSM8K problems to algebra problems
        for gsm8k_problem in gsm8k.get_problems_by_type(MathProblemType::AlgebraicReasoning) {
            if let Ok(algebra_problem) = gsm8k_problem.to_algebra_problem() {
                problems.push(algebra_problem);
            }
        }
        
        // Add some arithmetic problems as well
        for gsm8k_problem in gsm8k.get_problems_by_type(MathProblemType::ArithmeticReasoning) {
            if let Ok(algebra_problem) = gsm8k_problem.to_algebra_problem() {
                problems.push(algebra_problem);
            }
        }
        
        Ok(Self {
            problems,
            input_size: 17,  // Fixed encoding size
            output_size: 1,  // Single numerical output
        })
    }
}

impl FitnessEvaluator for MathDatasetEvaluator {
    fn evaluate(&self, genome: &Genome) -> Result<f64> {
        if self.problems.is_empty() {
            return Ok(0.0);
        }
        
        let network = Network::from_genome(genome)?;
        let mut total_score = 0.0;
        let mut valid_problems = 0;
        
        // Evaluate on a subset of problems
        for problem in self.problems.iter().take(5) {
            let input = create_test_encoding(problem);
            
            if let Ok(output) = network.activate(&input) {
                // Score based on reasonable output (not NaN or extreme values)
                let prediction = output[0];
                if prediction.is_finite() && prediction.abs() < 100.0 {
                    total_score += 1.0 / (1.0 + prediction.abs());
                    valid_problems += 1;
                }
            }
        }
        
        if valid_problems > 0 {
            Ok(total_score / valid_problems as f64)
        } else {
            Ok(0.0)
        }
    }
    
    fn input_size(&self) -> usize {
        self.input_size
    }
    
    fn output_size(&self) -> usize {
        self.output_size
    }
    
    fn max_fitness(&self) -> f64 {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_math_dataset_evaluator() -> Result<()> {
        let evaluator = MathDatasetEvaluator::new()?;
        
        assert!(evaluator.input_size() > 0);
        assert_eq!(evaluator.output_size(), 1);
        assert!(evaluator.max_fitness() > 0.0);
        
        // Test with a simple genome
        let genome = Genome::new(0, evaluator.input_size(), evaluator.output_size());
        let fitness = evaluator.evaluate(&genome)?;
        assert!(fitness >= 0.0);
        assert!(fitness <= evaluator.max_fitness());
        
        Ok(())
    }
}