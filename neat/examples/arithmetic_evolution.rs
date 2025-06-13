//! Arithmetic Learning Evolution Demo
//!
//! This example demonstrates NEAT learning arithmetic operations from scratch.
//! Watch as networks evolve to discover mathematical concepts!

use neat::{
    config::NEATConfig,
    calculator::{
        ArithmeticEvaluator, ArithmeticEvaluatorConfig, ArithmeticConfig, EncodingConfig,
        Operation, DifficultyLevel, EncodingScheme, MathProblem,
    },
    neat::{trainer::NEATTrainer, genome::Genome, fitness::FitnessEvaluator, network::Network, NodeType},
    error::Result,
};
use std::time::Instant;

fn main() -> Result<()> {
    println!("🧮 NEAT Arithmetic Learning Evolution");
    println!("=====================================");
    
    // Start with single-digit addition
    run_single_digit_addition()?;
    
    // Progress to multi-digit addition  
    run_multi_digit_addition()?;
    
    // Try learning multiplication
    run_multiplication_learning()?;
    
    // Mixed operations challenge
    run_mixed_operations()?;
    
    println!("\n🎉 All arithmetic evolution experiments completed!");
    Ok(())
}

/// Experiment 1: Learning single-digit addition (1+1=2, 5+3=8, etc.)
fn run_single_digit_addition() -> Result<()> {
    println!("\n🔢 Experiment 1: Single-Digit Addition Learning");
    println!("================================================");
    
    let arithmetic_config = ArithmeticConfig {
        operations: vec![Operation::Add],
        difficulty: DifficultyLevel::SingleDigit,
        allow_negative: false,
        max_result: 18, // Max possible: 9+9=18
        ..Default::default()
    };
    
    let encoding_config = EncodingConfig {
        scheme: EncodingScheme::Decimal,
        max_digits: 2,
        include_sign: false,
        fixed_length: true,
    };
    
    let evaluator_config = ArithmeticEvaluatorConfig {
        problems_per_evaluation: 30,
        encoding_config,
        arithmetic_config,
        exact_weight: 1.0,
        partial_weight: 0.2,
        complexity_penalty: 0.002,
        perfect_bonus: 0.3,
        ..Default::default()
    };
    
    let evaluator = ArithmeticEvaluator::new(evaluator_config);
    
    println!("📊 Problem setup:");
    println!("  Input size: {} neurons", evaluator.input_size());
    println!("  Output size: {} neurons", evaluator.output_size());
    println!("  Max fitness: {:.2}", evaluator.max_fitness());
    
    // Configure NEAT for arithmetic learning
    let mut neat_config = NEATConfig::default();
    neat_config.population.size = 100;
    neat_config.population.max_generations = 30;
    neat_config.population.target_fitness = 1.2; // High accuracy target
    
    // Encourage exploration for arithmetic discovery
    neat_config.mutation.weight_mutation_rate = 0.9;
    neat_config.mutation.add_connection_rate = 0.4;
    neat_config.mutation.add_node_rate = 0.15;
    neat_config.mutation.weight_perturbation_power = 1.0;
    
    println!("\n🧬 NEAT Configuration:");
    println!("  Population: {}", neat_config.population.size);
    println!("  Max generations: {}", neat_config.population.max_generations);
    println!("  Target fitness: {:.2}", neat_config.population.target_fitness);
    
    // Run evolution
    println!("\n🚀 Starting evolution...");
    let start_time = Instant::now();
    
    let mut trainer = NEATTrainer::new(evaluator, neat_config);
    let result = trainer.train()?;
    
    let evolution_time = start_time.elapsed();
    
    // Results analysis
    println!("\n📈 Evolution Results:");
    println!("  Best fitness: {:.3}", result.state.best_fitness);
    println!("  Generations: {}", result.state.generation);
    println!("  Training time: {:.2}s", evolution_time.as_secs_f64());
    println!("  Success: {}", result.success);
    
    // Test the best network on specific problems
    test_arithmetic_network(&result.best_genome, "Single-Digit Addition")?;
    
    Ok(())
}

/// Experiment 2: Multi-digit addition with carry operations
fn run_multi_digit_addition() -> Result<()> {
    println!("\n🔢 Experiment 2: Multi-Digit Addition with Carry");
    println!("================================================");
    
    let arithmetic_config = ArithmeticConfig {
        operations: vec![Operation::Add],
        difficulty: DifficultyLevel::TwoDigit,
        allow_negative: false,
        max_result: 200,
        ..Default::default()
    };
    
    let encoding_config = EncodingConfig {
        scheme: EncodingScheme::Decimal,
        max_digits: 3,
        include_sign: false,
        fixed_length: true,
    };
    
    let evaluator_config = ArithmeticEvaluatorConfig {
        problems_per_evaluation: 40,
        encoding_config,
        arithmetic_config,
        exact_weight: 1.0,
        partial_weight: 0.4, // Higher partial credit for harder problems
        complexity_penalty: 0.001,
        ..Default::default()
    };
    
    let evaluator = ArithmeticEvaluator::new(evaluator_config);
    
    let mut neat_config = NEATConfig::default();
    neat_config.population.size = 150; // Larger population for harder problem
    neat_config.population.max_generations = 50;
    neat_config.population.target_fitness = 1.0;
    
    println!("🚀 Learning multi-digit addition...");
    let mut trainer = NEATTrainer::new(evaluator, neat_config);
    let result = trainer.train()?;
    
    println!("📈 Multi-digit results: {:.3} fitness in {} generations", 
            result.state.best_fitness, result.state.generation);
    
    test_arithmetic_network(&result.best_genome, "Multi-Digit Addition")?;
    
    Ok(())
}

/// Experiment 3: Learning multiplication tables
fn run_multiplication_learning() -> Result<()> {
    println!("\n🔢 Experiment 3: Multiplication Table Learning");
    println!("==============================================");
    
    let arithmetic_config = ArithmeticConfig {
        operations: vec![Operation::Multiply],
        difficulty: DifficultyLevel::SingleDigit,
        allow_negative: false,
        max_result: 81, // Max: 9*9=81
        ..Default::default()
    };
    
    let encoding_config = EncodingConfig {
        scheme: EncodingScheme::Decimal,
        max_digits: 2,
        include_sign: false,
        fixed_length: true,
    };
    
    let evaluator_config = ArithmeticEvaluatorConfig {
        problems_per_evaluation: 35,
        encoding_config,
        arithmetic_config,
        exact_weight: 1.0,
        partial_weight: 0.3,
        complexity_penalty: 0.002,
        perfect_bonus: 0.5, // Big bonus for perfect multiplication
        ..Default::default()
    };
    
    let evaluator = ArithmeticEvaluator::new(evaluator_config);
    
    let mut neat_config = NEATConfig::default();
    neat_config.population.size = 120;
    neat_config.population.max_generations = 40;
    neat_config.population.target_fitness = 1.3;
    
    // Multiplication might need more complex networks
    neat_config.mutation.add_node_rate = 0.2;
    neat_config.speciation.target_species_count = 12;
    
    println!("🚀 Learning multiplication tables...");
    let mut trainer = NEATTrainer::new(evaluator, neat_config);
    let result = trainer.train()?;
    
    println!("📈 Multiplication results: {:.3} fitness in {} generations", 
            result.state.best_fitness, result.state.generation);
    
    test_arithmetic_network(&result.best_genome, "Multiplication")?;
    
    Ok(())
}

/// Experiment 4: Mixed operations (addition, subtraction, multiplication)
fn run_mixed_operations() -> Result<()> {
    println!("\n🔢 Experiment 4: Mixed Operations Challenge");
    println!("==========================================");
    
    let arithmetic_config = ArithmeticConfig {
        operations: vec![Operation::Add, Operation::Subtract, Operation::Multiply],
        difficulty: DifficultyLevel::SingleDigit,
        allow_negative: false, // Keep it simpler for this challenge
        max_result: 99,
        ..Default::default()
    };
    
    let encoding_config = EncodingConfig {
        scheme: EncodingScheme::Decimal,
        max_digits: 2,
        include_sign: false,
        fixed_length: true,
    };
    
    let evaluator_config = ArithmeticEvaluatorConfig {
        problems_per_evaluation: 60, // More problems to cover all operations
        encoding_config,
        arithmetic_config,
        exact_weight: 1.0,
        partial_weight: 0.5,
        complexity_penalty: 0.001,
        perfect_bonus: 1.0, // Huge bonus for mastering all operations
        ..Default::default()
    };
    
    let evaluator = ArithmeticEvaluator::new(evaluator_config);
    
    let mut neat_config = NEATConfig::default();
    neat_config.population.size = 200; // Large population for complex task
    neat_config.population.max_generations = 60;
    neat_config.population.target_fitness = 1.5;
    
    // Encourage diverse solutions
    neat_config.speciation.target_species_count = 15;
    neat_config.mutation.add_connection_rate = 0.5;
    neat_config.mutation.add_node_rate = 0.25;
    
    println!("🚀 Learning mixed arithmetic operations...");
    let mut trainer = NEATTrainer::new(evaluator, neat_config);
    let result = trainer.train()?;
    
    println!("📈 Mixed operations results: {:.3} fitness in {} generations", 
            result.state.best_fitness, result.state.generation);
    
    test_arithmetic_network(&result.best_genome, "Mixed Operations")?;
    
    Ok(())
}

/// Test a trained network on specific arithmetic problems
fn test_arithmetic_network(genome: &Genome, experiment_name: &str) -> Result<()> {
    println!("\n🧪 Testing {} Network", experiment_name);
    println!("------------------------------");
    
    // Create test problems based on experiment type
    let test_problems = match experiment_name {
        "Single-Digit Addition" => vec![
            MathProblem::new(1, 1, Operation::Add).unwrap(),
            MathProblem::new(5, 3, Operation::Add).unwrap(),
            MathProblem::new(9, 9, Operation::Add).unwrap(),
            MathProblem::new(7, 2, Operation::Add).unwrap(),
        ],
        "Multi-Digit Addition" => vec![
            MathProblem::new(23, 45, Operation::Add).unwrap(),
            MathProblem::new(67, 89, Operation::Add).unwrap(),
            MathProblem::new(19, 26, Operation::Add).unwrap(),
        ],
        "Multiplication" => vec![
            MathProblem::new(2, 3, Operation::Multiply).unwrap(),
            MathProblem::new(7, 8, Operation::Multiply).unwrap(),
            MathProblem::new(9, 6, Operation::Multiply).unwrap(),
            MathProblem::new(4, 4, Operation::Multiply).unwrap(),
        ],
        "Mixed Operations" => vec![
            MathProblem::new(5, 3, Operation::Add).unwrap(),
            MathProblem::new(8, 2, Operation::Subtract).unwrap(),
            MathProblem::new(6, 7, Operation::Multiply).unwrap(),
            MathProblem::new(9, 4, Operation::Add).unwrap(),
        ],
        _ => vec![],
    };
    
    // Set up evaluator for testing
    let encoding_config = EncodingConfig {
        scheme: EncodingScheme::Decimal,
        max_digits: if experiment_name.contains("Multi-Digit") { 3 } else { 2 },
        include_sign: false,
        fixed_length: true,
    };
    
    let evaluator_config = ArithmeticEvaluatorConfig {
        encoding_config,
        ..Default::default()
    };
    
    let evaluator = ArithmeticEvaluator::new(evaluator_config);
    let network = Network::from_genome(genome)?;
    
    // Test the network
    let results = evaluator.evaluate_on_problems(&network, &test_problems)?;
    
    // Display results
    results.print_detailed();
    
    // Network structure analysis
    println!("\n🏗️  Network Architecture:");
    println!("  Total nodes: {}", genome.nodes.len());
    println!("  Active connections: {}", genome.connections.iter().filter(|c| c.enabled).count());
    println!("  Hidden nodes: {}", genome.nodes.iter()
            .filter(|n| matches!(n.node_type, NodeType::Hidden)).count());
    
    if results.is_perfect() {
        println!("  🎉 PERFECT SCORE! Network mastered this arithmetic!");
    } else if results.exact_accuracy > 0.8 {
        println!("  ⭐ Excellent performance!");
    } else if results.exact_accuracy > 0.5 {
        println!("  👍 Good learning progress!");
    } else {
        println!("  🤔 Still learning...");
    }
    
    Ok(())
}

/// Demonstration of different encoding schemes
#[allow(dead_code)]
fn demo_encoding_schemes() -> Result<()> {
    println!("\n🔤 Encoding Scheme Comparison");
    println!("=============================");
    
    let schemes = [
        ("Decimal", EncodingScheme::Decimal),
        ("Binary", EncodingScheme::Binary),
        ("One-Hot", EncodingScheme::OneHot),
        ("Normalized", EncodingScheme::Normalized),
    ];
    
    for (name, scheme) in &schemes {
        let config = EncodingConfig {
            scheme: *scheme,
            max_digits: 2,
            include_sign: false,
            fixed_length: true,
        };
        
        let evaluator_config = ArithmeticEvaluatorConfig {
            encoding_config: config,
            problems_per_evaluation: 20,
            ..Default::default()
        };
        
        let evaluator = ArithmeticEvaluator::new(evaluator_config);
        
        println!("  {}: {} inputs, {} outputs", 
                name, 
                evaluator.input_size(), 
                evaluator.output_size());
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_arithmetic_evaluation() -> Result<()> {
        let evaluator_config = ArithmeticEvaluatorConfig {
            problems_per_evaluation: 5,
            ..Default::default()
        };
        
        let evaluator = ArithmeticEvaluator::new(evaluator_config);
        let genome = Genome::new(0, evaluator.input_size(), evaluator.output_size());
        
        let fitness = evaluator.evaluate(&genome)?;
        assert!(fitness >= 0.0);
        assert!(fitness <= evaluator.max_fitness());
        
        Ok(())
    }
}