//! Simple Addition Learning Demo
//!
//! This example focuses on the simplest possible arithmetic: single-digit addition.
//! Perfect for demonstrating how NEAT can discover mathematical concepts.

use neat_fashion_classifier::{
    config::NEATConfig,
    calculator::{
        ArithmeticEvaluator, ArithmeticEvaluatorConfig, ArithmeticConfig, EncodingConfig,
        Operation, DifficultyLevel, EncodingScheme, MathProblem,
    },
    neat::{trainer::NEATTrainer, genome::Genome, fitness::FitnessEvaluator, network::Network},
    error::Result,
};

fn main() -> Result<()> {
    println!("🧮 Simple Addition Learning with NEAT");
    println!("=====================================");
    
    // Configure for the simplest possible arithmetic: single-digit addition
    let arithmetic_config = ArithmeticConfig {
        operations: vec![Operation::Add],
        difficulty: DifficultyLevel::SingleDigit,
        allow_negative: false,
        max_result: 18, // 9+9=18 is the maximum
        bounded_results: true,
        random_seed: Some(42), // Reproducible results
    };
    
    // Use simple decimal encoding
    let encoding_config = EncodingConfig {
        scheme: EncodingScheme::Decimal,
        max_digits: 2, // Handle results up to 18
        include_sign: false,
        fixed_length: true,
    };
    
    // Configure evaluator for learning
    let evaluator_config = ArithmeticEvaluatorConfig {
        problems_per_evaluation: 20, // Small set for faster learning
        encoding_config,
        arithmetic_config,
        tolerance: 0.1,
        exact_weight: 1.0,     // Focus on exact answers
        partial_weight: 0.5,   // Some credit for close answers
        complexity_penalty: 0.005, // Small penalty for complexity
        perfect_bonus: 0.5,    // Big bonus for perfect performance
    };
    
    let evaluator = ArithmeticEvaluator::new(evaluator_config);
    
    println!("📊 Problem Setup:");
    println!("  Learning: Single-digit addition (0+0 to 9+9)");
    println!("  Input neurons: {}", evaluator.input_size());
    println!("  Output neurons: {}", evaluator.output_size());
    println!("  Max fitness: {:.2}", evaluator.max_fitness());
    
    // Configure NEAT for simple learning
    let mut neat_config = NEATConfig::default();
    neat_config.population.size = 50;           // Small population
    neat_config.population.max_generations = 20; // Quick learning
    neat_config.population.target_fitness = 1.3; // Achievable target
    
    // Encourage exploration
    neat_config.mutation.weight_mutation_rate = 0.8;
    neat_config.mutation.add_connection_rate = 0.3;
    neat_config.mutation.add_node_rate = 0.1;
    neat_config.mutation.weight_perturbation_power = 0.5;
    
    println!("\n🧬 NEAT Setup:");
    println!("  Population: {}", neat_config.population.size);
    println!("  Max generations: {}", neat_config.population.max_generations);
    println!("  Target fitness: {:.2}", neat_config.population.target_fitness);
    
    // Run evolution
    println!("\n🚀 Starting evolution...");
    let mut trainer = NEATTrainer::new(evaluator, neat_config);
    let result = trainer.train()?;
    
    println!("\n📈 Results:");
    println!("  Final fitness: {:.3}", result.state.best_fitness);
    println!("  Generations: {}", result.state.generation);
    println!("  Success: {}", result.success);
    
    // Test on specific addition problems
    println!("\n🧪 Testing the evolved network:");
    test_specific_problems(&result.best_genome)?;
    
    // Show network structure
    println!("\n🏗️  Network Architecture:");
    println!("  Total nodes: {}", result.best_genome.nodes.len());
    println!("  Active connections: {}", 
            result.best_genome.connections.iter().filter(|c| c.enabled).count());
    
    Ok(())
}

fn test_specific_problems(genome: &Genome) -> Result<()> {
    // Create test problems covering the addition table
    let test_problems = vec![
        MathProblem::new(0, 0, Operation::Add).unwrap(), // 0+0=0
        MathProblem::new(1, 1, Operation::Add).unwrap(), // 1+1=2
        MathProblem::new(2, 3, Operation::Add).unwrap(), // 2+3=5
        MathProblem::new(5, 4, Operation::Add).unwrap(), // 5+4=9
        MathProblem::new(6, 7, Operation::Add).unwrap(), // 6+7=13
        MathProblem::new(9, 9, Operation::Add).unwrap(), // 9+9=18 (hardest)
    ];
    
    // Set up the evaluator for testing
    let encoding_config = EncodingConfig {
        scheme: EncodingScheme::Decimal,
        max_digits: 2,
        include_sign: false,
        fixed_length: true,
    };
    
    let evaluator_config = ArithmeticEvaluatorConfig {
        encoding_config,
        ..Default::default()
    };
    
    let evaluator = ArithmeticEvaluator::new(evaluator_config);
    let network = Network::from_genome(genome)?;
    
    // Test each problem
    println!("  Testing key addition problems:");
    let mut correct = 0;
    
    for problem in &test_problems {
        let input = evaluator.encode_problem(problem)?;
        let output = network.activate(&input)?;
        let predicted = evaluator.decode_result(&output)?;
        
        let is_correct = predicted == problem.result;
        if is_correct { correct += 1; }
        
        let status = if is_correct { "✓" } else { "✗" };
        println!("    {} = {} (predicted: {}) {}", 
                problem.to_string().split(" = ").next().unwrap(),
                problem.result,
                predicted,
                status);
    }
    
    let accuracy = correct as f64 / test_problems.len() as f64;
    println!("  Overall accuracy: {:.1}% ({}/{})", 
            accuracy * 100.0, correct, test_problems.len());
    
    if accuracy >= 0.8 {
        println!("  🎉 Excellent! The network learned addition!");
    } else if accuracy >= 0.5 {
        println!("  👍 Good progress, learning in progress!");
    } else {
        println!("  🤔 Still figuring out the pattern...");
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_addition_setup() -> Result<()> {
        let evaluator_config = ArithmeticEvaluatorConfig {
            problems_per_evaluation: 5,
            ..Default::default()
        };
        
        let evaluator = ArithmeticEvaluator::new(evaluator_config);
        
        // Test basic functionality
        assert_eq!(evaluator.input_size(), 8);  // 2 numbers + 4 operation bits
        assert_eq!(evaluator.output_size(), 4); // 2-digit result
        
        let genome = Genome::new(0, evaluator.input_size(), evaluator.output_size());
        let fitness = evaluator.evaluate(&genome)?;
        assert!(fitness >= 0.0);
        
        Ok(())
    }
}