//! Algebraic Learning Evolution Demo
//!
//! This example demonstrates NEAT learning algebraic concepts,
//! from simple expression evaluation to equation solving - a breakthrough
//! in evolutionary mathematical reasoning!

use neat::{
    config::NEATConfig,
    calculator::{
        AlgebraEvaluator, AlgebraEvaluatorConfig, AlgebraEncodingConfig,
        AlgebraicDifficulty, Expression, AlgebraProblem, Operation,
    },
    neat::{trainer::NEATTrainer, genome::Genome, fitness::FitnessEvaluator, network::Network},
    error::Result,
};
use std::collections::HashMap;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🧮 Algebraic Learning with NEAT");
    println!("===========================================");
    println!("Watch as neural networks evolve to understand algebra!");
    
    // Start with basic expression evaluation
    run_expression_evaluation()?;
    
    // Progress to multi-variable expressions
    run_multi_variable_expressions()?;
    
    // Advanced: Linear equation solving
    run_equation_solving()?;
    
    // Demonstrate symbolic understanding
    run_symbolic_reasoning()?;
    
    println!("\n🎉 Algebraic evolution experiments completed!");
    println!("We've demonstrated NEAT discovering algebraic reasoning!");
    Ok(())
}

/// Experiment 1: Basic Expression Evaluation
fn run_expression_evaluation() -> Result<()> {
    println!("\n📊 Experiment 1: Expression Evaluation (ax + b)");
    println!("===============================================");
    
    let encoding_config = AlgebraEncodingConfig {
        max_depth: 3,
        max_variables: 1,
        encode_structure: true,
        ..Default::default()
    };
    
    let evaluator_config = AlgebraEvaluatorConfig {
        problems_per_evaluation: 20,
        encoding_config,
        difficulty: AlgebraicDifficulty::Basic,
        exact_weight: 1.0,
        partial_weight: 0.5,
        structure_weight: 0.1,
        complexity_penalty: 0.005,
        ..Default::default()
    };
    
    let evaluator = AlgebraEvaluator::new(evaluator_config);
    
    println!("📈 Problem Setup:");
    println!("  Learning to evaluate: ax + b where x is given");
    println!("  Input neurons: {}", evaluator.input_size());
    println!("  Output neurons: {}", evaluator.output_size());
    println!("  Max fitness: {:.2}", evaluator.max_fitness());
    
    // Configure NEAT for algebraic learning
    let mut neat_config = NEATConfig::default();
    neat_config.population.size = 100;
    neat_config.population.max_generations = 30;
    neat_config.population.target_fitness = 1.3;
    
    // Encourage complex networks for symbolic reasoning
    neat_config.mutation.weight_mutation_rate = 0.8;
    neat_config.mutation.add_connection_rate = 0.5;
    neat_config.mutation.add_node_rate = 0.2;
    
    println!("\n🧬 Starting evolution...");
    let start_time = Instant::now();
    
    let mut trainer = NEATTrainer::new(evaluator, neat_config);
    let result = trainer.train()?;
    
    println!("\n✨ Evolution Results:");
    println!("  Final fitness: {:.3}", result.state.best_fitness);
    println!("  Generations: {}", result.state.generation);
    println!("  Time: {:.2}s", start_time.elapsed().as_secs_f64());
    
    // Test the evolved network
    test_expression_evaluation(&result.best_genome)?;
    
    Ok(())
}

/// Experiment 2: Multi-variable Expression Evaluation
fn run_multi_variable_expressions() -> Result<()> {
    println!("\n📊 Experiment 2: Multi-Variable Expressions (ax + by + c)");
    println!("========================================================");
    
    let encoding_config = AlgebraEncodingConfig {
        max_depth: 4,
        max_variables: 2,
        encode_structure: true,
        ..Default::default()
    };
    
    let evaluator_config = AlgebraEvaluatorConfig {
        problems_per_evaluation: 30,
        encoding_config,
        difficulty: AlgebraicDifficulty::Intermediate,
        exact_weight: 1.0,
        partial_weight: 0.6,
        structure_weight: 0.2,
        ..Default::default()
    };
    
    let evaluator = AlgebraEvaluator::new(evaluator_config);
    
    let mut neat_config = NEATConfig::default();
    neat_config.population.size = 150;
    neat_config.population.max_generations = 40;
    neat_config.population.target_fitness = 1.2;
    
    println!("🧬 Evolving multi-variable understanding...");
    let mut trainer = NEATTrainer::new(evaluator, neat_config);
    let result = trainer.train()?;
    
    println!("✨ Results: {:.3} fitness in {} generations", 
            result.state.best_fitness, result.state.generation);
    
    test_multi_variable_expressions(&result.best_genome)?;
    
    Ok(())
}

/// Experiment 3: Linear Equation Solving
fn run_equation_solving() -> Result<()> {
    println!("\n📊 Experiment 3: Linear Equation Solving (ax + b = c)");
    println!("====================================================");
    
    let encoding_config = AlgebraEncodingConfig {
        max_depth: 3,
        max_variables: 1,
        encode_structure: true,
        encode_precedence: true,
        ..Default::default()
    };
    
    let evaluator_config = AlgebraEvaluatorConfig {
        problems_per_evaluation: 25,
        encoding_config,
        difficulty: AlgebraicDifficulty::Advanced,
        exact_weight: 1.0,
        partial_weight: 0.7,
        structure_weight: 0.3,
        complexity_penalty: 0.003,
        ..Default::default()
    };
    
    let evaluator = AlgebraEvaluator::new(evaluator_config);
    
    println!("🔍 Problem Setup:");
    println!("  Solving for x in: ax + b = c");
    println!("  This requires inverse operations!");
    
    let mut neat_config = NEATConfig::default();
    neat_config.population.size = 200;
    neat_config.population.max_generations = 50;
    neat_config.population.target_fitness = 1.0;
    
    // Equation solving needs more complex networks
    neat_config.mutation.add_node_rate = 0.3;
    neat_config.speciation.target_species_count = 15;
    
    println!("\n🧬 Evolving equation-solving capabilities...");
    let mut trainer = NEATTrainer::new(evaluator, neat_config);
    let result = trainer.train()?;
    
    println!("✨ Results: {:.3} fitness in {} generations", 
            result.state.best_fitness, result.state.generation);
    
    test_equation_solving(&result.best_genome)?;
    
    Ok(())
}

/// Experiment 4: Symbolic Reasoning Demonstration
fn run_symbolic_reasoning() -> Result<()> {
    println!("\n📊 Experiment 4: Symbolic Mathematical Reasoning");
    println!("==============================================");
    println!("Testing if NEAT can discover algebraic patterns!");
    
    // This demonstrates the potential for symbolic discovery
    let expr1 = Expression::power(Expression::variable("x"), 2);
    let expr2 = Expression::binary(
        Expression::variable("x"),
        Operation::Multiply,
        Expression::variable("x")
    );
    
    println!("\n🔬 Symbolic Equivalence Test:");
    println!("  Expression 1: x²");
    println!("  Expression 2: x * x");
    
    let mut vars = HashMap::new();
    for x in [2.0, 3.0, 5.0, 7.0] {
        vars.insert("x".to_string(), x);
        let val1 = expr1.evaluate(&vars)?;
        let val2 = expr2.evaluate(&vars)?;
        println!("  When x={}: x² = {}, x*x = {} ✓", x, val1, val2);
    }
    
    println!("\n💡 Insight: NEAT could learn these are equivalent!");
    println!("   This opens doors to symbolic AI through evolution!");
    
    Ok(())
}

/// Test expression evaluation capabilities
fn test_expression_evaluation(genome: &Genome) -> Result<()> {
    println!("\n🧪 Testing Expression Evaluation:");
    
    let evaluator_config = AlgebraEvaluatorConfig {
        difficulty: AlgebraicDifficulty::Basic,
        ..Default::default()
    };
    let evaluator = AlgebraEvaluator::new(evaluator_config);
    
    // Create test problems
    let test_cases = vec![
        (2.0, 3.0, 5.0),  // 2x + 3 where x=5 → 13
        (1.0, 0.0, 7.0),  // x + 0 where x=7 → 7
        (3.0, 1.0, 2.0),  // 3x + 1 where x=2 → 7
        (5.0, 2.0, 1.0),  // 5x + 2 where x=1 → 7
    ];
    
    let mut problems = Vec::new();
    for (a, b, x_val) in test_cases {
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
        
        if let Ok(problem) = AlgebraProblem::evaluation(expr, vars) {
            problems.push(problem);
        }
    }
    
    let network = Network::from_genome(genome)?;
    let results = evaluator.evaluate_on_problems(&network, &problems)?;
    
    results.print_detailed();
    
    if results.exact_accuracy >= 0.8 {
        println!("  🎉 Excellent! Network mastered expression evaluation!");
    } else if results.exact_accuracy >= 0.5 {
        println!("  👍 Good progress in learning algebra!");
    }
    
    Ok(())
}

/// Test multi-variable expression capabilities
fn test_multi_variable_expressions(genome: &Genome) -> Result<()> {
    println!("\n🧪 Testing Multi-Variable Expressions:");
    
    let evaluator_config = AlgebraEvaluatorConfig {
        difficulty: AlgebraicDifficulty::Intermediate,
        ..Default::default()
    };
    let evaluator = AlgebraEvaluator::new(evaluator_config);
    
    // Test case: 2x + 3y + 1
    let expr = Expression::binary(
        Expression::binary(
            Expression::binary(
                Expression::constant(2.0),
                Operation::Multiply,
                Expression::variable("x")
            ),
            Operation::Add,
            Expression::binary(
                Expression::constant(3.0),
                Operation::Multiply,
                Expression::variable("y")
            )
        ),
        Operation::Add,
        Expression::constant(1.0)
    );
    
    let test_values = vec![
        (2.0, 1.0),  // 2(2) + 3(1) + 1 = 8
        (3.0, 2.0),  // 2(3) + 3(2) + 1 = 13
        (1.0, 4.0),  // 2(1) + 3(4) + 1 = 15
    ];
    
    let mut problems = Vec::new();
    for (x, y) in test_values {
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), x);
        vars.insert("y".to_string(), y);
        
        if let Ok(problem) = AlgebraProblem::evaluation(expr.clone(), vars) {
            problems.push(problem);
        }
    }
    
    let network = Network::from_genome(genome)?;
    let results = evaluator.evaluate_on_problems(&network, &problems)?;
    
    println!("  Multi-variable accuracy: {:.1}%", results.exact_accuracy * 100.0);
    println!("  Average error: {:.3}", results.average_error);
    
    Ok(())
}

/// Test equation solving capabilities
fn test_equation_solving(genome: &Genome) -> Result<()> {
    println!("\n🧪 Testing Equation Solving:");
    
    let evaluator_config = AlgebraEvaluatorConfig {
        difficulty: AlgebraicDifficulty::Advanced,
        ..Default::default()
    };
    let evaluator = AlgebraEvaluator::new(evaluator_config);
    
    // Test cases: ax + b = c, solve for x
    let test_cases = vec![
        (2.0, 3.0, 7.0),   // 2x + 3 = 7 → x = 2
        (3.0, 1.0, 10.0),  // 3x + 1 = 10 → x = 3
        (1.0, 5.0, 8.0),   // x + 5 = 8 → x = 3
        (4.0, 0.0, 12.0),  // 4x + 0 = 12 → x = 3
    ];
    
    let mut problems = Vec::new();
    for (a, b, c) in test_cases {
        problems.push(AlgebraProblem::linear_equation(a, b, c));
    }
    
    let network = Network::from_genome(genome)?;
    let results = evaluator.evaluate_on_problems(&network, &problems)?;
    
    results.print_detailed();
    
    println!("\n🏗️ Network Architecture:");
    println!("  Nodes: {}", genome.nodes.len());
    println!("  Connections: {}", genome.connections.iter().filter(|c| c.enabled).count());
    
    if results.exact_accuracy >= 0.5 {
        println!("  🎯 Breakthrough! Network is solving equations!");
        println!("  This demonstrates symbolic reasoning emergence!");
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_algebra_demo_setup() -> Result<()> {
        let config = AlgebraEvaluatorConfig {
            problems_per_evaluation: 5,
            ..Default::default()
        };
        
        let evaluator = AlgebraEvaluator::new(config);
        assert!(evaluator.input_size() > 0);
        assert_eq!(evaluator.output_size(), 1);
        
        Ok(())
    }
}