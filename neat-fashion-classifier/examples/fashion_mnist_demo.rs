//! Fashion-MNIST NEAT Classification Demo
//!
//! This example demonstrates training a NEAT network on the Fashion-MNIST dataset.
//! Run with: cargo run --example fashion_mnist_demo

use neat_fashion_classifier::{
    config::NEATConfig,
    dataset::{
        fashion_mnist::{FashionMNISTEvaluator, FashionMNISTDataset},
        DatasetEvaluatorConfig,
    },
    neat::{trainer::NEATTrainer, population::Population},
    error::Result,
};
use std::time::Instant;
use std::env;

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    
    println!("🧠 NEAT Fashion-MNIST Classification Demo");
    println!("==========================================");
    
    // Configuration
    let data_dir = env::temp_dir().join("fashion_mnist_demo");
    let subset_size = 500; // Small subset for demo
    let test_size = 100;
    
    println!("📊 Loading Fashion-MNIST dataset...");
    let dataset_config = DatasetEvaluatorConfig {
        subset_size,
        validation_split: 0.2,
        batch_size: 50,
        shuffle_data: true,
        random_seed: Some(42),
        complexity_penalty: 0.01,
    };
    
    let start_time = Instant::now();
    let evaluator = FashionMNISTEvaluator::with_subset(
        &data_dir,
        subset_size,
        test_size,
        dataset_config,
    )?;
    println!("✅ Dataset loaded in {:.2}s", start_time.elapsed().as_secs_f64());
    
    // Print dataset information
    let dataset_stats = evaluator.get_dataset().get_statistics();
    println!("\n📈 Dataset Statistics:");
    println!("  Training samples: {}", dataset_stats.train_samples);
    println!("  Test samples: {}", dataset_stats.test_samples);
    println!("  Image dimensions: {}x{}", dataset_stats.image_dimensions.0, dataset_stats.image_dimensions.1);
    println!("  Feature size: {}", dataset_stats.feature_size);
    println!("  Number of classes: {}", dataset_stats.num_classes);
    
    // Configure NEAT
    let mut neat_config = NEATConfig::default();
    neat_config.population.size = 50; // Small population for demo
    neat_config.population.max_generations = 20;
    neat_config.population.target_fitness = 0.7; // 70% accuracy target
    
    // Mutation rates for Fashion-MNIST
    neat_config.mutation.weight_mutation_rate = 0.8;
    neat_config.mutation.add_connection_rate = 0.3;
    neat_config.mutation.add_node_rate = 0.1;
    neat_config.mutation.toggle_connection_rate = 0.1;
    
    // Speciation settings
    neat_config.speciation.target_species_count = 8;
    neat_config.speciation.dynamic_threshold = true;
    
    println!("\n🧬 NEAT Configuration:");
    println!("  Population size: {}", neat_config.population.size);
    println!("  Max generations: {}", neat_config.population.max_generations);
    println!("  Target fitness: {:.1}%", neat_config.population.target_fitness * 100.0);
    
    // Create and run trainer
    println!("\n🚀 Starting NEAT evolution...");
    let mut trainer = NEATTrainer::new(evaluator, neat_config);
    
    let evolution_start = Instant::now();
    let training_result = trainer.train()?;
    let evolution_time = evolution_start.elapsed();
    
    println!("\n🏆 Evolution completed in {:.2}s!", evolution_time.as_secs_f64());
    println!("📊 Training Results:");
    println!("  Generations: {}", training_result.generation);
    println!("  Best fitness: {:.3} ({:.1}%)", training_result.best_fitness, training_result.best_fitness * 100.0);
    println!("  Final population size: {}", training_result.final_population_size);
    
    // Evaluate best genome on test data
    if let Some(best_genome) = training_result.best_genome {
        println!("\n🔬 Evaluating best genome on test data...");
        
        // Re-create evaluator for testing
        let test_evaluator = FashionMNISTEvaluator::with_subset(
            &data_dir,
            subset_size,
            test_size,
            DatasetEvaluatorConfig {
                subset_size: test_size,
                validation_split: 0.0, // Use all test data
                batch_size: test_size,
                shuffle_data: false,
                random_seed: Some(42),
                complexity_penalty: 0.0, // No penalty for final evaluation
            },
        )?;
        
        let detailed_results = test_evaluator.evaluate_test_detailed(&best_genome)?;
        
        println!("\n📈 Detailed Test Results:");
        detailed_results.print_detailed();
        
        // Network complexity analysis
        println!("\n🏗️  Best Network Structure:");
        println!("  Nodes: {}", best_genome.nodes.len());
        println!("  Connections: {}", best_genome.connections.iter().filter(|c| c.enabled).count());
        println!("  Genome ID: {}", best_genome.id);
        
        // Performance per class
        println!("\n🎯 Fashion Categories Performance:");
        for (i, &accuracy) in detailed_results.class_accuracies.iter().enumerate() {
            println!("  {}: {:.1}%", 
                    FashionMNISTDataset::get_class_name(i as u8), 
                    accuracy * 100.0);
        }
    }
    
    // Training statistics
    println!("\n📊 Evolution Statistics:");
    println!("  Average generation time: {:.2}s", 
            evolution_time.as_secs_f64() / training_result.generation as f64);
    println!("  Fitness evaluations: ~{}", 
            neat_config.population.size * training_result.generation);
    
    // Clean up
    let _ = std::fs::remove_dir_all(&data_dir);
    
    println!("\n✨ Demo completed successfully!");
    Ok(())
}

/// Helper function to demonstrate network evolution progress
#[allow(dead_code)]
fn run_generation_analysis() -> Result<()> {
    let data_dir = env::temp_dir().join("fashion_mnist_analysis");
    
    // Quick evaluation config
    let config = DatasetEvaluatorConfig {
        subset_size: 200,
        validation_split: 0.2,
        batch_size: 40,
        shuffle_data: true,
        random_seed: Some(42),
        complexity_penalty: 0.005,
    };
    
    let evaluator = FashionMNISTEvaluator::with_subset(&data_dir, 200, 50, config)?;
    
    // Test different network sizes
    println!("🔍 Network Size Analysis:");
    for &num_hidden in &[0, 5, 10, 20] {
        let mut genome = neat_fashion_classifier::neat::genome::Genome::new(0, 784, 10);
        
        // Add hidden nodes (simplified for demo)
        for i in 0..num_hidden {
            let hidden_node = neat_fashion_classifier::neat::genome::NodeGene::new(
                1000 + i,
                neat_fashion_classifier::neat::genome::NodeType::Hidden,
                neat_fashion_classifier::neat::genome::ActivationType::Tanh,
            );
            let _ = genome.add_node(hidden_node);
        }
        
        let fitness = evaluator.evaluate(&genome)?;
        println!("  {} hidden nodes: {:.3} fitness", num_hidden, fitness);
    }
    
    // Clean up
    let _ = std::fs::remove_dir_all(&data_dir);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_demo_components() -> Result<()> {
        // Test that all components can be created
        let data_dir = env::temp_dir().join("fashion_mnist_test_demo");
        
        let config = DatasetEvaluatorConfig {
            subset_size: 10,
            validation_split: 0.2,
            batch_size: 5,
            ..Default::default()
        };
        
        let evaluator = FashionMNISTEvaluator::with_subset(&data_dir, 10, 5, config)?;
        
        // Test evaluation
        let genome = neat_fashion_classifier::neat::genome::Genome::new(0, 784, 10);
        let fitness = evaluator.evaluate(&genome)?;
        assert!(fitness >= 0.0);
        assert!(fitness <= 1.0);
        
        // Clean up
        let _ = std::fs::remove_dir_all(&data_dir);
        Ok(())
    }
}