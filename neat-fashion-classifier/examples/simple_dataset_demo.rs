//! Simple Dataset Integration Demo
//!
//! This example demonstrates the basic dataset functionality without full Fashion-MNIST.

use neat_fashion_classifier::{
    config::NEATConfig,
    dataset::{DatasetEvaluator, DatasetEvaluatorConfig, Dataset},
    neat::{trainer::NEATTrainer, genome::Genome, fitness::FitnessEvaluator},
    error::Result,
};
use ndarray::{Array1, Array2};

// Simple mock dataset for demonstration
struct SimpleDataset {
    train_images: Array2<f32>,
    train_labels: Array1<u8>,
    test_images: Array2<f32>,
    test_labels: Array1<u8>,
}

impl SimpleDataset {
    fn new() -> Self {
        // Create simple 4x4 images with 3 classes
        let train_images = Array2::from_shape_vec(
            (150, 16),
            (0..2400).map(|i| (i % 256) as f32 / 255.0).collect(),
        ).unwrap();
        
        let train_labels = Array1::from_vec(
            (0..150).map(|i| (i % 3) as u8).collect()
        );
        
        let test_images = Array2::from_shape_vec(
            (30, 16),
            (0..480).map(|i| (i % 256) as f32 / 255.0).collect(),
        ).unwrap();
        
        let test_labels = Array1::from_vec(
            (0..30).map(|i| (i % 3) as u8).collect()
        );
        
        Self {
            train_images,
            train_labels,
            test_images,
            test_labels,
        }
    }
}

impl Dataset for SimpleDataset {
    fn get_train_dimensions(&self) -> (usize, usize) {
        (self.train_images.ncols(), self.train_images.nrows())
    }
    
    fn get_test_dimensions(&self) -> (usize, usize) {
        (self.test_images.ncols(), self.test_images.nrows())
    }
    
    fn get_num_classes(&self) -> usize {
        3
    }
    
    fn get_train_batch(&self, indices: &[usize]) -> Result<(Array2<f32>, Array1<u8>)> {
        let mut batch_images = Array2::zeros((indices.len(), self.train_images.ncols()));
        let mut batch_labels = Array1::zeros(indices.len());
        
        for (i, &idx) in indices.iter().enumerate() {
            if idx < self.train_images.nrows() {
                batch_images.row_mut(i).assign(&self.train_images.row(idx));
                batch_labels[i] = self.train_labels[idx];
            }
        }
        
        Ok((batch_images, batch_labels))
    }
    
    fn get_test_batch(&self, indices: &[usize]) -> Result<(Array2<f32>, Array1<u8>)> {
        let mut batch_images = Array2::zeros((indices.len(), self.test_images.ncols()));
        let mut batch_labels = Array1::zeros(indices.len());
        
        for (i, &idx) in indices.iter().enumerate() {
            if idx < self.test_images.nrows() {
                batch_images.row_mut(i).assign(&self.test_images.row(idx));
                batch_labels[i] = self.test_labels[idx];
            }
        }
        
        Ok((batch_images, batch_labels))
    }
    
    fn get_train_data(&self) -> Result<(Array2<f32>, Array1<u8>)> {
        Ok((self.train_images.clone(), self.train_labels.clone()))
    }
    
    fn get_test_data(&self) -> Result<(Array2<f32>, Array1<u8>)> {
        Ok((self.test_images.clone(), self.test_labels.clone()))
    }
}

fn main() -> Result<()> {
    println!("🧠 Simple Dataset Integration Demo");
    println!("==================================");
    
    // Create simple dataset
    let dataset = SimpleDataset::new();
    println!("📊 Dataset created: {} features, {} classes", 
            dataset.get_train_dimensions().0, 
            dataset.get_num_classes());
    
    // Configure dataset evaluator
    let config = DatasetEvaluatorConfig {
        subset_size: 100,
        validation_split: 0.2,
        batch_size: 20,
        shuffle_data: true,
        random_seed: Some(42),
        complexity_penalty: 0.01,
    };
    
    let evaluator = DatasetEvaluator::new(dataset, config)?;
    println!("✅ Evaluator created with {} training, {} validation samples",
            evaluator.get_train_indices().len(),
            evaluator.get_validation_indices().len());
    
    // Test with a simple genome
    let genome = Genome::new(0, 16, 3);
    println!("🧬 Testing genome: {} inputs, {} outputs", 
            genome.get_input_count(), 
            genome.get_output_count());
    
    let fitness = evaluator.evaluate(&genome)?;
    println!("🎯 Fitness score: {:.3} ({:.1}%)", 
            fitness, 
            fitness * 100.0);
    
    // Quick NEAT evolution demo
    println!("\n🚀 Running mini NEAT evolution...");
    let mut neat_config = NEATConfig::default();
    neat_config.population.size = 20;
    neat_config.population.max_generations = 5;
    neat_config.population.target_fitness = 0.5;
    
    let mut trainer = NEATTrainer::new(evaluator, neat_config);
    let result = trainer.train()?;
    
    println!("🏆 Evolution Results:");
    println!("  Best fitness: {:.3} ({:.1}%)", 
            result.state.best_fitness, 
            result.state.best_fitness * 100.0);
    println!("  Generations: {}", result.state.generation);
    println!("  Success: {}", result.success);
    
    let best_genome = &result.best_genome;
    println!("  Best network: {} nodes, {} connections",
            best_genome.nodes.len(),
            best_genome.connections.iter().filter(|c| c.enabled).count());
    
    println!("\n✨ Demo completed successfully!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use neat_fashion_classifier::neat::fitness::FitnessEvaluator;
    
    #[test]
    fn test_simple_dataset() -> Result<()> {
        let dataset = SimpleDataset::new();
        assert_eq!(dataset.get_num_classes(), 3);
        assert_eq!(dataset.get_train_dimensions(), (16, 150));
        
        let config = DatasetEvaluatorConfig::default();
        let evaluator = DatasetEvaluator::new(dataset, config)?;
        
        let genome = Genome::new(0, 16, 3);
        let fitness = evaluator.evaluate(&genome)?;
        assert!(fitness >= 0.0);
        assert!(fitness <= 1.0);
        
        Ok(())
    }
}