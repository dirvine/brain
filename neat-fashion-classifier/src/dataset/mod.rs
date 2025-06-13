//! Dataset integration for NEAT training
//!
//! This module provides comprehensive dataset loading and preprocessing
//! capabilities for training NEAT networks on real-world problems.

use crate::error::{NEATError, Result};
use crate::neat::fitness::FitnessEvaluator;
use crate::neat::genome::Genome;
use crate::neat::network::Network;
use ndarray::{Array1, Array2};

pub mod fashion_mnist;

/// Common dataset interface for NEAT training
pub trait Dataset {
    /// Get training data dimensions (features, samples)
    fn get_train_dimensions(&self) -> (usize, usize);
    
    /// Get test data dimensions (features, samples)
    fn get_test_dimensions(&self) -> (usize, usize);
    
    /// Get number of output classes
    fn get_num_classes(&self) -> usize;
    
    /// Get a batch of training data
    fn get_train_batch(&self, indices: &[usize]) -> Result<(Array2<f32>, Array1<u8>)>;
    
    /// Get a batch of test data
    fn get_test_batch(&self, indices: &[usize]) -> Result<(Array2<f32>, Array1<u8>)>;
    
    /// Get all training data
    fn get_train_data(&self) -> Result<(Array2<f32>, Array1<u8>)>;
    
    /// Get all test data
    fn get_test_data(&self) -> Result<(Array2<f32>, Array1<u8>)>;
}

/// Configuration for dataset-based fitness evaluation
#[derive(Debug, Clone)]
pub struct DatasetEvaluatorConfig {
    /// Size of subset to use for training (0 = use all data)
    pub subset_size: usize,
    /// Fraction of data to use for validation (0.0 to 1.0)
    pub validation_split: f64,
    /// Batch size for evaluation
    pub batch_size: usize,
    /// Whether to shuffle data between evaluations
    pub shuffle_data: bool,
    /// Random seed for reproducible results
    pub random_seed: Option<u64>,
    /// Complexity penalty coefficient
    pub complexity_penalty: f64,
}

impl Default for DatasetEvaluatorConfig {
    fn default() -> Self {
        Self {
            subset_size: 1000,      // Use 1000 samples by default for fast training
            validation_split: 0.2,  // 80% train, 20% validation
            batch_size: 100,        // Process 100 samples at a time
            shuffle_data: true,     // Shuffle for better generalization
            random_seed: Some(42),  // Reproducible by default
            complexity_penalty: 0.01, // Small penalty for network complexity
        }
    }
}

/// Generic dataset-based fitness evaluator
pub struct DatasetEvaluator<D: Dataset> {
    dataset: D,
    config: DatasetEvaluatorConfig,
    train_indices: Vec<usize>,
    validation_indices: Vec<usize>,
}

impl<D: Dataset> DatasetEvaluator<D> {
    /// Create a new dataset evaluator
    pub fn new(dataset: D, config: DatasetEvaluatorConfig) -> Result<Self> {
        let (_, train_samples) = dataset.get_train_dimensions();
        
        // Validate configuration
        if config.validation_split < 0.0 || config.validation_split >= 1.0 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "validation_split".to_string(),
                value: config.validation_split.to_string(),
            });
        }
        
        if config.batch_size == 0 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "batch_size".to_string(),
                value: config.batch_size.to_string(),
            });
        }
        
        let subset_size = if config.subset_size == 0 {
            train_samples
        } else {
            config.subset_size.min(train_samples)
        };
        
        // Create indices for training and validation
        let mut indices: Vec<usize> = (0..subset_size).collect();
        
        if config.shuffle_data {
            use rand::prelude::*;
            let mut rng = match config.random_seed {
                Some(seed) => SmallRng::seed_from_u64(seed),
                None => SmallRng::from_entropy(),
            };
            indices.shuffle(&mut rng);
        }
        
        let validation_size = (subset_size as f64 * config.validation_split) as usize;
        let train_size = subset_size - validation_size;
        
        let train_indices = indices[..train_size].to_vec();
        let validation_indices = indices[train_size..].to_vec();
        
        Ok(Self {
            dataset,
            config,
            train_indices,
            validation_indices,
        })
    }
    
    /// Evaluate network accuracy on a batch of data
    fn evaluate_batch(&self, network: &Network, images: &Array2<f32>, labels: &Array1<u8>) -> Result<f64> {
        let mut correct_predictions = 0;
        let total_predictions = images.nrows();
        
        for i in 0..total_predictions {
            let input = images.row(i).to_owned();
            let input_f64: Vec<f64> = input.iter().map(|&x| x as f64).collect();
            let outputs = network.activate(&input_f64)?;
            
            // Find predicted class (argmax)
            let predicted_class = outputs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0) as u8;
            
            if predicted_class == labels[i] {
                correct_predictions += 1;
            }
        }
        
        Ok(correct_predictions as f64 / total_predictions as f64)
    }
    
    /// Calculate complexity penalty based on network structure
    fn calculate_complexity_penalty(&self, genome: &Genome) -> f64 {
        let node_count = genome.nodes.len() as f64;
        let connection_count = genome.connections.iter().filter(|c| c.enabled).count() as f64;
        
        // Penalty grows with network complexity
        self.config.complexity_penalty * (node_count + connection_count * 0.5)
    }
    
    /// Get configuration
    pub fn get_config(&self) -> &DatasetEvaluatorConfig {
        &self.config
    }
    
    /// Get dataset reference
    pub fn get_dataset(&self) -> &D {
        &self.dataset
    }
    
    /// Get training indices
    pub fn get_train_indices(&self) -> &[usize] {
        &self.train_indices
    }
    
    /// Get validation indices
    pub fn get_validation_indices(&self) -> &[usize] {
        &self.validation_indices
    }
}

impl<D: Dataset> FitnessEvaluator for DatasetEvaluator<D> {
    fn input_size(&self) -> usize {
        self.dataset.get_train_dimensions().0
    }
    
    fn output_size(&self) -> usize {
        self.dataset.get_num_classes()
    }
    fn evaluate(&self, genome: &Genome) -> Result<f64> {
        // Create network from genome
        let network = Network::from_genome(genome)?;
        
        // Evaluate on validation data in batches
        let mut total_accuracy = 0.0;
        let mut batch_count = 0;
        
        for batch_start in (0..self.validation_indices.len()).step_by(self.config.batch_size) {
            let batch_end = (batch_start + self.config.batch_size).min(self.validation_indices.len());
            let batch_indices = &self.validation_indices[batch_start..batch_end];
            
            let (batch_images, batch_labels) = self.dataset.get_train_batch(batch_indices)?;
            let batch_accuracy = self.evaluate_batch(&network, &batch_images, &batch_labels)?;
            
            total_accuracy += batch_accuracy;
            batch_count += 1;
        }
        
        let average_accuracy = if batch_count > 0 {
            total_accuracy / batch_count as f64
        } else {
            0.0
        };
        
        // Apply complexity penalty
        let complexity_penalty = self.calculate_complexity_penalty(genome);
        let fitness = (average_accuracy - complexity_penalty).max(0.0);
        
        Ok(fitness)
    }
    
    fn max_fitness(&self) -> f64 {
        1.0 // Perfect accuracy is 1.0
    }
}

/// Utility functions for dataset operations
pub mod utils {
    use super::*;
    use ndarray::{Array1, Array2};
    
    /// Normalize image data to [0, 1] range
    pub fn normalize_images(images: Array2<u8>) -> Array2<f32> {
        images.mapv(|x| x as f32 / 255.0)
    }
    
    /// Convert labels to one-hot encoding
    pub fn labels_to_one_hot(labels: &Array1<u8>, num_classes: usize) -> Array2<f32> {
        let mut one_hot = Array2::zeros((labels.len(), num_classes));
        
        for (i, &label) in labels.iter().enumerate() {
            if (label as usize) < num_classes {
                one_hot[[i, label as usize]] = 1.0;
            }
        }
        
        one_hot
    }
    
    /// Calculate classification accuracy
    pub fn calculate_accuracy(predictions: &Array1<usize>, labels: &Array1<u8>) -> f64 {
        if predictions.len() != labels.len() {
            return 0.0;
        }
        
        let correct = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(&pred, &label)| pred == label as usize)
            .count();
        
        correct as f64 / predictions.len() as f64
    }
    
    /// Split dataset into train/validation sets
    pub fn split_dataset(
        size: usize,
        validation_split: f64,
        shuffle: bool,
        seed: Option<u64>,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut indices: Vec<usize> = (0..size).collect();
        
        if shuffle {
            use rand::prelude::*;
            let mut rng = match seed {
                Some(s) => SmallRng::seed_from_u64(s),
                None => SmallRng::from_entropy(),
            };
            indices.shuffle(&mut rng);
        }
        
        let validation_size = (size as f64 * validation_split) as usize;
        let train_size = size - validation_size;
        
        let train_indices = indices[..train_size].to_vec();
        let validation_indices = indices[train_size..].to_vec();
        
        (train_indices, validation_indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;
    
    // Mock dataset for testing
    struct MockDataset {
        train_images: Array2<f32>,
        train_labels: Array1<u8>,
        test_images: Array2<f32>,
        test_labels: Array1<u8>,
    }
    
    impl MockDataset {
        fn new() -> Self {
            // Create simple 4x4 images with 2 classes
            let train_images = Array2::from_shape_vec(
                (100, 16),
                (0..1600).map(|i| (i % 256) as f32 / 255.0).collect(),
            ).unwrap();
            
            let train_labels = Array1::from_vec(
                (0..100).map(|i| (i % 2) as u8).collect()
            );
            
            let test_images = Array2::from_shape_vec(
                (20, 16),
                (0..320).map(|i| (i % 256) as f32 / 255.0).collect(),
            ).unwrap();
            
            let test_labels = Array1::from_vec(
                (0..20).map(|i| (i % 2) as u8).collect()
            );
            
            Self {
                train_images,
                train_labels,
                test_images,
                test_labels,
            }
        }
    }
    
    impl Dataset for MockDataset {
        fn get_train_dimensions(&self) -> (usize, usize) {
            (self.train_images.ncols(), self.train_images.nrows())
        }
        
        fn get_test_dimensions(&self) -> (usize, usize) {
            (self.test_images.ncols(), self.test_images.nrows())
        }
        
        fn get_num_classes(&self) -> usize {
            2
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
    
    #[test]
    fn test_dataset_evaluator_creation() -> Result<()> {
        let dataset = MockDataset::new();
        let config = DatasetEvaluatorConfig::default();
        let evaluator = DatasetEvaluator::new(dataset, config)?;
        
        assert!(evaluator.get_train_indices().len() > 0);
        assert!(evaluator.get_validation_indices().len() > 0);
        
        Ok(())
    }
    
    #[test]
    fn test_dataset_evaluator_with_genome() -> Result<()> {
        let dataset = MockDataset::new();
        let config = DatasetEvaluatorConfig {
            subset_size: 50,
            validation_split: 0.2,
            batch_size: 10,
            ..Default::default()
        };
        let evaluator = DatasetEvaluator::new(dataset, config)?;
        
        // Create a simple genome
        let genome = Genome::new(0, 16, 2);
        
        // Evaluate fitness
        let fitness = evaluator.evaluate(&genome)?;
        assert!(fitness >= 0.0);
        assert!(fitness <= evaluator.max_fitness());
        
        Ok(())
    }
    
    #[test]
    fn test_invalid_configuration() {
        let dataset = MockDataset::new();
        
        // Invalid validation split
        let invalid_config = DatasetEvaluatorConfig {
            validation_split: 1.5,
            ..Default::default()
        };
        assert!(DatasetEvaluator::new(dataset, invalid_config).is_err());
    }
    
    #[test]
    fn test_utils_functions() {
        // Test normalize_images
        let images = Array2::from_elem((2, 4), 255u8);
        let normalized = utils::normalize_images(images);
        assert_eq!(normalized[[0, 0]], 1.0);
        
        // Test labels_to_one_hot
        let labels = Array1::from_vec(vec![0, 1, 2]);
        let one_hot = utils::labels_to_one_hot(&labels, 3);
        assert_eq!(one_hot[[0, 0]], 1.0);
        assert_eq!(one_hot[[1, 1]], 1.0);
        assert_eq!(one_hot[[2, 2]], 1.0);
        
        // Test calculate_accuracy
        let predictions = Array1::from_vec(vec![0, 1, 1, 0]);
        let labels = Array1::from_vec(vec![0, 1, 0, 0]);
        let accuracy = utils::calculate_accuracy(&predictions, &labels);
        assert_eq!(accuracy, 0.75); // 3 out of 4 correct
        
        // Test split_dataset
        let (train_indices, val_indices) = utils::split_dataset(100, 0.2, false, Some(42));
        assert_eq!(train_indices.len(), 80);
        assert_eq!(val_indices.len(), 20);
    }
}