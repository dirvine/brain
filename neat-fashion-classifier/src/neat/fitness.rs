//! Fitness evaluation framework for NEAT
//!
//! This module provides traits and implementations for evaluating genome fitness
//! across different tasks, with specific support for classification problems.

use crate::neat::{Genome, Network};
use crate::error::{NEATError, Result};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Trait for evaluating genome fitness
pub trait FitnessEvaluator {
    /// Evaluate a single genome and return its fitness score
    fn evaluate(&self, genome: &Genome) -> Result<f64>;
    
    /// Evaluate multiple genomes in parallel (default implementation)
    fn evaluate_batch(&self, genomes: &[Genome]) -> Result<Vec<f64>> {
        genomes.iter().map(|g| self.evaluate(g)).collect()
    }
    
    /// Get the maximum possible fitness score (for normalization)
    fn max_fitness(&self) -> f64;
    
    /// Whether higher fitness is better (true) or lower is better (false)
    fn higher_is_better(&self) -> bool {
        true
    }
    
    /// Get the number of inputs expected by this evaluator
    fn input_size(&self) -> usize;
    
    /// Get the number of outputs expected by this evaluator
    fn output_size(&self) -> usize;
}

/// Classification fitness evaluator for datasets like Fashion-MNIST
#[derive(Debug, Clone)]
pub struct ClassificationEvaluator {
    /// Input data (samples × features)
    inputs: Array2<f64>,
    /// Target labels (one-hot encoded)
    targets: Array2<f64>,
    /// Number of classes
    num_classes: usize,
    /// Evaluation metrics to use
    metrics: ClassificationMetrics,
    /// Whether to penalize network complexity
    complexity_penalty: Option<ComplexityPenalty>,
}

/// Configuration for classification metrics
#[derive(Debug, Clone)]
pub struct ClassificationMetrics {
    /// Weight for accuracy (0.0 to 1.0)
    pub accuracy_weight: f64,
    /// Weight for cross-entropy loss (0.0 to 1.0)  
    pub cross_entropy_weight: f64,
    /// Weight for F1 score (0.0 to 1.0)
    pub f1_weight: f64,
}

/// Configuration for penalizing network complexity
#[derive(Debug, Clone)]
pub struct ComplexityPenalty {
    /// Penalty weight for number of connections
    pub connection_penalty: f64,
    /// Penalty weight for number of nodes
    pub node_penalty: f64,
    /// Penalty weight for network depth
    pub depth_penalty: f64,
}

impl Default for ClassificationMetrics {
    fn default() -> Self {
        Self {
            accuracy_weight: 1.0,
            cross_entropy_weight: 0.0,
            f1_weight: 0.0,
        }
    }
}

impl Default for ComplexityPenalty {
    fn default() -> Self {
        Self {
            connection_penalty: 0.01,
            node_penalty: 0.005,
            depth_penalty: 0.001,
        }
    }
}

impl ClassificationEvaluator {
    /// Create a new classification evaluator
    pub fn new(
        inputs: Array2<f64>,
        targets: Array2<f64>,
        num_classes: usize,
    ) -> Result<Self> {
        // Validate input dimensions
        if inputs.nrows() != targets.nrows() {
            return Err(NEATError::InvalidConfiguration {
                parameter: "dataset_size".to_string(),
                value: format!("inputs: {}, targets: {}", inputs.nrows(), targets.nrows()),
            });
        }
        
        if targets.ncols() != num_classes {
            return Err(NEATError::InvalidConfiguration {
                parameter: "num_classes".to_string(),
                value: format!("expected: {}, got: {}", num_classes, targets.ncols()),
            });
        }
        
        Ok(Self {
            inputs,
            targets,
            num_classes,
            metrics: ClassificationMetrics::default(),
            complexity_penalty: None,
        })
    }
    
    /// Set custom metrics configuration
    pub fn with_metrics(mut self, metrics: ClassificationMetrics) -> Self {
        self.metrics = metrics;
        self
    }
    
    /// Set complexity penalty configuration
    pub fn with_complexity_penalty(mut self, penalty: ComplexityPenalty) -> Self {
        self.complexity_penalty = Some(penalty);
        self
    }
    
    /// Calculate accuracy for predictions vs targets
    fn calculate_accuracy(&self, predictions: &Array2<f64>) -> f64 {
        let mut correct = 0;
        let total = predictions.nrows();
        
        for i in 0..total {
            let pred_class = predictions.row(i)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
                
            let true_class = self.targets.row(i)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
                
            if pred_class == true_class {
                correct += 1;
            }
        }
        
        correct as f64 / total as f64
    }
    
    /// Calculate cross-entropy loss
    fn calculate_cross_entropy(&self, predictions: &Array2<f64>) -> f64 {
        let mut total_loss = 0.0;
        let total = predictions.nrows();
        
        for i in 0..total {
            let mut sample_loss = 0.0;
            for j in 0..self.num_classes {
                let pred = predictions[[i, j]].max(1e-15).min(1.0 - 1e-15); // Clamp to avoid log(0)
                let target = self.targets[[i, j]];
                sample_loss -= target * pred.ln();
            }
            total_loss += sample_loss;
        }
        
        total_loss / total as f64
    }
    
    /// Calculate F1 score (macro-averaged)
    fn calculate_f1_score(&self, predictions: &Array2<f64>) -> f64 {
        let mut f1_scores = Vec::with_capacity(self.num_classes);
        
        for class in 0..self.num_classes {
            let mut tp = 0;
            let mut fp = 0;
            let mut fn_count = 0;
            
            for i in 0..predictions.nrows() {
                let pred_class = predictions.row(i)
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                    
                let true_class = self.targets.row(i)
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                
                match (pred_class == class, true_class == class) {
                    (true, true) => tp += 1,
                    (true, false) => fp += 1,
                    (false, true) => fn_count += 1,
                    (false, false) => {}, // TN - not needed for F1
                }
            }
            
            let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
            let recall = if tp + fn_count > 0 { tp as f64 / (tp + fn_count) as f64 } else { 0.0 };
            
            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };
            
            f1_scores.push(f1);
        }
        
        // Macro-averaged F1 score
        f1_scores.iter().sum::<f64>() / f1_scores.len() as f64
    }
    
    /// Calculate complexity penalty
    fn calculate_complexity_penalty(&self, network: &Network) -> f64 {
        if let Some(penalty_config) = &self.complexity_penalty {
            let info = network.get_info();
            penalty_config.connection_penalty * info.num_connections as f64
                + penalty_config.node_penalty * info.num_hidden as f64
                + penalty_config.depth_penalty * info.depth as f64
        } else {
            0.0
        }
    }
}

impl FitnessEvaluator for ClassificationEvaluator {
    fn evaluate(&self, genome: &Genome) -> Result<f64> {
        // Create network from genome
        let network = Network::from_genome(genome)?;
        
        // Make predictions
        let mut predictions = Array2::zeros((self.inputs.nrows(), self.num_classes));
        
        for (i, input_row) in self.inputs.rows().into_iter().enumerate() {
            let input_vec: Vec<f64> = input_row.to_vec();
            let output = network.activate(&input_vec)?;
            
            // Ensure output has correct size
            if output.len() != self.num_classes {
                return Err(NEATError::InvalidGenome {
                    message: format!(
                        "Network output size {} doesn't match expected classes {}",
                        output.len(), self.num_classes
                    ),
                });
            }
            
            // Apply softmax for classification
            let output_softmax = softmax(&output);
            for (j, &value) in output_softmax.iter().enumerate() {
                predictions[[i, j]] = value;
            }
        }
        
        // Calculate fitness components
        let mut fitness = 0.0;
        
        if self.metrics.accuracy_weight > 0.0 {
            let accuracy = self.calculate_accuracy(&predictions);
            fitness += self.metrics.accuracy_weight * accuracy;
        }
        
        if self.metrics.cross_entropy_weight > 0.0 {
            let cross_entropy = self.calculate_cross_entropy(&predictions);
            // Convert loss to fitness (lower loss = higher fitness)
            let ce_fitness = (-cross_entropy).exp();
            fitness += self.metrics.cross_entropy_weight * ce_fitness;
        }
        
        if self.metrics.f1_weight > 0.0 {
            let f1 = self.calculate_f1_score(&predictions);
            fitness += self.metrics.f1_weight * f1;
        }
        
        // Apply complexity penalty
        let penalty = self.calculate_complexity_penalty(&network);
        fitness -= penalty;
        
        // Ensure fitness is non-negative
        Ok(fitness.max(0.0))
    }
    
    fn max_fitness(&self) -> f64 {
        // Maximum possible fitness is the sum of all metric weights
        self.metrics.accuracy_weight + self.metrics.cross_entropy_weight + self.metrics.f1_weight
    }
    
    fn input_size(&self) -> usize {
        self.inputs.ncols()
    }
    
    fn output_size(&self) -> usize {
        self.num_classes
    }
}

/// Simple XOR fitness evaluator for testing
#[derive(Debug, Clone)]
pub struct XORFitnessEvaluator {
    /// Expected error threshold for perfect fitness
    pub error_threshold: f64,
}

impl Default for XORFitnessEvaluator {
    fn default() -> Self {
        Self {
            error_threshold: 0.01,
        }
    }
}

impl FitnessEvaluator for XORFitnessEvaluator {
    fn evaluate(&self, genome: &Genome) -> Result<f64> {
        let network = Network::from_genome(genome)?;
        
        // XOR truth table
        let test_cases = [
            (vec![0.0, 0.0], 0.0),
            (vec![0.0, 1.0], 1.0),
            (vec![1.0, 0.0], 1.0),
            (vec![1.0, 1.0], 0.0),
        ];
        
        let mut total_error = 0.0;
        
        for (inputs, expected) in &test_cases {
            let outputs = network.activate(inputs)?;
            
            if outputs.is_empty() {
                return Err(NEATError::InvalidGenome {
                    message: "Network has no outputs".to_string(),
                });
            }
            
            let error = (outputs[0] - expected).abs();
            total_error += error;
        }
        
        let average_error = total_error / test_cases.len() as f64;
        
        // Convert error to fitness: lower error = higher fitness
        let fitness = if average_error < self.error_threshold {
            1.0 // Perfect fitness
        } else {
            1.0 / (1.0 + average_error) // Inverse relationship
        };
        
        Ok(fitness)
    }
    
    fn max_fitness(&self) -> f64 {
        1.0
    }
    
    fn input_size(&self) -> usize {
        2 // XOR has 2 inputs
    }
    
    fn output_size(&self) -> usize {
        1 // XOR has 1 output
    }
}

/// Apply softmax normalization to outputs
fn softmax(outputs: &[f64]) -> Vec<f64> {
    let max_val = outputs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_vals: Vec<f64> = outputs.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exp_vals.iter().sum();
    
    if sum == 0.0 {
        // Handle edge case where all values are very negative
        vec![1.0 / outputs.len() as f64; outputs.len()]
    } else {
        exp_vals.iter().map(|&x| x / sum).collect()
    }
}

/// Fitness evaluation results with detailed metrics
#[derive(Debug, Clone)]
pub struct FitnessResults {
    /// Overall fitness score
    pub fitness: f64,
    /// Detailed metrics breakdown
    pub metrics: HashMap<String, f64>,
    /// Network complexity info
    pub complexity: (usize, usize), // (nodes, connections)
}

/// Batch fitness evaluator for parallel processing
pub struct BatchEvaluator<E: FitnessEvaluator> {
    evaluator: E,
}

impl<E: FitnessEvaluator> BatchEvaluator<E> {
    /// Create a new batch evaluator
    pub fn new(evaluator: E) -> Self {
        Self { evaluator }
    }
    
    /// Evaluate multiple genomes with detailed results
    pub fn evaluate_detailed(&self, genomes: &[Genome]) -> Result<Vec<FitnessResults>> {
        genomes.iter().map(|genome| {
            let fitness = self.evaluator.evaluate(genome)?;
            let network = Network::from_genome(genome)?;
            let info = network.get_info();
            
            let mut metrics = HashMap::new();
            metrics.insert("fitness".to_string(), fitness);
            
            Ok(FitnessResults {
                fitness,
                metrics,
                complexity: (info.num_hidden + info.num_inputs + info.num_outputs, info.num_connections),
            })
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neat::genome::{NodeGene, ConnectionGene, NodeType, ActivationType};
    use approx::assert_relative_eq;

    #[test]
    fn test_xor_evaluator() {
        let evaluator = XORFitnessEvaluator::default();
        
        // Create a simple genome (2 inputs, 1 output)
        let genome = Genome::new(0, 2, 1);
        
        // Should get some fitness even without training
        let fitness = evaluator.evaluate(&genome).unwrap();
        assert!(fitness >= 0.0);
        assert!(fitness <= 1.0);
        assert_eq!(evaluator.max_fitness(), 1.0);
    }
    
    #[test]
    fn test_classification_evaluator_creation() {
        // Create simple test data: 4 samples, 2 features, 2 classes
        let inputs = Array2::from_shape_vec((4, 2), vec![
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
            0.0, 0.0,
        ]).unwrap();
        
        let targets = Array2::from_shape_vec((4, 2), vec![
            1.0, 0.0,  // Class 0
            0.0, 1.0,  // Class 1
            0.0, 1.0,  // Class 1
            1.0, 0.0,  // Class 0
        ]).unwrap();
        
        let evaluator = ClassificationEvaluator::new(inputs, targets, 2).unwrap();
        assert_eq!(evaluator.max_fitness(), 1.0); // Default accuracy weight only
    }
    
    #[test]
    fn test_classification_evaluator_with_genome() {
        // Create test data
        let inputs = Array2::from_shape_vec((2, 2), vec![
            1.0, 0.0,
            0.0, 1.0,
        ]).unwrap();
        
        let targets = Array2::from_shape_vec((2, 2), vec![
            1.0, 0.0,
            0.0, 1.0,
        ]).unwrap();
        
        let evaluator = ClassificationEvaluator::new(inputs, targets, 2).unwrap();
        
        // Create genome (2 inputs, 2 outputs)
        let genome = Genome::new(0, 2, 2);
        
        let fitness = evaluator.evaluate(&genome).unwrap();
        assert!(fitness >= 0.0);
        assert!(fitness <= 1.0);
    }
    
    #[test]
    fn test_softmax_function() {
        let outputs = vec![1.0, 2.0, 3.0];
        let softmax_outputs = softmax(&outputs);
        
        // Should sum to 1.0
        let sum: f64 = softmax_outputs.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        
        // Should be monotonic (higher input = higher output)
        assert!(softmax_outputs[0] < softmax_outputs[1]);
        assert!(softmax_outputs[1] < softmax_outputs[2]);
    }
    
    #[test]
    fn test_complexity_penalty() {
        let inputs = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let targets = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        
        let penalty = ComplexityPenalty {
            connection_penalty: 0.1,
            node_penalty: 0.05,
            depth_penalty: 0.01,
        };
        
        let evaluator = ClassificationEvaluator::new(inputs, targets, 2)
            .unwrap()
            .with_complexity_penalty(penalty);
        
        // Create two genomes: one simple, one complex
        let simple_genome = Genome::new(0, 2, 2);
        
        let mut complex_genome = Genome::new(1, 2, 2);
        let hidden_node = NodeGene::new(10, NodeType::Hidden, ActivationType::Sigmoid);
        complex_genome.add_node(hidden_node).unwrap();
        let conn = ConnectionGene::new(0, 0, 10, 1.0);
        complex_genome.add_connection(conn).unwrap();
        
        let simple_fitness = evaluator.evaluate(&simple_genome).unwrap();
        let complex_fitness = evaluator.evaluate(&complex_genome).unwrap();
        
        // Complex genome should have lower fitness due to penalty
        assert!(simple_fitness >= complex_fitness);
    }
    
    #[test]
    fn test_batch_evaluator() {
        let evaluator = XORFitnessEvaluator::default();
        let batch_evaluator = BatchEvaluator::new(evaluator);
        
        let genomes = vec![
            Genome::new(0, 2, 1),
            Genome::new(1, 2, 1),
        ];
        
        let results = batch_evaluator.evaluate_detailed(&genomes).unwrap();
        assert_eq!(results.len(), 2);
        
        for result in &results {
            assert!(result.fitness >= 0.0);
            assert!(result.fitness <= 1.0);
            assert!(result.metrics.contains_key("fitness"));
        }
    }
    
    #[test]
    fn test_invalid_dataset_dimensions() {
        let inputs = Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]).unwrap();
        let targets = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0]).unwrap(); // Wrong number of samples
        
        let result = ClassificationEvaluator::new(inputs, targets, 2);
        assert!(result.is_err());
    }
}