# NEAT Network in Rust - Project Specification

## Project Overview

This project implements a NeuroEvolution of Augmenting Topologies (NEAT) algorithm in Rust, integrated with HuggingFace datasets and evaluation frameworks. The system will evolve neural networks for image classification tasks, starting with small, manageable datasets.

## 1. Dataset Selection

Based on research of HuggingFace datasets, we've identified optimal starting datasets:

### Primary Target: Fashion-MNIST
- **Dataset**: `zalando-datasets/fashion_mnist`
- **Size**: 70,000 grayscale images (60,000 training + 10,000 test)
- **Image Dimensions**: 28×28 pixels
- **Classes**: 10 clothing categories
- **Format**: Grayscale (single channel)
- **Advantages**: 
  - More challenging than MNIST digits
  - Same simple structure as MNIST
  - Well-established benchmarks
  - Manageable size for evolution experiments

### Secondary Options:
- **MNIST**: `mnist` - For initial testing and validation
- **CIFAR-10**: `cifar10` - For advanced experiments (32×32 color)

### Dataset Loading Strategy:
```python
from datasets import load_dataset
dataset = load_dataset("zalando-datasets/fashion_mnist")
```

## 2. Evaluation Framework

### HuggingFace Evaluate Integration
Using the `evaluate` library for comprehensive metrics:

#### Primary Metrics:
- **Accuracy**: Overall correctness percentage
- **Precision**: Per-class precision scores  
- **Recall**: Per-class recall scores
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

#### Evaluation Code Structure:
```python
import evaluate

# Load metrics
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall") 
f1 = evaluate.load("f1")

# Combined evaluation
combined = evaluate.combine([accuracy, precision, recall, f1])
results = combined.compute(predictions=preds, references=labels)
```

### NEAT-Specific Metrics:
- **Network Complexity**: Node and connection counts
- **Species Diversity**: Population variety measures
- **Innovation Tracking**: Historical topology changes
- **Convergence Rate**: Generations to reach target performance

## 3. NEAT Algorithm Specification

### Core NEAT Components

#### 3.1 Genome Representation
```rust
#[derive(Debug, Clone)]
pub struct Genome {
    pub nodes: Vec<NodeGene>,
    pub connections: Vec<ConnectionGene>,
    pub fitness: f64,
    pub adjusted_fitness: f64,
    pub species_id: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct NodeGene {
    pub id: usize,
    pub node_type: NodeType,
    pub activation: ActivationType,
}

#[derive(Debug, Clone)]
pub struct ConnectionGene {
    pub innovation_id: usize,
    pub input_node: usize,
    pub output_node: usize,
    pub weight: f64,
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub enum NodeType {
    Input,
    Output,
    Hidden,
}

#[derive(Debug, Clone)]
pub enum ActivationType {
    Sigmoid,
    Tanh,
    ReLU,
    Linear,
}
```

#### 3.2 Network Structure
- **Input Layer**: 784 nodes (28×28 pixels)
- **Output Layer**: 10 nodes (fashion categories)
- **Hidden Layers**: Evolved dynamically
- **Initial Topology**: Direct input-to-output connections

#### 3.3 Population Management
```rust
#[derive(Debug)]
pub struct Population {
    pub genomes: Vec<Genome>,
    pub species: Vec<Species>,
    pub generation: usize,
    pub innovation_tracker: InnovationTracker,
    pub config: NEATConfig,
}

#[derive(Debug)]
pub struct Species {
    pub id: usize,
    pub representative: Genome,
    pub members: Vec<usize>, // Genome indices
    pub staleness: usize,
    pub best_fitness: f64,
}
```

#### 3.4 Evolution Operators

##### Structural Mutations:
- **Add Node**: Split existing connection with new node
- **Add Connection**: Create new connection between nodes
- **Remove Connection**: Disable existing connections

##### Weight Mutations:
- **Perturb Weights**: Small random adjustments
- **Replace Weights**: Complete weight replacement
- **Uniform Perturbation**: Consistent weight changes

##### Crossover:
- **Historical Marking**: Align genes by innovation numbers
- **Excess Genes**: From more fit parent
- **Disjoint Genes**: Random selection
- **Matching Genes**: Average or random selection

### 3.5 Speciation Algorithm
```rust
pub fn genetic_distance(genome1: &Genome, genome2: &Genome, config: &NEATConfig) -> f64 {
    let (excess, disjoint, matching) = analyze_compatibility(genome1, genome2);
    let weight_diff = calculate_weight_difference(genome1, genome2);
    
    config.excess_coefficient * excess as f64 +
    config.disjoint_coefficient * disjoint as f64 +
    config.weight_difference_coefficient * weight_diff
}
```

## 4. Rust Implementation Architecture

### 4.1 Project Structure
```
neat-fashion-classifier/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── neat/
│   │   ├── mod.rs
│   │   ├── genome.rs
│   │   ├── population.rs
│   │   ├── species.rs
│   │   ├── innovation.rs
│   │   ├── crossover.rs
│   │   ├── mutation.rs
│   │   └── network.rs
│   ├── data/
│   │   ├── mod.rs
│   │   ├── loader.rs
│   │   └── preprocessor.rs
│   ├── evaluation/
│   │   ├── mod.rs
│   │   ├── fitness.rs
│   │   └── metrics.rs
│   └── config/
│       ├── mod.rs
│       └── neat_config.rs
├── tests/
│   ├── integration_tests.rs
│   └── unit_tests.rs
├── benches/
│   └── neat_benchmarks.rs
├── examples/
│   └── fashion_mnist_evolution.rs
└── data/
    └── cache/
```

### 4.2 Core Dependencies
```toml
[dependencies]
# Core neural network
ndarray = "0.15"
ndarray-rand = "0.14"
rand = "0.8"
serde = { version = "1.0", features = ["derive"] }

# Data handling
hf-hub = "0.3"
image = "0.24"
csv = "1.1"

# Parallel processing
rayon = "1.7"
crossbeam = "0.8"

# Logging and monitoring
log = "0.4"
env_logger = "0.10"

# Performance
criterion = { version = "0.5", features = ["html_reports"] }

# Python integration (for HF datasets)
pyo3 = { version = "0.19", features = ["auto-initialize"] }

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
proptest = "1.0"
```

### 4.3 Key Traits and Interfaces
```rust
pub trait Network {
    fn activate(&mut self, inputs: &[f64]) -> Vec<f64>;
    fn get_complexity(&self) -> (usize, usize); // (nodes, connections)
    fn reset(&mut self);
}

pub trait Fitness {
    fn evaluate(&self, network: &mut dyn Network, dataset: &Dataset) -> f64;
    fn evaluate_batch(&self, networks: &mut [Box<dyn Network>], dataset: &Dataset) -> Vec<f64>;
}

pub trait Mutation {
    fn mutate_structure(&self, genome: &mut Genome, innovation_tracker: &mut InnovationTracker);
    fn mutate_weights(&self, genome: &mut Genome);
}
```

## 5. Integration with HuggingFace

### 5.1 Dataset Loading Pipeline
```rust
use pyo3::prelude::*;

#[derive(Debug)]
pub struct HFDataset {
    train_images: Array2<f32>,
    train_labels: Array1<usize>,
    test_images: Array2<f32>,
    test_labels: Array1<usize>,
}

impl HFDataset {
    pub fn load_fashion_mnist() -> Result<Self, Box<dyn std::error::Error>> {
        Python::with_gil(|py| -> PyResult<HFDataset> {
            let datasets = py.import("datasets")?;
            let dataset = datasets.call_method1("load_dataset", ("zalando-datasets/fashion_mnist",))?;
            
            // Extract training data
            let train = dataset.get_item("train")?;
            let train_images = Self::extract_images(train)?;
            let train_labels = Self::extract_labels(train)?;
            
            // Extract test data
            let test = dataset.get_item("test")?;
            let test_images = Self::extract_images(test)?;
            let test_labels = Self::extract_labels(test)?;
            
            Ok(HFDataset {
                train_images,
                train_labels,
                test_images,
                test_labels,
            })
        })
    }
}
```

### 5.2 Evaluation Integration
```rust
pub struct HFEvaluator {
    accuracy: PyObject,
    precision: PyObject,
    recall: PyObject,
    f1: PyObject,
}

impl HFEvaluator {
    pub fn new() -> Result<Self, PyErr> {
        Python::with_gil(|py| {
            let evaluate = py.import("evaluate")?;
            Ok(HFEvaluator {
                accuracy: evaluate.call_method1("load", ("accuracy",))?.into(),
                precision: evaluate.call_method1("load", ("precision",))?.into(),
                recall: evaluate.call_method1("load", ("recall",))?.into(),
                f1: evaluate.call_method1("load", ("f1",))?.into(),
            })
        })
    }
    
    pub fn compute_metrics(&self, predictions: &[usize], references: &[usize]) -> EvaluationResults {
        Python::with_gil(|py| {
            // Compute all metrics and return structured results
            // Implementation details...
        })
    }
}
```

## 6. Configuration Parameters

### 6.1 NEAT Parameters
```rust
#[derive(Debug, Clone)]
pub struct NEATConfig {
    // Population
    pub population_size: usize,           // 150
    pub max_generations: usize,           // 500
    
    // Speciation
    pub excess_coefficient: f64,          // 1.0
    pub disjoint_coefficient: f64,        // 1.0
    pub weight_difference_coefficient: f64, // 0.4
    pub compatibility_threshold: f64,     // 3.0
    pub species_staleness_threshold: usize, // 15
    
    // Mutation rates
    pub add_node_mutation_rate: f64,      // 0.03
    pub add_connection_mutation_rate: f64, // 0.05
    pub weight_mutation_rate: f64,        // 0.8
    pub weight_perturbation_rate: f64,    // 0.9
    pub weight_replacement_rate: f64,     // 0.1
    
    // Selection
    pub survival_threshold: f64,          // 0.2
    pub interspecies_mating_rate: f64,    // 0.001
    
    // Network
    pub activation_functions: Vec<ActivationType>,
    pub bias_enabled: bool,               // true
    pub recurrent_connections: bool,      // false
}
```

### 6.2 Training Parameters
```rust
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub fitness_evaluation_size: usize,   // 1000 (subset for speed)
    pub validation_frequency: usize,      // 10 generations
    pub checkpoint_frequency: usize,      // 50 generations
    pub early_stopping_patience: usize,   // 100 generations
    pub target_accuracy: f64,             // 0.85
    pub parallel_evaluation: bool,        // true
    pub num_threads: usize,               // CPU cores
}
```

## 7. Performance Targets

### 7.1 Accuracy Benchmarks
- **Baseline Target**: 70% accuracy on Fashion-MNIST
- **Good Performance**: 80% accuracy
- **Excellent Performance**: 85%+ accuracy
- **Comparison**: Standard CNN achieves ~91%

### 7.2 Computational Efficiency
- **Training Time**: <2 hours on modern CPU
- **Generation Time**: <30 seconds per generation
- **Memory Usage**: <4GB RAM
- **Network Size**: <1000 nodes maximum

### 7.3 Evolution Metrics
- **Convergence**: Stable improvement over 50 generations
- **Diversity**: Maintain 3+ species
- **Innovation**: Regular topology improvements
- **Scalability**: Handle population size 100-300

## 8. Testing Strategy

### 8.1 Unit Tests
- Genome operations (mutation, crossover)
- Network activation and forward pass
- Speciation algorithms
- Innovation tracking

### 8.2 Integration Tests  
- Full evolution pipeline
- Dataset loading and preprocessing
- Evaluation metrics computation
- Checkpoint save/load functionality

### 8.3 Benchmarks
- Evolution performance profiling
- Memory usage analysis
- Parallel processing efficiency
- Network activation speed

### 8.4 Property-Based Testing
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn crossover_preserves_innovation_numbers(
        parent1 in genome_strategy(),
        parent2 in genome_strategy()
    ) {
        let offspring = crossover(&parent1, &parent2);
        // Verify innovation number consistency
    }
}
```

## 9. Implementation Phases

### Phase 1: Core NEAT Implementation (Weeks 1-2)
- Basic genome representation
- Mutation operators
- Crossover algorithms
- Simple fitness evaluation

### Phase 2: Dataset Integration (Week 3)
- HuggingFace dataset loading
- Data preprocessing pipeline
- Batch evaluation system

### Phase 3: Advanced Evolution (Week 4)
- Speciation implementation
- Population management
- Innovation tracking

### Phase 4: Evaluation & Metrics (Week 5)
- HuggingFace evaluate integration
- Comprehensive metrics
- Performance monitoring

### Phase 5: Optimization & Testing (Week 6)
- Parallel processing
- Performance optimization
- Comprehensive testing suite

### Phase 6: Documentation & Examples (Week 7)
- API documentation
- Usage examples
- Performance analysis

## 10. Success Criteria

### Functional Requirements:
✅ Successfully load Fashion-MNIST from HuggingFace
✅ Evolve network topologies using NEAT algorithm
✅ Achieve >70% classification accuracy
✅ Integrate HuggingFace evaluation metrics
✅ Support parallel evolution
✅ Provide comprehensive logging and monitoring

### Non-Functional Requirements:
✅ Maintain clean, documented Rust code
✅ Follow NEAT algorithm principles faithfully
✅ Enable reproducible experiments
✅ Support easy configuration changes
✅ Provide performance benchmarks

This specification provides a comprehensive foundation for implementing a NEAT network in Rust with HuggingFace integration, balancing theoretical rigor with practical implementation considerations.