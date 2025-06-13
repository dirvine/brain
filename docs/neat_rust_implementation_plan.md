# NEAT Rust Implementation Plan

## Overview

This document provides a detailed implementation plan for building a NEAT (NeuroEvolution of Augmenting Topologies) network in Rust with HuggingFace dataset integration. The plan follows specification-driven development principles and includes comprehensive testing strategies.

## Phase 1: Project Setup and Core Structures (Week 1)

### 1.1 Project Initialization
```bash
# Create new Rust project
cargo new neat-fashion-classifier --lib
cd neat-fashion-classifier

# Initialize git repository
git init
git add .
git commit -m "Initial project setup"
```

### 1.2 Cargo.toml Configuration
```toml
[package]
name = "neat-fashion-classifier"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <your.email@example.com>"]
description = "NEAT algorithm implementation for Fashion-MNIST classification"
license = "MIT"
repository = "https://github.com/dirvine/brain"

[dependencies]
# Core dependencies
ndarray = "0.15"
ndarray-rand = "0.14"
rand = { version = "0.8", features = ["small_rng"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Data handling
hf-hub = "0.3"
image = "0.24"
csv = "1.1"

# Parallel processing
rayon = "1.7"
crossbeam = "0.8"

# Logging
log = "0.4"
env_logger = "0.10"

# Python integration
pyo3 = { version = "0.19", features = ["auto-initialize"] }

# Error handling
thiserror = "1.0"
anyhow = "1.0"

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
proptest = "1.0"
tempfile = "3.0"

[[bench]]
name = "neat_benchmarks"
harness = false

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

### 1.3 Project Structure Setup
```bash
mkdir -p src/{neat,data,evaluation,config}
mkdir -p tests benches examples data/cache
```

### 1.4 Core Data Structures Implementation

#### src/neat/genome.rs
```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genome {
    pub nodes: Vec<NodeGene>,
    pub connections: Vec<ConnectionGene>,
    pub fitness: f64,
    pub adjusted_fitness: f64,
    pub species_id: Option<usize>,
    pub id: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeGene {
    pub id: usize,
    pub node_type: NodeType,
    pub activation: ActivationType,
    pub bias: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionGene {
    pub innovation_id: usize,
    pub input_node: usize,
    pub output_node: usize,
    pub weight: f64,
    pub enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    Input,
    Output,
    Hidden,
    Bias,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationType {
    Sigmoid,
    Tanh,
    ReLU,
    Linear,
    Gaussian,
}

impl Genome {
    pub fn new(id: usize, input_size: usize, output_size: usize) -> Self {
        let mut nodes = Vec::new();
        let mut connections = Vec::new();
        
        // Add input nodes
        for i in 0..input_size {
            nodes.push(NodeGene {
                id: i,
                node_type: NodeType::Input,
                activation: ActivationType::Linear,
                bias: 0.0,
            });
        }
        
        // Add bias node
        nodes.push(NodeGene {
            id: input_size,
            node_type: NodeType::Bias,
            activation: ActivationType::Linear,
            bias: 1.0,
        });
        
        // Add output nodes
        for i in 0..output_size {
            nodes.push(NodeGene {
                id: input_size + 1 + i,
                node_type: NodeType::Output,
                activation: ActivationType::Sigmoid,
                bias: 0.0,
            });
        }
        
        Self {
            nodes,
            connections,
            fitness: 0.0,
            adjusted_fitness: 0.0,
            species_id: None,
            id,
        }
    }
    
    pub fn get_complexity(&self) -> (usize, usize) {
        (self.nodes.len(), self.connections.iter().filter(|c| c.enabled).count())
    }
}
```

#### src/neat/innovation.rs
```rust
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Innovation {
    pub innovation_id: usize,
    pub input_node: usize,
    pub output_node: usize,
    pub innovation_type: InnovationType,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InnovationType {
    NewConnection,
    NewNode { split_connection: usize },
}

#[derive(Debug, Default)]
pub struct InnovationTracker {
    innovations: HashMap<(usize, usize), usize>,
    next_innovation_id: usize,
    innovation_history: Vec<Innovation>,
}

impl InnovationTracker {
    pub fn new() -> Self {
        Self {
            innovations: HashMap::new(),
            next_innovation_id: 0,
            innovation_history: Vec::new(),
        }
    }
    
    pub fn get_innovation_id(&mut self, input: usize, output: usize) -> usize {
        if let Some(&id) = self.innovations.get(&(input, output)) {
            id
        } else {
            let id = self.next_innovation_id;
            self.innovations.insert((input, output), id);
            self.innovation_history.push(Innovation {
                innovation_id: id,
                input_node: input,
                output_node: output,
                innovation_type: InnovationType::NewConnection,
            });
            self.next_innovation_id += 1;
            id
        }
    }
    
    pub fn get_node_innovation_id(&mut self, split_connection: usize, input: usize, output: usize) -> (usize, usize) {
        // Return innovation IDs for the two new connections created when adding a node
        let in_to_new = self.get_innovation_id(input, self.next_innovation_id);
        let new_to_out = self.get_innovation_id(self.next_innovation_id, output);
        
        self.innovation_history.push(Innovation {
            innovation_id: self.next_innovation_id,
            input_node: input,
            output_node: output,
            innovation_type: InnovationType::NewNode { split_connection },
        });
        
        (in_to_new, new_to_out)
    }
}
```

### 1.5 Configuration System

#### src/config/neat_config.rs
```rust
use serde::{Deserialize, Serialize};
use crate::neat::genome::ActivationType;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NEATConfig {
    // Population parameters
    pub population_size: usize,
    pub max_generations: usize,
    pub target_fitness: f64,
    
    // Speciation parameters
    pub excess_coefficient: f64,
    pub disjoint_coefficient: f64,
    pub weight_difference_coefficient: f64,
    pub compatibility_threshold: f64,
    pub species_staleness_threshold: usize,
    pub min_species_size: usize,
    
    // Mutation rates
    pub add_node_mutation_rate: f64,
    pub add_connection_mutation_rate: f64,
    pub weight_mutation_rate: f64,
    pub weight_perturbation_rate: f64,
    pub weight_replacement_rate: f64,
    pub weight_perturbation_power: f64,
    pub disable_connection_rate: f64,
    pub enable_connection_rate: f64,
    
    // Selection parameters
    pub survival_threshold: f64,
    pub interspecies_mating_rate: f64,
    pub elitism_rate: f64,
    
    // Network parameters
    pub activation_functions: Vec<ActivationType>,
    pub bias_enabled: bool,
    pub recurrent_connections: bool,
    pub max_nodes: usize,
    pub max_connections: usize,
    
    // Training parameters
    pub fitness_evaluation_size: usize,
    pub validation_frequency: usize,
    pub checkpoint_frequency: usize,
    pub early_stopping_patience: usize,
}

impl Default for NEATConfig {
    fn default() -> Self {
        Self {
            population_size: 150,
            max_generations: 500,
            target_fitness: 0.85,
            
            excess_coefficient: 1.0,
            disjoint_coefficient: 1.0,
            weight_difference_coefficient: 0.4,
            compatibility_threshold: 3.0,
            species_staleness_threshold: 15,
            min_species_size: 5,
            
            add_node_mutation_rate: 0.03,
            add_connection_mutation_rate: 0.05,
            weight_mutation_rate: 0.8,
            weight_perturbation_rate: 0.9,
            weight_replacement_rate: 0.1,
            weight_perturbation_power: 0.5,
            disable_connection_rate: 0.01,
            enable_connection_rate: 0.01,
            
            survival_threshold: 0.2,
            interspecies_mating_rate: 0.001,
            elitism_rate: 0.1,
            
            activation_functions: vec![
                ActivationType::Sigmoid,
                ActivationType::Tanh,
                ActivationType::ReLU,
            ],
            bias_enabled: true,
            recurrent_connections: false,
            max_nodes: 1000,
            max_connections: 5000,
            
            fitness_evaluation_size: 1000,
            validation_frequency: 10,
            checkpoint_frequency: 50,
            early_stopping_patience: 100,
        }
    }
}
```

## Phase 2: Network Implementation (Week 2)

### 2.1 Neural Network Activation

#### src/neat/network.rs
```rust
use crate::neat::genome::{Genome, NodeType, ActivationType};
use std::collections::HashMap;
use ndarray::Array1;

pub struct NeuralNetwork {
    nodes: Vec<NetworkNode>,
    connections: Vec<NetworkConnection>,
    node_map: HashMap<usize, usize>, // genome_id -> network_index
    input_indices: Vec<usize>,
    output_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
struct NetworkNode {
    id: usize,
    node_type: NodeType,
    activation: ActivationType,
    bias: f64,
    value: f64,
    activated: bool,
}

#[derive(Debug, Clone)]
struct NetworkConnection {
    input_index: usize,
    output_index: usize,
    weight: f64,
    enabled: bool,
}

impl NeuralNetwork {
    pub fn from_genome(genome: &Genome) -> Self {
        let mut nodes = Vec::new();
        let mut node_map = HashMap::new();
        let mut input_indices = Vec::new();
        let mut output_indices = Vec::new();
        
        // Create nodes
        for (index, node_gene) in genome.nodes.iter().enumerate() {
            nodes.push(NetworkNode {
                id: node_gene.id,
                node_type: node_gene.node_type,
                activation: node_gene.activation,
                bias: node_gene.bias,
                value: 0.0,
                activated: false,
            });
            
            node_map.insert(node_gene.id, index);
            
            match node_gene.node_type {
                NodeType::Input | NodeType::Bias => input_indices.push(index),
                NodeType::Output => output_indices.push(index),
                _ => {}
            }
        }
        
        // Create connections
        let mut connections = Vec::new();
        for conn_gene in &genome.connections {
            if conn_gene.enabled {
                if let (Some(&input_idx), Some(&output_idx)) = (
                    node_map.get(&conn_gene.input_node),
                    node_map.get(&conn_gene.output_node)
                ) {
                    connections.push(NetworkConnection {
                        input_index: input_idx,
                        output_index: output_idx,
                        weight: conn_gene.weight,
                        enabled: true,
                    });
                }
            }
        }
        
        Self {
            nodes,
            connections,
            node_map,
            input_indices,
            output_indices,
        }
    }
    
    pub fn activate(&mut self, inputs: &[f64]) -> Vec<f64> {
        // Reset network state
        for node in &mut self.nodes {
            node.value = 0.0;
            node.activated = false;
        }
        
        // Set input values
        for (i, &input) in inputs.iter().enumerate() {
            if i < self.input_indices.len() {
                let idx = self.input_indices[i];
                self.nodes[idx].value = input;
                self.nodes[idx].activated = true;
            }
        }
        
        // Set bias values
        for &idx in &self.input_indices {
            if self.nodes[idx].node_type == NodeType::Bias {
                self.nodes[idx].value = self.nodes[idx].bias;
                self.nodes[idx].activated = true;
            }
        }
        
        // Propagate signals through network
        let max_iterations = self.nodes.len() * 2; // Prevent infinite loops
        for _ in 0..max_iterations {
            let mut changed = false;
            
            for connection in &self.connections {
                if connection.enabled && self.nodes[connection.input_index].activated {
                    self.nodes[connection.output_index].value += 
                        self.nodes[connection.input_index].value * connection.weight;
                    changed = true;
                }
            }
            
            // Activate nodes that have received all inputs
            for node in &mut self.nodes {
                if !node.activated && node.node_type != NodeType::Input && node.node_type != NodeType::Bias {
                    node.value = self.apply_activation(node.value + node.bias, node.activation);
                    node.activated = true;
                }
            }
            
            if !changed {
                break;
            }
        }
        
        // Collect output values
        self.output_indices.iter()
            .map(|&idx| self.nodes[idx].value)
            .collect()
    }
    
    fn apply_activation(&self, x: f64, activation: ActivationType) -> f64 {
        match activation {
            ActivationType::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationType::Tanh => x.tanh(),
            ActivationType::ReLU => x.max(0.0),
            ActivationType::Linear => x,
            ActivationType::Gaussian => (-x * x).exp(),
        }
    }
    
    pub fn get_complexity(&self) -> (usize, usize) {
        (self.nodes.len(), self.connections.len())
    }
}

pub trait Network {
    fn activate(&mut self, inputs: &[f64]) -> Vec<f64>;
    fn get_complexity(&self) -> (usize, usize);
    fn reset(&mut self);
}

impl Network for NeuralNetwork {
    fn activate(&mut self, inputs: &[f64]) -> Vec<f64> {
        self.activate(inputs)
    }
    
    fn get_complexity(&self) -> (usize, usize) {
        self.get_complexity()
    }
    
    fn reset(&mut self) {
        for node in &mut self.nodes {
            node.value = 0.0;
            node.activated = false;
        }
    }
}
```

### 2.2 Unit Tests for Core Components

#### tests/genome_tests.rs
```rust
use neat_fashion_classifier::neat::genome::*;
use neat_fashion_classifier::neat::innovation::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genome_creation() {
        let genome = Genome::new(0, 784, 10);
        assert_eq!(genome.nodes.len(), 795); // 784 inputs + 1 bias + 10 outputs
        assert_eq!(genome.connections.len(), 0); // No initial connections
        assert_eq!(genome.id, 0);
    }

    #[test]
    fn test_complexity_calculation() {
        let mut genome = Genome::new(0, 2, 1);
        let (nodes, connections) = genome.get_complexity();
        assert_eq!(nodes, 4); // 2 inputs + 1 bias + 1 output
        assert_eq!(connections, 0);
    }

    #[test]
    fn test_innovation_tracker() {
        let mut tracker = InnovationTracker::new();
        let id1 = tracker.get_innovation_id(0, 1);
        let id2 = tracker.get_innovation_id(0, 1); // Same connection
        let id3 = tracker.get_innovation_id(1, 2); // Different connection
        
        assert_eq!(id1, id2); // Same innovation should get same ID
        assert_ne!(id1, id3); // Different innovations get different IDs
    }
}
```

## Phase 3: Mutation and Crossover (Week 3)

### 3.1 Mutation Operators

#### src/neat/mutation.rs
```rust
use crate::neat::genome::*;
use crate::neat::innovation::InnovationTracker;
use crate::config::neat_config::NEATConfig;
use rand::prelude::*;

pub struct Mutator {
    config: NEATConfig,
    rng: SmallRng,
}

impl Mutator {
    pub fn new(config: NEATConfig) -> Self {
        Self {
            config,
            rng: SmallRng::from_entropy(),
        }
    }
    
    pub fn mutate(&mut self, genome: &mut Genome, innovation_tracker: &mut InnovationTracker) {
        // Weight mutations
        if self.rng.gen::<f64>() < self.config.weight_mutation_rate {
            self.mutate_weights(genome);
        }
        
        // Structural mutations
        if self.rng.gen::<f64>() < self.config.add_node_mutation_rate {
            self.add_node_mutation(genome, innovation_tracker);
        }
        
        if self.rng.gen::<f64>() < self.config.add_connection_mutation_rate {
            self.add_connection_mutation(genome, innovation_tracker);
        }
        
        // Connection enable/disable mutations
        if self.rng.gen::<f64>() < self.config.disable_connection_rate {
            self.disable_connection_mutation(genome);
        }
        
        if self.rng.gen::<f64>() < self.config.enable_connection_rate {
            self.enable_connection_mutation(genome);
        }
    }
    
    fn mutate_weights(&mut self, genome: &mut Genome) {
        for connection in &mut genome.connections {
            if self.rng.gen::<f64>() < self.config.weight_perturbation_rate {
                // Perturb existing weight
                let perturbation = self.rng.gen_range(-1.0..1.0) * self.config.weight_perturbation_power;
                connection.weight += perturbation;
            } else {
                // Replace weight entirely
                connection.weight = self.rng.gen_range(-2.0..2.0);
            }
        }
    }
    
    fn add_node_mutation(&mut self, genome: &mut Genome, innovation_tracker: &mut InnovationTracker) {
        if genome.connections.is_empty() || genome.nodes.len() >= self.config.max_nodes {
            return;
        }
        
        // Select random enabled connection to split
        let enabled_connections: Vec<usize> = genome.connections
            .iter()
            .enumerate()
            .filter(|(_, c)| c.enabled)
            .map(|(i, _)| i)
            .collect();
        
        if enabled_connections.is_empty() {
            return;
        }
        
        let connection_idx = enabled_connections[self.rng.gen_range(0..enabled_connections.len())];
        let connection = &mut genome.connections[connection_idx];
        
        // Disable the split connection
        connection.enabled = false;
        
        // Create new node
        let new_node_id = genome.nodes.len();
        genome.nodes.push(NodeGene {
            id: new_node_id,
            node_type: NodeType::Hidden,
            activation: *self.config.activation_functions.choose(&mut self.rng).unwrap(),
            bias: 0.0,
        });
        
        // Get innovation IDs for new connections
        let (in_to_new_id, new_to_out_id) = innovation_tracker.get_node_innovation_id(
            connection.innovation_id,
            connection.input_node,
            connection.output_node,
        );
        
        // Add connection from input to new node (weight = 1.0)
        genome.connections.push(ConnectionGene {
            innovation_id: in_to_new_id,
            input_node: connection.input_node,
            output_node: new_node_id,
            weight: 1.0,
            enabled: true,
        });
        
        // Add connection from new node to output (weight = original weight)
        genome.connections.push(ConnectionGene {
            innovation_id: new_to_out_id,
            input_node: new_node_id,
            output_node: connection.output_node,
            weight: connection.weight,
            enabled: true,
        });
    }
    
    fn add_connection_mutation(&mut self, genome: &mut Genome, innovation_tracker: &mut InnovationTracker) {
        if genome.connections.len() >= self.config.max_connections {
            return;
        }
        
        // Find potential new connections
        let mut possible_connections = Vec::new();
        
        for input_node in &genome.nodes {
            for output_node in &genome.nodes {
                // Skip if same node or invalid connection
                if input_node.id == output_node.id 
                    || output_node.node_type == NodeType::Input 
                    || output_node.node_type == NodeType::Bias 
                    || input_node.node_type == NodeType::Output {
                    continue;
                }
                
                // Skip if connection already exists
                if genome.connections.iter().any(|c| 
                    c.input_node == input_node.id && c.output_node == output_node.id) {
                    continue;
                }
                
                // Skip recurrent connections if not allowed
                if !self.config.recurrent_connections && 
                   self.would_create_cycle(genome, input_node.id, output_node.id) {
                    continue;
                }
                
                possible_connections.push((input_node.id, output_node.id));
            }
        }
        
        if possible_connections.is_empty() {
            return;
        }
        
        // Select random connection to add
        let (input_id, output_id) = possible_connections[self.rng.gen_range(0..possible_connections.len())];
        let innovation_id = innovation_tracker.get_innovation_id(input_id, output_id);
        
        genome.connections.push(ConnectionGene {
            innovation_id,
            input_node: input_id,
            output_node: output_id,
            weight: self.rng.gen_range(-2.0..2.0),
            enabled: true,
        });
    }
    
    fn would_create_cycle(&self, genome: &Genome, from: usize, to: usize) -> bool {
        // Simple cycle detection - can be optimized
        // For now, just prevent connections that would obviously create cycles
        from >= to
    }
    
    fn disable_connection_mutation(&mut self, genome: &mut Genome) {
        let enabled_connections: Vec<usize> = genome.connections
            .iter()
            .enumerate()
            .filter(|(_, c)| c.enabled)
            .map(|(i, _)| i)
            .collect();
        
        if !enabled_connections.is_empty() {
            let idx = enabled_connections[self.rng.gen_range(0..enabled_connections.len())];
            genome.connections[idx].enabled = false;
        }
    }
    
    fn enable_connection_mutation(&mut self, genome: &mut Genome) {
        let disabled_connections: Vec<usize> = genome.connections
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.enabled)
            .map(|(i, _)| i)
            .collect();
        
        if !disabled_connections.is_empty() {
            let idx = disabled_connections[self.rng.gen_range(0..disabled_connections.len())];
            genome.connections[idx].enabled = true;
        }
    }
}
```

### 3.2 Crossover Implementation

#### src/neat/crossover.rs
```rust
use crate::neat::genome::*;
use rand::prelude::*;
use std::collections::HashMap;

pub fn crossover(parent1: &Genome, parent2: &Genome, mut rng: impl Rng) -> Genome {
    // Determine more fit parent
    let (fitter_parent, other_parent) = if parent1.fitness >= parent2.fitness {
        (parent1, parent2)
    } else {
        (parent2, parent1)
    };
    
    let mut child = Genome::new(0, 0, 0); // Will be populated
    child.nodes = fitter_parent.nodes.clone();
    
    // Create connection maps for alignment
    let mut p1_connections: HashMap<usize, &ConnectionGene> = HashMap::new();
    let mut p2_connections: HashMap<usize, &ConnectionGene> = HashMap::new();
    
    for conn in &parent1.connections {
        p1_connections.insert(conn.innovation_id, conn);
    }
    
    for conn in &parent2.connections {
        p2_connections.insert(conn.innovation_id, conn);
    }
    
    // Get all innovation IDs
    let mut all_innovations: Vec<usize> = p1_connections.keys().cloned().collect();
    all_innovations.extend(p2_connections.keys().cloned());
    all_innovations.sort_unstable();
    all_innovations.dedup();
    
    for &innovation_id in &all_innovations {
        let conn1 = p1_connections.get(&innovation_id);
        let conn2 = p2_connections.get(&innovation_id);
        
        match (conn1, conn2) {
            // Matching genes - randomly inherit from either parent
            (Some(c1), Some(c2)) => {
                let chosen_conn = if rng.gen_bool(0.5) { c1 } else { c2 };
                let mut new_conn = (*chosen_conn).clone();
                
                // If one parent has disabled gene, 75% chance to disable in child
                if !c1.enabled || !c2.enabled {
                    new_conn.enabled = rng.gen_bool(0.25);
                }
                
                child.connections.push(new_conn);
            }
            
            // Disjoint/excess genes - inherit from fitter parent only
            (Some(c1), None) if std::ptr::eq(fitter_parent, parent1) => {
                child.connections.push((*c1).clone());
            }
            (None, Some(c2)) if std::ptr::eq(fitter_parent, parent2) => {
                child.connections.push((*c2).clone());
            }
            
            // Gene only in less fit parent - don't inherit
            _ => {}
        }
    }
    
    child
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neat::innovation::InnovationTracker;

    #[test]
    fn test_crossover_preserves_structure() {
        let mut parent1 = Genome::new(1, 2, 1);
        let mut parent2 = Genome::new(2, 2, 1);
        
        parent1.fitness = 0.8;
        parent2.fitness = 0.6;
        
        let mut rng = SmallRng::seed_from_u64(42);
        let child = crossover(&parent1, &parent2, &mut rng);
        
        assert_eq!(child.nodes.len(), parent1.nodes.len());
    }
}
```

## Phase 4: Speciation and Population Management (Week 4)

This phase will implement the speciation algorithm and population management system that are core to NEAT's success.

## Phase 5: Dataset Integration and Evaluation (Week 5)

This phase will integrate HuggingFace datasets and implement the fitness evaluation system.

## Phase 6: Optimization and Testing (Week 6)

This phase focuses on performance optimization, parallel processing, and comprehensive testing.

## Next Steps

1. Review and approve this implementation plan
2. Set up the development environment
3. Begin Phase 1 implementation
4. Establish continuous integration and testing pipeline
5. Create benchmarking framework for performance tracking

This plan provides a solid foundation for building a robust NEAT implementation in Rust with modern best practices and comprehensive testing.