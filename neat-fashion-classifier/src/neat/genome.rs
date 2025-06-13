//! Genome representation for NEAT algorithm
//!
//! This module provides the core genome structure that represents neural networks
//! in the NEAT algorithm, including nodes and connections with innovation tracking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::error::{NEATError, Result};

/// Node type in the neural network
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeType {
    /// Input node (receives external input)
    Input,
    /// Output node (produces network output)
    Output,
    /// Hidden node (internal processing)
    Hidden,
    /// Bias node (constant output of 1.0)
    Bias,
}

/// Activation function type for nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActivationType {
    /// Sigmoid activation function: 1 / (1 + e^(-x))
    Sigmoid,
    /// Hyperbolic tangent: tanh(x)
    Tanh,
    /// Rectified Linear Unit: max(0, x)
    ReLU,
    /// Linear activation: x (typically for input nodes)
    Linear,
    /// Gaussian activation: e^(-x^2)
    Gaussian,
}

impl ActivationType {
    /// Apply the activation function to a value
    pub fn activate(self, x: f64) -> f64 {
        match self {
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Self::Tanh => x.tanh(),
            Self::ReLU => x.max(0.0),
            Self::Linear => x,
            Self::Gaussian => (-x * x).exp(),
        }
    }
    
    /// Get the derivative of the activation function (for future gradient-based methods)
    pub fn derivative(self, x: f64) -> f64 {
        match self {
            Self::Sigmoid => {
                let s = self.activate(x);
                s * (1.0 - s)
            },
            Self::Tanh => 1.0 - x.tanh().powi(2),
            Self::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            Self::Linear => 1.0,
            Self::Gaussian => -2.0 * x * (-x * x).exp(),
        }
    }
}

/// Gene representing a node in the neural network
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeGene {
    /// Unique identifier for this node
    pub id: usize,
    /// Type of node (input, output, hidden, bias)
    pub node_type: NodeType,
    /// Activation function for this node
    pub activation: ActivationType,
    /// Bias value added to node input
    pub bias: f64,
}

impl NodeGene {
    /// Create a new node gene
    pub fn new(id: usize, node_type: NodeType, activation: ActivationType) -> Self {
        Self {
            id,
            node_type,
            activation,
            bias: 0.0,
        }
    }
    
    /// Create a new bias node
    pub fn new_bias(id: usize) -> Self {
        Self {
            id,
            node_type: NodeType::Bias,
            activation: ActivationType::Linear,
            bias: 1.0,
        }
    }
}

/// Gene representing a connection between nodes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConnectionGene {
    /// Innovation number for this connection
    pub innovation_id: usize,
    /// ID of the input node
    pub input_node: usize,
    /// ID of the output node
    pub output_node: usize,
    /// Weight of the connection
    pub weight: f64,
    /// Whether this connection is enabled
    pub enabled: bool,
}

impl ConnectionGene {
    /// Create a new connection gene
    pub fn new(innovation_id: usize, input_node: usize, output_node: usize, weight: f64) -> Self {
        Self {
            innovation_id,
            input_node,
            output_node,
            weight,
            enabled: true,
        }
    }
}

/// Complete genome representing a neural network
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Genome {
    /// All nodes in the network
    pub nodes: Vec<NodeGene>,
    /// All connections in the network
    pub connections: Vec<ConnectionGene>,
    /// Fitness score from evaluation
    pub fitness: f64,
    /// Fitness adjusted for speciation
    pub adjusted_fitness: f64,
    /// Species assignment (None if unassigned)
    pub species_id: Option<usize>,
    /// Unique identifier for this genome
    pub id: usize,
}

impl Genome {
    /// Create a new genome with minimal topology for Fashion-MNIST
    /// 
    /// Creates a genome with the specified number of input and output nodes,
    /// plus one bias node. No connections are created initially, following
    /// NEAT's principle of starting with minimal structure.
    /// 
    /// # Arguments
    /// 
    /// * `id` - Unique identifier for this genome
    /// * `input_size` - Number of input nodes (e.g., 784 for Fashion-MNIST)
    /// * `output_size` - Number of output nodes (e.g., 10 for classification)
    /// 
    /// # Examples
    /// 
    /// ```
    /// use neat_fashion_classifier::neat::Genome;
    /// 
    /// let genome = Genome::new(0, 784, 10);
    /// assert_eq!(genome.get_input_count(), 784);
    /// assert_eq!(genome.get_output_count(), 10);
    /// assert_eq!(genome.connections.len(), 0); // No initial connections
    /// ```
    pub fn new(id: usize, input_size: usize, output_size: usize) -> Self {
        let mut nodes = Vec::new();
        let mut next_node_id = 0;
        
        // Add input nodes
        for _ in 0..input_size {
            nodes.push(NodeGene::new(next_node_id, NodeType::Input, ActivationType::Linear));
            next_node_id += 1;
        }
        
        // Add bias node
        nodes.push(NodeGene::new_bias(next_node_id));
        next_node_id += 1;
        
        // Add output nodes
        for _ in 0..output_size {
            nodes.push(NodeGene::new(next_node_id, NodeType::Output, ActivationType::Sigmoid));
            next_node_id += 1;
        }
        
        Self {
            nodes,
            connections: Vec::new(),
            fitness: 0.0,
            adjusted_fitness: 0.0,
            species_id: None,
            id,
        }
    }
    
    /// Get the number of input nodes
    pub fn get_input_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.node_type == NodeType::Input).count()
    }
    
    /// Get the number of output nodes
    pub fn get_output_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.node_type == NodeType::Output).count()
    }
    
    /// Get the number of hidden nodes
    pub fn get_hidden_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.node_type == NodeType::Hidden).count()
    }
    
    /// Get network complexity as (nodes, enabled_connections)
    pub fn get_complexity(&self) -> (usize, usize) {
        let enabled_connections = self.connections.iter().filter(|c| c.enabled).count();
        (self.nodes.len(), enabled_connections)
    }
    
    /// Get all input node IDs
    pub fn get_input_node_ids(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|n| n.node_type == NodeType::Input)
            .map(|n| n.id)
            .collect()
    }
    
    /// Get all output node IDs
    pub fn get_output_node_ids(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|n| n.node_type == NodeType::Output)
            .map(|n| n.id)
            .collect()
    }
    
    /// Get the bias node ID (if exists)
    pub fn get_bias_node_id(&self) -> Option<usize> {
        self.nodes
            .iter()
            .find(|n| n.node_type == NodeType::Bias)
            .map(|n| n.id)
    }
    
    /// Find node by ID
    pub fn get_node(&self, node_id: usize) -> Option<&NodeGene> {
        self.nodes.iter().find(|n| n.id == node_id)
    }
    
    /// Find mutable node by ID
    pub fn get_node_mut(&mut self, node_id: usize) -> Option<&mut NodeGene> {
        self.nodes.iter_mut().find(|n| n.id == node_id)
    }
    
    /// Find connection by innovation ID
    pub fn get_connection(&self, innovation_id: usize) -> Option<&ConnectionGene> {
        self.connections.iter().find(|c| c.innovation_id == innovation_id)
    }
    
    /// Check if a connection exists between two nodes
    pub fn has_connection(&self, input_id: usize, output_id: usize) -> bool {
        self.connections
            .iter()
            .any(|c| c.input_node == input_id && c.output_node == output_id)
    }
    
    /// Add a new node to the genome
    pub fn add_node(&mut self, node: NodeGene) -> Result<()> {
        // Check for duplicate node IDs
        if self.nodes.iter().any(|n| n.id == node.id) {
            return Err(NEATError::InvalidGenome {
                message: format!("Node with ID {} already exists", node.id),
            });
        }
        
        self.nodes.push(node);
        Ok(())
    }
    
    /// Add a new connection to the genome
    pub fn add_connection(&mut self, connection: ConnectionGene) -> Result<()> {
        // Validate that nodes exist
        if !self.nodes.iter().any(|n| n.id == connection.input_node) {
            return Err(NEATError::InvalidGenome {
                message: format!("Input node {} does not exist", connection.input_node),
            });
        }
        
        if !self.nodes.iter().any(|n| n.id == connection.output_node) {
            return Err(NEATError::InvalidGenome {
                message: format!("Output node {} does not exist", connection.output_node),
            });
        }
        
        // Check for duplicate connections
        if self.connections.iter().any(|c| c.innovation_id == connection.innovation_id) {
            return Err(NEATError::InvalidGenome {
                message: format!("Connection with innovation ID {} already exists", connection.innovation_id),
            });
        }
        
        self.connections.push(connection);
        Ok(())
    }
    
    /// Validate genome structure
    pub fn validate(&self) -> Result<()> {
        // Check for at least one input and output node
        if self.get_input_count() == 0 {
            return Err(NEATError::InvalidGenome {
                message: "Genome must have at least one input node".to_string(),
            });
        }
        
        if self.get_output_count() == 0 {
            return Err(NEATError::InvalidGenome {
                message: "Genome must have at least one output node".to_string(),
            });
        }
        
        // Check for unique node IDs
        let mut node_ids = std::collections::HashSet::new();
        for node in &self.nodes {
            if !node_ids.insert(node.id) {
                return Err(NEATError::InvalidGenome {
                    message: format!("Duplicate node ID: {}", node.id),
                });
            }
        }
        
        // Check for unique innovation IDs
        let mut innovation_ids = std::collections::HashSet::new();
        for connection in &self.connections {
            if !innovation_ids.insert(connection.innovation_id) {
                return Err(NEATError::InvalidGenome {
                    message: format!("Duplicate innovation ID: {}", connection.innovation_id),
                });
            }
        }
        
        // Validate connections reference existing nodes
        for connection in &self.connections {
            if !self.nodes.iter().any(|n| n.id == connection.input_node) {
                return Err(NEATError::InvalidGenome {
                    message: format!("Connection references non-existent input node: {}", connection.input_node),
                });
            }
            
            if !self.nodes.iter().any(|n| n.id == connection.output_node) {
                return Err(NEATError::InvalidGenome {
                    message: format!("Connection references non-existent output node: {}", connection.output_node),
                });
            }
        }
        
        Ok(())
    }
    
    /// Get a map of node ID to index for efficient lookup
    pub fn create_node_index_map(&self) -> HashMap<usize, usize> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.id, index))
            .collect()
    }
    
    /// Reset fitness scores
    pub fn reset_fitness(&mut self) {
        self.fitness = 0.0;
        self.adjusted_fitness = 0.0;
    }
    
    /// Create a copy of this genome with a new ID
    pub fn clone_with_id(&self, new_id: usize) -> Self {
        let mut cloned = self.clone();
        cloned.id = new_id;
        cloned.reset_fitness();
        cloned.species_id = None;
        cloned
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_activation_functions() {
        // Test sigmoid
        assert_relative_eq!(ActivationType::Sigmoid.activate(0.0), 0.5, epsilon = 1e-10);
        assert!(ActivationType::Sigmoid.activate(-10.0) < 0.01);
        assert!(ActivationType::Sigmoid.activate(10.0) > 0.99);
        
        // Test tanh
        assert_relative_eq!(ActivationType::Tanh.activate(0.0), 0.0, epsilon = 1e-10);
        assert!(ActivationType::Tanh.activate(-10.0) < -0.99);
        assert!(ActivationType::Tanh.activate(10.0) > 0.99);
        
        // Test ReLU
        assert_eq!(ActivationType::ReLU.activate(-1.0), 0.0);
        assert_eq!(ActivationType::ReLU.activate(0.0), 0.0);
        assert_eq!(ActivationType::ReLU.activate(1.0), 1.0);
        
        // Test Linear
        assert_eq!(ActivationType::Linear.activate(-1.0), -1.0);
        assert_eq!(ActivationType::Linear.activate(0.0), 0.0);
        assert_eq!(ActivationType::Linear.activate(1.0), 1.0);
        
        // Test Gaussian
        assert_relative_eq!(ActivationType::Gaussian.activate(0.0), 1.0, epsilon = 1e-10);
        assert!(ActivationType::Gaussian.activate(2.0) < 0.02); // e^(-4) ≈ 0.0183
    }

    #[test]
    fn test_node_gene_creation() {
        let node = NodeGene::new(0, NodeType::Input, ActivationType::Linear);
        assert_eq!(node.id, 0);
        assert_eq!(node.node_type, NodeType::Input);
        assert_eq!(node.activation, ActivationType::Linear);
        assert_eq!(node.bias, 0.0);
        
        let bias_node = NodeGene::new_bias(1);
        assert_eq!(bias_node.id, 1);
        assert_eq!(bias_node.node_type, NodeType::Bias);
        assert_eq!(bias_node.bias, 1.0);
    }

    #[test]
    fn test_connection_gene_creation() {
        let connection = ConnectionGene::new(0, 1, 2, 0.5);
        assert_eq!(connection.innovation_id, 0);
        assert_eq!(connection.input_node, 1);
        assert_eq!(connection.output_node, 2);
        assert_eq!(connection.weight, 0.5);
        assert!(connection.enabled);
    }

    #[test]
    fn test_genome_creation() {
        let genome = Genome::new(0, 784, 10);
        
        assert_eq!(genome.id, 0);
        assert_eq!(genome.get_input_count(), 784);
        assert_eq!(genome.get_output_count(), 10);
        assert_eq!(genome.get_hidden_count(), 0);
        assert_eq!(genome.connections.len(), 0);
        
        // Should have 784 inputs + 1 bias + 10 outputs = 795 nodes
        assert_eq!(genome.nodes.len(), 795);
        
        // Verify node types
        assert!(genome.get_bias_node_id().is_some());
        assert_eq!(genome.get_input_node_ids().len(), 784);
        assert_eq!(genome.get_output_node_ids().len(), 10);
    }

    #[test]
    fn test_genome_validation() {
        let mut genome = Genome::new(0, 2, 1);
        assert!(genome.validate().is_ok());
        
        // Test adding valid node
        let node = NodeGene::new(100, NodeType::Hidden, ActivationType::Sigmoid);
        assert!(genome.add_node(node).is_ok());
        
        // Test adding duplicate node
        let duplicate_node = NodeGene::new(100, NodeType::Hidden, ActivationType::Tanh);
        assert!(genome.add_node(duplicate_node).is_err());
        
        // Test adding valid connection
        let connection = ConnectionGene::new(0, 0, 3, 0.5); // input 0 to output 3
        assert!(genome.add_connection(connection).is_ok());
        
        // Test adding connection with non-existent node
        let invalid_connection = ConnectionGene::new(1, 999, 3, 0.5);
        assert!(genome.add_connection(invalid_connection).is_err());
    }

    #[test]
    fn test_genome_complexity() {
        let mut genome = Genome::new(0, 2, 1);
        let (nodes, connections) = genome.get_complexity();
        assert_eq!(nodes, 4); // 2 inputs + 1 bias + 1 output
        assert_eq!(connections, 0);
        
        // Add a connection
        let connection = ConnectionGene::new(0, 0, 3, 0.5);
        genome.add_connection(connection).unwrap();
        
        let (_nodes, connections) = genome.get_complexity();
        assert_eq!(connections, 1);
        
        // Disable the connection
        genome.connections[0].enabled = false;
        let (_nodes, connections) = genome.get_complexity();
        assert_eq!(connections, 0); // Disabled connections don't count
    }

    #[test]
    fn test_genome_node_lookup() {
        let genome = Genome::new(0, 2, 1);
        
        // Test getting existing node
        assert!(genome.get_node(0).is_some());
        assert!(genome.get_node(1).is_some());
        assert!(genome.get_node(2).is_some()); // bias
        assert!(genome.get_node(3).is_some()); // output
        
        // Test getting non-existent node
        assert!(genome.get_node(999).is_none());
        
        // Test connection existence
        assert!(!genome.has_connection(0, 3));
    }

    #[test]
    fn test_genome_cloning() {
        let mut genome = Genome::new(0, 2, 1);
        genome.fitness = 0.8;
        genome.species_id = Some(5);
        
        let cloned = genome.clone_with_id(42);
        assert_eq!(cloned.id, 42);
        assert_eq!(cloned.fitness, 0.0); // Should reset fitness
        assert!(cloned.species_id.is_none()); // Should reset species
        assert_eq!(cloned.nodes.len(), genome.nodes.len());
    }

    #[test]
    fn test_fashion_mnist_genome() {
        // Test creating a genome suitable for Fashion-MNIST
        let genome = Genome::new(0, 784, 10);
        
        assert!(genome.validate().is_ok());
        assert_eq!(genome.get_input_count(), 784);
        assert_eq!(genome.get_output_count(), 10);
        
        // Should be able to identify all node types
        let input_ids = genome.get_input_node_ids();
        let output_ids = genome.get_output_node_ids();
        let bias_id = genome.get_bias_node_id();
        
        assert_eq!(input_ids.len(), 784);
        assert_eq!(output_ids.len(), 10);
        assert!(bias_id.is_some());
        
        // All IDs should be unique
        let mut all_ids = input_ids;
        all_ids.extend(output_ids);
        all_ids.push(bias_id.unwrap());
        all_ids.sort_unstable();
        all_ids.dedup();
        assert_eq!(all_ids.len(), 795); // 784 + 10 + 1
    }
}