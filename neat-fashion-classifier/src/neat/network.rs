//! Neural network activation and processing
//!
//! This module implements the network activation system that converts a genome
//! into an executable neural network capable of processing input data.

use crate::neat::genome::{Genome, NodeType, ActivationType};
use crate::error::{NEATError, Result};
use std::collections::{HashMap, HashSet, VecDeque};
use ndarray::Array1;

/// Represents an executable neural network derived from a genome
#[derive(Debug, Clone)]
pub struct Network {
    /// Network topology as an adjacency list
    topology: HashMap<usize, Vec<(usize, f64)>>, // node_id -> [(output_node, weight), ...]
    /// Node information indexed by node ID
    nodes: HashMap<usize, NodeInfo>,
    /// Input node IDs in order
    input_nodes: Vec<usize>,
    /// Output node IDs in order
    output_nodes: Vec<usize>,
    /// Bias node ID (if any)
    bias_node: Option<usize>,
    /// Activation order for forward propagation
    activation_order: Vec<usize>,
    /// Network depth (maximum path length from input to output)
    depth: usize,
}

/// Information about a network node
#[derive(Debug, Clone)]
struct NodeInfo {
    /// Node type
    node_type: NodeType,
    /// Activation function
    activation: ActivationType,
    /// Bias value
    bias: f64,
}

impl Network {
    /// Create a new network from a genome
    pub fn from_genome(genome: &Genome) -> Result<Self> {
        // Validate genome first
        genome.validate()?;
        
        let mut network = Self {
            topology: HashMap::new(),
            nodes: HashMap::new(),
            input_nodes: Vec::new(),
            output_nodes: Vec::new(),
            bias_node: None,
            activation_order: Vec::new(),
            depth: 0,
        };
        
        // Build node information
        network.build_nodes(genome)?;
        
        // Build topology from connections
        network.build_topology(genome)?;
        
        // Calculate activation order and depth
        network.calculate_activation_order()?;
        
        Ok(network)
    }
    
    /// Build node information from genome
    fn build_nodes(&mut self, genome: &Genome) -> Result<()> {
        for node in &genome.nodes {
            let node_info = NodeInfo {
                node_type: node.node_type,
                activation: node.activation,
                bias: node.bias,
            };
            
            self.nodes.insert(node.id, node_info);
            
            match node.node_type {
                NodeType::Input => self.input_nodes.push(node.id),
                NodeType::Output => self.output_nodes.push(node.id),
                NodeType::Bias => self.bias_node = Some(node.id),
                NodeType::Hidden => {}, // Handled in activation order calculation
            }
        }
        
        // Sort input and output nodes for consistent ordering
        self.input_nodes.sort_unstable();
        self.output_nodes.sort_unstable();
        
        Ok(())
    }
    
    /// Build network topology from connections
    fn build_topology(&mut self, genome: &Genome) -> Result<()> {
        // Initialize empty adjacency lists for all nodes
        for node_id in self.nodes.keys() {
            self.topology.insert(*node_id, Vec::new());
        }
        
        // Add enabled connections
        for connection in &genome.connections {
            if connection.enabled {
                // Verify nodes exist
                if !self.nodes.contains_key(&connection.input_node) {
                    return Err(NEATError::InvalidGenome {
                        message: format!("Connection references non-existent input node: {}", connection.input_node),
                    });
                }
                
                if !self.nodes.contains_key(&connection.output_node) {
                    return Err(NEATError::InvalidGenome {
                        message: format!("Connection references non-existent output node: {}", connection.output_node),
                    });
                }
                
                // Add connection to topology
                self.topology
                    .get_mut(&connection.input_node)
                    .unwrap()
                    .push((connection.output_node, connection.weight));
            }
        }
        
        Ok(())
    }
    
    /// Calculate activation order using topological sort
    fn calculate_activation_order(&mut self) -> Result<()> {
        // Calculate in-degrees for topological sort
        let mut in_degree: HashMap<usize, usize> = HashMap::new();
        for node_id in self.nodes.keys() {
            in_degree.insert(*node_id, 0);
        }
        
        // Count incoming connections
        for (_, connections) in &self.topology {
            for &(target_node, _) in connections {
                *in_degree.get_mut(&target_node).unwrap() += 1;
            }
        }
        
        // Start with nodes that have no incoming connections (inputs and bias)
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        
        // Add input nodes first (they should be processed first)
        for &node_id in &self.input_nodes {
            queue.push_back(node_id);
            visited.insert(node_id);
        }
        
        // Add bias node if it exists
        if let Some(bias_id) = self.bias_node {
            if !visited.contains(&bias_id) {
                queue.push_back(bias_id);
                visited.insert(bias_id);
            }
        }
        
        // Add any other nodes with in-degree 0
        for (&node_id, &degree) in &in_degree {
            if degree == 0 && !visited.contains(&node_id) {
                queue.push_back(node_id);
                visited.insert(node_id);
            }
        }
        
        self.activation_order.clear();
        let mut max_depth = 0;
        let mut node_depths: HashMap<usize, usize> = HashMap::new();
        
        // Set input and bias nodes to depth 0
        for &node_id in &self.input_nodes {
            node_depths.insert(node_id, 0);
        }
        if let Some(bias_id) = self.bias_node {
            node_depths.insert(bias_id, 0);
        }
        
        // Process nodes in topological order
        while let Some(current_node) = queue.pop_front() {
            self.activation_order.push(current_node);
            
            let current_depth = node_depths.get(&current_node).copied().unwrap_or(0);
            
            // Process all outgoing connections
            if let Some(connections) = self.topology.get(&current_node) {
                for &(target_node, _) in connections {
                    // Decrease in-degree
                    let new_degree = in_degree[&target_node] - 1;
                    in_degree.insert(target_node, new_degree);
                    
                    // Update depth
                    let target_depth = current_depth + 1;
                    let existing_depth = node_depths.get(&target_node).copied().unwrap_or(0);
                    if target_depth > existing_depth {
                        node_depths.insert(target_node, target_depth);
                        max_depth = max_depth.max(target_depth);
                    }
                    
                    // If in-degree reaches 0, add to queue
                    if new_degree == 0 {
                        queue.push_back(target_node);
                    }
                }
            }
        }
        
        // Check for cycles (if not all nodes were processed)
        if self.activation_order.len() != self.nodes.len() {
            return Err(NEATError::InvalidGenome {
                message: "Network contains cycles - cannot create activation order".to_string(),
            });
        }
        
        self.depth = max_depth;
        Ok(())
    }
    
    /// Activate the network with given inputs
    pub fn activate(&self, inputs: &[f64]) -> Result<Vec<f64>> {
        if inputs.len() != self.input_nodes.len() {
            return Err(NEATError::InvalidGenome {
                message: format!(
                    "Input size mismatch: expected {}, got {}",
                    self.input_nodes.len(),
                    inputs.len()
                ),
            });
        }
        
        // Node activations
        let mut activations: HashMap<usize, f64> = HashMap::new();
        
        // Set input values
        for (i, &node_id) in self.input_nodes.iter().enumerate() {
            activations.insert(node_id, inputs[i]);
        }
        
        // Set bias value
        if let Some(bias_id) = self.bias_node {
            activations.insert(bias_id, 1.0);
        }
        
        // Process nodes in activation order
        for &node_id in &self.activation_order {
            // Skip if already set (input/bias nodes)
            if activations.contains_key(&node_id) {
                continue;
            }
            
            let node_info = &self.nodes[&node_id];
            let mut sum = node_info.bias;
            
            // Sum weighted inputs from all incoming connections
            for (&input_node, connections) in &self.topology {
                if let Some(&input_value) = activations.get(&input_node) {
                    for &(target_node, weight) in connections {
                        if target_node == node_id {
                            sum += input_value * weight;
                        }
                    }
                }
            }
            
            // Apply activation function
            let output = node_info.activation.activate(sum);
            activations.insert(node_id, output);
        }
        
        // Collect outputs
        let mut outputs = Vec::with_capacity(self.output_nodes.len());
        for &node_id in &self.output_nodes {
            let output = activations.get(&node_id).copied().unwrap_or(0.0);
            outputs.push(output);
        }
        
        Ok(outputs)
    }
    
    /// Get network information
    pub fn get_info(&self) -> NetworkInfo {
        NetworkInfo {
            num_inputs: self.input_nodes.len(),
            num_outputs: self.output_nodes.len(),
            num_hidden: self.nodes.len() - self.input_nodes.len() - self.output_nodes.len() 
                        - if self.bias_node.is_some() { 1 } else { 0 },
            num_connections: self.topology.values().map(|v| v.len()).sum(),
            depth: self.depth,
            has_bias: self.bias_node.is_some(),
        }
    }
    
    /// Check if the network is feedforward (no recurrent connections)
    pub fn is_feedforward(&self) -> bool {
        // If we successfully calculated activation order, it's feedforward
        // (cycles would have been detected in calculate_activation_order)
        !self.activation_order.is_empty()
    }
    
    /// Get the activation order of nodes
    pub fn get_activation_order(&self) -> &[usize] {
        &self.activation_order
    }
    
    /// Get network depth (maximum path length from input to output)
    pub fn get_depth(&self) -> usize {
        self.depth
    }
    
    /// Convert inputs to ndarray format for potential ML library integration
    pub fn activate_ndarray(&self, inputs: &Array1<f64>) -> Result<Array1<f64>> {
        let inputs_slice = inputs.as_slice().ok_or_else(|| NEATError::Other(
            anyhow::anyhow!("Failed to convert ndarray to slice")
        ))?;
        
        let outputs = self.activate(inputs_slice)?;
        Ok(Array1::from_vec(outputs))
    }
}

/// Network information summary
#[derive(Debug, Clone, PartialEq)]
pub struct NetworkInfo {
    /// Number of input nodes
    pub num_inputs: usize,
    /// Number of output nodes
    pub num_outputs: usize,
    /// Number of hidden nodes
    pub num_hidden: usize,
    /// Number of connections
    pub num_connections: usize,
    /// Network depth
    pub depth: usize,
    /// Whether network has a bias node
    pub has_bias: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neat::genome::{NodeGene, ConnectionGene};
    use approx::assert_relative_eq;

    #[test]
    fn test_simple_network_creation() {
        // Create a simple genome: 2 inputs, 1 bias, 1 output, no connections
        let genome = Genome::new(0, 2, 1);
        let network = Network::from_genome(&genome).unwrap();
        
        let info = network.get_info();
        assert_eq!(info.num_inputs, 2);
        assert_eq!(info.num_outputs, 1);
        assert_eq!(info.num_hidden, 0);
        assert_eq!(info.num_connections, 0);
        assert!(info.has_bias);
        assert!(network.is_feedforward());
    }
    
    #[test]
    fn test_network_activation_no_connections() {
        // Network with no connections should output zeros (bias effect only)
        let genome = Genome::new(0, 2, 1);
        let network = Network::from_genome(&genome).unwrap();
        
        let inputs = vec![1.0, -1.0];
        let outputs = network.activate(&inputs).unwrap();
        
        assert_eq!(outputs.len(), 1);
        // Output should be sigmoid(0) = 0.5 (no connections, just bias=0)
        assert_relative_eq!(outputs[0], 0.5, epsilon = 1e-10);
    }
    
    #[test]
    fn test_network_with_direct_connections() {
        let mut genome = Genome::new(0, 2, 1);
        
        // Add direct connections from inputs to output
        let conn1 = ConnectionGene::new(0, 0, 3, 1.0); // input 0 -> output (weight 1.0)
        let conn2 = ConnectionGene::new(1, 1, 3, -0.5); // input 1 -> output (weight -0.5)
        let conn3 = ConnectionGene::new(2, 2, 3, 2.0); // bias -> output (weight 2.0)
        
        genome.add_connection(conn1).unwrap();
        genome.add_connection(conn2).unwrap();
        genome.add_connection(conn3).unwrap();
        
        let network = Network::from_genome(&genome).unwrap();
        
        let inputs = vec![1.0, 1.0];
        let outputs = network.activate(&inputs).unwrap();
        
        assert_eq!(outputs.len(), 1);
        // Expected: sigmoid(1.0 * 1.0 + 1.0 * (-0.5) + 1.0 * 2.0) = sigmoid(2.5)
        let expected = ActivationType::Sigmoid.activate(2.5);
        assert_relative_eq!(outputs[0], expected, epsilon = 1e-10);
    }
    
    #[test]
    fn test_network_with_hidden_node() {
        let mut genome = Genome::new(0, 2, 1);
        
        // Add a hidden node
        let hidden_node = NodeGene::new(10, NodeType::Hidden, ActivationType::Tanh);
        genome.add_node(hidden_node).unwrap();
        
        // Add connections: input -> hidden -> output
        let conn1 = ConnectionGene::new(0, 0, 10, 1.0); // input 0 -> hidden
        let conn2 = ConnectionGene::new(1, 1, 10, 1.0); // input 1 -> hidden
        let conn3 = ConnectionGene::new(2, 10, 3, 1.0); // hidden -> output
        
        genome.add_connection(conn1).unwrap();
        genome.add_connection(conn2).unwrap();
        genome.add_connection(conn3).unwrap();
        
        let network = Network::from_genome(&genome).unwrap();
        
        let info = network.get_info();
        assert_eq!(info.num_hidden, 1);
        assert_eq!(info.depth, 2); // input -> hidden -> output
        
        let inputs = vec![0.5, -0.5];
        let outputs = network.activate(&inputs).unwrap();
        
        assert_eq!(outputs.len(), 1);
        // Expected: sigmoid(tanh(0.5 + (-0.5)) * 1.0) = sigmoid(tanh(0) * 1.0) = sigmoid(0) = 0.5
        assert_relative_eq!(outputs[0], 0.5, epsilon = 1e-10);
    }
    
    #[test]
    fn test_network_activation_order() {
        let mut genome = Genome::new(0, 2, 2);
        
        // Add two hidden nodes
        let hidden1 = NodeGene::new(10, NodeType::Hidden, ActivationType::Linear);
        let hidden2 = NodeGene::new(11, NodeType::Hidden, ActivationType::Linear);
        genome.add_node(hidden1).unwrap();
        genome.add_node(hidden2).unwrap();
        
        // Create a specific topology: input -> hidden1 -> hidden2 -> output
        let conn1 = ConnectionGene::new(0, 0, 10, 1.0); // input 0 -> hidden1
        let conn2 = ConnectionGene::new(1, 10, 11, 1.0); // hidden1 -> hidden2
        let conn3 = ConnectionGene::new(2, 11, 3, 1.0); // hidden2 -> output1
        let conn4 = ConnectionGene::new(3, 11, 4, 1.0); // hidden2 -> output2
        
        genome.add_connection(conn1).unwrap();
        genome.add_connection(conn2).unwrap();
        genome.add_connection(conn3).unwrap();
        genome.add_connection(conn4).unwrap();
        
        let network = Network::from_genome(&genome).unwrap();
        
        let activation_order = network.get_activation_order();
        
        // Inputs should come first
        assert!(activation_order.iter().position(|&x| x == 0).unwrap() < 
                activation_order.iter().position(|&x| x == 10).unwrap());
        // hidden1 should come before hidden2
        assert!(activation_order.iter().position(|&x| x == 10).unwrap() < 
                activation_order.iter().position(|&x| x == 11).unwrap());
        // hidden2 should come before outputs
        assert!(activation_order.iter().position(|&x| x == 11).unwrap() < 
                activation_order.iter().position(|&x| x == 3).unwrap());
    }
    
    #[test]
    fn test_invalid_input_size() {
        let genome = Genome::new(0, 2, 1);
        let network = Network::from_genome(&genome).unwrap();
        
        // Wrong input size should fail
        let inputs = vec![1.0]; // Should be 2 inputs
        let result = network.activate(&inputs);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_disconnected_nodes() {
        let mut genome = Genome::new(0, 2, 1);
        
        // Add a hidden node with no connections
        let hidden_node = NodeGene::new(10, NodeType::Hidden, ActivationType::Sigmoid);
        genome.add_node(hidden_node).unwrap();
        
        let network = Network::from_genome(&genome).unwrap();
        
        // Should still work - disconnected nodes get default activation
        let inputs = vec![1.0, -1.0];
        let outputs = network.activate(&inputs).unwrap();
        assert_eq!(outputs.len(), 1);
    }
    
    #[test]
    fn test_ndarray_integration() {
        let genome = Genome::new(0, 3, 2);
        let network = Network::from_genome(&genome).unwrap();
        
        let inputs = Array1::from_vec(vec![1.0, 0.5, -0.5]);
        let outputs = network.activate_ndarray(&inputs).unwrap();
        
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs.ndim(), 1);
    }
}