//! Mutation operations for NEAT genomes
//!
//! This module implements the core mutation operations that drive evolution
//! in the NEAT algorithm, including structural and parametric mutations.

use crate::neat::genome::{Genome, NodeGene, ConnectionGene, NodeType, ActivationType};
use crate::neat::innovation::InnovationTracker;
use crate::config::NEATConfig;
use crate::error::{NEATError, Result};
use rand::prelude::*;
// use rand::distributions::{Uniform, WeightedIndex};

/// Mutation context containing configuration and shared state
pub struct MutationContext<'a> {
    /// Configuration for mutation parameters
    pub config: &'a NEATConfig,
    /// Innovation tracker for historical markings
    pub innovation_tracker: &'a mut InnovationTracker,
    /// Random number generator
    pub rng: &'a mut dyn RngCore,
}

/// Trait for mutation operations
pub trait Mutation {
    /// Apply the mutation to a genome
    fn mutate(&self, genome: &mut Genome, context: &mut MutationContext) -> Result<bool>;
    
    /// Get the probability of this mutation occurring
    fn probability(&self, config: &NEATConfig) -> f64;
}

/// Add a new connection between existing nodes
#[derive(Debug, Clone)]
pub struct AddConnectionMutation;

/// Add a new node by splitting an existing connection
#[derive(Debug, Clone)]
pub struct AddNodeMutation;

/// Mutate the weight of an existing connection
#[derive(Debug, Clone)]
pub struct WeightMutation;

/// Toggle the enabled state of a connection
#[derive(Debug, Clone)]
pub struct ToggleConnectionMutation;

/// Mutate node bias values
#[derive(Debug, Clone)]
pub struct BiasMutation;

/// Change activation function of a node
#[derive(Debug, Clone)]
pub struct ActivationMutation;

impl Mutation for AddConnectionMutation {
    fn mutate(&self, genome: &mut Genome, context: &mut MutationContext) -> Result<bool> {
        if genome.nodes.len() < 2 {
            return Ok(false); // Need at least 2 nodes to create a connection
        }
        
        // Get potential input and output nodes
        let mut potential_inputs = Vec::new();
        let mut potential_outputs = Vec::new();
        
        for node in &genome.nodes {
            match node.node_type {
                NodeType::Input | NodeType::Bias | NodeType::Hidden => {
                    potential_inputs.push(node.id);
                }
                _ => {}
            }
            
            match node.node_type {
                NodeType::Output | NodeType::Hidden => {
                    potential_outputs.push(node.id);
                }
                _ => {}
            }
        }
        
        // Try to find a valid connection that doesn't already exist
        let max_attempts = 50;
        for _ in 0..max_attempts {
            if potential_inputs.is_empty() || potential_outputs.is_empty() {
                break;
            }
            
            let input_idx = context.rng.gen_range(0..potential_inputs.len());
            let output_idx = context.rng.gen_range(0..potential_outputs.len());
            
            let input_id = potential_inputs[input_idx];
            let output_id = potential_outputs[output_idx];
            
            // Don't allow self-connections or duplicate connections
            if input_id == output_id || genome.has_connection(input_id, output_id) {
                continue;
            }
            
            // Check for cycles (simple check: no output -> input connections)
            if let (Some(input_node), Some(output_node)) = (genome.get_node(input_id), genome.get_node(output_id)) {
                if output_node.node_type == NodeType::Input || 
                   (input_node.node_type == NodeType::Output && output_node.node_type != NodeType::Output) {
                    continue; // Would create invalid topology
                }
            }
            
            // Create new connection
            let innovation_id = context.innovation_tracker.get_innovation_id(input_id, output_id);
            
            let weight_range = context.config.mutation.weight_range;
            let weight = context.rng.gen_range(weight_range.0..=weight_range.1);
            
            let connection = ConnectionGene::new(innovation_id, input_id, output_id, weight);
            
            match genome.add_connection(connection) {
                Ok(()) => return Ok(true),
                Err(_) => continue, // Try again
            }
        }
        
        Ok(false) // Failed to add connection
    }
    
    fn probability(&self, config: &NEATConfig) -> f64 {
        config.mutation.add_connection_rate
    }
}

impl Mutation for AddNodeMutation {
    fn mutate(&self, genome: &mut Genome, context: &mut MutationContext) -> Result<bool> {
        // Get enabled connections indices
        let enabled_indices: Vec<usize> = genome.connections
            .iter()
            .enumerate()
            .filter(|(_, conn)| conn.enabled)
            .map(|(idx, _)| idx)
            .collect();
            
        if enabled_indices.is_empty() {
            return Ok(false); // No connections to split
        }
        
        // Select random connection to split
        let conn_idx = *enabled_indices.choose(context.rng).unwrap();
        let split_connection = genome.connections[conn_idx].clone();
        
        // Create new node
        let (in_to_new_id, new_to_out_id, new_node_id) = context.innovation_tracker
            .get_node_innovation_ids(
                split_connection.innovation_id,
                split_connection.input_node,
                split_connection.output_node,
            );
        
        // Choose activation function for new node
        let activation = choose_activation_function(context.rng, &context.config.network.activation_functions);
        
        let new_node = NodeGene::new(new_node_id, NodeType::Hidden, activation);
        
        // Add new node
        genome.add_node(new_node)?;
        
        // Disable old connection
        genome.connections[conn_idx].enabled = false;
        
        // Add two new connections
        let in_to_new_conn = ConnectionGene::new(
            in_to_new_id,
            split_connection.input_node,
            new_node_id,
            1.0, // Weight = 1.0 to preserve signal
        );
        
        let new_to_out_conn = ConnectionGene::new(
            new_to_out_id,
            new_node_id,
            split_connection.output_node,
            split_connection.weight, // Preserve original weight
        );
        
        genome.add_connection(in_to_new_conn)?;
        genome.add_connection(new_to_out_conn)?;
        
        Ok(true)
    }
    
    fn probability(&self, config: &NEATConfig) -> f64 {
        config.mutation.add_node_rate
    }
}

impl Mutation for WeightMutation {
    fn mutate(&self, genome: &mut Genome, context: &mut MutationContext) -> Result<bool> {
        if genome.connections.is_empty() {
            return Ok(false);
        }
        
        let mut mutated = false;
        
        for connection in &mut genome.connections {
            if context.rng.gen::<f64>() < context.config.mutation.weight_mutation_rate {
                if context.rng.gen::<f64>() < context.config.mutation.weight_perturbation_rate {
                    // Perturb existing weight
                    let perturbation = context.rng.gen_range(
                        -context.config.mutation.weight_perturbation_power..=context.config.mutation.weight_perturbation_power
                    );
                    connection.weight += perturbation;
                } else {
                    // Replace with new random weight
                    let weight_range = context.config.mutation.weight_range;
                    connection.weight = context.rng.gen_range(weight_range.0..=weight_range.1);
                }
                
                mutated = true;
            }
        }
        
        Ok(mutated)
    }
    
    fn probability(&self, config: &NEATConfig) -> f64 {
        config.mutation.weight_mutation_rate
    }
}

impl Mutation for ToggleConnectionMutation {
    fn mutate(&self, genome: &mut Genome, context: &mut MutationContext) -> Result<bool> {
        if genome.connections.is_empty() {
            return Ok(false);
        }
        
        // Choose random connection
        let conn_idx = context.rng.gen_range(0..genome.connections.len());
        
        // Toggle enabled state
        genome.connections[conn_idx].enabled = !genome.connections[conn_idx].enabled;
        
        Ok(true)
    }
    
    fn probability(&self, config: &NEATConfig) -> f64 {
        config.mutation.disable_connection_rate
    }
}

impl Mutation for BiasMutation {
    fn mutate(&self, genome: &mut Genome, context: &mut MutationContext) -> Result<bool> {
        let mut mutated = false;
        
        for node in &mut genome.nodes {
            // Don't mutate bias of bias nodes (they should always be 1.0)
            if node.node_type == NodeType::Bias {
                continue;
            }
            
            if context.rng.gen::<f64>() < context.config.mutation.bias_mutation_rate {
                if context.rng.gen::<f64>() < context.config.mutation.bias_perturbation_rate {
                    // Perturb existing bias
                    let perturbation = context.rng.gen_range(
                        -context.config.mutation.bias_perturbation_power..=context.config.mutation.bias_perturbation_power
                    );
                    node.bias += perturbation;
                } else {
                    // Replace with new random bias
                    let bias_range = context.config.mutation.bias_range;
                    node.bias = context.rng.gen_range(bias_range.0..=bias_range.1);
                }
                
                mutated = true;
            }
        }
        
        Ok(mutated)
    }
    
    fn probability(&self, config: &NEATConfig) -> f64 {
        config.mutation.bias_mutation_rate
    }
}

impl Mutation for ActivationMutation {
    fn mutate(&self, genome: &mut Genome, context: &mut MutationContext) -> Result<bool> {
        // Only mutate hidden nodes (input/output activations are typically fixed)
        let hidden_nodes: Vec<usize> = genome.nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.node_type == NodeType::Hidden)
            .map(|(idx, _)| idx)
            .collect();
            
        if hidden_nodes.is_empty() {
            return Ok(false);
        }
        
        let mut mutated = false;
        
        for &node_idx in &hidden_nodes {
            if context.rng.gen::<f64>() < context.config.mutation.activation_mutation_rate {
                let new_activation = choose_activation_function(
                    context.rng, 
                    &context.config.network.activation_functions
                );
                
                // Only change if it's actually different
                if genome.nodes[node_idx].activation != new_activation {
                    genome.nodes[node_idx].activation = new_activation;
                    mutated = true;
                }
            }
        }
        
        Ok(mutated)
    }
    
    fn probability(&self, config: &NEATConfig) -> f64 {
        config.mutation.activation_mutation_rate
    }
}

/// Choose a random activation function from allowed list
fn choose_activation_function(rng: &mut dyn RngCore, allowed: &[ActivationType]) -> ActivationType {
    if allowed.is_empty() {
        ActivationType::Sigmoid // Default fallback
    } else {
        allowed.choose(rng).copied().unwrap_or(ActivationType::Sigmoid)
    }
}

/// Complete mutation pipeline that applies multiple mutations
pub struct MutationPipeline {
    mutations: Vec<Box<dyn Mutation>>,
}

impl Default for MutationPipeline {
    fn default() -> Self {
        Self {
            mutations: vec![
                Box::new(WeightMutation),
                Box::new(BiasMutation),
                Box::new(AddConnectionMutation),
                Box::new(AddNodeMutation),
                Box::new(ToggleConnectionMutation),
                Box::new(ActivationMutation),
            ],
        }
    }
}

impl MutationPipeline {
    /// Create a new mutation pipeline with custom mutations
    pub fn new(mutations: Vec<Box<dyn Mutation>>) -> Self {
        Self { mutations }
    }
    
    /// Apply all mutations to a genome based on their probabilities
    pub fn mutate_genome(&self, genome: &mut Genome, context: &mut MutationContext) -> Result<Vec<String>> {
        let mut applied_mutations = Vec::new();
        
        for mutation in &self.mutations {
            let probability = mutation.probability(context.config);
            
            if context.rng.gen::<f64>() < probability {
                match mutation.mutate(genome, context) {
                    Ok(true) => {
                        applied_mutations.push("Mutation applied".to_string());
                    }
                    Ok(false) => {
                        // Mutation attempted but failed (e.g., no valid connections)
                    }
                    Err(e) => {
                        // Mutation failed with error
                        log::warn!("Mutation failed: {}", e);
                    }
                }
            }
        }
        
        Ok(applied_mutations)
    }
    
    /// Mutate a batch of genomes
    pub fn mutate_batch(&self, genomes: &mut [Genome], context: &mut MutationContext) -> Result<Vec<Vec<String>>> {
        genomes.iter_mut()
            .map(|genome| self.mutate_genome(genome, context))
            .collect()
    }
}

/// Utility functions for mutation analysis
pub mod utils {
    use super::*;
    
    /// Calculate mutation statistics for a genome
    pub fn calculate_mutation_potential(genome: &Genome) -> MutationPotential {
        let enabled_connections = genome.connections.iter().filter(|c| c.enabled).count();
        let _disabled_connections = genome.connections.len() - enabled_connections;
        
        // Calculate potential new connections (avoiding cycles and duplicates)
        let mut potential_connections = 0;
        for input_node in &genome.nodes {
            for output_node in &genome.nodes {
                if can_connect(input_node, output_node) && 
                   !genome.has_connection(input_node.id, output_node.id) {
                    potential_connections += 1;
                }
            }
        }
        
        MutationPotential {
            potential_new_connections: potential_connections,
            splittable_connections: enabled_connections,
            mutable_weights: genome.connections.len(),
            toggleable_connections: genome.connections.len(),
            mutable_biases: genome.nodes.iter().filter(|n| n.node_type != NodeType::Bias).count(),
            mutable_activations: genome.nodes.iter().filter(|n| n.node_type == NodeType::Hidden).count(),
        }
    }
    
    fn can_connect(input_node: &crate::neat::genome::NodeGene, output_node: &crate::neat::genome::NodeGene) -> bool {
        use crate::neat::genome::NodeType;
        
        // Basic rules for valid connections
        match (input_node.node_type, output_node.node_type) {
            (NodeType::Output, _) => false, // Outputs can't be inputs
            (_, NodeType::Input) => false,  // Inputs can't be outputs
            (_, NodeType::Bias) => false,   // Nothing connects to bias
            (NodeType::Input, NodeType::Output) => true,
            (NodeType::Input, NodeType::Hidden) => true,
            (NodeType::Hidden, NodeType::Output) => true,
            (NodeType::Hidden, NodeType::Hidden) => input_node.id != output_node.id, // No self-connections
            (NodeType::Bias, NodeType::Output) => true,
            (NodeType::Bias, NodeType::Hidden) => true,
        }
    }
}

/// Statistics about mutation potential for a genome
#[derive(Debug, Clone, PartialEq)]
pub struct MutationPotential {
    /// Number of new connections that could be added
    pub potential_new_connections: usize,
    /// Number of connections that could be split with nodes
    pub splittable_connections: usize,
    /// Number of weights that could be mutated
    pub mutable_weights: usize,
    /// Number of connections that could be toggled
    pub toggleable_connections: usize,
    /// Number of biases that could be mutated
    pub mutable_biases: usize,
    /// Number of activation functions that could be mutated
    pub mutable_activations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NEATConfig;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn create_test_context() -> (NEATConfig, InnovationTracker, SmallRng) {
        let config = NEATConfig::default();
        // Start innovation tracker with ID 1000 to avoid conflicts with genome node IDs
        let tracker = InnovationTracker::with_starting_id(1000);
        let rng = SmallRng::seed_from_u64(42);
        (config, tracker, rng)
    }

    #[test]
    fn test_add_connection_mutation() {
        let mut genome = Genome::new(0, 2, 1);
        let (config, mut tracker, mut rng) = create_test_context();
        
        let mut context = MutationContext {
            config: &config,
            innovation_tracker: &mut tracker,
            rng: &mut rng,
        };
        
        let mutation = AddConnectionMutation;
        let initial_connections = genome.connections.len();
        
        let result = mutation.mutate(&mut genome, &mut context).unwrap();
        
        if result {
            assert!(genome.connections.len() > initial_connections);
        }
    }

    #[test]
    fn test_add_node_mutation() {
        let mut genome = Genome::new(0, 2, 1);
        let (config, mut tracker, mut rng) = create_test_context();
        
        // First add a connection so we have something to split
        let innovation_id = tracker.get_innovation_id(0, 3);
        let conn = ConnectionGene::new(innovation_id, 0, 3, 1.0); // input -> output
        genome.add_connection(conn).unwrap();
        
        let mut context = MutationContext {
            config: &config,
            innovation_tracker: &mut tracker,
            rng: &mut rng,
        };
        
        let mutation = AddNodeMutation;
        let initial_nodes = genome.nodes.len();
        let initial_enabled_connections = genome.connections.iter().filter(|c| c.enabled).count();
        
        let result = mutation.mutate(&mut genome, &mut context).unwrap();
        
        if result {
            assert_eq!(genome.nodes.len(), initial_nodes + 1);
            assert_eq!(genome.connections.iter().filter(|c| c.enabled).count(), initial_enabled_connections + 1);
            
            // Should have one disabled connection (the split one)
            assert_eq!(genome.connections.iter().filter(|c| !c.enabled).count(), 1);
        }
    }

    #[test]
    fn test_weight_mutation() {
        let mut genome = Genome::new(0, 2, 1);
        let (config, mut tracker, mut rng) = create_test_context();
        
        // Add a connection to mutate
        let innovation_id = tracker.get_innovation_id(0, 3);
        let conn = ConnectionGene::new(innovation_id, 0, 3, 1.0);
        genome.add_connection(conn).unwrap();
        
        let mut context = MutationContext {
            config: &config,
            innovation_tracker: &mut tracker,
            rng: &mut rng,
        };
        
        let original_weight = genome.connections[0].weight;
        let mutation = WeightMutation;
        
        // Force mutation by setting high rate temporarily
        let mut test_config = config.clone();
        test_config.mutation.weight_mutation_rate = 1.0;
        context.config = &test_config;
        
        let result = mutation.mutate(&mut genome, &mut context).unwrap();
        
        if result {
            // Weight should have changed
            assert_ne!(genome.connections[0].weight, original_weight);
        }
    }

    #[test]
    fn test_toggle_connection_mutation() {
        let mut genome = Genome::new(0, 2, 1);
        let (config, mut tracker, mut rng) = create_test_context();
        
        // Add a connection to toggle
        let innovation_id = tracker.get_innovation_id(0, 3);
        let conn = ConnectionGene::new(innovation_id, 0, 3, 1.0);
        genome.add_connection(conn).unwrap();
        
        let mut context = MutationContext {
            config: &config,
            innovation_tracker: &mut tracker,
            rng: &mut rng,
        };
        
        let original_enabled = genome.connections[0].enabled;
        let mutation = ToggleConnectionMutation;
        
        let result = mutation.mutate(&mut genome, &mut context).unwrap();
        
        assert!(result);
        assert_ne!(genome.connections[0].enabled, original_enabled);
    }

    #[test]
    fn test_bias_mutation() {
        let mut genome = Genome::new(0, 2, 1);
        let (config, mut tracker, mut rng) = create_test_context();
        
        let mut context = MutationContext {
            config: &config,
            innovation_tracker: &mut tracker,
            rng: &mut rng,
        };
        
        let original_biases: Vec<f64> = genome.nodes.iter().map(|n| n.bias).collect();
        let mutation = BiasMutation;
        
        // Force mutation by setting high rate
        let mut test_config = config.clone();
        test_config.mutation.bias_mutation_rate = 1.0;
        context.config = &test_config;
        
        let result = mutation.mutate(&mut genome, &mut context).unwrap();
        
        if result {
            // At least one bias should have changed (excluding bias node which stays at 1.0)
            let new_biases: Vec<f64> = genome.nodes.iter().map(|n| n.bias).collect();
            let mut bias_changed = false;
            
            for (i, node) in genome.nodes.iter().enumerate() {
                if node.node_type != NodeType::Bias && original_biases[i] != new_biases[i] {
                    bias_changed = true;
                    break;
                }
            }
            
            assert!(bias_changed);
        }
    }

    #[test]
    fn test_mutation_pipeline() {
        let mut genome = Genome::new(0, 2, 1);
        let (config, mut tracker, mut rng) = create_test_context();
        
        let mut context = MutationContext {
            config: &config,
            innovation_tracker: &mut tracker,
            rng: &mut rng,
        };
        
        let pipeline = MutationPipeline::default();
        let applied = pipeline.mutate_genome(&mut genome, &mut context).unwrap();
        
        // Should return list of applied mutations (could be empty if probabilities are low)
        assert!(applied.len() <= 6); // Maximum number of mutation types
    }

    #[test]
    fn test_mutation_potential() {
        let genome = Genome::new(0, 2, 1);
        let potential = utils::calculate_mutation_potential(&genome);
        
        assert!(potential.potential_new_connections > 0);
        assert_eq!(potential.splittable_connections, 0); // No connections initially
        assert_eq!(potential.mutable_weights, 0);
        assert_eq!(potential.toggleable_connections, 0);
        assert!(potential.mutable_biases > 0); // Input and output nodes can have bias mutated
        assert_eq!(potential.mutable_activations, 0); // No hidden nodes initially
    }

    #[test]
    fn test_activation_function_choice() {
        let mut rng = SmallRng::seed_from_u64(42);
        let allowed = vec![ActivationType::Sigmoid, ActivationType::Tanh, ActivationType::ReLU];
        
        let chosen = choose_activation_function(&mut rng, &allowed);
        assert!(allowed.contains(&chosen));
        
        // Test empty list fallback
        let empty: Vec<ActivationType> = vec![];
        let fallback = choose_activation_function(&mut rng, &empty);
        assert_eq!(fallback, ActivationType::Sigmoid);
    }
}