//! Integration tests for Week 2 NEAT functionality
//!
//! This module contains comprehensive tests that validate the interaction
//! between all Week 2 components: network activation, fitness evaluation,
//! mutation operations, and topology analysis.

#[cfg(test)]
mod tests {
    use crate::neat::*;
    use crate::config::NEATConfig;
    use crate::error::Result;
    use rand::{SeedableRng, rngs::SmallRng};
    use ndarray::Array2;
    use approx::assert_relative_eq;

    /// Test complete workflow: genome -> network -> activation -> fitness
    #[test]
    fn test_complete_activation_workflow() -> Result<()> {
        // Create a simple genome
        let mut genome = Genome::new(0, 3, 2);
        
        // Add some connections to make it interesting
        let mut tracker = InnovationTracker::with_starting_id(1000);
        let conn1 = ConnectionGene::new(tracker.get_innovation_id(0, 4), 0, 4, 0.5); // input0 -> output0
        let conn2 = ConnectionGene::new(tracker.get_innovation_id(1, 5), 1, 5, -0.3); // input1 -> output1
        let conn3 = ConnectionGene::new(tracker.get_innovation_id(3, 4), 3, 4, 1.0); // bias -> output0
        
        genome.add_connection(conn1)?;
        genome.add_connection(conn2)?;
        genome.add_connection(conn3)?;
        
        // Create network and activate
        let network = Network::from_genome(&genome)?;
        let inputs = vec![1.0, -0.5, 0.8];
        let outputs = network.activate(&inputs)?;
        
        // Verify outputs
        assert_eq!(outputs.len(), 2);
        
        // Expected output0: sigmoid(1.0 * 0.5 + 1.0 * 1.0) = sigmoid(1.5)
        let expected_output0 = ActivationType::Sigmoid.activate(1.5);
        assert_relative_eq!(outputs[0], expected_output0, epsilon = 1e-10);
        
        // Expected output1: sigmoid(-0.5 * -0.3) = sigmoid(0.15)
        let expected_output1 = ActivationType::Sigmoid.activate(0.15);
        assert_relative_eq!(outputs[1], expected_output1, epsilon = 1e-10);
        
        Ok(())
    }

    /// Test fitness evaluation with XOR problem
    #[test]
    fn test_xor_fitness_evaluation() -> Result<()> {
        // Create a genome for XOR (2 inputs, 1 output)
        let mut genome = Genome::new(0, 2, 1);
        
        // Add hidden node and connections to solve XOR
        let hidden_node = NodeGene::new(10, NodeType::Hidden, ActivationType::Sigmoid);
        genome.add_node(hidden_node)?;
        
        let mut tracker = InnovationTracker::with_starting_id(1000);
        
        // Create a network that can potentially solve XOR
        let connections = vec![
            (0, 10, 2.0),  // input0 -> hidden
            (1, 10, 2.0),  // input1 -> hidden
            (2, 10, -1.0), // bias -> hidden
            (10, 3, 2.0),  // hidden -> output
        ];
        
        for (input, output, weight) in connections {
            let innovation_id = tracker.get_innovation_id(input, output);
            let conn = ConnectionGene::new(innovation_id, input, output, weight);
            genome.add_connection(conn)?;
        }
        
        // Evaluate fitness
        let evaluator = XORFitnessEvaluator::default();
        let fitness = evaluator.evaluate(&genome)?;
        
        // Should get reasonable fitness (this configuration should do reasonably well on XOR)
        assert!(fitness > 0.0);
        assert!(fitness <= 1.0);
        
        Ok(())
    }

    /// Test classification fitness with simple dataset
    #[test]
    fn test_classification_fitness_evaluation() -> Result<()> {
        // Create simple 2D classification dataset
        let inputs = Array2::from_shape_vec((4, 2), vec![
            1.0, 1.0,   // Class 0
            1.0, -1.0,  // Class 1
            -1.0, 1.0,  // Class 1
            -1.0, -1.0, // Class 0
        ]).unwrap();
        
        let targets = Array2::from_shape_vec((4, 2), vec![
            1.0, 0.0,  // Class 0
            0.0, 1.0,  // Class 1
            0.0, 1.0,  // Class 1
            1.0, 0.0,  // Class 0
        ]).unwrap();
        
        let evaluator = ClassificationEvaluator::new(inputs, targets, 2)?;
        
        // Test with simple genome
        let genome = Genome::new(0, 2, 2);
        let fitness = evaluator.evaluate(&genome)?;
        
        // Should get some fitness (even random should get ~0.5 accuracy)
        assert!(fitness >= 0.0);
        assert!(fitness <= 1.0);
        
        Ok(())
    }

    /// Test mutation operations comprehensively
    #[test]
    fn test_comprehensive_mutation() -> Result<()> {
        let mut genome = Genome::new(0, 3, 2);
        let config = NEATConfig::default();
        let mut tracker = InnovationTracker::with_starting_id(1000);
        let mut rng = SmallRng::seed_from_u64(12345);
        
        // Add initial connections
        let conn1 = ConnectionGene::new(tracker.get_innovation_id(0, 4), 0, 4, 1.0);
        let conn2 = ConnectionGene::new(tracker.get_innovation_id(1, 5), 1, 5, -0.5);
        genome.add_connection(conn1)?;
        genome.add_connection(conn2)?;
        
        let mut context = MutationContext {
            config: &config,
            innovation_tracker: &mut tracker,
            rng: &mut rng,
        };
        
        // Test individual mutations
        let initial_nodes = genome.nodes.len();
        let initial_connections = genome.connections.len();
        
        // Test add node mutation
        let add_node = mutation::AddNodeMutation;
        let node_added = add_node.mutate(&mut genome, &mut context)?;
        if node_added {
            assert!(genome.nodes.len() > initial_nodes);
            assert!(genome.connections.len() > initial_connections);
        }
        
        // Test add connection mutation
        let add_connection = mutation::AddConnectionMutation;
        let _connection_added = add_connection.mutate(&mut genome, &mut context)?;
        // Connection might not be added if no valid connections exist
        
        // Test weight mutation (force it to happen)
        let original_weights: Vec<f64> = genome.connections.iter().map(|c| c.weight).collect();
        let weight_mutation = mutation::WeightMutation;
        
        // Temporarily increase mutation rate
        let mut test_config = config.clone();
        test_config.mutation.weight_mutation_rate = 1.0;
        context.config = &test_config;
        
        let weights_mutated = weight_mutation.mutate(&mut genome, &mut context)?;
        if weights_mutated && !genome.connections.is_empty() {
            let new_weights: Vec<f64> = genome.connections.iter().map(|c| c.weight).collect();
            // At least one weight should have changed
            assert_ne!(original_weights, new_weights);
        }
        
        Ok(())
    }

    /// Test mutation pipeline with multiple mutations
    #[test]
    fn test_mutation_pipeline() -> Result<()> {
        let mut genome = Genome::new(0, 2, 1);
        let config = NEATConfig::default();
        let mut tracker = InnovationTracker::with_starting_id(1000);
        let mut rng = SmallRng::seed_from_u64(42);
        
        // Add initial connection
        let conn = ConnectionGene::new(tracker.get_innovation_id(0, 3), 0, 3, 0.5);
        genome.add_connection(conn)?;
        
        let mut context = MutationContext {
            config: &config,
            innovation_tracker: &mut tracker,
            rng: &mut rng,
        };
        
        let pipeline = MutationPipeline::default();
        let applied_mutations = pipeline.mutate_genome(&mut genome, &mut context)?;
        
        // Should return list of applied mutations
        assert!(applied_mutations.len() <= 6); // Maximum number of mutation types
        
        // Genome should still be valid after mutations
        genome.validate()?;
        
        Ok(())
    }

    /// Test topology analysis integration
    #[test]
    fn test_topology_analysis_integration() -> Result<()> {
        let mut genome = Genome::new(0, 3, 2);
        
        // Add some structure
        let hidden1 = NodeGene::new(10, NodeType::Hidden, ActivationType::Tanh);
        let hidden2 = NodeGene::new(11, NodeType::Hidden, ActivationType::ReLU);
        genome.add_node(hidden1)?;
        genome.add_node(hidden2)?;
        
        // Add connections to create interesting topology
        let mut tracker = InnovationTracker::with_starting_id(1000);
        let connections = vec![
            (0, 10), // input0 -> hidden1
            (1, 11), // input1 -> hidden2
            (10, 4), // hidden1 -> output0
            (11, 5), // hidden2 -> output1
            (10, 11), // hidden1 -> hidden2 (creates depth)
        ];
        
        for (input, output) in connections {
            let innovation_id = tracker.get_innovation_id(input, output);
            let conn = ConnectionGene::new(innovation_id, input, output, 1.0);
            genome.add_connection(conn)?;
        }
        
        // Analyze topology
        let analysis = TopologyAnalyzer::analyze_genome(&genome)?;
        
        // Verify analysis results
        assert_eq!(analysis.structure.node_counts[&NodeType::Input], 3);
        assert_eq!(analysis.structure.node_counts[&NodeType::Output], 2);
        assert_eq!(analysis.structure.node_counts[&NodeType::Hidden], 2);
        assert_eq!(analysis.structure.node_counts[&NodeType::Bias], 1);
        assert_eq!(analysis.structure.enabled_connections, 5);
        assert!(analysis.structure.is_feedforward);
        assert!(analysis.structure.depth >= 2); // Due to hidden1 -> hidden2 -> output
        
        // Network should still be activatable
        let network = Network::from_genome(&genome)?;
        let inputs = vec![1.0, -1.0, 0.5];
        let outputs = network.activate(&inputs)?;
        assert_eq!(outputs.len(), 2);
        
        Ok(())
    }

    /// Test evolution simulation (simplified)
    #[test]
    fn test_evolution_simulation() -> Result<()> {
        let config = NEATConfig::default();
        let mut tracker = InnovationTracker::with_starting_id(1000);
        let mut rng = SmallRng::seed_from_u64(123);
        
        // Create initial population
        let mut population = Vec::new();
        for i in 0..10 {
            population.push(Genome::new(i, 2, 1));
        }
        
        // Simple XOR evaluator
        let evaluator = XORFitnessEvaluator::default();
        
        // Simulate a few generations
        for generation in 0..3 {
            // Evaluate fitness
            let mut fitness_scores = Vec::new();
            for genome in &population {
                let fitness = evaluator.evaluate(genome)?;
                fitness_scores.push(fitness);
            }
            
            // Find best fitness
            let best_fitness = fitness_scores.iter().fold(0.0f64, |a, &b| a.max(b));
            
            println!("Generation {}: Best fitness = {:.4}", generation, best_fitness);
            
            // Mutate population (simplified)
            let pipeline = MutationPipeline::default();
            for genome in &mut population {
                let mut context = MutationContext {
                    config: &config,
                    innovation_tracker: &mut tracker,
                    rng: &mut rng,
                };
                
                let _applied = pipeline.mutate_genome(genome, &mut context)?;
                
                // Ensure genome is still valid
                genome.validate()?;
            }
            
            tracker.next_generation();
        }
        
        Ok(())
    }

    /// Test network with complex topology
    #[test]
    fn test_complex_network_activation() -> Result<()> {
        let mut genome = Genome::new(0, 4, 3);
        
        // Add multiple hidden layers
        let hidden_nodes = vec![
            NodeGene::new(20, NodeType::Hidden, ActivationType::Tanh),
            NodeGene::new(21, NodeType::Hidden, ActivationType::ReLU),
            NodeGene::new(22, NodeType::Hidden, ActivationType::Sigmoid),
            NodeGene::new(23, NodeType::Hidden, ActivationType::Linear),
        ];
        
        for node in hidden_nodes {
            genome.add_node(node)?;
        }
        
        // Create complex connectivity
        let mut tracker = InnovationTracker::with_starting_id(1000);
        let connections = vec![
            // Input layer to first hidden layer
            (0, 20, 0.5),
            (1, 20, -0.3),
            (2, 21, 0.8),
            (3, 21, 0.2),
            // Bias connections
            (4, 20, 0.1),
            (4, 21, -0.1),
            // Hidden to hidden
            (20, 22, 1.0),
            (21, 23, -0.5),
            // Hidden to output
            (22, 5, 0.7),
            (23, 6, 0.9),
            (20, 7, 0.4),
            // Skip connections
            (0, 5, 0.1),
            (1, 6, -0.1),
        ];
        
        for (input, output, weight) in connections {
            let innovation_id = tracker.get_innovation_id(input, output);
            let conn = ConnectionGene::new(innovation_id, input, output, weight);
            genome.add_connection(conn)?;
        }
        
        // Test network creation and activation
        let network = Network::from_genome(&genome)?;
        let inputs = vec![1.0, -0.5, 0.3, 0.8];
        let outputs = network.activate(&inputs)?;
        
        assert_eq!(outputs.len(), 3);
        
        // All outputs should be finite
        for &output in &outputs {
            assert!(output.is_finite());
        }
        
        // Test network info
        let info = network.get_info();
        assert_eq!(info.num_inputs, 4);
        assert_eq!(info.num_outputs, 3);
        assert_eq!(info.num_hidden, 4);
        assert!(info.depth >= 2);
        
        Ok(())
    }

    /// Test fitness evaluation with complexity penalty
    #[test]
    fn test_fitness_with_complexity_penalty() -> Result<()> {
        // Create simple dataset
        let inputs = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
        let targets = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
        
        let penalty = fitness::ComplexityPenalty {
            connection_penalty: 0.1,
            node_penalty: 0.05,
            depth_penalty: 0.01,
        };
        
        let evaluator = ClassificationEvaluator::new(inputs, targets, 1)?
            .with_complexity_penalty(penalty);
        
        // Compare simple vs complex genome
        let simple_genome = Genome::new(0, 1, 1);
        
        let mut complex_genome = Genome::new(1, 1, 1);
        for i in 0..5 {
            let hidden = NodeGene::new(100 + i, NodeType::Hidden, ActivationType::Sigmoid);
            complex_genome.add_node(hidden)?;
        }
        
        let simple_fitness = evaluator.evaluate(&simple_genome)?;
        let complex_fitness = evaluator.evaluate(&complex_genome)?;
        
        // Simple genome should have higher fitness due to lower complexity penalty
        assert!(simple_fitness >= complex_fitness);
        
        Ok(())
    }

    /// Test error handling and edge cases
    #[test]
    fn test_error_handling() -> Result<()> {
        // Test network creation with invalid genome
        let mut invalid_genome = Genome::new(0, 2, 1);
        let invalid_conn = ConnectionGene::new(0, 999, 1000, 1.0); // Non-existent nodes
        invalid_genome.connections.push(invalid_conn); // Bypass validation
        
        // Should fail to create network
        let result = Network::from_genome(&invalid_genome);
        assert!(result.is_err());
        
        // Test fitness evaluation with wrong input size
        let genome = Genome::new(0, 2, 1);
        let network = Network::from_genome(&genome)?;
        
        let wrong_inputs = vec![1.0]; // Should be 2 inputs
        let result = network.activate(&wrong_inputs);
        assert!(result.is_err());
        
        // Test invalid classification dataset
        let inputs = Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]).unwrap();
        let targets = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0]).unwrap(); // Wrong size
        
        let result = ClassificationEvaluator::new(inputs, targets, 2);
        assert!(result.is_err());
        
        Ok(())
    }
}