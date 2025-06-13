//! Crossover operations for NEAT genomes
//!
//! This module implements the crossover (breeding) operations that are fundamental
//! to NEAT's genetic algorithm. NEAT uses historical markings to enable meaningful
//! crossover between genomes with different topologies.

use crate::neat::genome::{Genome, NodeGene, ConnectionGene, NodeType, ActivationType};
use crate::neat::innovation::InnovationTracker;
use crate::config::NEATConfig;
use crate::error::{NEATError, Result};
use rand::prelude::*;
use std::collections::{HashMap, HashSet};

/// Crossover context containing configuration and shared state
pub struct CrossoverContext<'a> {
    /// Configuration for crossover parameters
    pub config: &'a NEATConfig,
    /// Innovation tracker for historical markings
    pub innovation_tracker: &'a InnovationTracker,
    /// Random number generator
    pub rng: &'a mut dyn RngCore,
}

/// Result of a crossover operation
#[derive(Debug, Clone)]
pub struct CrossoverResult {
    /// The resulting offspring genome
    pub offspring: Genome,
    /// Statistics about the crossover operation
    pub stats: CrossoverStats,
}

/// Statistics about a crossover operation
#[derive(Debug, Clone, PartialEq)]
pub struct CrossoverStats {
    /// Number of matching genes (same innovation ID)
    pub matching_genes: usize,
    /// Number of disjoint genes (innovation IDs in one parent but not the other, within range)
    pub disjoint_genes: usize,
    /// Number of excess genes (innovation IDs beyond the range of the other parent)
    pub excess_genes: usize,
    /// Parent that was considered more fit (or chosen in case of equal fitness)
    pub fitter_parent: usize, // 0 for parent1, 1 for parent2
    /// Whether structural genes were inherited from both parents
    pub mixed_structure: bool,
}

/// Main crossover implementation for NEAT genomes
pub struct NEATCrossover;

impl NEATCrossover {
    /// Perform crossover between two genomes
    /// 
    /// This implements the NEAT crossover algorithm where:
    /// - Matching genes are randomly inherited from either parent
    /// - Disjoint and excess genes are inherited from the more fit parent
    /// - If fitness is equal, genes are inherited from both parents or the larger genome
    pub fn crossover(
        parent1: &Genome,
        parent2: &Genome,
        context: &mut CrossoverContext,
    ) -> Result<CrossoverResult> {
        // Validate parents
        parent1.validate()?;
        parent2.validate()?;
        
        // Determine which parent is more fit
        let (fitter_parent, less_fit_parent, fitter_idx) = if parent1.fitness > parent2.fitness {
            (parent1, parent2, 0)
        } else if parent2.fitness > parent1.fitness {
            (parent2, parent1, 1)
        } else {
            // Equal fitness - choose the larger genome, or parent1 if equal size
            if parent1.connections.len() >= parent2.connections.len() {
                (parent1, parent2, 0)
            } else {
                (parent2, parent1, 1)
            }
        };
        
        // Analyze gene alignment
        let alignment = Self::analyze_gene_alignment(parent1, parent2);
        
        // Create offspring genome
        let offspring_id = context.rng.gen::<usize>();
        let offspring = Self::create_offspring(
            parent1,
            parent2,
            fitter_parent,
            less_fit_parent,
            &alignment,
            offspring_id,
            context,
        )?;
        
        let stats = CrossoverStats {
            matching_genes: alignment.matching_connections.len(),
            disjoint_genes: alignment.disjoint_connections.len(),
            excess_genes: alignment.excess_connections.len(),
            fitter_parent: fitter_idx,
            mixed_structure: alignment.matching_connections.len() > 0 && 
                           (alignment.disjoint_connections.len() > 0 || alignment.excess_connections.len() > 0),
        };
        
        Ok(CrossoverResult { offspring, stats })
    }
    
    /// Analyze the alignment of genes between two parents
    fn analyze_gene_alignment(parent1: &Genome, parent2: &Genome) -> GeneAlignment {
        let mut p1_connections: HashMap<usize, &ConnectionGene> = HashMap::new();
        let mut p2_connections: HashMap<usize, &ConnectionGene> = HashMap::new();
        
        // Index connections by innovation ID
        for conn in &parent1.connections {
            p1_connections.insert(conn.innovation_id, conn);
        }
        
        for conn in &parent2.connections {
            p2_connections.insert(conn.innovation_id, conn);
        }
        
        // Find min and max innovation IDs for each parent
        let p1_max = p1_connections.keys().max().copied().unwrap_or(0);
        let p2_max = p2_connections.keys().max().copied().unwrap_or(0);
        let overall_max = p1_max.max(p2_max);
        
        let mut matching_connections = Vec::new();
        let mut disjoint_connections = Vec::new();
        let mut excess_connections = Vec::new();
        
        // Collect all innovation IDs
        let mut all_innovation_ids: HashSet<usize> = HashSet::new();
        all_innovation_ids.extend(p1_connections.keys());
        all_innovation_ids.extend(p2_connections.keys());
        
        for &innovation_id in &all_innovation_ids {
            match (p1_connections.get(&innovation_id), p2_connections.get(&innovation_id)) {
                (Some(conn1), Some(conn2)) => {
                    // Matching gene
                    matching_connections.push(MatchingConnection {
                        innovation_id,
                        parent1_gene: (*conn1).clone(),
                        parent2_gene: (*conn2).clone(),
                    });
                },
                (Some(conn), None) => {
                    // Gene only in parent1
                    if innovation_id > p2_max {
                        excess_connections.push(ExcessConnection {
                            innovation_id,
                            gene: (*conn).clone(),
                            parent: 1,
                        });
                    } else {
                        disjoint_connections.push(DisjointConnection {
                            innovation_id,
                            gene: (*conn).clone(),
                            parent: 1,
                        });
                    }
                },
                (None, Some(conn)) => {
                    // Gene only in parent2
                    if innovation_id > p1_max {
                        excess_connections.push(ExcessConnection {
                            innovation_id,
                            gene: (*conn).clone(),
                            parent: 2,
                        });
                    } else {
                        disjoint_connections.push(DisjointConnection {
                            innovation_id,
                            gene: (*conn).clone(),
                            parent: 2,
                        });
                    }
                },
                (None, None) => unreachable!(), // Can't happen since we iterate over existing keys
            }
        }
        
        GeneAlignment {
            matching_connections,
            disjoint_connections,
            excess_connections,
        }
    }
    
    /// Create offspring genome from aligned gene analysis
    fn create_offspring(
        parent1: &Genome,
        parent2: &Genome,
        fitter_parent: &Genome,
        _less_fit_parent: &Genome,
        alignment: &GeneAlignment,
        offspring_id: usize,
        context: &mut CrossoverContext,
    ) -> Result<Genome> {
        // Start with the input/output structure from parent1 (they should be the same)
        let input_count = parent1.get_input_count();
        let output_count = parent1.get_output_count();
        let mut offspring = Genome::new(offspring_id, input_count, output_count);
        
        // Collect all nodes that will be needed
        let mut required_nodes: HashSet<usize> = HashSet::new();
        let mut selected_connections = Vec::new();
        
        // Process matching genes - randomly inherit from either parent
        for matching in &alignment.matching_connections {
            let chosen_gene = if context.rng.gen::<f64>() < 0.5 {
                &matching.parent1_gene
            } else {
                &matching.parent2_gene
            };
            
            selected_connections.push(chosen_gene.clone());
            required_nodes.insert(chosen_gene.input_node);
            required_nodes.insert(chosen_gene.output_node);
        }
        
        // Process disjoint and excess genes - inherit from fitter parent only
        // (or from both if fitness is equal)
        let inherit_from_both = parent1.fitness == parent2.fitness;
        
        for disjoint in &alignment.disjoint_connections {
            if inherit_from_both || 
               (disjoint.parent == 1 && std::ptr::eq(fitter_parent, parent1)) ||
               (disjoint.parent == 2 && std::ptr::eq(fitter_parent, parent2)) {
                selected_connections.push(disjoint.gene.clone());
                required_nodes.insert(disjoint.gene.input_node);
                required_nodes.insert(disjoint.gene.output_node);
            }
        }
        
        for excess in &alignment.excess_connections {
            if inherit_from_both || 
               (excess.parent == 1 && std::ptr::eq(fitter_parent, parent1)) ||
               (excess.parent == 2 && std::ptr::eq(fitter_parent, parent2)) {
                selected_connections.push(excess.gene.clone());
                required_nodes.insert(excess.gene.input_node);
                required_nodes.insert(excess.gene.output_node);
            }
        }
        
        // Add all required nodes from both parents
        let mut all_parent_nodes: HashMap<usize, NodeGene> = HashMap::new();
        
        for node in &parent1.nodes {
            all_parent_nodes.insert(node.id, node.clone());
        }
        
        for node in &parent2.nodes {
            if !all_parent_nodes.contains_key(&node.id) {
                all_parent_nodes.insert(node.id, node.clone());
            } else {
                // If node exists in both parents, randomly choose which version to use
                if context.rng.gen::<f64>() < 0.5 {
                    all_parent_nodes.insert(node.id, node.clone());
                }
            }
        }
        
        // Add required nodes to offspring (skip input/output/bias nodes that are already there)
        for &node_id in &required_nodes {
            if let Some(node) = all_parent_nodes.get(&node_id) {
                if !offspring.nodes.iter().any(|n| n.id == node_id) {
                    offspring.add_node(node.clone())?;
                }
            }
        }
        
        // Add selected connections to offspring
        for connection in selected_connections {
            offspring.add_connection(connection)?;
        }
        
        // Validate the resulting offspring
        offspring.validate()?;
        
        Ok(offspring)
    }
    
    /// Calculate compatibility distance between two genomes
    /// 
    /// This is used for speciation and follows the NEAT compatibility function:
    /// δ = (c1 * E / N) + (c2 * D / N) + c3 * W̄
    /// 
    /// Where:
    /// - E = number of excess genes
    /// - D = number of disjoint genes  
    /// - W̄ = average weight difference of matching genes
    /// - N = number of genes in larger genome (1 if both genomes are small)
    /// - c1, c2, c3 = compatibility coefficients
    pub fn compatibility_distance(
        genome1: &Genome,
        genome2: &Genome,
        config: &NEATConfig,
    ) -> f64 {
        let alignment = Self::analyze_gene_alignment(genome1, genome2);
        
        let excess_genes = alignment.excess_connections.len() as f64;
        let disjoint_genes = alignment.disjoint_connections.len() as f64;
        
        // Calculate average weight difference for matching genes
        let mut weight_diff_sum = 0.0;
        let mut weight_comparisons = 0;
        
        for matching in &alignment.matching_connections {
            weight_diff_sum += (matching.parent1_gene.weight - matching.parent2_gene.weight).abs();
            weight_comparisons += 1;
        }
        
        let avg_weight_diff = if weight_comparisons > 0 {
            weight_diff_sum / weight_comparisons as f64
        } else {
            0.0
        };
        
        // N = number of genes in larger genome (minimum 1)
        let n = (genome1.connections.len().max(genome2.connections.len()) as f64).max(1.0);
        
        // Compatibility distance formula
        let compatibility_distance = 
            (config.speciation.excess_coefficient * excess_genes / n) +
            (config.speciation.disjoint_coefficient * disjoint_genes / n) +
            (config.speciation.weight_difference_coefficient * avg_weight_diff);
        
        compatibility_distance
    }
}

/// Gene alignment analysis between two parents
#[derive(Debug, Clone)]
struct GeneAlignment {
    /// Connections that exist in both parents (same innovation ID)
    matching_connections: Vec<MatchingConnection>,
    /// Connections that exist in one parent but not the other (within innovation range)
    disjoint_connections: Vec<DisjointConnection>,
    /// Connections that exist beyond the innovation range of the other parent
    excess_connections: Vec<ExcessConnection>,
}

/// A connection gene that exists in both parents
#[derive(Debug, Clone)]
struct MatchingConnection {
    innovation_id: usize,
    parent1_gene: ConnectionGene,
    parent2_gene: ConnectionGene,
}

/// A connection gene that exists in only one parent (disjoint)
#[derive(Debug, Clone)]
struct DisjointConnection {
    innovation_id: usize,
    gene: ConnectionGene,
    parent: usize, // 1 or 2
}

/// A connection gene that exists beyond the other parent's range (excess)
#[derive(Debug, Clone)]
struct ExcessConnection {
    innovation_id: usize,
    gene: ConnectionGene,
    parent: usize, // 1 or 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neat::innovation::InnovationTracker;
    use rand::{SeedableRng, rngs::SmallRng};

    fn create_test_context() -> (NEATConfig, InnovationTracker, SmallRng) {
        let config = NEATConfig::default();
        let tracker = InnovationTracker::with_starting_id(1000);
        let rng = SmallRng::seed_from_u64(42);
        (config, tracker, rng)
    }

    #[test]
    fn test_simple_crossover() -> Result<()> {
        let (config, tracker, mut rng) = create_test_context();
        
        // Create two simple parent genomes
        let mut parent1 = Genome::new(1, 2, 1);
        let mut parent2 = Genome::new(2, 2, 1);
        
        // Give parent1 higher fitness
        parent1.fitness = 0.8;
        parent2.fitness = 0.6;
        
        // Add some connections to parent1
        let conn1 = ConnectionGene::new(1000, 0, 3, 0.5);
        let conn2 = ConnectionGene::new(1001, 1, 3, -0.3);
        parent1.add_connection(conn1)?;
        parent1.add_connection(conn2)?;
        
        // Add overlapping and different connections to parent2
        let conn3 = ConnectionGene::new(1000, 0, 3, 0.7); // Same innovation ID as parent1's first connection
        let conn4 = ConnectionGene::new(1002, 2, 3, 0.2); // Different innovation ID (bias->output)
        parent2.add_connection(conn3)?;
        parent2.add_connection(conn4)?;
        
        let mut context = CrossoverContext {
            config: &config,
            innovation_tracker: &tracker,
            rng: &mut rng,
        };
        
        let result = NEATCrossover::crossover(&parent1, &parent2, &mut context)?;
        
        // Verify offspring is valid
        result.offspring.validate()?;
        
        // Should have input and output nodes
        assert_eq!(result.offspring.get_input_count(), 2);
        assert_eq!(result.offspring.get_output_count(), 1);
        
        // Stats should reflect the gene analysis
        assert_eq!(result.stats.matching_genes, 1); // Innovation ID 1000
        assert!(result.stats.disjoint_genes >= 1); // Innovation IDs 1001 and 1002
        assert_eq!(result.stats.fitter_parent, 0); // Parent1 has higher fitness
        
        println!("Crossover stats: {:?}", result.stats);
        println!("Offspring connections: {}", result.offspring.connections.len());
        
        Ok(())
    }

    #[test]
    fn test_equal_fitness_crossover() -> Result<()> {
        let (config, tracker, mut rng) = create_test_context();
        
        // Create two genomes with equal fitness
        let mut parent1 = Genome::new(1, 2, 1);
        let mut parent2 = Genome::new(2, 2, 1);
        
        parent1.fitness = 0.7;
        parent2.fitness = 0.7; // Equal fitness
        
        // Add different connections to each parent
        let conn1 = ConnectionGene::new(1000, 0, 3, 0.5);
        let conn2 = ConnectionGene::new(1001, 1, 3, -0.3);
        parent1.add_connection(conn1)?;
        parent1.add_connection(conn2)?;
        
        let conn3 = ConnectionGene::new(1002, 2, 3, 0.2);
        parent2.add_connection(conn3)?;
        
        let mut context = CrossoverContext {
            config: &config,
            innovation_tracker: &tracker,
            rng: &mut rng,
        };
        
        let result = NEATCrossover::crossover(&parent1, &parent2, &mut context)?;
        
        // Verify offspring is valid
        result.offspring.validate()?;
        
        // With equal fitness, offspring might inherit genes from both parents
        assert!(result.offspring.connections.len() > 0);
        
        Ok(())
    }

    #[test]
    fn test_compatibility_distance() {
        let (config, _tracker, _rng) = create_test_context();
        
        // Create two identical genomes
        let genome1 = Genome::new(1, 2, 1);
        let genome2 = Genome::new(2, 2, 1);
        
        let distance = NEATCrossover::compatibility_distance(&genome1, &genome2, &config);
        assert_eq!(distance, 0.0); // Identical genomes should have zero distance
        
        // Create genomes with different connections
        let mut genome3 = Genome::new(3, 2, 1);
        let mut genome4 = Genome::new(4, 2, 1);
        
        let conn1 = ConnectionGene::new(1000, 0, 3, 0.5);
        let conn2 = ConnectionGene::new(1001, 1, 3, -0.3);
        genome3.add_connection(conn1).unwrap();
        genome3.add_connection(conn2).unwrap();
        
        let conn3 = ConnectionGene::new(1002, 2, 3, 0.7);
        genome4.add_connection(conn3).unwrap();
        
        let distance2 = NEATCrossover::compatibility_distance(&genome3, &genome4, &config);
        assert!(distance2 > 0.0); // Different genomes should have positive distance
        
        println!("Compatibility distance: {:.4}", distance2);
    }

    #[test]
    fn test_crossover_with_hidden_nodes() -> Result<()> {
        let (config, tracker, mut rng) = create_test_context();
        
        // Create parent genomes with hidden nodes
        let mut parent1 = Genome::new(1, 2, 1);
        let mut parent2 = Genome::new(2, 2, 1);
        
        parent1.fitness = 0.9;
        parent2.fitness = 0.7;
        
        // Add hidden node to parent1
        let hidden1 = NodeGene::new(10, NodeType::Hidden, ActivationType::Sigmoid);
        parent1.add_node(hidden1)?;
        
        // Add connections involving hidden node
        let conn1 = ConnectionGene::new(1000, 0, 10, 0.5); // input -> hidden
        let conn2 = ConnectionGene::new(1001, 10, 3, 0.8); // hidden -> output
        parent1.add_connection(conn1)?;
        parent1.add_connection(conn2)?;
        
        // Parent2 has direct connection
        let conn3 = ConnectionGene::new(1002, 1, 3, 0.3); // input -> output
        parent2.add_connection(conn3)?;
        
        let mut context = CrossoverContext {
            config: &config,
            innovation_tracker: &tracker,
            rng: &mut rng,
        };
        
        let result = NEATCrossover::crossover(&parent1, &parent2, &mut context)?;
        
        // Verify offspring is valid and can include hidden nodes
        result.offspring.validate()?;
        
        // Should inherit structure from fitter parent (parent1)
        let has_hidden = result.offspring.nodes.iter().any(|n| n.node_type == NodeType::Hidden);
        println!("Offspring has hidden node: {}", has_hidden);
        println!("Offspring nodes: {}", result.offspring.nodes.len());
        println!("Offspring connections: {}", result.offspring.connections.len());
        
        Ok(())
    }

    #[test]
    fn test_gene_alignment_analysis() {
        // Create test genomes with specific innovation patterns
        let mut genome1 = Genome::new(1, 2, 1);
        let mut genome2 = Genome::new(2, 2, 1);
        
        // Genome1: innovations 1000, 1001, 1003
        let conn1 = ConnectionGene::new(1000, 0, 3, 0.5);
        let conn2 = ConnectionGene::new(1001, 1, 3, 0.3);
        let conn3 = ConnectionGene::new(1003, 2, 3, 0.1);
        genome1.add_connection(conn1).unwrap();
        genome1.add_connection(conn2).unwrap();
        genome1.add_connection(conn3).unwrap();
        
        // Genome2: innovations 1000, 1002, 1004
        let conn4 = ConnectionGene::new(1000, 0, 3, 0.7); // Matching
        let conn5 = ConnectionGene::new(1002, 1, 3, 0.4); // Disjoint
        let conn6 = ConnectionGene::new(1004, 2, 3, 0.2); // Excess
        genome2.add_connection(conn4).unwrap();
        genome2.add_connection(conn5).unwrap();
        genome2.add_connection(conn6).unwrap();
        
        let alignment = NEATCrossover::analyze_gene_alignment(&genome1, &genome2);
        
        assert_eq!(alignment.matching_connections.len(), 1); // Innovation 1000
        assert_eq!(alignment.disjoint_connections.len(), 3); // Innovations 1001, 1002, 1003
        assert_eq!(alignment.excess_connections.len(), 1); // Innovation 1004
        
        println!("Gene alignment: matching={}, disjoint={}, excess={}", 
                alignment.matching_connections.len(),
                alignment.disjoint_connections.len(), 
                alignment.excess_connections.len());
    }

    #[test]
    fn test_crossover_preserves_basic_structure() -> Result<()> {
        let (config, tracker, mut rng) = create_test_context();
        
        // Test with minimal genomes
        let parent1 = Genome::new(1, 3, 2);
        let parent2 = Genome::new(2, 3, 2);
        
        let mut context = CrossoverContext {
            config: &config,
            innovation_tracker: &tracker,
            rng: &mut rng,
        };
        
        let result = NEATCrossover::crossover(&parent1, &parent2, &mut context)?;
        
        // Offspring should preserve basic structure
        assert_eq!(result.offspring.get_input_count(), 3);
        assert_eq!(result.offspring.get_output_count(), 2);
        assert!(result.offspring.get_bias_node_id().is_some());
        
        result.offspring.validate()?;
        
        Ok(())
    }
}