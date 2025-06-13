//! Network topology analysis and utilities
//!
//! This module provides tools for analyzing the structure of NEAT networks,
//! including connectivity patterns, path analysis, and structural metrics.

use crate::neat::genome::{Genome, NodeType};
use crate::neat::network::Network;
use crate::error::Result;
use std::collections::{HashMap, HashSet, VecDeque};

/// Comprehensive topology analysis for a genome/network
#[derive(Debug, Clone, PartialEq)]
pub struct TopologyAnalysis {
    /// Basic network structure information
    pub structure: NetworkStructure,
    /// Connectivity analysis
    pub connectivity: ConnectivityAnalysis,
    /// Path analysis
    pub paths: PathAnalysis,
    /// Modularity and clustering information
    pub modularity: ModularityAnalysis,
}

/// Basic structure information
#[derive(Debug, Clone, PartialEq)]
pub struct NetworkStructure {
    /// Total number of nodes by type
    pub node_counts: HashMap<NodeType, usize>,
    /// Total number of connections
    pub connection_count: usize,
    /// Number of enabled connections
    pub enabled_connections: usize,
    /// Network depth (longest path from input to output)
    pub depth: usize,
    /// Average degree (connections per node)
    pub average_degree: f64,
    /// Is the network feedforward (no cycles)
    pub is_feedforward: bool,
}

/// Connectivity analysis
#[derive(Debug, Clone, PartialEq)]
pub struct ConnectivityAnalysis {
    /// Nodes with no incoming connections (besides inputs/bias)
    pub disconnected_inputs: Vec<usize>,
    /// Nodes with no outgoing connections (besides outputs)
    pub disconnected_outputs: Vec<usize>,
    /// Strongly connected components
    pub connected_components: Vec<Vec<usize>>,
    /// Input-output reachability matrix
    pub reachability: HashMap<usize, HashSet<usize>>,
}

/// Path analysis
#[derive(Debug, Clone, PartialEq)]
pub struct PathAnalysis {
    /// Shortest paths from each input to each output
    pub shortest_paths: HashMap<(usize, usize), Option<usize>>,
    /// Average path length from inputs to outputs
    pub average_path_length: f64,
    /// Number of disjoint paths from inputs to outputs
    pub disjoint_paths: usize,
    /// Critical nodes (nodes that are on all input-output paths)
    pub critical_nodes: HashSet<usize>,
}

/// Modularity and clustering analysis
#[derive(Debug, Clone, PartialEq)]
pub struct ModularityAnalysis {
    /// Clustering coefficient for each node
    pub clustering_coefficients: HashMap<usize, f64>,
    /// Average clustering coefficient
    pub average_clustering: f64,
    /// Potential modules (highly interconnected groups)
    pub modules: Vec<Vec<usize>>,
}

/// Analyzer for network topology
pub struct TopologyAnalyzer;

impl TopologyAnalyzer {
    /// Perform complete topology analysis on a genome
    pub fn analyze_genome(genome: &Genome) -> Result<TopologyAnalysis> {
        let network = Network::from_genome(genome)?;
        Self::analyze_network(genome, &network)
    }
    
    /// Perform complete topology analysis on a network with its genome
    pub fn analyze_network(genome: &Genome, _network: &Network) -> Result<TopologyAnalysis> {
        let structure = Self::analyze_structure(genome);
        let connectivity = Self::analyze_connectivity(genome);
        let paths = Self::analyze_paths(genome);
        let modularity = Self::analyze_modularity(genome);
        
        Ok(TopologyAnalysis {
            structure,
            connectivity,
            paths,
            modularity,
        })
    }
    
    /// Analyze basic network structure
    fn analyze_structure(genome: &Genome) -> NetworkStructure {
        let mut node_counts = HashMap::new();
        for node in &genome.nodes {
            *node_counts.entry(node.node_type).or_insert(0) += 1;
        }
        
        let connection_count = genome.connections.len();
        let enabled_connections = genome.connections.iter().filter(|c| c.enabled).count();
        
        // Calculate average degree
        let total_degree: usize = genome.connections
            .iter()
            .filter(|c| c.enabled)
            .map(|_| 2) // Each connection adds 1 to input degree and 1 to output degree
            .sum();
        let average_degree = if genome.nodes.is_empty() {
            0.0
        } else {
            total_degree as f64 / genome.nodes.len() as f64
        };
        
        // Check if feedforward (simplified check)
        let is_feedforward = Self::is_feedforward(genome);
        
        // Calculate depth
        let depth = Self::calculate_depth(genome);
        
        NetworkStructure {
            node_counts,
            connection_count,
            enabled_connections,
            depth,
            average_degree,
            is_feedforward,
        }
    }
    
    /// Analyze connectivity patterns
    fn analyze_connectivity(genome: &Genome) -> ConnectivityAnalysis {
        let adjacency = Self::build_adjacency_map(genome);
        
        // Find disconnected nodes
        let mut disconnected_inputs = Vec::new();
        let mut disconnected_outputs = Vec::new();
        
        for node in &genome.nodes {
            let has_inputs = adjacency.values().any(|connections| {
                connections.iter().any(|(target, _)| *target == node.id)
            });
            let has_outputs = adjacency.get(&node.id).map_or(false, |connections| !connections.is_empty());
            
            match node.node_type {
                NodeType::Input | NodeType::Bias => {
                    if !has_outputs {
                        disconnected_outputs.push(node.id);
                    }
                }
                NodeType::Output => {
                    if !has_inputs {
                        disconnected_inputs.push(node.id);
                    }
                }
                NodeType::Hidden => {
                    if !has_inputs {
                        disconnected_inputs.push(node.id);
                    }
                    if !has_outputs {
                        disconnected_outputs.push(node.id);
                    }
                }
            }
        }
        
        // Find connected components (simplified)
        let connected_components = Self::find_connected_components(genome);
        
        // Build reachability matrix
        let reachability = Self::build_reachability_matrix(genome);
        
        ConnectivityAnalysis {
            disconnected_inputs,
            disconnected_outputs,
            connected_components,
            reachability,
        }
    }
    
    /// Analyze paths in the network
    fn analyze_paths(genome: &Genome) -> PathAnalysis {
        let input_nodes: Vec<usize> = genome.nodes
            .iter()
            .filter(|n| n.node_type == NodeType::Input)
            .map(|n| n.id)
            .collect();
            
        let output_nodes: Vec<usize> = genome.nodes
            .iter()
            .filter(|n| n.node_type == NodeType::Output)
            .map(|n| n.id)
            .collect();
        
        let mut shortest_paths = HashMap::new();
        let mut total_paths = 0;
        let mut total_length = 0;
        
        // Calculate shortest paths between all input-output pairs
        for &input in &input_nodes {
            for &output in &output_nodes {
                let path_length = Self::shortest_path_length(genome, input, output);
                shortest_paths.insert((input, output), path_length);
                
                if let Some(length) = path_length {
                    total_paths += 1;
                    total_length += length;
                }
            }
        }
        
        let average_path_length = if total_paths > 0 {
            total_length as f64 / total_paths as f64
        } else {
            0.0
        };
        
        // Count disjoint paths (simplified)
        let disjoint_paths = total_paths; // TODO: Implement proper disjoint path counting
        
        // Find critical nodes (simplified)
        let critical_nodes = HashSet::new(); // TODO: Implement critical node detection
        
        PathAnalysis {
            shortest_paths,
            average_path_length,
            disjoint_paths,
            critical_nodes,
        }
    }
    
    /// Analyze modularity and clustering
    fn analyze_modularity(genome: &Genome) -> ModularityAnalysis {
        let mut clustering_coefficients = HashMap::new();
        
        // Calculate clustering coefficient for each node
        for node in &genome.nodes {
            let coefficient = Self::calculate_clustering_coefficient(genome, node.id);
            clustering_coefficients.insert(node.id, coefficient);
        }
        
        let average_clustering = if clustering_coefficients.is_empty() {
            0.0
        } else {
            clustering_coefficients.values().sum::<f64>() / clustering_coefficients.len() as f64
        };
        
        // Detect modules (simplified)
        let modules = Vec::new(); // TODO: Implement module detection
        
        ModularityAnalysis {
            clustering_coefficients,
            average_clustering,
            modules,
        }
    }
    
    /// Check if network is feedforward
    fn is_feedforward(genome: &Genome) -> bool {
        // Use topological sort to detect cycles
        let adjacency = Self::build_adjacency_map(genome);
        let mut in_degree: HashMap<usize, usize> = HashMap::new();
        
        // Initialize in-degrees
        for node in &genome.nodes {
            in_degree.insert(node.id, 0);
        }
        
        // Count incoming edges
        for connections in adjacency.values() {
            for &(target, _) in connections {
                *in_degree.entry(target).or_insert(0) += 1;
            }
        }
        
        // Topological sort
        let mut queue = VecDeque::new();
        let mut processed = 0;
        
        // Start with nodes that have no incoming edges
        for (&node_id, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(node_id);
            }
        }
        
        while let Some(node_id) = queue.pop_front() {
            processed += 1;
            
            if let Some(connections) = adjacency.get(&node_id) {
                for &(target, _) in connections {
                    let new_degree = in_degree[&target] - 1;
                    in_degree.insert(target, new_degree);
                    
                    if new_degree == 0 {
                        queue.push_back(target);
                    }
                }
            }
        }
        
        // If all nodes were processed, it's feedforward
        processed == genome.nodes.len()
    }
    
    /// Calculate network depth
    fn calculate_depth(genome: &Genome) -> usize {
        let input_nodes: Vec<usize> = genome.nodes
            .iter()
            .filter(|n| matches!(n.node_type, NodeType::Input | NodeType::Bias))
            .map(|n| n.id)
            .collect();
            
        let output_nodes: Vec<usize> = genome.nodes
            .iter()
            .filter(|n| n.node_type == NodeType::Output)
            .map(|n| n.id)
            .collect();
        
        let mut max_depth = 0;
        
        for &input in &input_nodes {
            for &output in &output_nodes {
                if let Some(depth) = Self::shortest_path_length(genome, input, output) {
                    max_depth = max_depth.max(depth);
                }
            }
        }
        
        max_depth
    }
    
    /// Build adjacency map from genome
    fn build_adjacency_map(genome: &Genome) -> HashMap<usize, Vec<(usize, f64)>> {
        let mut adjacency = HashMap::new();
        
        // Initialize empty adjacency lists
        for node in &genome.nodes {
            adjacency.insert(node.id, Vec::new());
        }
        
        // Add enabled connections
        for connection in &genome.connections {
            if connection.enabled {
                adjacency
                    .get_mut(&connection.input_node)
                    .unwrap()
                    .push((connection.output_node, connection.weight));
            }
        }
        
        adjacency
    }
    
    /// Find connected components
    fn find_connected_components(genome: &Genome) -> Vec<Vec<usize>> {
        let mut visited = HashSet::new();
        let mut components = Vec::new();
        
        for node in &genome.nodes {
            if !visited.contains(&node.id) {
                let component = Self::dfs_component(genome, node.id, &mut visited);
                components.push(component);
            }
        }
        
        components
    }
    
    /// DFS to find connected component
    fn dfs_component(genome: &Genome, start: usize, visited: &mut HashSet<usize>) -> Vec<usize> {
        let mut component = Vec::new();
        let mut stack = vec![start];
        
        while let Some(node_id) = stack.pop() {
            if !visited.contains(&node_id) {
                visited.insert(node_id);
                component.push(node_id);
                
                // Add neighbors (both directions for undirected connectivity)
                for connection in &genome.connections {
                    if connection.enabled {
                        if connection.input_node == node_id && !visited.contains(&connection.output_node) {
                            stack.push(connection.output_node);
                        }
                        if connection.output_node == node_id && !visited.contains(&connection.input_node) {
                            stack.push(connection.input_node);
                        }
                    }
                }
            }
        }
        
        component
    }
    
    /// Build reachability matrix using BFS
    fn build_reachability_matrix(genome: &Genome) -> HashMap<usize, HashSet<usize>> {
        let mut reachability = HashMap::new();
        
        for node in &genome.nodes {
            let reachable = Self::find_reachable_nodes(genome, node.id);
            reachability.insert(node.id, reachable);
        }
        
        reachability
    }
    
    /// Find all nodes reachable from a starting node
    fn find_reachable_nodes(genome: &Genome, start: usize) -> HashSet<usize> {
        let mut reachable = HashSet::new();
        let mut queue = VecDeque::new();
        
        queue.push_back(start);
        
        while let Some(node_id) = queue.pop_front() {
            if !reachable.contains(&node_id) {
                reachable.insert(node_id);
                
                // Add all nodes reachable via enabled connections
                for connection in &genome.connections {
                    if connection.enabled && connection.input_node == node_id {
                        queue.push_back(connection.output_node);
                    }
                }
            }
        }
        
        reachable
    }
    
    /// Calculate shortest path length between two nodes
    fn shortest_path_length(genome: &Genome, start: usize, end: usize) -> Option<usize> {
        if start == end {
            return Some(0);
        }
        
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        
        queue.push_back((start, 0));
        
        while let Some((node_id, distance)) = queue.pop_front() {
            if visited.contains(&node_id) {
                continue;
            }
            
            visited.insert(node_id);
            
            if node_id == end {
                return Some(distance);
            }
            
            // Add neighbors
            for connection in &genome.connections {
                if connection.enabled && connection.input_node == node_id {
                    if !visited.contains(&connection.output_node) {
                        queue.push_back((connection.output_node, distance + 1));
                    }
                }
            }
        }
        
        None // No path found
    }
    
    /// Calculate clustering coefficient for a node
    fn calculate_clustering_coefficient(genome: &Genome, node_id: usize) -> f64 {
        let neighbors = Self::get_neighbors(genome, node_id);
        
        if neighbors.len() < 2 {
            return 0.0; // Need at least 2 neighbors for clustering
        }
        
        let mut connections_between_neighbors = 0;
        
        for &neighbor1 in &neighbors {
            for &neighbor2 in &neighbors {
                if neighbor1 != neighbor2 {
                    if Self::has_connection(genome, neighbor1, neighbor2) {
                        connections_between_neighbors += 1;
                    }
                }
            }
        }
        
        let possible_connections = neighbors.len() * (neighbors.len() - 1);
        
        if possible_connections == 0 {
            0.0
        } else {
            connections_between_neighbors as f64 / possible_connections as f64
        }
    }
    
    /// Get all neighbors of a node (both incoming and outgoing)
    fn get_neighbors(genome: &Genome, node_id: usize) -> Vec<usize> {
        let mut neighbors = HashSet::new();
        
        for connection in &genome.connections {
            if connection.enabled {
                if connection.input_node == node_id {
                    neighbors.insert(connection.output_node);
                }
                if connection.output_node == node_id {
                    neighbors.insert(connection.input_node);
                }
            }
        }
        
        neighbors.into_iter().collect()
    }
    
    /// Check if there's a connection between two nodes
    fn has_connection(genome: &Genome, from: usize, to: usize) -> bool {
        genome.connections.iter().any(|c| {
            c.enabled && c.input_node == from && c.output_node == to
        })
    }
}

/// Utility functions for topology analysis
pub mod utils {
    use super::*;
    use crate::neat::genome::Genome;
    
    /// Compare topology complexity between two genomes
    pub fn compare_complexity(genome1: &Genome, genome2: &Genome) -> Result<ComplexityComparison> {
        let analysis1 = TopologyAnalyzer::analyze_genome(genome1)?;
        let analysis2 = TopologyAnalyzer::analyze_genome(genome2)?;
        
        Ok(ComplexityComparison {
            node_difference: analysis1.structure.node_counts.values().sum::<usize>() as i32
                - analysis2.structure.node_counts.values().sum::<usize>() as i32,
            connection_difference: analysis1.structure.enabled_connections as i32
                - analysis2.structure.enabled_connections as i32,
            depth_difference: analysis1.structure.depth as i32 - analysis2.structure.depth as i32,
            clustering_difference: analysis1.modularity.average_clustering - analysis2.modularity.average_clustering,
        })
    }
    
    /// Calculate topology diversity in a population
    pub fn calculate_diversity(genomes: &[Genome]) -> Result<f64> {
        if genomes.len() < 2 {
            return Ok(0.0);
        }
        
        let mut total_distance = 0.0;
        let mut comparisons = 0;
        
        for i in 0..genomes.len() {
            for j in (i + 1)..genomes.len() {
                let comparison = compare_complexity(&genomes[i], &genomes[j])?;
                let distance = comparison.total_distance();
                total_distance += distance;
                comparisons += 1;
            }
        }
        
        Ok(total_distance / comparisons as f64)
    }
}

/// Comparison of complexity between two genomes
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexityComparison {
    /// Difference in number of nodes (positive means first genome has more)
    pub node_difference: i32,
    /// Difference in number of connections
    pub connection_difference: i32,
    /// Difference in network depth
    pub depth_difference: i32,
    /// Difference in clustering coefficient
    pub clustering_difference: f64,
}

impl ComplexityComparison {
    /// Calculate total distance metric
    pub fn total_distance(&self) -> f64 {
        (self.node_difference.abs() as f64 * 1.0)
            + (self.connection_difference.abs() as f64 * 0.5)
            + (self.depth_difference.abs() as f64 * 2.0)
            + (self.clustering_difference.abs() * 3.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neat::genome::{NodeGene, ConnectionGene, ActivationType};

    #[test]
    fn test_basic_topology_analysis() {
        let genome = Genome::new(0, 2, 1);
        let analysis = TopologyAnalyzer::analyze_genome(&genome).unwrap();
        
        assert_eq!(analysis.structure.node_counts[&NodeType::Input], 2);
        assert_eq!(analysis.structure.node_counts[&NodeType::Output], 1);
        assert_eq!(analysis.structure.node_counts[&NodeType::Bias], 1);
        assert_eq!(analysis.structure.connection_count, 0);
        assert_eq!(analysis.structure.enabled_connections, 0);
        assert!(analysis.structure.is_feedforward);
    }
    
    #[test]
    fn test_topology_with_connections() {
        let mut genome = Genome::new(0, 2, 1);
        
        // Add some connections
        let conn1 = ConnectionGene::new(0, 0, 3, 1.0); // input -> output
        let conn2 = ConnectionGene::new(1, 1, 3, 0.5); // input -> output
        
        genome.add_connection(conn1).unwrap();
        genome.add_connection(conn2).unwrap();
        
        let analysis = TopologyAnalyzer::analyze_genome(&genome).unwrap();
        
        assert_eq!(analysis.structure.connection_count, 2);
        assert_eq!(analysis.structure.enabled_connections, 2);
        assert!(analysis.structure.average_degree > 0.0);
        assert!(analysis.structure.is_feedforward);
    }
    
    #[test]
    fn test_connectivity_analysis() {
        let mut genome = Genome::new(0, 2, 1);
        
        // Add a hidden node
        let hidden_node = NodeGene::new(10, NodeType::Hidden, ActivationType::Sigmoid);
        genome.add_node(hidden_node).unwrap();
        
        // Add connections
        let conn1 = ConnectionGene::new(0, 0, 10, 1.0); // input -> hidden
        let conn2 = ConnectionGene::new(1, 10, 3, 1.0); // hidden -> output
        
        genome.add_connection(conn1).unwrap();
        genome.add_connection(conn2).unwrap();
        
        let analysis = TopologyAnalyzer::analyze_genome(&genome).unwrap();
        
        // Should have path from input to output through hidden node
        assert!(analysis.paths.shortest_paths.get(&(0, 3)).unwrap().is_some());
        assert_eq!(analysis.structure.depth, 2);
    }
    
    #[test]
    fn test_disconnected_nodes() {
        let mut genome = Genome::new(0, 2, 1);
        
        // Add a disconnected hidden node
        let hidden_node = NodeGene::new(10, NodeType::Hidden, ActivationType::Sigmoid);
        genome.add_node(hidden_node).unwrap();
        
        let analysis = TopologyAnalyzer::analyze_genome(&genome).unwrap();
        
        // Hidden node should be disconnected
        assert!(analysis.connectivity.disconnected_inputs.contains(&10));
        assert!(analysis.connectivity.disconnected_outputs.contains(&10));
    }
    
    #[test]
    fn test_complexity_comparison() {
        let genome1 = Genome::new(0, 2, 1);
        
        let mut genome2 = Genome::new(1, 2, 1);
        let hidden_node = NodeGene::new(10, NodeType::Hidden, ActivationType::Sigmoid);
        genome2.add_node(hidden_node).unwrap();
        
        let comparison = utils::compare_complexity(&genome1, &genome2).unwrap();
        
        assert_eq!(comparison.node_difference, -1); // genome2 has one more node
        assert!(comparison.total_distance() > 0.0);
    }
    
    #[test]
    fn test_is_feedforward() {
        let mut genome = Genome::new(0, 2, 1);
        
        // Add hidden nodes
        let hidden1 = NodeGene::new(10, NodeType::Hidden, ActivationType::Sigmoid);
        let hidden2 = NodeGene::new(11, NodeType::Hidden, ActivationType::Sigmoid);
        genome.add_node(hidden1).unwrap();
        genome.add_node(hidden2).unwrap();
        
        // Add feedforward connections
        let conn1 = ConnectionGene::new(0, 0, 10, 1.0); // input -> hidden1
        let conn2 = ConnectionGene::new(1, 10, 11, 1.0); // hidden1 -> hidden2
        let conn3 = ConnectionGene::new(2, 11, 3, 1.0); // hidden2 -> output
        
        genome.add_connection(conn1).unwrap();
        genome.add_connection(conn2).unwrap();
        genome.add_connection(conn3).unwrap();
        
        assert!(TopologyAnalyzer::is_feedforward(&genome));
        
        // Add a cycle
        let cycle_conn = ConnectionGene::new(3, 11, 10, 1.0); // hidden2 -> hidden1 (cycle)
        genome.add_connection(cycle_conn).unwrap();
        
        assert!(!TopologyAnalyzer::is_feedforward(&genome));
    }
}