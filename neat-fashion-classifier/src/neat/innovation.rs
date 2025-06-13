//! Innovation tracking for NEAT algorithm
//!
//! This module implements the innovation tracking system that enables meaningful
//! crossover between genomes with different topologies by maintaining historical
//! records of structural innovations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Type of innovation that occurred
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InnovationType {
    /// New connection was added between existing nodes
    NewConnection,
    /// New node was added by splitting an existing connection
    NewNode {
        /// Innovation ID of the connection that was split
        split_connection: usize,
    },
}

/// Record of a structural innovation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Innovation {
    /// Unique innovation ID
    pub innovation_id: usize,
    /// Input node ID
    pub input_node: usize,
    /// Output node ID
    pub output_node: usize,
    /// Type of innovation
    pub innovation_type: InnovationType,
    /// Generation when this innovation first occurred
    pub generation: usize,
}

impl Innovation {
    /// Create a new connection innovation
    pub fn new_connection(
        innovation_id: usize,
        input_node: usize,
        output_node: usize,
        generation: usize,
    ) -> Self {
        Self {
            innovation_id,
            input_node,
            output_node,
            innovation_type: InnovationType::NewConnection,
            generation,
        }
    }
    
    /// Create a new node innovation
    pub fn new_node(
        innovation_id: usize,
        input_node: usize,
        output_node: usize,
        split_connection: usize,
        generation: usize,
    ) -> Self {
        Self {
            innovation_id,
            input_node,
            output_node,
            innovation_type: InnovationType::NewNode { split_connection },
            generation,
        }
    }
}

/// Innovation tracker that maintains historical records of all innovations
#[derive(Debug, Default)]
pub struct InnovationTracker {
    /// Map from (input_node, output_node) to innovation ID
    innovations: HashMap<(usize, usize), usize>,
    /// Next available innovation ID
    next_innovation_id: usize,
    /// Complete history of all innovations
    innovation_history: Vec<Innovation>,
    /// Current generation number
    current_generation: usize,
}

impl InnovationTracker {
    /// Create a new innovation tracker
    pub fn new() -> Self {
        Self {
            innovations: HashMap::new(),
            next_innovation_id: 0,
            innovation_history: Vec::new(),
            current_generation: 0,
        }
    }
    
    /// Create a new innovation tracker with a specific starting ID
    pub fn with_starting_id(starting_id: usize) -> Self {
        Self {
            innovations: HashMap::new(),
            next_innovation_id: starting_id,
            innovation_history: Vec::new(),
            current_generation: 0,
        }
    }
    
    /// Advance to the next generation
    pub fn next_generation(&mut self) {
        self.current_generation += 1;
    }
    
    /// Get the current generation number
    pub fn get_generation(&self) -> usize {
        self.current_generation
    }
    
    /// Get innovation ID for a connection, creating new one if necessary
    pub fn get_innovation_id(&mut self, input_node: usize, output_node: usize) -> usize {
        let key = (input_node, output_node);
        
        if let Some(&existing_id) = self.innovations.get(&key) {
            existing_id
        } else {
            let innovation_id = self.next_innovation_id;
            self.innovations.insert(key, innovation_id);
            
            let innovation = Innovation::new_connection(
                innovation_id,
                input_node,
                output_node,
                self.current_generation,
            );
            self.innovation_history.push(innovation);
            
            self.next_innovation_id += 1;
            innovation_id
        }
    }
    
    /// Get innovation IDs for adding a node (splits existing connection)
    /// 
    /// Returns (in_to_new_innovation_id, new_to_out_innovation_id, new_node_id)
    pub fn get_node_innovation_ids(
        &mut self,
        split_connection_id: usize,
        input_node: usize,
        output_node: usize,
    ) -> (usize, usize, usize) {
        // The new node gets an ID based on the current innovation counter
        let new_node_id = self.next_innovation_id;
        
        // Record the node innovation first (doesn't increment next_innovation_id)
        let node_innovation = Innovation::new_node(
            new_node_id,
            input_node,
            output_node,
            split_connection_id,
            self.current_generation,
        );
        self.innovation_history.push(node_innovation);
        self.next_innovation_id += 1;
        
        // Get innovation IDs for the two new connections
        let in_to_new_id = self.get_innovation_id(input_node, new_node_id);
        let new_to_out_id = self.get_innovation_id(new_node_id, output_node);
        
        (in_to_new_id, new_to_out_id, new_node_id)
    }
    
    /// Check if an innovation already exists
    pub fn has_innovation(&self, input_node: usize, output_node: usize) -> bool {
        self.innovations.contains_key(&(input_node, output_node))
    }
    
    /// Get innovation by ID
    pub fn get_innovation(&self, innovation_id: usize) -> Option<&Innovation> {
        self.innovation_history
            .iter()
            .find(|i| i.innovation_id == innovation_id)
    }
    
    /// Get all innovations from a specific generation
    pub fn get_innovations_from_generation(&self, generation: usize) -> Vec<&Innovation> {
        self.innovation_history
            .iter()
            .filter(|i| i.generation == generation)
            .collect()
    }
    
    /// Get the next available innovation ID without incrementing
    pub fn peek_next_innovation_id(&self) -> usize {
        self.next_innovation_id
    }
    
    /// Get total number of innovations
    pub fn innovation_count(&self) -> usize {
        self.innovation_history.len()
    }
    
    /// Get innovation statistics
    pub fn get_statistics(&self) -> InnovationStatistics {
        let mut connection_count = 0;
        let mut node_count = 0;
        
        for innovation in &self.innovation_history {
            match innovation.innovation_type {
                InnovationType::NewConnection => connection_count += 1,
                InnovationType::NewNode { .. } => node_count += 1,
            }
        }
        
        InnovationStatistics {
            total_innovations: self.innovation_history.len(),
            connection_innovations: connection_count,
            node_innovations: node_count,
            current_generation: self.current_generation,
            next_innovation_id: self.next_innovation_id,
        }
    }
    
    /// Reset the tracker (for testing or restarting evolution)
    pub fn reset(&mut self) {
        self.innovations.clear();
        self.innovation_history.clear();
        self.next_innovation_id = 0;
        self.current_generation = 0;
    }
    
    /// Create a thread-safe version of the innovation tracker
    pub fn into_shared(self) -> SharedInnovationTracker {
        SharedInnovationTracker {
            inner: Arc::new(Mutex::new(self)),
        }
    }
}

/// Statistics about innovations
#[derive(Debug, Clone, PartialEq)]
pub struct InnovationStatistics {
    /// Total number of innovations
    pub total_innovations: usize,
    /// Number of connection innovations
    pub connection_innovations: usize,
    /// Number of node innovations
    pub node_innovations: usize,
    /// Current generation
    pub current_generation: usize,
    /// Next available innovation ID
    pub next_innovation_id: usize,
}

/// Thread-safe wrapper around InnovationTracker
#[derive(Debug, Clone)]
pub struct SharedInnovationTracker {
    inner: Arc<Mutex<InnovationTracker>>,
}

impl SharedInnovationTracker {
    /// Create a new shared innovation tracker
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(InnovationTracker::new())),
        }
    }
    
    /// Execute a function with exclusive access to the tracker
    pub fn with_tracker<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut InnovationTracker) -> R,
    {
        let mut tracker = self.inner.lock().unwrap();
        f(&mut tracker)
    }
    
    /// Get innovation ID for a connection
    pub fn get_innovation_id(&self, input_node: usize, output_node: usize) -> usize {
        self.with_tracker(|tracker| tracker.get_innovation_id(input_node, output_node))
    }
    
    /// Get innovation IDs for adding a node
    pub fn get_node_innovation_ids(
        &self,
        split_connection_id: usize,
        input_node: usize,
        output_node: usize,
    ) -> (usize, usize, usize) {
        self.with_tracker(|tracker| {
            tracker.get_node_innovation_ids(split_connection_id, input_node, output_node)
        })
    }
    
    /// Advance to next generation
    pub fn next_generation(&self) {
        self.with_tracker(|tracker| tracker.next_generation());
    }
    
    /// Get statistics
    pub fn get_statistics(&self) -> InnovationStatistics {
        self.with_tracker(|tracker| tracker.get_statistics())
    }
}

impl Default for SharedInnovationTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_innovation_creation() {
        let innovation = Innovation::new_connection(0, 1, 2, 5);
        assert_eq!(innovation.innovation_id, 0);
        assert_eq!(innovation.input_node, 1);
        assert_eq!(innovation.output_node, 2);
        assert_eq!(innovation.generation, 5);
        assert_eq!(innovation.innovation_type, InnovationType::NewConnection);
        
        let node_innovation = Innovation::new_node(1, 3, 4, 0, 6);
        assert_eq!(node_innovation.innovation_id, 1);
        assert_eq!(node_innovation.input_node, 3);
        assert_eq!(node_innovation.output_node, 4);
        assert_eq!(node_innovation.generation, 6);
        assert_eq!(
            node_innovation.innovation_type,
            InnovationType::NewNode { split_connection: 0 }
        );
    }

    #[test]
    fn test_innovation_tracker_basic() {
        let mut tracker = InnovationTracker::new();
        assert_eq!(tracker.get_generation(), 0);
        assert_eq!(tracker.innovation_count(), 0);
        
        // First connection innovation
        let id1 = tracker.get_innovation_id(0, 1);
        assert_eq!(id1, 0);
        assert_eq!(tracker.innovation_count(), 1);
        
        // Same connection should return same ID
        let id2 = tracker.get_innovation_id(0, 1);
        assert_eq!(id1, id2);
        assert_eq!(tracker.innovation_count(), 1); // No new innovation created
        
        // Different connection should get new ID
        let id3 = tracker.get_innovation_id(1, 2);
        assert_eq!(id3, 1);
        assert_eq!(tracker.innovation_count(), 2);
    }

    #[test]
    fn test_innovation_tracker_consistency() {
        let mut tracker = InnovationTracker::new();
        
        // Create several innovations
        let id_0_1 = tracker.get_innovation_id(0, 1);
        let id_1_2 = tracker.get_innovation_id(1, 2);
        let id_0_2 = tracker.get_innovation_id(0, 2);
        
        // Verify consistency
        assert_eq!(tracker.get_innovation_id(0, 1), id_0_1);
        assert_eq!(tracker.get_innovation_id(1, 2), id_1_2);
        assert_eq!(tracker.get_innovation_id(0, 2), id_0_2);
        
        // Verify uniqueness
        assert_ne!(id_0_1, id_1_2);
        assert_ne!(id_1_2, id_0_2);
        assert_ne!(id_0_1, id_0_2);
    }

    #[test]
    fn test_node_innovation() {
        let mut tracker = InnovationTracker::new();
        
        // Create initial connection
        let connection_id = tracker.get_innovation_id(0, 1);
        assert_eq!(connection_id, 0);
        
        // Add node by splitting connection
        let (in_to_new, new_to_out, new_node_id) = 
            tracker.get_node_innovation_ids(connection_id, 0, 1);
        
        // Verify the node got a unique ID
        assert_eq!(new_node_id, 1); // Next available ID after connection
        
        // Verify the new connections got appropriate IDs
        assert_ne!(in_to_new, new_to_out);
        // Note: the connections might reuse the original connection_id since they're new innovations
        
        // Should be able to retrieve the same IDs
        assert_eq!(tracker.get_innovation_id(0, new_node_id), in_to_new);
        assert_eq!(tracker.get_innovation_id(new_node_id, 1), new_to_out);
    }

    #[test]
    fn test_generation_tracking() {
        let mut tracker = InnovationTracker::new();
        
        // Create innovation in generation 0
        let id1 = tracker.get_innovation_id(0, 1);
        let innovations_gen0 = tracker.get_innovations_from_generation(0);
        assert_eq!(innovations_gen0.len(), 1);
        assert_eq!(innovations_gen0[0].innovation_id, id1);
        
        // Move to next generation
        tracker.next_generation();
        assert_eq!(tracker.get_generation(), 1);
        
        // Create innovation in generation 1
        let id2 = tracker.get_innovation_id(1, 2);
        let innovations_gen1 = tracker.get_innovations_from_generation(1);
        assert_eq!(innovations_gen1.len(), 1);
        assert_eq!(innovations_gen1[0].innovation_id, id2);
        
        // Verify generation 0 still has its innovation
        let innovations_gen0 = tracker.get_innovations_from_generation(0);
        assert_eq!(innovations_gen0.len(), 1);
    }

    #[test]
    fn test_innovation_lookup() {
        let mut tracker = InnovationTracker::new();
        
        let id1 = tracker.get_innovation_id(0, 1);
        let id2 = tracker.get_innovation_id(1, 2);
        
        // Test has_innovation
        assert!(tracker.has_innovation(0, 1));
        assert!(tracker.has_innovation(1, 2));
        assert!(!tracker.has_innovation(2, 3));
        
        // Test get_innovation
        let innovation1 = tracker.get_innovation(id1).unwrap();
        assert_eq!(innovation1.input_node, 0);
        assert_eq!(innovation1.output_node, 1);
        
        let innovation2 = tracker.get_innovation(id2).unwrap();
        assert_eq!(innovation2.input_node, 1);
        assert_eq!(innovation2.output_node, 2);
        
        // Test non-existent innovation
        assert!(tracker.get_innovation(999).is_none());
    }

    #[test]
    fn test_statistics() {
        let mut tracker = InnovationTracker::new();
        
        let initial_stats = tracker.get_statistics();
        assert_eq!(initial_stats.total_innovations, 0);
        assert_eq!(initial_stats.connection_innovations, 0);
        assert_eq!(initial_stats.node_innovations, 0);
        assert_eq!(initial_stats.current_generation, 0);
        
        // Add some connections
        tracker.get_innovation_id(0, 1);
        tracker.get_innovation_id(1, 2);
        
        // Add a node
        tracker.get_node_innovation_ids(0, 0, 1);
        
        let stats = tracker.get_statistics();
        assert_eq!(stats.total_innovations, 5); // 4 connections + 1 node
        assert_eq!(stats.connection_innovations, 4); // Original 2 + 2 from node split
        assert_eq!(stats.node_innovations, 1);
    }

    #[test]
    fn test_tracker_reset() {
        let mut tracker = InnovationTracker::new();
        
        // Add some innovations
        tracker.get_innovation_id(0, 1);
        tracker.get_innovation_id(1, 2);
        tracker.next_generation();
        
        assert_eq!(tracker.innovation_count(), 2);
        assert_eq!(tracker.get_generation(), 1);
        
        // Reset
        tracker.reset();
        
        assert_eq!(tracker.innovation_count(), 0);
        assert_eq!(tracker.get_generation(), 0);
        assert_eq!(tracker.peek_next_innovation_id(), 0);
    }

    #[test]
    fn test_shared_innovation_tracker() {
        let tracker = SharedInnovationTracker::new();
        
        // Test basic operations
        let id1 = tracker.get_innovation_id(0, 1);
        let id2 = tracker.get_innovation_id(0, 1); // Should be same
        assert_eq!(id1, id2);
        
        let id3 = tracker.get_innovation_id(1, 2); // Should be different
        assert_ne!(id1, id3);
        
        // Test node innovation
        let (_in_to_new, _new_to_out, new_node_id) = 
            tracker.get_node_innovation_ids(id1, 0, 1);
        assert_eq!(new_node_id, 2); // Should be next available
        
        // Test statistics
        let stats = tracker.get_statistics();
        assert_eq!(stats.total_innovations, 5); // 4 connections + 1 node
        
        // Test generation advancement
        tracker.next_generation();
        let stats = tracker.get_statistics();
        assert_eq!(stats.current_generation, 1);
    }

    #[test]
    fn test_starting_id() {
        let mut tracker = InnovationTracker::with_starting_id(100);
        
        let id1 = tracker.get_innovation_id(0, 1);
        assert_eq!(id1, 100);
        
        let id2 = tracker.get_innovation_id(1, 2);
        assert_eq!(id2, 101);
        
        let (_, _, node_id) = tracker.get_node_innovation_ids(id1, 0, 1);
        assert_eq!(node_id, 102);
    }

    #[test]
    fn test_complex_innovation_sequence() {
        let mut tracker = InnovationTracker::new();
        
        // Simulate a complex sequence of innovations
        let conn_0_1 = tracker.get_innovation_id(0, 1); // 0
        let conn_1_3 = tracker.get_innovation_id(1, 3); // 1
        
        // Split connection 0->1 with new node (node_new will be 2)
        let (conn_0_new, conn_new_1, node_new) = 
            tracker.get_node_innovation_ids(conn_0_1, 0, 1); // node: 2, conns: 3, 4
        
        // Add more connections with truly unique endpoints
        let conn_4_new = tracker.get_innovation_id(4, node_new); // 5 (4 -> 2)
        let conn_new_5 = tracker.get_innovation_id(node_new, 5); // 6 (2 -> 5)
        
        // Verify all IDs are unique
        let ids = vec![conn_0_1, conn_1_3, node_new, conn_0_new, conn_new_1, conn_4_new, conn_new_5];
        let mut sorted_ids = ids.clone();
        sorted_ids.sort_unstable();
        sorted_ids.dedup();
        assert_eq!(ids.len(), sorted_ids.len()); // All unique
        
        // Verify statistics
        let stats = tracker.get_statistics();
        assert_eq!(stats.total_innovations, 7); // 6 connections + 1 node  
        assert_eq!(stats.connection_innovations, 6);
        assert_eq!(stats.node_innovations, 1);
    }
}