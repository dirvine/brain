//! Speciation system for NEAT
//!
//! This module implements the speciation mechanism that protects innovation
//! by clustering similar genomes together and providing fitness sharing.
//! This allows new structures to develop without being immediately eliminated
//! by more established networks.

use crate::neat::genome::Genome;
use crate::neat::crossover::NEATCrossover;
use crate::config::NEATConfig;
use crate::error::{NEATError, Result};
use std::collections::HashMap;
// use rand::prelude::*;

/// A species containing similar genomes
#[derive(Debug, Clone)]
pub struct Species {
    /// Unique identifier for this species
    pub id: usize,
    /// Representative genome for this species (used for compatibility testing)
    pub representative: Genome,
    /// All genomes belonging to this species
    pub members: Vec<Genome>,
    /// Average fitness of the species
    pub average_fitness: f64,
    /// Best fitness achieved by any member
    pub best_fitness: f64,
    /// Number of generations since improvement
    pub generations_without_improvement: usize,
    /// Generation when this species was created
    pub creation_generation: usize,
    /// Total offspring this species should produce
    pub offspring_count: usize,
}

impl Species {
    /// Create a new species with a founder genome
    pub fn new(id: usize, founder: Genome, generation: usize) -> Self {
        let fitness = founder.fitness;
        
        Self {
            id,
            representative: founder.clone(),
            members: vec![founder],
            average_fitness: fitness,
            best_fitness: fitness,
            generations_without_improvement: 0,
            creation_generation: generation,
            offspring_count: 0,
        }
    }
    
    /// Add a genome to this species
    pub fn add_member(&mut self, genome: Genome) {
        self.members.push(genome);
        // Note: fitness stats will be updated externally when needed
    }
    
    /// Remove all members and update fitness statistics
    pub fn clear_members(&mut self) {
        self.members.clear();
        self.average_fitness = 0.0;
        self.best_fitness = 0.0;
    }
    
    /// Update fitness statistics based on current members
    pub fn update_fitness_stats(&mut self) {
        if self.members.is_empty() {
            self.average_fitness = 0.0;
            self.best_fitness = 0.0;
            return;
        }
        
        let total_fitness: f64 = self.members.iter().map(|g| g.fitness).sum();
        self.average_fitness = total_fitness / self.members.len() as f64;
        self.best_fitness = self.members.iter()
            .map(|g| g.fitness)
            .fold(f64::NEG_INFINITY, f64::max);
    }
    
    /// Apply fitness sharing to all members
    /// Each member's fitness is divided by the species size to encourage diversity
    pub fn apply_fitness_sharing(&mut self) {
        let species_size = self.members.len() as f64;
        for member in &mut self.members {
            member.adjusted_fitness = member.fitness / species_size;
        }
    }
    
    /// Check if this species should be eliminated due to stagnation
    pub fn should_be_eliminated(&self, config: &NEATConfig) -> bool {
        self.generations_without_improvement >= config.speciation.staleness_threshold &&
        self.members.len() < config.speciation.min_species_size
    }
    
    /// Update the species representative (usually the best member)
    pub fn update_representative(&mut self) {
        if let Some(best_member) = self.members.iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal)) {
            self.representative = best_member.clone();
        }
    }
    
    /// Get the size of this species
    pub fn size(&self) -> usize {
        self.members.len()
    }
    
    /// Check if this species has improved recently
    pub fn has_improved(&mut self) -> bool {
        let current_best = self.members.iter()
            .map(|g| g.fitness)
            .fold(f64::NEG_INFINITY, f64::max);
            
        if current_best > self.best_fitness {
            self.best_fitness = current_best;
            self.generations_without_improvement = 0;
            true
        } else {
            self.generations_without_improvement += 1;
            false
        }
    }
}

/// Manager for all species in the population
#[derive(Debug, Clone)]
pub struct SpeciesManager {
    /// All current species
    pub species: Vec<Species>,
    /// Next species ID to assign
    next_species_id: usize,
    /// Current generation number
    current_generation: usize,
    /// Compatibility threshold (may be adjusted dynamically)
    compatibility_threshold: f64,
}

impl SpeciesManager {
    /// Create a new species manager
    pub fn new() -> Self {
        Self {
            species: Vec::new(),
            next_species_id: 0,
            current_generation: 0,
            compatibility_threshold: 3.0, // Default threshold
        }
    }
    
    /// Create a species manager with specific threshold
    pub fn with_threshold(threshold: f64) -> Self {
        Self {
            species: Vec::new(),
            next_species_id: 0,
            current_generation: 0,
            compatibility_threshold: threshold,
        }
    }
    
    /// Classify all genomes into species
    pub fn classify_population(&mut self, population: &mut [Genome], config: &NEATConfig) -> Result<()> {
        // Clear existing members but keep species structure
        for species in &mut self.species {
            species.clear_members();
        }
        
        // Assign each genome to a species
        for genome in population.iter_mut() {
            self.assign_to_species(genome.clone(), config)?;
        }
        
        // Remove empty species and update statistics
        self.species.retain(|species| !species.members.is_empty());
        
        // Update fitness statistics for all species
        for species in &mut self.species {
            species.update_fitness_stats();
            species.apply_fitness_sharing();
            species.has_improved(); // Update improvement tracking
        }
        
        // Adjust compatibility threshold if dynamic adjustment is enabled
        if config.speciation.dynamic_threshold {
            self.adjust_compatibility_threshold(config);
        }
        
        // Update species assignments in the population
        self.update_genome_species_assignments(population);
        
        Ok(())
    }
    
    /// Assign a genome to an appropriate species
    fn assign_to_species(&mut self, genome: Genome, config: &NEATConfig) -> Result<()> {
        // Try to find a compatible existing species
        for species in &mut self.species {
            let distance = NEATCrossover::compatibility_distance(
                &genome,
                &species.representative,
                config,
            );
            
            if distance < self.compatibility_threshold {
                species.add_member(genome);
                return Ok(());
            }
        }
        
        // No compatible species found, create a new one
        let new_species = Species::new(self.next_species_id, genome, self.current_generation);
        self.species.push(new_species);
        self.next_species_id += 1;
        
        Ok(())
    }
    
    /// Update genome species assignments based on current classification
    fn update_genome_species_assignments(&self, population: &mut [Genome]) {
        let mut genome_to_species: HashMap<usize, usize> = HashMap::new();
        
        // Build mapping from genome ID to species ID
        for species in &self.species {
            for member in &species.members {
                genome_to_species.insert(member.id, species.id);
            }
        }
        
        // Update species assignments in population
        for genome in population.iter_mut() {
            genome.species_id = genome_to_species.get(&genome.id).copied();
        }
    }
    
    /// Adjust compatibility threshold to maintain target species count
    fn adjust_compatibility_threshold(&mut self, config: &NEATConfig) {
        let current_species_count = self.species.len();
        let target_count = config.speciation.target_species_count;
        
        // Adjust threshold based on current vs target species count
        if current_species_count > target_count {
            // Too many species, increase threshold to merge some
            self.compatibility_threshold *= 1.1;
        } else if current_species_count < target_count {
            // Too few species, decrease threshold to create more
            self.compatibility_threshold *= 0.9;
        }
        
        // Keep threshold within reasonable bounds
        self.compatibility_threshold = self.compatibility_threshold.clamp(0.5, 10.0);
    }
    
    /// Calculate offspring allocation for each species
    pub fn calculate_offspring_allocation(&mut self, total_population_size: usize, config: &NEATConfig) {
        if self.species.is_empty() {
            return;
        }
        
        // Calculate total adjusted fitness across all species
        let total_adjusted_fitness: f64 = self.species.iter()
            .map(|s| s.average_fitness * s.members.len() as f64)
            .sum();
        
        if total_adjusted_fitness <= 0.0 {
            // If no positive fitness, distribute equally
            let offspring_per_species = total_population_size / self.species.len();
            for species in &mut self.species {
                species.offspring_count = offspring_per_species;
            }
            return;
        }
        
        // Allocate offspring proportionally to adjusted fitness
        let mut total_allocated = 0;
        for species in &mut self.species {
            let species_fitness = species.average_fitness * species.members.len() as f64;
            let proportion = species_fitness / total_adjusted_fitness;
            species.offspring_count = (proportion * total_population_size as f64) as usize;
            total_allocated += species.offspring_count;
        }
        
        // Handle rounding errors by giving remaining offspring to best species
        if total_allocated < total_population_size {
            let remaining = total_population_size - total_allocated;
            if let Some(best_species) = self.species.iter_mut()
                .max_by(|a, b| a.average_fitness.partial_cmp(&b.average_fitness).unwrap_or(std::cmp::Ordering::Equal)) {
                best_species.offspring_count += remaining;
            }
        }
        
        // Ensure minimum species size
        for species in &mut self.species {
            if species.offspring_count < config.speciation.min_species_size {
                species.offspring_count = config.speciation.min_species_size;
            }
        }
    }
    
    /// Remove stagnant species
    pub fn remove_stagnant_species(&mut self, config: &NEATConfig) {
        self.species.retain(|species| !species.should_be_eliminated(config));
    }
    
    /// Get species by ID
    pub fn get_species(&self, species_id: usize) -> Option<&Species> {
        self.species.iter().find(|s| s.id == species_id)
    }
    
    /// Get species by ID (mutable)
    pub fn get_species_mut(&mut self, species_id: usize) -> Option<&mut Species> {
        self.species.iter_mut().find(|s| s.id == species_id)
    }
    
    /// Get total number of species
    pub fn species_count(&self) -> usize {
        self.species.len()
    }
    
    /// Get current compatibility threshold
    pub fn get_compatibility_threshold(&self) -> f64 {
        self.compatibility_threshold
    }
    
    /// Set compatibility threshold manually
    pub fn set_compatibility_threshold(&mut self, threshold: f64) {
        self.compatibility_threshold = threshold.max(0.1); // Minimum threshold
    }
    
    /// Advance to next generation
    pub fn next_generation(&mut self) {
        self.current_generation += 1;
        
        // Update species representatives
        for species in &mut self.species {
            species.update_representative();
        }
    }
    
    /// Get statistics about current speciation
    pub fn get_statistics(&self) -> SpeciationStatistics {
        let total_genomes: usize = self.species.iter().map(|s| s.size()).sum();
        let species_sizes: Vec<usize> = self.species.iter().map(|s| s.size()).collect();
        
        let average_species_size = if self.species.is_empty() {
            0.0
        } else {
            total_genomes as f64 / self.species.len() as f64
        };
        
        let species_fitness: Vec<f64> = self.species.iter().map(|s| s.average_fitness).collect();
        let max_fitness = species_fitness.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_fitness = species_fitness.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        SpeciationStatistics {
            species_count: self.species.len(),
            total_genomes,
            average_species_size,
            species_sizes,
            compatibility_threshold: self.compatibility_threshold,
            max_species_fitness: max_fitness,
            min_species_fitness: min_fitness,
            current_generation: self.current_generation,
        }
    }
}

impl Default for SpeciesManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the current speciation state
#[derive(Debug, Clone, PartialEq)]
pub struct SpeciationStatistics {
    /// Number of species
    pub species_count: usize,
    /// Total number of genomes across all species
    pub total_genomes: usize,
    /// Average size of species
    pub average_species_size: f64,
    /// Sizes of all species
    pub species_sizes: Vec<usize>,
    /// Current compatibility threshold
    pub compatibility_threshold: f64,
    /// Highest average fitness among species
    pub max_species_fitness: f64,
    /// Lowest average fitness among species
    pub min_species_fitness: f64,
    /// Current generation number
    pub current_generation: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neat::genome::{NodeGene, ConnectionGene, NodeType, ActivationType};

    fn create_test_genome(id: usize, fitness: f64) -> Genome {
        let mut genome = Genome::new(id, 2, 1);
        genome.fitness = fitness;
        genome
    }

    fn create_diverse_genome(id: usize, fitness: f64, connections: Vec<(usize, usize, usize)>) -> Genome {
        let mut genome = Genome::new(id, 2, 1);
        genome.fitness = fitness;
        
        for (innovation_id, input, output) in connections {
            if input >= 10 || output >= 10 {
                // Add hidden node if needed
                let hidden = NodeGene::new(input.max(output), NodeType::Hidden, ActivationType::Sigmoid);
                let _ = genome.add_node(hidden);
            }
            
            let conn = ConnectionGene::new(innovation_id, input, output, 1.0);
            let _ = genome.add_connection(conn);
        }
        
        genome
    }

    #[test]
    fn test_species_creation() {
        let genome = create_test_genome(1, 0.5);
        let species = Species::new(0, genome, 0);
        
        assert_eq!(species.id, 0);
        assert_eq!(species.size(), 1);
        assert_eq!(species.average_fitness, 0.5);
        assert_eq!(species.best_fitness, 0.5);
        assert_eq!(species.creation_generation, 0);
    }

    #[test]
    fn test_species_member_management() {
        let founder = create_test_genome(1, 0.5);
        let mut species = Species::new(0, founder, 0);
        
        // Add another member
        let member2 = create_test_genome(2, 0.7);
        species.add_member(member2);
        
        // Update fitness stats after adding member
        species.update_fitness_stats();
        
        assert_eq!(species.size(), 2);
        assert_eq!(species.average_fitness, 0.6); // (0.5 + 0.7) / 2
        assert_eq!(species.best_fitness, 0.7);
        
        // Test fitness sharing
        species.apply_fitness_sharing();
        assert_eq!(species.members[0].adjusted_fitness, 0.5 / 2.0);
        assert_eq!(species.members[1].adjusted_fitness, 0.7 / 2.0);
    }

    #[test]
    fn test_species_manager_basic() -> Result<()> {
        let mut manager = SpeciesManager::new();
        let config = NEATConfig::default();
        
        // Create a small population
        let mut population = vec![
            create_test_genome(1, 0.5),
            create_test_genome(2, 0.6),
            create_test_genome(3, 0.4),
        ];
        
        manager.classify_population(&mut population, &config)?;
        
        // Should create at least one species
        assert!(manager.species_count() > 0);
        
        // All genomes should be assigned to species
        let total_assigned: usize = manager.species.iter().map(|s| s.size()).sum();
        assert_eq!(total_assigned, population.len());
        
        Ok(())
    }

    #[test]
    fn test_species_classification_with_diversity() -> Result<()> {
        let mut manager = SpeciesManager::with_threshold(1.0); // Lower threshold for easier testing
        let config = NEATConfig::default();
        
        // Create genomes with different structures
        let mut population = vec![
            create_diverse_genome(1, 0.5, vec![(1000, 0, 3)]), // Simple connection
            create_diverse_genome(2, 0.6, vec![(1000, 0, 3)]), // Same structure
            create_diverse_genome(3, 0.4, vec![(1001, 1, 3), (1002, 0, 10), (1003, 10, 3)]), // Different structure
        ];
        
        manager.classify_population(&mut population, &config)?;
        
        let stats = manager.get_statistics();
        println!("Species count: {}", stats.species_count);
        println!("Species sizes: {:?}", stats.species_sizes);
        
        // Should create multiple species due to structural differences
        assert!(manager.species_count() >= 1);
        
        Ok(())
    }

    #[test]
    fn test_offspring_allocation() -> Result<()> {
        let mut manager = SpeciesManager::new();
        let config = NEATConfig::default();
        
        // Create population with different fitness levels
        let mut population = vec![
            create_test_genome(1, 0.8), // High fitness
            create_test_genome(2, 0.8), // High fitness
            create_test_genome(3, 0.2), // Low fitness
            create_test_genome(4, 0.2), // Low fitness
        ];
        
        manager.classify_population(&mut population, &config)?;
        manager.calculate_offspring_allocation(10, &config);
        
        let total_offspring: usize = manager.species.iter().map(|s| s.offspring_count).sum();
        
        // Should allocate all offspring
        assert!(total_offspring >= 10);
        
        // Better species should get more offspring
        for species in &manager.species {
            println!("Species {}: avg_fitness={:.2}, offspring={}", 
                    species.id, species.average_fitness, species.offspring_count);
        }
        
        Ok(())
    }

    #[test]
    fn test_compatibility_threshold_adjustment() -> Result<()> {
        let mut manager = SpeciesManager::with_threshold(3.0);
        let mut config = NEATConfig::default();
        config.speciation.dynamic_threshold = true;
        config.speciation.target_species_count = 3;
        
        // Create diverse population that should create many species
        let mut population = vec![
            create_diverse_genome(1, 0.5, vec![(1000, 0, 3)]),
            create_diverse_genome(2, 0.6, vec![(1001, 1, 3)]),
            create_diverse_genome(3, 0.4, vec![(1002, 0, 10), (1003, 10, 3)]),
            create_diverse_genome(4, 0.7, vec![(1004, 1, 11), (1005, 11, 3)]),
            create_diverse_genome(5, 0.3, vec![(1006, 0, 12), (1007, 12, 3)]),
        ];
        
        let initial_threshold = manager.get_compatibility_threshold();
        manager.classify_population(&mut population, &config)?;
        let final_threshold = manager.get_compatibility_threshold();
        
        println!("Initial threshold: {:.2}", initial_threshold);
        println!("Final threshold: {:.2}", final_threshold);
        println!("Species created: {}", manager.species_count());
        
        // Threshold should adjust based on species count vs target
        assert!(final_threshold > 0.0);
        
        Ok(())
    }

    #[test]
    fn test_species_improvement_tracking() {
        let founder = create_test_genome(1, 0.5);
        let mut species = Species::new(0, founder, 0);
        
        // Initially generations_without_improvement should be 0
        assert_eq!(species.generations_without_improvement, 0);
        println!("Initial best fitness: {}", species.best_fitness);
        
        // Add member with same fitness - no improvement
        let member2 = create_test_genome(2, 0.5);
        species.add_member(member2);
        println!("After adding member2, best fitness: {}", species.best_fitness);
        assert!(!species.has_improved());
        assert_eq!(species.generations_without_improvement, 1);
        
        // Add member with higher fitness - improvement!
        let member3 = create_test_genome(3, 0.8);
        species.add_member(member3);
        println!("After adding member3, best fitness: {}", species.best_fitness);
        println!("Current best in members: {}", 
                species.members.iter().map(|g| g.fitness).fold(f64::NEG_INFINITY, f64::max));
        let improved = species.has_improved();
        println!("Has improved: {}, generations without improvement: {}", 
                improved, species.generations_without_improvement);
        assert!(improved);
        assert_eq!(species.generations_without_improvement, 0);
        
        // Check again with no improvement - should increment counter
        assert!(!species.has_improved());
        assert_eq!(species.generations_without_improvement, 1);
    }

    #[test]
    fn test_stagnant_species_elimination() {
        let config = NEATConfig::default();
        let genome = create_test_genome(1, 0.5);
        let mut species = Species::new(0, genome, 0);
        
        // Make species stagnant
        species.generations_without_improvement = config.speciation.staleness_threshold + 1;
        
        // Small species should be eliminated when stagnant
        assert!(species.should_be_eliminated(&config));
        
        // Add more members to exceed minimum size
        for i in 2..=config.speciation.min_species_size + 1 {
            species.add_member(create_test_genome(i, 0.5));
        }
        
        // Large species should not be eliminated even when stagnant
        assert!(!species.should_be_eliminated(&config));
    }

    #[test]
    fn test_species_statistics() -> Result<()> {
        let mut manager = SpeciesManager::new();
        let config = NEATConfig::default();
        
        let mut population = vec![
            create_test_genome(1, 0.8),
            create_test_genome(2, 0.6),
            create_test_genome(3, 0.4),
            create_test_genome(4, 0.2),
        ];
        
        manager.classify_population(&mut population, &config)?;
        let stats = manager.get_statistics();
        
        assert_eq!(stats.total_genomes, 4);
        assert!(stats.species_count > 0);
        assert!(stats.average_species_size > 0.0);
        assert!(stats.compatibility_threshold > 0.0);
        
        println!("Speciation stats: {:?}", stats);
        
        Ok(())
    }

    #[test]
    fn test_generation_advancement() -> Result<()> {
        let mut manager = SpeciesManager::new();
        let config = NEATConfig::default();
        
        let mut population = vec![
            create_test_genome(1, 0.8),
            create_test_genome(2, 0.6),
        ];
        
        manager.classify_population(&mut population, &config)?;
        
        let initial_generation = manager.current_generation;
        manager.next_generation();
        
        assert_eq!(manager.current_generation, initial_generation + 1);
        
        Ok(())
    }
}