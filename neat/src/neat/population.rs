//! Population management for NEAT
//!
//! This module implements population-level operations including selection,
//! reproduction, and evolutionary dynamics for the NEAT algorithm.

use crate::neat::genome::Genome;
use crate::neat::speciation::{SpeciesManager, Species};
use crate::neat::crossover::{NEATCrossover, CrossoverContext};
use crate::neat::mutation::{MutationPipeline, MutationContext};
use crate::neat::fitness::FitnessEvaluator;
use crate::neat::innovation::InnovationTracker;
use crate::config::NEATConfig;
use crate::error::Result;
use rand::prelude::*;

/// Population manager for NEAT evolution
pub struct PopulationManager {
    /// Current population of genomes
    pub population: Vec<Genome>,
    /// Species manager for population clustering
    pub species_manager: SpeciesManager,
    /// Innovation tracker for historical markings
    pub innovation_tracker: InnovationTracker,
    /// Current generation number
    pub generation: usize,
    /// Population size to maintain
    pub population_size: usize,
    /// Best genome ever found
    pub champion: Option<Genome>,
    /// Random number generator
    rng: SmallRng,
}

impl PopulationManager {
    /// Create a new population manager
    pub fn new(population_size: usize, input_count: usize, output_count: usize) -> Self {
        let rng = SmallRng::from_entropy();
        
        // Create initial population
        let mut population = Vec::with_capacity(population_size);
        for i in 0..population_size {
            population.push(Genome::new(i, input_count, output_count));
        }
        
        Self {
            population,
            species_manager: SpeciesManager::new(),
            innovation_tracker: InnovationTracker::with_starting_id(1000),
            generation: 0,
            population_size,
            champion: None,
            rng,
        }
    }
    
    /// Create population manager with specific seed for reproducibility
    pub fn with_seed(
        population_size: usize, 
        input_count: usize, 
        output_count: usize, 
        seed: u64
    ) -> Self {
        let rng = SmallRng::seed_from_u64(seed);
        
        // Create initial population
        let mut population = Vec::with_capacity(population_size);
        for i in 0..population_size {
            population.push(Genome::new(i, input_count, output_count));
        }
        
        Self {
            population,
            species_manager: SpeciesManager::new(),
            innovation_tracker: InnovationTracker::with_starting_id(1000),
            generation: 0,
            population_size,
            champion: None,
            rng,
        }
    }
    
    /// Evolve the population for one generation
    pub fn evolve_generation<E: FitnessEvaluator>(
        &mut self,
        evaluator: &E,
        config: &NEATConfig,
    ) -> Result<EvolutionStats> {
        // Evaluate fitness for all genomes
        self.evaluate_fitness(evaluator)?;
        
        // Classify population into species
        self.species_manager.classify_population(&mut self.population, config)?;
        
        // Calculate offspring allocation for each species
        self.species_manager.calculate_offspring_allocation(self.population_size, config);
        
        // Remove stagnant species
        self.species_manager.remove_stagnant_species(config);
        
        // Record statistics before reproduction
        let stats = self.calculate_generation_stats();
        
        // Create next generation
        self.reproduce_population(config)?;
        
        // Advance to next generation
        self.generation += 1;
        self.species_manager.next_generation();
        self.innovation_tracker.next_generation();
        
        Ok(stats)
    }
    
    /// Evaluate fitness for all genomes in the population
    fn evaluate_fitness<E: FitnessEvaluator>(&mut self, evaluator: &E) -> Result<()> {
        for genome in &mut self.population {
            let fitness = evaluator.evaluate(genome)?;
            genome.fitness = fitness;
        }
        
        // Update champion
        if let Some(best_genome) = self.population.iter().max_by(|a, b| 
            a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal)
        ) {
            if self.champion.as_ref().map_or(true, |champ| best_genome.fitness > champ.fitness) {
                self.champion = Some(best_genome.clone());
            }
        }
        
        Ok(())
    }
    
    /// Reproduce population to create next generation
    fn reproduce_population(&mut self, config: &NEATConfig) -> Result<()> {
        let mut new_population = Vec::with_capacity(self.population_size);
        let mut next_genome_id = self.population.len();
        
        // Elite preservation - keep best genome
        let elitism_count = (self.population_size as f64 * config.selection.elitism_rate).ceil() as usize;
        if elitism_count > 0 {
            if let Some(champion) = &self.champion {
                let mut elite = champion.clone();
                elite.id = next_genome_id;
                new_population.push(elite);
                next_genome_id += 1;
            }
        }
        
        // Collect species reproduction info to avoid borrowing issues
        let mut species_info = Vec::new();
        for species in &self.species_manager.species {
            if species.offspring_count == 0 {
                continue;
            }
            
            // Skip elites already added
            let offspring_needed = if new_population.len() < elitism_count {
                species.offspring_count.saturating_sub(1)
            } else {
                species.offspring_count
            };
            
            species_info.push((species.clone(), offspring_needed));
        }
        
        // Reproduce each species
        for (mut species, offspring_needed) in species_info {
            let offspring = self.reproduce_species(&mut species, offspring_needed, &mut next_genome_id, config)?;
            new_population.extend(offspring);
        }
        
        // Fill remaining slots if needed
        while new_population.len() < self.population_size {
            // Create random new genome
            let genome = Genome::new(next_genome_id, 
                self.population[0].get_input_count(), 
                self.population[0].get_output_count());
            new_population.push(genome);
            next_genome_id += 1;
        }
        
        // Truncate if over capacity
        new_population.truncate(self.population_size);
        
        self.population = new_population;
        Ok(())
    }
    
    /// Reproduce offspring for a single species
    fn reproduce_species(
        &mut self,
        species: &mut Species,
        offspring_count: usize,
        next_genome_id: &mut usize,
        config: &NEATConfig,
    ) -> Result<Vec<Genome>> {
        let mut offspring = Vec::with_capacity(offspring_count);
        
        if species.members.is_empty() {
            return Ok(offspring);
        }
        
        // Sort members by fitness (best first)
        species.members.sort_by(|a, b| 
            b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal)
        );
        
        // Calculate survival threshold
        let survival_count = (species.members.len() as f64 * config.selection.survival_threshold).ceil() as usize;
        let survivors = &species.members[..survival_count.min(species.members.len())];
        
        if survivors.is_empty() {
            return Ok(offspring);
        }
        
        for _ in 0..offspring_count {
            let child = if survivors.len() == 1 || self.rng.gen::<f64>() < (1.0 - config.selection.crossover_probability) {
                // Asexual reproduction (mutation only)
                self.create_mutated_offspring(&survivors[0], *next_genome_id, config)?
            } else {
                // Sexual reproduction (crossover + mutation)
                let parent1 = self.select_parent(survivors);
                let parent2 = self.select_parent(survivors);
                self.create_crossover_offspring(parent1, parent2, *next_genome_id, config)?
            };
            
            offspring.push(child);
            *next_genome_id += 1;
        }
        
        Ok(offspring)
    }
    
    /// Create offspring through mutation only
    fn create_mutated_offspring(
        &mut self,
        parent: &Genome,
        child_id: usize,
        config: &NEATConfig,
    ) -> Result<Genome> {
        let mut child = parent.clone();
        child.id = child_id;
        child.fitness = 0.0;
        
        // Apply mutations
        let pipeline = MutationPipeline::default();
        let mut context = MutationContext {
            config,
            innovation_tracker: &mut self.innovation_tracker,
            rng: &mut self.rng,
        };
        
        let _mutations_applied = pipeline.mutate_genome(&mut child, &mut context)?;
        
        Ok(child)
    }
    
    /// Create offspring through crossover and mutation
    fn create_crossover_offspring(
        &mut self,
        parent1: &Genome,
        parent2: &Genome,
        child_id: usize,
        config: &NEATConfig,
    ) -> Result<Genome> {
        // Perform crossover
        let mut context = CrossoverContext {
            config,
            innovation_tracker: &self.innovation_tracker,
            rng: &mut self.rng,
        };
        
        let crossover_result = NEATCrossover::crossover(parent1, parent2, &mut context)?;
        let mut child = crossover_result.offspring;
        child.id = child_id;
        child.fitness = 0.0;
        
        // Apply mutations to offspring
        let pipeline = MutationPipeline::default();
        let mut mutation_context = MutationContext {
            config,
            innovation_tracker: &mut self.innovation_tracker,
            rng: &mut self.rng,
        };
        
        let _mutations_applied = pipeline.mutate_genome(&mut child, &mut mutation_context)?;
        
        Ok(child)
    }
    
    /// Select a parent using fitness-proportionate selection
    fn select_parent<'a>(&mut self, candidates: &'a [Genome]) -> &'a Genome {
        if candidates.len() == 1 {
            return &candidates[0];
        }
        
        // Tournament selection
        let tournament_size = 3.min(candidates.len());
        let mut best = &candidates[0];
        
        for _ in 0..tournament_size {
            let candidate = &candidates[self.rng.gen_range(0..candidates.len())];
            if candidate.fitness > best.fitness {
                best = candidate;
            }
        }
        
        best
    }
    
    /// Calculate statistics for the current generation
    fn calculate_generation_stats(&self) -> EvolutionStats {
        let fitness_values: Vec<f64> = self.population.iter().map(|g| g.fitness).collect();
        
        let (max_fitness, min_fitness, avg_fitness) = if fitness_values.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            // Filter out NaN and infinite values
            let valid_fitness: Vec<f64> = fitness_values.iter()
                .copied()
                .filter(|f| f.is_finite())
                .collect();
            
            if valid_fitness.is_empty() {
                (0.0, 0.0, 0.0)
            } else {
                let max = valid_fitness.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let min = valid_fitness.iter().copied().fold(f64::INFINITY, f64::min);
                let avg = valid_fitness.iter().sum::<f64>() / valid_fitness.len() as f64;
                (max, min, avg)
            }
        };
        
        let total_nodes: usize = self.population.iter().map(|g| g.nodes.len()).sum();
        let total_connections: usize = self.population.iter().map(|g| g.connections.len()).sum();
        
        let avg_nodes = total_nodes as f64 / self.population.len() as f64;
        let avg_connections = total_connections as f64 / self.population.len() as f64;
        
        EvolutionStats {
            generation: self.generation,
            population_size: self.population.len(),
            species_count: self.species_manager.species_count(),
            max_fitness,
            min_fitness,
            avg_fitness,
            avg_nodes,
            avg_connections,
            champion_fitness: self.champion.as_ref().map(|c| c.fitness).unwrap_or(0.0),
        }
    }
    
    /// Get the best genome in the current population
    pub fn get_best_genome(&self) -> Option<&Genome> {
        self.population.iter().max_by(|a, b| 
            a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal)
        )
    }
    
    /// Get the champion genome (best ever found)
    pub fn get_champion(&self) -> Option<&Genome> {
        self.champion.as_ref()
    }
    
    /// Get current generation number
    pub fn get_generation(&self) -> usize {
        self.generation
    }
    
    /// Get current population size
    pub fn get_population_size(&self) -> usize {
        self.population.len()
    }
    
    /// Get reference to the population
    pub fn get_population(&self) -> &[Genome] {
        &self.population
    }
    
    /// Get mutable reference to the population
    pub fn get_population_mut(&mut self) -> &mut [Genome] {
        &mut self.population
    }
    
    /// Get statistics about current speciation
    pub fn get_speciation_stats(&self) -> crate::neat::speciation::SpeciationStatistics {
        self.species_manager.get_statistics()
    }
    
    /// Set population fitness values (for external evaluation)
    pub fn set_population_fitness(&mut self, fitness_values: Vec<f64>) -> Result<()> {
        if fitness_values.len() != self.population.len() {
            return Err(crate::error::NEATError::InvalidGenome {
                message: format!("Fitness values length {} doesn't match population size {}", 
                       fitness_values.len(), self.population.len())
            }.into());
        }
        
        for (genome, fitness) in self.population.iter_mut().zip(fitness_values.iter()) {
            genome.fitness = *fitness;
        }
        
        // Update champion
        if let Some(best_genome) = self.population.iter().max_by(|a, b| 
            a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal)
        ) {
            if self.champion.as_ref().map_or(true, |champ| best_genome.fitness > champ.fitness) {
                self.champion = Some(best_genome.clone());
            }
        }
        
        Ok(())
    }
    
    /// Reset population to initial state
    pub fn reset(&mut self, input_count: usize, output_count: usize) {
        self.population.clear();
        for i in 0..self.population_size {
            self.population.push(Genome::new(i, input_count, output_count));
        }
        
        self.species_manager = SpeciesManager::new();
        self.innovation_tracker = InnovationTracker::with_starting_id(1000);
        self.generation = 0;
        self.champion = None;
    }
}

/// Statistics for a single generation
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EvolutionStats {
    /// Generation number
    pub generation: usize,
    /// Population size
    pub population_size: usize,
    /// Number of species
    pub species_count: usize,
    /// Maximum fitness in population
    pub max_fitness: f64,
    /// Minimum fitness in population
    pub min_fitness: f64,
    /// Average fitness in population
    pub avg_fitness: f64,
    /// Average number of nodes per genome
    pub avg_nodes: f64,
    /// Average number of connections per genome
    pub avg_connections: f64,
    /// Fitness of champion genome
    pub champion_fitness: f64,
}

impl EvolutionStats {
    /// Calculate fitness diversity (standard deviation)
    pub fn fitness_diversity(&self) -> f64 {
        // Simplified - would need population data for accurate calculation
        self.max_fitness - self.min_fitness
    }
}

/// Selection methods for parent selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SelectionMethod {
    /// Tournament selection with given tournament size
    Tournament(usize),
    /// Fitness proportionate selection (roulette wheel)
    FitnessProportionate,
    /// Rank-based selection
    Rank,
}

impl Default for SelectionMethod {
    fn default() -> Self {
        SelectionMethod::Tournament(3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neat::fitness::XORFitnessEvaluator;
    use crate::config::NEATConfig;

    #[test]
    fn test_population_creation() {
        let pop_manager = PopulationManager::with_seed(50, 2, 1, 42);
        
        assert_eq!(pop_manager.population.len(), 50);
        assert_eq!(pop_manager.generation, 0);
        assert_eq!(pop_manager.population_size, 50);
        assert!(pop_manager.champion.is_none());
        
        // All genomes should have correct structure
        for genome in &pop_manager.population {
            assert_eq!(genome.get_input_count(), 2);
            assert_eq!(genome.get_output_count(), 1);
        }
    }
    
    #[test]
    fn test_fitness_evaluation() -> Result<()> {
        let mut pop_manager = PopulationManager::with_seed(10, 2, 1, 42);
        let evaluator = XORFitnessEvaluator::default();
        
        pop_manager.evaluate_fitness(&evaluator)?;
        
        // All genomes should have been evaluated
        for genome in &pop_manager.population {
            assert!(genome.fitness >= 0.0);
            assert!(genome.fitness <= 1.0);
        }
        
        // Champion should be set
        assert!(pop_manager.champion.is_some());
        
        Ok(())
    }
    
    #[test]
    fn test_single_generation_evolution() -> Result<()> {
        let mut pop_manager = PopulationManager::with_seed(20, 2, 1, 42);
        let evaluator = XORFitnessEvaluator::default();
        let config = NEATConfig::default();
        
        let stats = pop_manager.evolve_generation(&evaluator, &config)?;
        
        assert_eq!(stats.generation, 0);
        assert_eq!(stats.population_size, 20);
        assert!(stats.max_fitness >= stats.avg_fitness || (stats.max_fitness - stats.avg_fitness).abs() < 1e-10);
        assert!(stats.avg_fitness >= stats.min_fitness || (stats.avg_fitness - stats.min_fitness).abs() < 1e-10);
        assert!(stats.champion_fitness >= 0.0);
        
        // Generation should have advanced
        assert_eq!(pop_manager.generation, 1);
        
        Ok(())
    }
    
    #[test]
    fn test_multiple_generations() -> Result<()> {
        let mut pop_manager = PopulationManager::with_seed(30, 2, 1, 42);
        let evaluator = XORFitnessEvaluator::default();
        let config = NEATConfig::default();
        
        let mut prev_champion_fitness = 0.0;
        
        for gen in 0..5 {
            let stats = pop_manager.evolve_generation(&evaluator, &config)?;
            
            assert_eq!(stats.generation, gen);
            assert_eq!(pop_manager.generation, gen + 1);
            
            // Champion fitness should not decrease
            assert!(stats.champion_fitness >= prev_champion_fitness);
            prev_champion_fitness = stats.champion_fitness;
        }
        
        Ok(())
    }
    
    #[test]
    fn test_population_size_maintained() -> Result<()> {
        let mut pop_manager = PopulationManager::with_seed(25, 2, 1, 42);
        let evaluator = XORFitnessEvaluator::default();
        let config = NEATConfig::default();
        
        for _ in 0..3 {
            pop_manager.evolve_generation(&evaluator, &config)?;
            assert_eq!(pop_manager.population.len(), 25);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_external_fitness_setting() -> Result<()> {
        let mut pop_manager = PopulationManager::with_seed(10, 2, 1, 42);
        
        let fitness_values = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        pop_manager.set_population_fitness(fitness_values.clone())?;
        
        for (genome, expected_fitness) in pop_manager.population.iter().zip(fitness_values.iter()) {
            assert_eq!(genome.fitness, *expected_fitness);
        }
        
        // Champion should be the one with highest fitness
        assert_eq!(pop_manager.champion.as_ref().unwrap().fitness, 1.0);
        
        Ok(())
    }
    
    #[test]
    fn test_reset_population() {
        let mut pop_manager = PopulationManager::with_seed(15, 2, 1, 42);
        
        // Evolve a few generations
        let evaluator = XORFitnessEvaluator::default();
        let config = NEATConfig::default();
        
        for _ in 0..3 {
            pop_manager.evolve_generation(&evaluator, &config).unwrap();
        }
        
        assert!(pop_manager.generation > 0);
        assert!(pop_manager.champion.is_some());
        
        // Reset
        pop_manager.reset(3, 2);
        
        assert_eq!(pop_manager.generation, 0);
        assert!(pop_manager.champion.is_none());
        assert_eq!(pop_manager.population.len(), 15);
        
        // All genomes should have new structure
        for genome in &pop_manager.population {
            assert_eq!(genome.get_input_count(), 3);
            assert_eq!(genome.get_output_count(), 2);
            assert_eq!(genome.fitness, 0.0);
        }
    }
    
    #[test]
    fn test_species_integration() -> Result<()> {
        let mut pop_manager = PopulationManager::with_seed(20, 2, 1, 42);
        let evaluator = XORFitnessEvaluator::default();
        let config = NEATConfig::default();
        
        // Evolve for several generations to allow speciation
        for _ in 0..3 {
            let stats = pop_manager.evolve_generation(&evaluator, &config)?;
            println!("Generation {}: {} species, max fitness: {:.4}", 
                    stats.generation, stats.species_count, stats.max_fitness);
        }
        
        let speciation_stats = pop_manager.get_speciation_stats();
        assert!(speciation_stats.species_count > 0);
        assert_eq!(speciation_stats.total_genomes, 20);
        
        Ok(())
    }
    
    #[test]
    fn test_evolution_statistics() -> Result<()> {
        let mut pop_manager = PopulationManager::with_seed(15, 2, 1, 42);
        let evaluator = XORFitnessEvaluator::default();
        let config = NEATConfig::default();
        
        let stats = pop_manager.evolve_generation(&evaluator, &config)?;
        
        // Basic statistics validation
        assert_eq!(stats.population_size, 15);
        assert!(stats.max_fitness >= stats.min_fitness || (stats.max_fitness - stats.min_fitness).abs() < 1e-10);
        assert!(stats.avg_fitness >= stats.min_fitness || (stats.avg_fitness - stats.min_fitness).abs() < 1e-10);
        assert!(stats.avg_fitness <= stats.max_fitness || (stats.avg_fitness - stats.max_fitness).abs() < 1e-10);
        assert!(stats.avg_nodes >= 4.0); // At least input + output + bias
        assert!(stats.avg_connections >= 0.0);
        assert!(stats.fitness_diversity() >= 0.0);
        
        Ok(())
    }
}