//! NEAT training coordinator
//!
//! This module provides the main training loop and experiment management
//! for the NEAT algorithm. It orchestrates population evolution, fitness
//! evaluation, and convergence detection.

use crate::neat::population::{PopulationManager, EvolutionStats};
use crate::neat::fitness::FitnessEvaluator;
use crate::neat::genome::Genome;
use crate::config::NEATConfig;
use crate::error::Result;
use std::time::{Duration, Instant};
use std::path::{Path, PathBuf};
use serde::{Serialize, Deserialize};

/// Main trainer for NEAT algorithm
pub struct NEATTrainer<E: FitnessEvaluator> {
    /// Population manager
    population_manager: PopulationManager,
    /// Fitness evaluator
    evaluator: E,
    /// Training configuration
    config: NEATConfig,
    /// Training state
    state: TrainingState,
    /// Training statistics
    stats: TrainingStatistics,
    /// Checkpoint directory
    checkpoint_dir: Option<PathBuf>,
}

/// Current state of training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState {
    /// Whether training is currently running
    pub is_training: bool,
    /// Current generation number
    pub generation: usize,
    /// Best fitness achieved so far
    pub best_fitness: f64,
    /// Generation when best fitness was achieved
    pub best_generation: usize,
    /// Generations without improvement
    pub stagnation_count: usize,
    /// Training start time
    #[serde(skip)]
    pub start_time: Option<Instant>,
    /// Total training duration
    pub training_duration: Duration,
    /// Whether target fitness has been reached
    pub target_reached: bool,
    /// Whether early stopping was triggered
    pub early_stopped: bool,
}

/// Comprehensive training statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStatistics {
    /// Evolution statistics per generation
    pub generation_stats: Vec<EvolutionStats>,
    /// Best genome per generation
    pub best_genomes: Vec<Genome>,
    /// Fitness history
    pub fitness_history: Vec<f64>,
    /// Population size history
    pub population_history: Vec<usize>,
    /// Species count history
    pub species_history: Vec<usize>,
    /// Average complexity history (nodes + connections)
    pub complexity_history: Vec<f64>,
    /// Training milestones
    pub milestones: Vec<TrainingMilestone>,
}

/// Important training milestones
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMilestone {
    /// Generation when milestone occurred
    pub generation: usize,
    /// Type of milestone
    pub milestone_type: MilestoneType,
    /// Fitness value at milestone
    pub fitness: f64,
    /// Additional data
    pub data: String,
}

/// Types of training milestones
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MilestoneType {
    /// New fitness record achieved
    FitnessRecord,
    /// Target fitness reached
    TargetReached,
    /// Convergence detected
    Convergence,
    /// Species milestone (new species record, etc.)
    Species,
    /// Complexity milestone
    Complexity,
    /// Time milestone
    Duration,
}

/// Training result summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    /// Final training state
    pub state: TrainingState,
    /// Training statistics
    pub stats: TrainingStatistics,
    /// Best genome found
    pub best_genome: Genome,
    /// Success status
    pub success: bool,
    /// Termination reason
    pub termination_reason: TerminationReason,
}

/// Reasons for training termination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TerminationReason {
    /// Target fitness achieved
    TargetFitnessReached,
    /// Maximum generations reached
    MaxGenerationsReached,
    /// Early stopping triggered
    EarlyStopping,
    /// Manual termination
    ManualStop,
    /// Error occurred
    Error(String),
}

impl<E: FitnessEvaluator> NEATTrainer<E> {
    /// Create a new NEAT trainer
    pub fn new(evaluator: E, config: NEATConfig) -> Self {
        let input_count = evaluator.input_size();
        let output_count = evaluator.output_size();
        
        let population_manager = if let Some(seed) = config.population.random_seed {
            PopulationManager::with_seed(config.population.size, input_count, output_count, seed)
        } else {
            PopulationManager::new(config.population.size, input_count, output_count)
        };
        
        Self {
            population_manager,
            evaluator,
            config,
            state: TrainingState::new(),
            stats: TrainingStatistics::new(),
            checkpoint_dir: None,
        }
    }
    
    /// Set checkpoint directory for saving training progress
    pub fn with_checkpoint_dir<P: AsRef<Path>>(mut self, dir: P) -> Result<Self> {
        let path = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&path)?;
        self.checkpoint_dir = Some(path);
        Ok(self)
    }
    
    /// Run complete training
    pub fn train(&mut self) -> Result<TrainingResult> {
        self.start_training()?;
        
        while self.should_continue_training() {
            self.train_generation()?;
            
            if self.state.generation % self.config.training.checkpoint_frequency == 0 {
                self.save_checkpoint()?;
            }
        }
        
        self.finish_training()
    }
    
    /// Start training session
    fn start_training(&mut self) -> Result<()> {
        self.state.is_training = true;
        self.state.start_time = Some(Instant::now());
        
        println!("Starting NEAT training...");
        println!("Population size: {}", self.config.population.size);
        println!("Target fitness: {}", self.config.population.target_fitness);
        println!("Max generations: {}", self.config.population.max_generations);
        
        // Initial evaluation
        self.evaluate_initial_population()?;
        
        Ok(())
    }
    
    /// Evaluate initial population
    fn evaluate_initial_population(&mut self) -> Result<()> {
        // Evaluate fitness for initial population
        for genome in self.population_manager.get_population_mut() {
            genome.fitness = self.evaluator.evaluate(genome)?;
        }
        
        // Update champion and record initial stats
        let best_genome = self.population_manager.get_best_genome()
            .ok_or_else(|| crate::error::NEATError::InvalidGenome {
                message: "No genomes in population".to_string()
            })?;
        
        self.state.best_fitness = best_genome.fitness;
        self.state.best_generation = 0;
        
        println!("Initial best fitness: {:.6}", self.state.best_fitness);
        
        Ok(())
    }
    
    /// Train for one generation
    fn train_generation(&mut self) -> Result<()> {
        let generation_start = Instant::now();
        
        // Evolve population for one generation
        let evolution_stats = self.population_manager.evolve_generation(&self.evaluator, &self.config)?;
        
        // Update training state
        self.state.generation = evolution_stats.generation + 1;
        
        // Check for fitness improvement
        if evolution_stats.champion_fitness > self.state.best_fitness {
            self.state.best_fitness = evolution_stats.champion_fitness;
            self.state.best_generation = self.state.generation;
            self.state.stagnation_count = 0;
            
            // Record fitness milestone
            self.record_milestone(MilestoneType::FitnessRecord, evolution_stats.champion_fitness, 
                format!("New best fitness: {:.6}", evolution_stats.champion_fitness));
        } else {
            self.state.stagnation_count += 1;
        }
        
        // Check if target fitness reached
        if evolution_stats.champion_fitness >= self.config.population.target_fitness {
            self.state.target_reached = true;
            self.record_milestone(MilestoneType::TargetReached, evolution_stats.champion_fitness,
                format!("Target fitness {:.6} reached", self.config.population.target_fitness));
        }
        
        // Record statistics
        self.record_generation_stats(evolution_stats, generation_start);
        
        // Print progress
        if self.state.generation % 10 == 0 || self.state.target_reached {
            self.print_progress();
        }
        
        Ok(())
    }
    
    /// Record statistics for current generation
    fn record_generation_stats(&mut self, evolution_stats: EvolutionStats, generation_start: Instant) {
        self.stats.generation_stats.push(evolution_stats.clone());
        self.stats.fitness_history.push(evolution_stats.champion_fitness);
        self.stats.population_history.push(evolution_stats.population_size);
        self.stats.species_history.push(evolution_stats.species_count);
        
        let complexity = evolution_stats.avg_nodes + evolution_stats.avg_connections;
        self.stats.complexity_history.push(complexity);
        
        // Record best genome
        if let Some(best_genome) = self.population_manager.get_champion() {
            self.stats.best_genomes.push(best_genome.clone());
        }
        
        // Check for milestones
        let generation_time = generation_start.elapsed();
        if generation_time > Duration::from_secs(60) {
            self.record_milestone(MilestoneType::Duration, evolution_stats.champion_fitness,
                format!("Long generation: {:.2}s", generation_time.as_secs_f64()));
        }
        
        if evolution_stats.species_count > 20 {
            self.record_milestone(MilestoneType::Species, evolution_stats.champion_fitness,
                format!("High species diversity: {}", evolution_stats.species_count));
        }
    }
    
    /// Record a training milestone
    fn record_milestone(&mut self, milestone_type: MilestoneType, fitness: f64, data: String) {
        let milestone = TrainingMilestone {
            generation: self.state.generation,
            milestone_type,
            fitness,
            data,
        };
        self.stats.milestones.push(milestone);
    }
    
    /// Check if training should continue
    fn should_continue_training(&mut self) -> bool {
        if !self.state.is_training {
            return false;
        }
        
        // Check target fitness
        if self.state.target_reached {
            return false;
        }
        
        // Check max generations
        if self.state.generation >= self.config.population.max_generations {
            return false;
        }
        
        // Check early stopping
        if self.state.stagnation_count >= self.config.training.early_stopping_patience {
            self.state.early_stopped = true;
            return false;
        }
        
        true
    }
    
    /// Finish training and return results
    fn finish_training(&mut self) -> Result<TrainingResult> {
        self.state.is_training = false;
        
        if let Some(start_time) = self.state.start_time {
            self.state.training_duration = start_time.elapsed();
        }
        
        let best_genome = self.population_manager.get_champion()
            .ok_or_else(|| crate::error::NEATError::InvalidGenome {
                message: "No champion genome found".to_string()
            })?
            .clone();
        
        let termination_reason = if self.state.target_reached {
            TerminationReason::TargetFitnessReached
        } else if self.state.generation >= self.config.population.max_generations {
            TerminationReason::MaxGenerationsReached
        } else if self.state.early_stopped {
            TerminationReason::EarlyStopping
        } else {
            TerminationReason::ManualStop
        };
        
        let success = self.state.target_reached || self.state.best_fitness > 0.8;
        
        self.print_final_summary();
        
        Ok(TrainingResult {
            state: self.state.clone(),
            stats: self.stats.clone(),
            best_genome,
            success,
            termination_reason,
        })
    }
    
    /// Print training progress
    fn print_progress(&self) {
        let duration = self.state.start_time
            .map(|start| start.elapsed())
            .unwrap_or_default();
        
        println!(
            "Gen {}: Best={:.6} Avg={:.6} Species={} Stagnation={} Time={:.1}s",
            self.state.generation,
            self.state.best_fitness,
            self.stats.generation_stats.last().map(|s| s.avg_fitness).unwrap_or(0.0),
            self.stats.generation_stats.last().map(|s| s.species_count).unwrap_or(0),
            self.state.stagnation_count,
            duration.as_secs_f64()
        );
    }
    
    /// Print final training summary
    fn print_final_summary(&self) {
        println!("\n=== Training Complete ===");
        println!("Generations: {}", self.state.generation);
        println!("Best fitness: {:.6} (Gen {})", self.state.best_fitness, self.state.best_generation);
        println!("Training time: {:.2}s", self.state.training_duration.as_secs_f64());
        println!("Target reached: {}", self.state.target_reached);
        println!("Early stopped: {}", self.state.early_stopped);
        
        if let Some(final_stats) = self.stats.generation_stats.last() {
            println!("Final population: {}", final_stats.population_size);
            println!("Final species: {}", final_stats.species_count);
            println!("Avg complexity: {:.1}", final_stats.avg_nodes + final_stats.avg_connections);
        }
        
        println!("Milestones recorded: {}", self.stats.milestones.len());
    }
    
    /// Save training checkpoint
    fn save_checkpoint(&self) -> Result<()> {
        if let Some(checkpoint_dir) = &self.checkpoint_dir {
            let checkpoint_path = checkpoint_dir.join(format!("checkpoint_gen_{}.json", self.state.generation));
            
            let checkpoint = TrainingCheckpoint {
                state: self.state.clone(),
                stats: self.stats.clone(),
                config: self.config.clone(),
                generation: self.state.generation,
            };
            
            let json = serde_json::to_string_pretty(&checkpoint)?;
            std::fs::write(checkpoint_path, json)?;
            
            // Also save best genome separately
            if let Some(best_genome) = self.population_manager.get_champion() {
                let genome_path = checkpoint_dir.join(format!("best_genome_gen_{}.json", self.state.generation));
                let genome_json = serde_json::to_string_pretty(best_genome)?;
                std::fs::write(genome_path, genome_json)?;
            }
        }
        
        Ok(())
    }
    
    /// Get current training state
    pub fn get_state(&self) -> &TrainingState {
        &self.state
    }
    
    /// Get training statistics
    pub fn get_stats(&self) -> &TrainingStatistics {
        &self.stats
    }
    
    /// Get current best genome
    pub fn get_best_genome(&self) -> Option<&Genome> {
        self.population_manager.get_champion()
    }
    
    /// Stop training manually
    pub fn stop(&mut self) {
        self.state.is_training = false;
    }
    
    /// Resume training from a checkpoint
    pub fn resume_from_checkpoint<P: AsRef<Path>>(checkpoint_path: P, evaluator: E) -> Result<Self> {
        let content = std::fs::read_to_string(checkpoint_path)?;
        let checkpoint: TrainingCheckpoint = serde_json::from_str(&content)?;
        
        let input_count = evaluator.input_size();
        let output_count = evaluator.output_size();
        
        let population_manager = PopulationManager::new(
            checkpoint.config.population.size,
            input_count,
            output_count
        );
        
        Ok(Self {
            population_manager,
            evaluator,
            config: checkpoint.config,
            state: checkpoint.state,
            stats: checkpoint.stats,
            checkpoint_dir: None,
        })
    }
}

/// Training checkpoint data
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrainingCheckpoint {
    /// Training state
    state: TrainingState,
    /// Training statistics
    stats: TrainingStatistics,
    /// Configuration
    config: NEATConfig,
    /// Generation number
    generation: usize,
}

impl TrainingState {
    /// Create new training state
    fn new() -> Self {
        Self {
            is_training: false,
            generation: 0,
            best_fitness: 0.0,
            best_generation: 0,
            stagnation_count: 0,
            start_time: None,
            training_duration: Duration::default(),
            target_reached: false,
            early_stopped: false,
        }
    }
}

impl TrainingStatistics {
    /// Create new training statistics
    fn new() -> Self {
        Self {
            generation_stats: Vec::new(),
            best_genomes: Vec::new(),
            fitness_history: Vec::new(),
            population_history: Vec::new(),
            species_history: Vec::new(),
            complexity_history: Vec::new(),
            milestones: Vec::new(),
        }
    }
    
    /// Get fitness improvement rate
    pub fn fitness_improvement_rate(&self) -> f64 {
        if self.fitness_history.len() < 2 {
            return 0.0;
        }
        
        let first = self.fitness_history[0];
        let last = *self.fitness_history.last().unwrap();
        
        if first == 0.0 {
            return last;
        }
        
        (last - first) / first
    }
    
    /// Get average species count
    pub fn average_species_count(&self) -> f64 {
        if self.species_history.is_empty() {
            return 0.0;
        }
        
        self.species_history.iter().sum::<usize>() as f64 / self.species_history.len() as f64
    }
    
    /// Get complexity growth rate
    pub fn complexity_growth_rate(&self) -> f64 {
        if self.complexity_history.len() < 2 {
            return 0.0;
        }
        
        let first = self.complexity_history[0];
        let last = *self.complexity_history.last().unwrap();
        
        if first == 0.0 {
            return last;
        }
        
        (last - first) / first
    }
}

impl Default for TrainingState {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for TrainingStatistics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neat::fitness::XORFitnessEvaluator;
    use tempfile::tempdir;

    #[test]
    fn test_trainer_creation() {
        let evaluator = XORFitnessEvaluator::default();
        let config = NEATConfig::for_testing();
        
        let trainer = NEATTrainer::new(evaluator, config.clone());
        
        assert_eq!(trainer.config.population.size, config.population.size);
        assert!(!trainer.state.is_training);
        assert_eq!(trainer.state.generation, 0);
    }
    
    #[test]
    fn test_training_state() {
        let mut state = TrainingState::new();
        
        assert!(!state.is_training);
        assert_eq!(state.generation, 0);
        assert_eq!(state.best_fitness, 0.0);
        assert!(!state.target_reached);
        assert!(!state.early_stopped);
        
        state.is_training = true;
        state.generation = 10;
        state.best_fitness = 0.8;
        
        assert!(state.is_training);
        assert_eq!(state.generation, 10);
        assert_eq!(state.best_fitness, 0.8);
    }
    
    #[test]
    fn test_training_statistics() {
        let mut stats = TrainingStatistics::new();
        
        assert!(stats.generation_stats.is_empty());
        assert!(stats.fitness_history.is_empty());
        assert_eq!(stats.fitness_improvement_rate(), 0.0);
        
        stats.fitness_history = vec![0.2, 0.4, 0.6, 0.8];
        assert!((stats.fitness_improvement_rate() - 3.0).abs() < 1e-10); // 300% improvement
        
        stats.species_history = vec![5, 8, 6, 7];
        assert_eq!(stats.average_species_count(), 6.5);
        
        stats.complexity_history = vec![10.0, 15.0];
        assert_eq!(stats.complexity_growth_rate(), 0.5); // 50% growth
    }
    
    #[test]
    fn test_milestone_recording() {
        let evaluator = XORFitnessEvaluator::default();
        let config = NEATConfig::for_testing();
        
        let mut trainer = NEATTrainer::new(evaluator, config);
        
        trainer.record_milestone(
            MilestoneType::FitnessRecord,
            0.85,
            "New record".to_string()
        );
        
        assert_eq!(trainer.stats.milestones.len(), 1);
        assert_eq!(trainer.stats.milestones[0].fitness, 0.85);
        assert_eq!(trainer.stats.milestones[0].data, "New record");
    }
    
    #[test]
    fn test_checkpoint_directory() -> Result<()> {
        let evaluator = XORFitnessEvaluator::default();
        let config = NEATConfig::for_testing();
        let temp_dir = tempdir()?;
        
        let trainer = NEATTrainer::new(evaluator, config)
            .with_checkpoint_dir(temp_dir.path())?;
        
        assert!(trainer.checkpoint_dir.is_some());
        assert!(temp_dir.path().exists());
        
        Ok(())
    }
    
    #[test]
    fn test_training_termination_conditions() {
        let evaluator = XORFitnessEvaluator::default();
        let mut config = NEATConfig::for_testing();
        config.population.target_fitness = 0.9;
        config.population.max_generations = 100;
        config.training.early_stopping_patience = 50;
        
        let mut trainer = NEATTrainer::new(evaluator, config);
        
        // Should continue initially
        trainer.state.is_training = true;
        assert!(trainer.should_continue_training());
        
        // Should stop when target reached
        trainer.state.target_reached = true;
        assert!(!trainer.should_continue_training());
        
        // Reset and test max generations
        trainer.state.target_reached = false;
        trainer.state.generation = 100;
        assert!(!trainer.should_continue_training());
        
        // Reset and test early stopping
        trainer.state.generation = 50;
        trainer.state.stagnation_count = 50;
        assert!(!trainer.should_continue_training());
    }
    
    #[test]
    fn test_short_training_run() -> Result<()> {
        let evaluator = XORFitnessEvaluator::default();
        let mut config = NEATConfig::for_testing();
        config.population.size = 10;
        config.population.max_generations = 3;
        config.training.checkpoint_frequency = 10; // No checkpoints in short run
        
        let mut trainer = NEATTrainer::new(evaluator, config);
        let result = trainer.train()?;
        
        assert!(!result.success || result.best_genome.fitness > 0.0);
        assert!(matches!(result.termination_reason, TerminationReason::MaxGenerationsReached));
        assert_eq!(result.state.generation, 3);
        assert!(result.stats.generation_stats.len() >= 3);
        
        Ok(())
    }
    
    #[test]
    fn test_checkpoint_serialization() -> Result<()> {
        let state = TrainingState {
            is_training: true,
            generation: 42,
            best_fitness: 0.85,
            best_generation: 35,
            stagnation_count: 7,
            start_time: Some(Instant::now()),
            training_duration: Duration::from_secs(120),
            target_reached: false,
            early_stopped: false,
        };
        
        let stats = TrainingStatistics {
            fitness_history: vec![0.1, 0.3, 0.6, 0.85],
            species_history: vec![5, 8, 6, 4],
            ..Default::default()
        };
        
        let checkpoint = TrainingCheckpoint {
            state: state.clone(),
            stats: stats.clone(),
            config: NEATConfig::for_testing(),
            generation: 42,
        };
        
        let json = serde_json::to_string(&checkpoint)?;
        let deserialized: TrainingCheckpoint = serde_json::from_str(&json)?;
        
        assert_eq!(deserialized.generation, 42);
        assert_eq!(deserialized.state.best_fitness, 0.85);
        assert_eq!(deserialized.stats.fitness_history.len(), 4);
        
        Ok(())
    }
}