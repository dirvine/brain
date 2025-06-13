//! NEAT algorithm configuration
//!
//! This module defines the complete configuration structure for the NEAT algorithm,
//! with defaults based on the original research papers and validation to ensure
//! parameters are within reasonable ranges.

use serde::{Deserialize, Serialize};
use std::path::Path;
use crate::neat::ActivationType;
use crate::error::{NEATError, Result};

/// Complete configuration for the NEAT algorithm
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NEATConfig {
    /// Population parameters
    pub population: PopulationConfig,
    
    /// Speciation parameters  
    pub speciation: SpeciationConfig,
    
    /// Mutation parameters
    pub mutation: MutationConfig,
    
    /// Selection and reproduction parameters
    pub selection: SelectionConfig,
    
    /// Network structure parameters
    pub network: NetworkConfig,
    
    /// Training and evaluation parameters
    pub training: TrainingConfig,
}

/// Population management configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PopulationConfig {
    /// Number of genomes in the population
    pub size: usize,
    
    /// Maximum number of generations to evolve
    pub max_generations: usize,
    
    /// Target fitness to stop evolution early
    pub target_fitness: f64,
    
    /// Random seed for reproducible results (None for random)
    pub random_seed: Option<u64>,
}

/// Speciation algorithm configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeciationConfig {
    /// Coefficient for excess genes in compatibility distance
    pub excess_coefficient: f64,
    
    /// Coefficient for disjoint genes in compatibility distance
    pub disjoint_coefficient: f64,
    
    /// Coefficient for weight differences in compatibility distance
    pub weight_difference_coefficient: f64,
    
    /// Distance threshold for same species
    pub compatibility_threshold: f64,
    
    /// Generations without improvement before species extinction
    pub staleness_threshold: usize,
    
    /// Minimum number of genomes in a species
    pub min_species_size: usize,
    
    /// Whether to dynamically adjust compatibility threshold
    pub dynamic_threshold: bool,
    
    /// Target number of species (for dynamic threshold)
    pub target_species_count: usize,
}

/// Mutation operator configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MutationConfig {
    /// Probability of adding a new node
    pub add_node_rate: f64,
    
    /// Probability of adding a new connection
    pub add_connection_rate: f64,
    
    /// Probability of mutating weights
    pub weight_mutation_rate: f64,
    
    /// Probability of perturbing vs replacing weights
    pub weight_perturbation_rate: f64,
    
    /// Magnitude of weight perturbations
    pub weight_perturbation_power: f64,
    
    /// Range for new random weights
    pub weight_range: (f64, f64),
    
    /// Probability of disabling a connection
    pub disable_connection_rate: f64,
    
    /// Probability of enabling a disabled connection
    pub enable_connection_rate: f64,
    
    /// Probability of mutating node bias
    pub bias_mutation_rate: f64,
    
    /// Probability of perturbing vs replacing bias
    pub bias_perturbation_rate: f64,
    
    /// Magnitude of bias perturbations
    pub bias_perturbation_power: f64,
    
    /// Range for new random bias values
    pub bias_range: (f64, f64),
    
    /// Probability of changing activation function
    pub activation_mutation_rate: f64,
}

/// Selection and reproduction configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectionConfig {
    /// Fraction of population that survives each generation
    pub survival_threshold: f64,
    
    /// Rate of mating between different species
    pub interspecies_mating_rate: f64,
    
    /// Fraction of population preserved unchanged (elitism)
    pub elitism_rate: f64,
    
    /// Size of tournament for tournament selection
    pub tournament_size: usize,
    
    /// Probability of mutation when reproducing
    pub mutation_probability: f64,
    
    /// Probability of crossover when reproducing
    pub crossover_probability: f64,
}

/// Network structure configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Available activation functions for hidden/output nodes
    pub activation_functions: Vec<ActivationType>,
    
    /// Whether to include bias nodes
    pub bias_enabled: bool,
    
    /// Whether to allow recurrent connections
    pub recurrent_connections: bool,
    
    /// Maximum number of nodes per network
    pub max_nodes: usize,
    
    /// Maximum number of connections per network
    pub max_connections: usize,
    
    /// Initial connection probability (0.0 for minimal start)
    pub initial_connection_rate: f64,
    
    /// Response multiplier for activation functions
    pub response_multiplier: f64,
}

/// Training and evaluation configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of samples for fitness evaluation (subset for speed)
    pub fitness_evaluation_size: usize,
    
    /// Frequency of validation evaluation (generations)
    pub validation_frequency: usize,
    
    /// Frequency of full dataset evaluation (generations) 
    pub full_evaluation_frequency: usize,
    
    /// Generations without improvement for early stopping
    pub early_stopping_patience: usize,
    
    /// Whether to enable parallel fitness evaluation
    pub parallel_evaluation: bool,
    
    /// Number of threads for parallel evaluation (None for auto)
    pub num_threads: Option<usize>,
    
    /// Frequency of saving checkpoints (generations)
    pub checkpoint_frequency: usize,
    
    /// Whether to save detailed evolution statistics
    pub save_statistics: bool,
}

impl Default for NEATConfig {
    fn default() -> Self {
        Self {
            population: PopulationConfig {
                size: 150,
                max_generations: 500,
                target_fitness: 0.85,
                random_seed: None,
            },
            
            speciation: SpeciationConfig {
                excess_coefficient: 1.0,
                disjoint_coefficient: 1.0,
                weight_difference_coefficient: 0.4,
                compatibility_threshold: 3.0,
                staleness_threshold: 15,
                min_species_size: 5,
                dynamic_threshold: true,
                target_species_count: 10,
            },
            
            mutation: MutationConfig {
                add_node_rate: 0.03,
                add_connection_rate: 0.05,
                weight_mutation_rate: 0.8,
                weight_perturbation_rate: 0.9,
                weight_perturbation_power: 0.5,
                weight_range: (-2.0, 2.0),
                disable_connection_rate: 0.01,
                enable_connection_rate: 0.01,
                bias_mutation_rate: 0.1,
                bias_perturbation_rate: 0.9,
                bias_perturbation_power: 0.1,
                bias_range: (-1.0, 1.0),
                activation_mutation_rate: 0.01,
            },
            
            selection: SelectionConfig {
                survival_threshold: 0.2,
                interspecies_mating_rate: 0.001,
                elitism_rate: 0.1,
                tournament_size: 3,
                mutation_probability: 0.25,
                crossover_probability: 0.75,
            },
            
            network: NetworkConfig {
                activation_functions: vec![
                    ActivationType::Sigmoid,
                    ActivationType::Tanh,
                    ActivationType::ReLU,
                ],
                bias_enabled: true,
                recurrent_connections: false,
                max_nodes: 1000,
                max_connections: 5000,
                initial_connection_rate: 0.0,
                response_multiplier: 1.0,
            },
            
            training: TrainingConfig {
                fitness_evaluation_size: 1000,
                validation_frequency: 10,
                full_evaluation_frequency: 50,
                early_stopping_patience: 100,
                parallel_evaluation: true,
                num_threads: None,
                checkpoint_frequency: 50,
                save_statistics: true,
            },
        }
    }
}

impl NEATConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Load configuration from a TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)
            .map_err(|e| NEATError::Other(e.into()))?;
        config.validate()?;
        Ok(config)
    }
    
    /// Save configuration to a TOML file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| NEATError::Other(e.into()))?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Load configuration from JSON
    pub fn from_json(json_str: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(json_str)?;
        config.validate()?;
        Ok(config)
    }
    
    /// Convert configuration to JSON
    pub fn to_json(&self) -> Result<String> {
        let json = serde_json::to_string_pretty(self)?;
        Ok(json)
    }
    
    /// Validate all configuration parameters
    pub fn validate(&self) -> Result<()> {
        self.validate_population()?;
        self.validate_speciation()?;
        self.validate_mutation()?;
        self.validate_selection()?;
        self.validate_network()?;
        self.validate_training()?;
        Ok(())
    }
    
    fn validate_population(&self) -> Result<()> {
        if self.population.size == 0 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "population.size".to_string(),
                value: self.population.size.to_string(),
            });
        }
        
        if self.population.max_generations == 0 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "population.max_generations".to_string(),
                value: self.population.max_generations.to_string(),
            });
        }
        
        if !(0.0..=1.0).contains(&self.population.target_fitness) {
            return Err(NEATError::InvalidConfiguration {
                parameter: "population.target_fitness".to_string(),
                value: self.population.target_fitness.to_string(),
            });
        }
        
        Ok(())
    }
    
    fn validate_speciation(&self) -> Result<()> {
        if self.speciation.excess_coefficient < 0.0 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "speciation.excess_coefficient".to_string(),
                value: self.speciation.excess_coefficient.to_string(),
            });
        }
        
        if self.speciation.disjoint_coefficient < 0.0 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "speciation.disjoint_coefficient".to_string(),
                value: self.speciation.disjoint_coefficient.to_string(),
            });
        }
        
        if self.speciation.weight_difference_coefficient < 0.0 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "speciation.weight_difference_coefficient".to_string(),
                value: self.speciation.weight_difference_coefficient.to_string(),
            });
        }
        
        if self.speciation.compatibility_threshold <= 0.0 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "speciation.compatibility_threshold".to_string(),
                value: self.speciation.compatibility_threshold.to_string(),
            });
        }
        
        if self.speciation.min_species_size == 0 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "speciation.min_species_size".to_string(),
                value: self.speciation.min_species_size.to_string(),
            });
        }
        
        if self.speciation.target_species_count == 0 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "speciation.target_species_count".to_string(),
                value: self.speciation.target_species_count.to_string(),
            });
        }
        
        Ok(())
    }
    
    fn validate_mutation(&self) -> Result<()> {
        let rates = [
            ("add_node_rate", self.mutation.add_node_rate),
            ("add_connection_rate", self.mutation.add_connection_rate),
            ("weight_mutation_rate", self.mutation.weight_mutation_rate),
            ("weight_perturbation_rate", self.mutation.weight_perturbation_rate),
            ("disable_connection_rate", self.mutation.disable_connection_rate),
            ("enable_connection_rate", self.mutation.enable_connection_rate),
            ("bias_mutation_rate", self.mutation.bias_mutation_rate),
        ];
        
        for (name, rate) in &rates {
            if !(0.0..=1.0).contains(rate) {
                return Err(NEATError::InvalidConfiguration {
                    parameter: format!("mutation.{}", name),
                    value: rate.to_string(),
                });
            }
        }
        
        if self.mutation.weight_perturbation_power <= 0.0 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "mutation.weight_perturbation_power".to_string(),
                value: self.mutation.weight_perturbation_power.to_string(),
            });
        }
        
        let (min_weight, max_weight) = self.mutation.weight_range;
        if min_weight >= max_weight {
            return Err(NEATError::InvalidConfiguration {
                parameter: "mutation.weight_range".to_string(),
                value: format!("({}, {})", min_weight, max_weight),
            });
        }
        
        Ok(())
    }
    
    fn validate_selection(&self) -> Result<()> {
        let rates = [
            ("survival_threshold", self.selection.survival_threshold),
            ("interspecies_mating_rate", self.selection.interspecies_mating_rate),
            ("elitism_rate", self.selection.elitism_rate),
            ("mutation_probability", self.selection.mutation_probability),
            ("crossover_probability", self.selection.crossover_probability),
        ];
        
        for (name, rate) in &rates {
            if !(0.0..=1.0).contains(rate) {
                return Err(NEATError::InvalidConfiguration {
                    parameter: format!("selection.{}", name),
                    value: rate.to_string(),
                });
            }
        }
        
        if self.selection.tournament_size == 0 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "selection.tournament_size".to_string(),
                value: self.selection.tournament_size.to_string(),
            });
        }
        
        if self.selection.tournament_size > self.population.size {
            return Err(NEATError::InvalidConfiguration {
                parameter: "selection.tournament_size".to_string(),
                value: format!("tournament_size ({}) > population_size ({})", 
                              self.selection.tournament_size, self.population.size),
            });
        }
        
        Ok(())
    }
    
    fn validate_network(&self) -> Result<()> {
        if self.network.activation_functions.is_empty() {
            return Err(NEATError::InvalidConfiguration {
                parameter: "network.activation_functions".to_string(),
                value: "empty".to_string(),
            });
        }
        
        if self.network.max_nodes == 0 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "network.max_nodes".to_string(),
                value: self.network.max_nodes.to_string(),
            });
        }
        
        if self.network.max_connections == 0 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "network.max_connections".to_string(),
                value: self.network.max_connections.to_string(),
            });
        }
        
        if !(0.0..=1.0).contains(&self.network.initial_connection_rate) {
            return Err(NEATError::InvalidConfiguration {
                parameter: "network.initial_connection_rate".to_string(),
                value: self.network.initial_connection_rate.to_string(),
            });
        }
        
        if self.network.response_multiplier <= 0.0 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "network.response_multiplier".to_string(),
                value: self.network.response_multiplier.to_string(),
            });
        }
        
        Ok(())
    }
    
    fn validate_training(&self) -> Result<()> {
        if self.training.fitness_evaluation_size == 0 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "training.fitness_evaluation_size".to_string(),
                value: self.training.fitness_evaluation_size.to_string(),
            });
        }
        
        if self.training.validation_frequency == 0 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "training.validation_frequency".to_string(),
                value: self.training.validation_frequency.to_string(),
            });
        }
        
        if self.training.full_evaluation_frequency == 0 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "training.full_evaluation_frequency".to_string(),
                value: self.training.full_evaluation_frequency.to_string(),
            });
        }
        
        if self.training.checkpoint_frequency == 0 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "training.checkpoint_frequency".to_string(),
                value: self.training.checkpoint_frequency.to_string(),
            });
        }
        
        if let Some(num_threads) = self.training.num_threads {
            if num_threads == 0 {
                return Err(NEATError::InvalidConfiguration {
                    parameter: "training.num_threads".to_string(),
                    value: num_threads.to_string(),
                });
            }
        }
        
        Ok(())
    }
    
    /// Create a configuration optimized for fast testing
    pub fn for_testing() -> Self {
        Self {
            population: PopulationConfig {
                size: 20,
                max_generations: 10,
                target_fitness: 0.7,
                random_seed: Some(42),
            },
            training: TrainingConfig {
                fitness_evaluation_size: 50,
                validation_frequency: 5,
                full_evaluation_frequency: 10,
                early_stopping_patience: 5,
                parallel_evaluation: false,
                num_threads: Some(1),
                checkpoint_frequency: 10,
                save_statistics: false,
            },
            ..Self::default()
        }
    }
    
    /// Create a configuration optimized for small-scale experiments
    pub fn for_small_scale() -> Self {
        Self {
            population: PopulationConfig {
                size: 50,
                max_generations: 100,
                target_fitness: 0.8,
                random_seed: None,
            },
            training: TrainingConfig {
                fitness_evaluation_size: 500,
                parallel_evaluation: true,
                num_threads: Some(4),
                ..Self::default().training
            },
            ..Self::default()
        }
    }
    
    /// Create a configuration optimized for production use
    pub fn for_production() -> Self {
        Self {
            population: PopulationConfig {
                size: 300,
                max_generations: 1000,
                target_fitness: 0.9,
                random_seed: None,
            },
            training: TrainingConfig {
                fitness_evaluation_size: 5000,
                parallel_evaluation: true,
                num_threads: None, // Use all available cores
                ..Self::default().training
            },
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config_validation() {
        let config = NEATConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_presets() {
        let testing_config = NEATConfig::for_testing();
        assert!(testing_config.validate().is_ok());
        assert_eq!(testing_config.population.size, 20);
        assert_eq!(testing_config.population.random_seed, Some(42));
        
        let small_config = NEATConfig::for_small_scale();
        assert!(small_config.validate().is_ok());
        assert_eq!(small_config.population.size, 50);
        
        let production_config = NEATConfig::for_production();
        assert!(production_config.validate().is_ok());
        assert_eq!(production_config.population.size, 300);
    }

    #[test]
    fn test_invalid_population_config() {
        let mut config = NEATConfig::default();
        
        // Test invalid population size
        config.population.size = 0;
        assert!(config.validate().is_err());
        
        config.population.size = 150; // Reset
        
        // Test invalid target fitness
        config.population.target_fitness = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_mutation_rates() {
        let mut config = NEATConfig::default();
        
        // Test invalid mutation rate
        config.mutation.add_node_rate = -0.1;
        assert!(config.validate().is_err());
        
        config.mutation.add_node_rate = 1.1;
        assert!(config.validate().is_err());
        
        config.mutation.add_node_rate = 0.03; // Reset
        
        // Test invalid weight range
        config.mutation.weight_range = (2.0, -2.0); // min > max
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_selection_config() {
        let mut config = NEATConfig::default();
        
        // Test tournament size larger than population
        config.selection.tournament_size = config.population.size + 1;
        assert!(config.validate().is_err());
        
        config.selection.tournament_size = 3; // Reset
        
        // Test invalid survival threshold
        config.selection.survival_threshold = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_network_config() {
        let mut config = NEATConfig::default();
        
        // Test empty activation functions
        config.network.activation_functions.clear();
        assert!(config.validate().is_err());
        
        config.network.activation_functions = vec![ActivationType::Sigmoid]; // Reset
        
        // Test zero max nodes
        config.network.max_nodes = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_serialization() {
        let config = NEATConfig::default();
        
        // Test JSON serialization
        let json = config.to_json().unwrap();
        let deserialized = NEATConfig::from_json(&json).unwrap();
        assert_eq!(config, deserialized);
        
        // Test TOML serialization
        let toml_str = toml::to_string(&config).unwrap();
        let from_toml: NEATConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(config, from_toml);
    }

    #[test]
    fn test_config_file_io() {
        let config = NEATConfig::for_testing();
        
        // Create a temporary file
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path();
        
        // Save and load config
        config.save_to_file(file_path).unwrap();
        let loaded_config = NEATConfig::from_file(file_path).unwrap();
        
        assert_eq!(config, loaded_config);
    }

    #[test]
    fn test_activation_function_serialization() {
        let config = NEATConfig {
            network: NetworkConfig {
                activation_functions: vec![
                    ActivationType::Sigmoid,
                    ActivationType::Tanh,
                    ActivationType::ReLU,
                    ActivationType::Linear,
                    ActivationType::Gaussian,
                ],
                ..NetworkConfig::default()
            },
            ..NEATConfig::default()
        };
        
        let json = config.to_json().unwrap();
        let deserialized = NEATConfig::from_json(&json).unwrap();
        assert_eq!(config.network.activation_functions, deserialized.network.activation_functions);
    }
    
    #[test]
    fn test_comprehensive_validation() {
        let config = NEATConfig::default();
        
        // Should validate successfully
        assert!(config.validate().is_ok());
        
        // Verify all subsection validations are called
        assert!(config.validate_population().is_ok());
        assert!(config.validate_speciation().is_ok());
        assert!(config.validate_mutation().is_ok());
        assert!(config.validate_selection().is_ok());
        assert!(config.validate_network().is_ok());
        assert!(config.validate_training().is_ok());
    }
}

// Provide Default implementations for sub-configs
impl Default for PopulationConfig {
    fn default() -> Self {
        NEATConfig::default().population
    }
}

impl Default for SpeciationConfig {
    fn default() -> Self {
        NEATConfig::default().speciation
    }
}

impl Default for MutationConfig {
    fn default() -> Self {
        NEATConfig::default().mutation
    }
}

impl Default for SelectionConfig {
    fn default() -> Self {
        NEATConfig::default().selection
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        NEATConfig::default().network
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        NEATConfig::default().training
    }
}