//! # NEAT Fashion-MNIST Classifier
//!
//! A complete implementation of the NeuroEvolution of Augmenting Topologies (NEAT) algorithm
//! for Fashion-MNIST classification. This library provides efficient, parallel evolution of
//! neural network topologies and weights using modern Rust best practices.
//!
//! ## Features
//!
//! - Complete NEAT algorithm implementation with historical markings
//! - Efficient parallel fitness evaluation
//! - Species-based evolution with innovation protection
//! - Comprehensive testing and benchmarking
//! - Future HuggingFace dataset integration
//!
//! ## Quick Start
//!
//! ```rust
//! use neat_fashion_classifier::neat::Genome;
//! use neat_fashion_classifier::config::NEATConfig;
//!
//! // Create a genome for Fashion-MNIST (784 inputs, 10 outputs)
//! let genome = Genome::new(0, 784, 10);
//! assert_eq!(genome.get_input_count(), 784);
//! assert_eq!(genome.get_output_count(), 10);
//!
//! // Use default NEAT configuration
//! let config = NEATConfig::default();
//! assert_eq!(config.population.size, 150);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod config;
pub mod neat;
pub mod dataset;
// Temporarily disabled while fixing compilation issues
// pub mod benchmarks;

// Re-export commonly used types
pub use crate::config::NEATConfig;
pub use crate::neat::{Genome, NodeGene, ConnectionGene, NodeType, ActivationType};

/// Common error types used throughout the library
pub mod error {
    use thiserror::Error;

    /// Main error type for NEAT operations
    #[derive(Error, Debug)]
    pub enum NEATError {
        /// Invalid genome structure
        #[error("Invalid genome structure: {message}")]
        InvalidGenome { 
            /// Error description
            message: String 
        },
        
        /// Configuration validation failed
        #[error("Configuration validation failed: {parameter} = {value}")]
        InvalidConfiguration { 
            /// Parameter name
            parameter: String, 
            /// Invalid value
            value: String 
        },
        
        /// Serialization/deserialization error
        #[error("Serialization error: {0}")]
        SerializationError(#[from] serde_json::Error),
        
        /// I/O error
        #[error("I/O error: {0}")]
        IoError(#[from] std::io::Error),
        
        /// Generic error with context
        #[error("NEAT error: {0}")]
        Other(#[from] anyhow::Error),
    }

    /// Result type alias for NEAT operations
    pub type Result<T> = std::result::Result<T, NEATError>;
}

/// Utility functions and common types
pub mod utils {
    use rand::prelude::*;
    
    /// Initialize logging for the library (only available with env_logger)
    #[cfg(feature = "logging")]
    pub fn init_logging() {
        env_logger::init();
    }
    
    /// Create a seeded random number generator for reproducible results
    pub fn create_rng(seed: Option<u64>) -> SmallRng {
        match seed {
            Some(s) => SmallRng::seed_from_u64(s),
            None => SmallRng::from_entropy(),
        }
    }
}
