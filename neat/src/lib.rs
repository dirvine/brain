//! # NEAT Mathematical Discovery and Educational Platform
//!
//! A comprehensive implementation combining NeuroEvolution of Augmenting Topologies (NEAT) with
//! advanced mathematical problem solving and educational technology. This platform evolves neural
//! networks to solve mathematical problems while providing adaptive tutoring and personalized
//! learning experiences.
//!
//! ## Features
//!
//! ### Core NEAT Implementation
//! - Complete NEAT algorithm with historical markings and speciation
//! - Efficient parallel fitness evaluation and evolution
//! - Species-based evolution with innovation protection
//!
//! ### Mathematical Problem Solving
//! - 21 specialized mathematical modules (arithmetic, algebra, calculus, etc.)
//! - Algebraic expression parsing and evaluation
//! - Automated mathematical discovery system
//!
//! ### Educational Technology
//! - Adaptive tutoring system with multiple teaching strategies
//! - Personalized learning paths and curriculum generation
//! - Comprehensive assessment engine with difficulty calibration
//! - Real-time learning analytics and progress tracking
//! - Step-by-step solution explanations with learning style adaptation
//!
//! ## Quick Start
//!
//! ### Mathematical Problem Solving
//! ```rust
//! use neat::calculator::Calculator;
//! use neat::calculator::algebra::AlgebraProblem;
//!
//! // Create a calculator and solve a linear equation
//! let mut calculator = Calculator::new();
//! let problem = AlgebraProblem::linear_equation(2.0, 3.0, 7.0); // 2x + 3 = 7
//! match calculator.solve_problem(&problem) {
//!     Ok(solution) => println!("Solution: x = {}", solution),
//!     Err(e) => println!("Error: {}", e),
//! }
//! ```
//!
//! ### Educational Platform
//! ```rust
//! use neat::education::{EducationalPlatform, EducationConfig, LearningStyle};
//!
//! // Create educational platform and register a student
//! let config = EducationConfig::default();
//! let mut platform = EducationalPlatform::new(config);
//! platform.register_student("student1".to_string(), 15, LearningStyle::Visual).unwrap();
//!
//! // Start a tutoring session
//! let session = platform.start_tutoring_session("student1").unwrap();
//! println!("Started session: {}", session.session_id);
//! ```
//!
//! ### NEAT Evolution
//! ```rust
//! use neat::neat::Genome;
//! use neat::config::NEATConfig;
//!
//! // Create a genome for mathematical problem solving
//! let genome = Genome::new(0, 10, 1); // 10 inputs, 1 output
//! assert_eq!(genome.get_input_count(), 10);
//! assert_eq!(genome.get_output_count(), 1);
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
pub mod calculator;
pub mod education;
// Temporarily disabled while fixing compilation issues
// pub mod benchmarks;

// Re-export commonly used types
pub use crate::config::NEATConfig;
pub use crate::neat::{Genome, NodeGene, ConnectionGene, NodeType, ActivationType};

// Re-export educational platform types
pub use crate::education::{
    EducationalPlatform, 
    EducationConfig, 
    StudentModel, 
    LearningStyle,
    AdaptiveTutor,
    AssessmentEngine,
    CurriculumGenerator,
    LearningAnalytics,
    EducationalProblemGenerator,
    ExplanationEngine,
};

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
