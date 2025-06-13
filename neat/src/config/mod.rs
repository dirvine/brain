//! Configuration system for NEAT algorithm
//!
//! This module provides comprehensive configuration management for the NEAT algorithm,
//! including parameter validation, serialization, and default values based on the
//! original NEAT research.

pub mod neat_config;

// Re-export main configuration type
pub use neat_config::NEATConfig;