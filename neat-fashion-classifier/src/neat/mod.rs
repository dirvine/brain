//! NEAT algorithm implementation module
//!
//! This module contains the core NEAT algorithm components including genome representation,
//! innovation tracking, network activation, fitness evaluation, genetic operators, and topology analysis.

pub mod genome;
pub mod innovation;
pub mod network;
pub mod fitness;
pub mod mutation;
pub mod topology;

#[cfg(test)]
pub mod integration_tests;

// Re-export commonly used types
pub use genome::{Genome, NodeGene, ConnectionGene, NodeType, ActivationType};
pub use innovation::{Innovation, InnovationType, InnovationTracker};
pub use network::{Network, NetworkInfo};
pub use fitness::{FitnessEvaluator, ClassificationEvaluator, XORFitnessEvaluator, FitnessResults};
pub use mutation::{MutationContext, MutationPipeline, Mutation};
pub use topology::{TopologyAnalyzer, TopologyAnalysis};