pub mod problem_generator;
pub mod network_visualizer;

use serde::{Deserialize, Serialize};

// Data structures for frontend communication
#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkVisualization {
    pub nodes: Vec<NetworkNode>,
    pub edges: Vec<NetworkEdge>,
    pub metrics: NetworkMetrics,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkNode {
    pub id: String,
    pub label: String,
    pub node_type: String,
    pub value: f64,
    pub x: f64,
    pub y: f64,
    pub color: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkEdge {
    pub from: String,
    pub to: String,
    pub weight: f64,
    pub color: String,
    pub width: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkMetrics {
    pub accuracy: f64,
    pub efficiency: f64,
    pub complexity: f64,
    pub nodes_count: usize,
    pub edges_count: usize,
}

pub use problem_generator::*;
pub use network_visualizer::*;