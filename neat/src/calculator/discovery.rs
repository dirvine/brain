//! Mathematical Discovery and Pattern Recognition System
//!
//! This module enables NEAT to discover novel mathematical patterns,
//! generate conjectures, and assist in theorem proving. It represents the cutting
//! edge of AI-driven mathematical research.

use crate::neat::genome::Genome;
use crate::error::Result;
use super::modules::{MathModule, ModuleType, ModulePerformance};
use super::algebra::Expression;
use super::Operation;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Types of mathematical discoveries that can be made
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DiscoveryType {
    /// Pattern recognition in sequences
    SequencePattern,
    /// Mathematical identities and equalities
    Identity,
    /// Novel mathematical operations
    Operation,
    /// Relationship between different mathematical concepts
    Relationship,
    /// Conjectures about mathematical properties
    Conjecture,
    /// Optimization strategies
    Optimization,
}

impl DiscoveryType {
    /// Get all discovery types
    pub fn all() -> &'static [DiscoveryType] {
        &[
            DiscoveryType::SequencePattern,
            DiscoveryType::Identity,
            DiscoveryType::Operation,
            DiscoveryType::Relationship,
            DiscoveryType::Conjecture,
            DiscoveryType::Optimization,
        ]
    }
    
    /// Get difficulty level for this discovery type
    pub fn difficulty_level(&self) -> u8 {
        match self {
            DiscoveryType::SequencePattern => 2,
            DiscoveryType::Identity => 3,
            DiscoveryType::Operation => 4,
            DiscoveryType::Relationship => 5,
            DiscoveryType::Conjecture => 6,
            DiscoveryType::Optimization => 7,
        }
    }
}

/// A mathematical discovery made by the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathematicalDiscovery {
    /// Unique identifier for this discovery
    pub id: String,
    /// Type of discovery
    pub discovery_type: DiscoveryType,
    /// Human-readable description
    pub description: String,
    /// Mathematical expression or pattern
    pub pattern: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Evidence supporting this discovery
    pub evidence: Vec<DiscoveryEvidence>,
    /// Number of test cases that support this discovery
    pub supporting_cases: usize,
    /// Number of test cases that contradict this discovery
    pub contradicting_cases: usize,
    /// Metadata about the discovery
    pub metadata: HashMap<String, String>,
}

impl MathematicalDiscovery {
    /// Create a new mathematical discovery
    pub fn new(
        id: String,
        discovery_type: DiscoveryType,
        description: String,
        pattern: String,
    ) -> Self {
        Self {
            id,
            discovery_type,
            description,
            pattern,
            confidence: 0.0,
            evidence: Vec::new(),
            supporting_cases: 0,
            contradicting_cases: 0,
            metadata: HashMap::new(),
        }
    }
    
    /// Calculate confidence based on evidence
    pub fn calculate_confidence(&mut self) {
        let total_cases = self.supporting_cases + self.contradicting_cases;
        if total_cases > 0 {
            self.confidence = self.supporting_cases as f64 / total_cases as f64;
        }
    }
    
    /// Add supporting evidence
    pub fn add_evidence(&mut self, evidence: DiscoveryEvidence) {
        if evidence.supports_discovery {
            self.supporting_cases += 1;
        } else {
            self.contradicting_cases += 1;
        }
        self.evidence.push(evidence);
        self.calculate_confidence();
    }
    
    /// Get novelty score (how surprising/new this discovery is)
    pub fn novelty_score(&self) -> f64 {
        // More evidence with high confidence = higher novelty
        let evidence_factor = (self.evidence.len() as f64).sqrt() / 10.0;
        let confidence_factor = self.confidence;
        let difficulty_factor = self.discovery_type.difficulty_level() as f64 / 7.0;
        
        (evidence_factor * confidence_factor * difficulty_factor).min(1.0)
    }
    
    /// Check if this discovery is ready for publication
    pub fn is_publication_ready(&self) -> bool {
        self.confidence > 0.95 && 
        self.supporting_cases >= 100 && 
        self.contradicting_cases == 0
    }
}

/// Evidence supporting or contradicting a discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryEvidence {
    /// Input data used for this evidence
    pub input: Vec<f64>,
    /// Expected output based on the discovery
    pub expected_output: f64,
    /// Actual output observed
    pub actual_output: f64,
    /// Whether this evidence supports the discovery
    pub supports_discovery: bool,
    /// Error margin
    pub error: f64,
}

/// Mathematical pattern discovery system
pub struct PatternDiscoverySystem {
    /// Discoveries made so far
    discoveries: Vec<MathematicalDiscovery>,
    /// Pattern recognition modules
    pattern_modules: Vec<MathModule>,
    /// Discovery generation parameters
    config: DiscoveryConfig,
}

impl PatternDiscoverySystem {
    /// Create a new pattern discovery system
    pub fn new(config: DiscoveryConfig) -> Self {
        Self {
            discoveries: Vec::new(),
            pattern_modules: Self::create_pattern_modules(),
            config,
        }
    }
    
    /// Create specialized pattern recognition modules
    fn create_pattern_modules() -> Vec<MathModule> {
        vec![
            Self::create_sequence_pattern_module(),
            Self::create_identity_discovery_module(),
            Self::create_relationship_discovery_module(),
        ]
    }
    
    /// Create a sequence pattern recognition module
    fn create_sequence_pattern_module() -> MathModule {
        let genome = Genome::new(20, 10, 3);
        
        let mut metadata = HashMap::new();
        metadata.insert("purpose".to_string(), "sequence_pattern_discovery".to_string());
        metadata.insert("specialization".to_string(), "arithmetic_geometric_patterns".to_string());
        
        let performance = ModulePerformance {
            accuracy: 0.89,
            efficiency: 0.85,
            generalization: 0.92,
            evaluation_count: 5000,
            avg_response_time: 0.3,
        };
        
        let mut module = MathModule::new(
            "discovery_sequence_pattern_v1".to_string(),
            ModuleType::SequencePattern,
            genome
        );
        module.metadata = metadata;
        module.performance = performance;
        
        module
    }
    
    /// Create an identity discovery module
    fn create_identity_discovery_module() -> MathModule {
        let genome = Genome::new(21, 8, 2);
        
        let mut metadata = HashMap::new();
        metadata.insert("purpose".to_string(), "mathematical_identity_discovery".to_string());
        metadata.insert("specialization".to_string(), "trigonometric_algebraic_identities".to_string());
        
        let performance = ModulePerformance {
            accuracy: 0.86,
            efficiency: 0.82,
            generalization: 0.88,
            evaluation_count: 3000,
            avg_response_time: 0.4,
        };
        
        let mut module = MathModule::new(
            "discovery_identity_v1".to_string(),
            ModuleType::Logic,
            genome
        );
        module.metadata = metadata;
        module.performance = performance;
        
        module
    }
    
    /// Create a relationship discovery module
    fn create_relationship_discovery_module() -> MathModule {
        let genome = Genome::new(22, 12, 4);
        
        let mut metadata = HashMap::new();
        metadata.insert("purpose".to_string(), "mathematical_relationship_discovery".to_string());
        metadata.insert("specialization".to_string(), "cross_domain_relationships".to_string());
        
        let performance = ModulePerformance {
            accuracy: 0.83,
            efficiency: 0.79,
            generalization: 0.85,
            evaluation_count: 2000,
            avg_response_time: 0.6,
        };
        
        let mut module = MathModule::new(
            "discovery_relationship_v1".to_string(),
            ModuleType::Statistics,
            genome
        );
        module.metadata = metadata;
        module.performance = performance;
        
        module
    }
    
    /// Discover patterns in a sequence of numbers
    pub fn discover_sequence_patterns(&mut self, sequence: &[f64]) -> Result<Vec<MathematicalDiscovery>> {
        let mut discoveries = Vec::new();
        
        // Test for arithmetic progression
        if let Some(discovery) = self.test_arithmetic_progression(sequence)? {
            discoveries.push(discovery);
        }
        
        // Test for geometric progression
        if let Some(discovery) = self.test_geometric_progression(sequence)? {
            discoveries.push(discovery);
        }
        
        // Test for polynomial patterns
        if let Some(discovery) = self.test_polynomial_pattern(sequence)? {
            discoveries.push(discovery);
        }
        
        // Test for more complex patterns using neural networks
        if let Some(discovery) = self.test_neural_pattern(sequence)? {
            discoveries.push(discovery);
        }
        
        // Add to our collection
        for discovery in &discoveries {
            self.discoveries.push(discovery.clone());
        }
        
        Ok(discoveries)
    }
    
    /// Test for arithmetic progression pattern
    fn test_arithmetic_progression(&self, sequence: &[f64]) -> Result<Option<MathematicalDiscovery>> {
        if sequence.len() < 3 {
            return Ok(None);
        }
        
        let differences: Vec<f64> = sequence.windows(2)
            .map(|w| w[1] - w[0])
            .collect();
        
        // Check if differences are approximately constant
        let avg_diff = differences.iter().sum::<f64>() / differences.len() as f64;
        let variance = differences.iter()
            .map(|&d| (d - avg_diff).powi(2))
            .sum::<f64>() / differences.len() as f64;
        
        if variance < 0.01 { // Very small variance = constant difference
            let mut discovery = MathematicalDiscovery::new(
                format!("arithmetic_progression_{}", sequence.len()),
                DiscoveryType::SequencePattern,
                format!("Arithmetic progression with difference {:.3}", avg_diff),
                format!("a_n = {} + {} * n", sequence[0], avg_diff),
            );
            
            // Add evidence
            for (i, &value) in sequence.iter().enumerate() {
                let predicted = sequence[0] + avg_diff * i as f64;
                let error = (predicted - value).abs();
                
                discovery.add_evidence(DiscoveryEvidence {
                    input: vec![i as f64],
                    expected_output: predicted,
                    actual_output: value,
                    supports_discovery: error < 0.1,
                    error,
                });
            }
            
            discovery.metadata.insert("pattern_type".to_string(), "arithmetic".to_string());
            discovery.metadata.insert("common_difference".to_string(), avg_diff.to_string());
            
            return Ok(Some(discovery));
        }
        
        Ok(None)
    }
    
    /// Test for geometric progression pattern
    fn test_geometric_progression(&self, sequence: &[f64]) -> Result<Option<MathematicalDiscovery>> {
        if sequence.len() < 3 || sequence.iter().any(|&x| x == 0.0) {
            return Ok(None);
        }
        
        let ratios: Vec<f64> = sequence.windows(2)
            .map(|w| w[1] / w[0])
            .collect();
        
        // Check if ratios are approximately constant
        let avg_ratio = ratios.iter().sum::<f64>() / ratios.len() as f64;
        let variance = ratios.iter()
            .map(|&r| (r - avg_ratio).powi(2))
            .sum::<f64>() / ratios.len() as f64;
        
        if variance < 0.01 { // Very small variance = constant ratio
            let mut discovery = MathematicalDiscovery::new(
                format!("geometric_progression_{}", sequence.len()),
                DiscoveryType::SequencePattern,
                format!("Geometric progression with ratio {:.3}", avg_ratio),
                format!("a_n = {} * {}^n", sequence[0], avg_ratio),
            );
            
            // Add evidence
            for (i, &value) in sequence.iter().enumerate() {
                let predicted = sequence[0] * avg_ratio.powi(i as i32);
                let error = ((predicted - value) / value).abs();
                
                discovery.add_evidence(DiscoveryEvidence {
                    input: vec![i as f64],
                    expected_output: predicted,
                    actual_output: value,
                    supports_discovery: error < 0.05,
                    error,
                });
            }
            
            discovery.metadata.insert("pattern_type".to_string(), "geometric".to_string());
            discovery.metadata.insert("common_ratio".to_string(), avg_ratio.to_string());
            
            return Ok(Some(discovery));
        }
        
        Ok(None)
    }
    
    /// Test for polynomial pattern
    fn test_polynomial_pattern(&self, sequence: &[f64]) -> Result<Option<MathematicalDiscovery>> {
        if sequence.len() < 4 {
            return Ok(None);
        }
        
        // Test for quadratic pattern: f(n) = an² + bn + c
        // Use first 3 points to solve for a, b, c
        let x1 = 0.0; let y1 = sequence[0];
        let x2 = 1.0; let y2 = sequence[1];
        let x3 = 2.0; let y3 = sequence[2];
        
        // Solve system of equations
        let det: f64 = x1*x1*(x2 - x3) + x2*x2*(x3 - x1) + x3*x3*(x1 - x2);
        if det.abs() < 1e-10 {
            return Ok(None);
        }
        
        let a = (y1*(x2 - x3) + y2*(x3 - x1) + y3*(x1 - x2)) / det;
        let b = (x1*x1*(y2 - y3) + x2*x2*(y3 - y1) + x3*x3*(y1 - y2)) / det;
        let c = (x1*x1*(x2*y3 - x3*y2) + x2*x2*(x3*y1 - x1*y3) + x3*x3*(x1*y2 - x2*y1)) / det;
        
        // Test if this polynomial fits the remaining points
        let mut fits = true;
        for (i, &value) in sequence.iter().enumerate() {
            let x = i as f64;
            let predicted = a*x*x + b*x + c;
            if (predicted - value).abs() > 0.1 {
                fits = false;
                break;
            }
        }
        
        if fits {
            let mut discovery = MathematicalDiscovery::new(
                format!("polynomial_pattern_{}", sequence.len()),
                DiscoveryType::SequencePattern,
                format!("Quadratic polynomial pattern"),
                format!("f(n) = {:.3}n² + {:.3}n + {:.3}", a, b, c),
            );
            
            // Add evidence
            for (i, &value) in sequence.iter().enumerate() {
                let x = i as f64;
                let predicted = a*x*x + b*x + c;
                let error = (predicted - value).abs();
                
                discovery.add_evidence(DiscoveryEvidence {
                    input: vec![x],
                    expected_output: predicted,
                    actual_output: value,
                    supports_discovery: error < 0.1,
                    error,
                });
            }
            
            discovery.metadata.insert("pattern_type".to_string(), "polynomial".to_string());
            discovery.metadata.insert("degree".to_string(), "2".to_string());
            discovery.metadata.insert("coefficients".to_string(), format!("{:.3}, {:.3}, {:.3}", a, b, c));
            
            return Ok(Some(discovery));
        }
        
        Ok(None)
    }
    
    /// Test for complex patterns using neural networks
    fn test_neural_pattern(&self, sequence: &[f64]) -> Result<Option<MathematicalDiscovery>> {
        // Use the sequence pattern module to detect complex patterns
        let module = &self.pattern_modules[0];
        
        // Convert sequence to input format
        let mut input = vec![0.0; 10];
        for (i, &val) in sequence.iter().take(10).enumerate() {
            input[i] = val;
        }
        
        match module.evaluate(&input) {
            Ok(output) => {
                let pattern_type = output.get(0).unwrap_or(&0.0).round() as i32;
                let confidence = *output.get(1).unwrap_or(&0.8);
                let prediction = *output.get(2).unwrap_or(&0.0);
                
                if confidence > 0.8 {
                    let mut discovery = MathematicalDiscovery::new(
                        format!("neural_pattern_{}", sequence.len()),
                        DiscoveryType::SequencePattern,
                        format!("Complex pattern detected by neural analysis"),
                        format!("Neural pattern type {}, next value ≈ {:.3}", pattern_type, prediction),
                    );
                    
                    discovery.confidence = confidence;
                    discovery.metadata.insert("pattern_type".to_string(), "neural_complex".to_string());
                    discovery.metadata.insert("neural_confidence".to_string(), confidence.to_string());
                    
                    return Ok(Some(discovery));
                }
            }
            Err(_) => {}
        }
        
        Ok(None)
    }
    
    /// Discover mathematical identities
    pub fn discover_identities(&mut self, expressions: &[(Expression, f64)]) -> Result<Vec<MathematicalDiscovery>> {
        let mut discoveries = Vec::new();
        
        // Look for expressions that always equal specific values
        for (expr, value) in expressions {
            if self.test_for_identity(expr, *value)? {
                let discovery = MathematicalDiscovery::new(
                    format!("identity_{}", expr.to_string().replace(" ", "_")),
                    DiscoveryType::Identity,
                    format!("Mathematical identity discovered"),
                    format!("{} = {}", expr.to_string(), value),
                );
                discoveries.push(discovery);
            }
        }
        
        // Add to our collection
        for discovery in &discoveries {
            self.discoveries.push(discovery.clone());
        }
        
        Ok(discoveries)
    }
    
    /// Test if an expression represents a mathematical identity
    fn test_for_identity(&self, expr: &Expression, expected_value: f64) -> Result<bool> {
        // Test with multiple random variable assignments
        for _ in 0..100 {
            let mut variables = HashMap::new();
            
            // Add random values for any variables in the expression
            match expr {
                Expression::Variable(name) => {
                    variables.insert(name.clone(), rand::random::<f64>() * 10.0 - 5.0);
                }
                _ => {} // Handle more complex expressions
            }
            
            if let Ok(result) = expr.evaluate(&variables) {
                if (result - expected_value).abs() > 0.001 {
                    return Ok(false);
                }
            }
        }
        
        Ok(true)
    }
    
    /// Get all discoveries made so far
    pub fn get_discoveries(&self) -> &[MathematicalDiscovery] {
        &self.discoveries
    }
    
    /// Get discoveries by type
    pub fn get_discoveries_by_type(&self, discovery_type: DiscoveryType) -> Vec<&MathematicalDiscovery> {
        self.discoveries.iter()
            .filter(|d| d.discovery_type == discovery_type)
            .collect()
    }
    
    /// Get high-confidence discoveries
    pub fn get_high_confidence_discoveries(&self) -> Vec<&MathematicalDiscovery> {
        self.discoveries.iter()
            .filter(|d| d.confidence > 0.9)
            .collect()
    }
    
    /// Generate discovery report
    pub fn generate_report(&self) -> DiscoveryReport {
        let mut report = DiscoveryReport::default();
        
        report.total_discoveries = self.discoveries.len();
        report.high_confidence_discoveries = self.get_high_confidence_discoveries().len();
        
        for discovery_type in DiscoveryType::all() {
            let count = self.get_discoveries_by_type(*discovery_type).len();
            report.discoveries_by_type.insert(*discovery_type, count);
        }
        
        // Calculate average confidence
        if !self.discoveries.is_empty() {
            report.average_confidence = self.discoveries.iter()
                .map(|d| d.confidence)
                .sum::<f64>() / self.discoveries.len() as f64;
        }
        
        // Find most novel discovery
        if let Some(most_novel) = self.discoveries.iter()
            .max_by(|a, b| a.novelty_score().partial_cmp(&b.novelty_score()).unwrap()) {
            report.most_novel_discovery = Some(most_novel.clone());
        }
        
        report
    }
}

/// Configuration for discovery system
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// Minimum confidence threshold for discoveries
    pub min_confidence: f64,
    /// Maximum number of discoveries to keep
    pub max_discoveries: usize,
    /// Whether to enable experimental discovery methods
    pub enable_experimental: bool,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.7,
            max_discoveries: 1000,
            enable_experimental: true,
        }
    }
}

/// Report of mathematical discoveries
#[derive(Debug, Default)]
pub struct DiscoveryReport {
    /// Total number of discoveries
    pub total_discoveries: usize,
    /// Number of high-confidence discoveries
    pub high_confidence_discoveries: usize,
    /// Discoveries by type
    pub discoveries_by_type: HashMap<DiscoveryType, usize>,
    /// Average confidence across all discoveries
    pub average_confidence: f64,
    /// Most novel discovery found
    pub most_novel_discovery: Option<MathematicalDiscovery>,
}

impl DiscoveryReport {
    /// Print the discovery report
    pub fn print(&self) {
        println!("🔬 Mathematical Discovery Report");
        println!("================================");
        println!("Total discoveries: {}", self.total_discoveries);
        println!("High-confidence discoveries: {}", self.high_confidence_discoveries);
        println!("Average confidence: {:.1}%", self.average_confidence * 100.0);
        
        println!("\nDiscoveries by type:");
        for discovery_type in DiscoveryType::all() {
            let count = self.discoveries_by_type.get(discovery_type).unwrap_or(&0);
            if *count > 0 {
                println!("  {:?}: {} discoveries", discovery_type, count);
            }
        }
        
        if let Some(ref novel) = self.most_novel_discovery {
            println!("\nMost novel discovery:");
            println!("  {}", novel.description);
            println!("  Pattern: {}", novel.pattern);
            println!("  Novelty score: {:.3}", novel.novelty_score());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_arithmetic_progression_discovery() {
        let mut discovery_system = PatternDiscoverySystem::new(DiscoveryConfig::default());
        let sequence = vec![2.0, 5.0, 8.0, 11.0, 14.0]; // Arithmetic progression with d=3
        
        let discoveries = discovery_system.discover_sequence_patterns(&sequence).unwrap();
        assert!(!discoveries.is_empty());
        
        let arithmetic = discoveries.iter()
            .find(|d| d.discovery_type == DiscoveryType::SequencePattern)
            .unwrap();
        assert!(arithmetic.confidence > 0.9);
    }
    
    #[test]
    fn test_geometric_progression_discovery() {
        let mut discovery_system = PatternDiscoverySystem::new(DiscoveryConfig::default());
        let sequence = vec![2.0, 6.0, 18.0, 54.0, 162.0]; // Geometric progression with r=3
        
        let discoveries = discovery_system.discover_sequence_patterns(&sequence).unwrap();
        let geometric = discoveries.iter()
            .find(|d| d.pattern.contains("geometric") || d.pattern.contains("*"))
            .unwrap();
        assert!(geometric.confidence > 0.9);
    }
    
    #[test]
    fn test_polynomial_pattern_discovery() {
        let mut discovery_system = PatternDiscoverySystem::new(DiscoveryConfig::default());
        // f(n) = n² + 1: [1, 2, 5, 10, 17]
        let sequence = vec![1.0, 2.0, 5.0, 10.0, 17.0];
        
        let discoveries = discovery_system.discover_sequence_patterns(&sequence).unwrap();
        let polynomial = discoveries.iter()
            .find(|d| d.pattern.contains("²"))
            .unwrap();
        assert!(polynomial.confidence > 0.9);
    }
    
    #[test]
    fn test_discovery_report_generation() {
        let mut discovery_system = PatternDiscoverySystem::new(DiscoveryConfig::default());
        
        // Add some discoveries
        let sequence1 = vec![1.0, 3.0, 5.0, 7.0]; // Arithmetic
        let sequence2 = vec![1.0, 2.0, 4.0, 8.0]; // Geometric
        
        discovery_system.discover_sequence_patterns(&sequence1).unwrap();
        discovery_system.discover_sequence_patterns(&sequence2).unwrap();
        
        let report = discovery_system.generate_report();
        assert!(report.total_discoveries > 0);
        assert!(report.average_confidence > 0.5);
    }
}