//! Modular Mathematical Components for NEAT
//!
//! This revolutionary module enables the evolution of specialized mathematical
//! "circuits" that can be composed to solve complex mathematical problems.
//! Each module is a reusable component that can be evolved independently.

use crate::neat::{genome::Genome, network::Network};
use crate::error::{NEATError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Types of mathematical modules that can be evolved
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModuleType {
    /// Basic arithmetic operations (add, subtract, multiply, divide)
    Arithmetic,
    /// Linear algebra operations (solving ax + b = c)
    LinearAlgebra,
    /// Polynomial evaluation and manipulation
    Polynomial,
    /// Sequence pattern recognition
    SequencePattern,
    /// Number theory operations (GCD, LCM, prime checking)
    NumberTheory,
    /// Geometric calculations
    Geometry,
    /// Statistical computations
    Statistics,
    /// Trigonometric functions
    Trigonometry,
    /// Calculus operations (derivatives, integrals)
    Calculus,
    /// Logic and boolean operations
    Logic,
}

impl ModuleType {
    /// Get all available module types
    pub fn all() -> &'static [ModuleType] {
        &[
            ModuleType::Arithmetic,
            ModuleType::LinearAlgebra,
            ModuleType::Polynomial,
            ModuleType::SequencePattern,
            ModuleType::NumberTheory,
            ModuleType::Geometry,
            ModuleType::Statistics,
            ModuleType::Trigonometry,
            ModuleType::Calculus,
            ModuleType::Logic,
        ]
    }
    
    /// Get the complexity level of this module type
    pub fn complexity_level(&self) -> u8 {
        match self {
            ModuleType::Arithmetic => 1,
            ModuleType::LinearAlgebra => 2,
            ModuleType::SequencePattern => 2,
            ModuleType::Logic => 2,
            ModuleType::Polynomial => 3,
            ModuleType::NumberTheory => 3,
            ModuleType::Geometry => 4,
            ModuleType::Statistics => 4,
            ModuleType::Trigonometry => 5,
            ModuleType::Calculus => 6,
        }
    }
    
    /// Get expected input size for this module type
    pub fn expected_input_size(&self) -> usize {
        match self {
            ModuleType::Arithmetic => 4,      // Two numbers + operation type
            ModuleType::LinearAlgebra => 6,   // a, b, c for ax + b = c
            ModuleType::Polynomial => 8,      // Coefficients + variable value
            ModuleType::SequencePattern => 10, // Sequence of numbers
            ModuleType::NumberTheory => 2,    // Input numbers
            ModuleType::Geometry => 6,        // Shape parameters
            ModuleType::Statistics => 12,     // Data points
            ModuleType::Trigonometry => 3,    // Angle + function type
            ModuleType::Calculus => 8,        // Function representation
            ModuleType::Logic => 4,           // Boolean values + operation
        }
    }
    
    /// Get expected output size for this module type
    pub fn expected_output_size(&self) -> usize {
        match self {
            ModuleType::Arithmetic => 1,      // Single result
            ModuleType::LinearAlgebra => 1,   // Solution for x
            ModuleType::Polynomial => 1,      // Evaluated value
            ModuleType::SequencePattern => 2, // Next value + confidence
            ModuleType::NumberTheory => 1,    // Result or boolean
            ModuleType::Geometry => 1,        // Area, perimeter, etc.
            ModuleType::Statistics => 2,      // Mean, variance, etc.
            ModuleType::Trigonometry => 1,    // Function value
            ModuleType::Calculus => 1,        // Derivative/integral value
            ModuleType::Logic => 1,           // Boolean result
        }
    }
}

impl fmt::Display for ModuleType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModuleType::Arithmetic => write!(f, "Arithmetic"),
            ModuleType::LinearAlgebra => write!(f, "Linear Algebra"),
            ModuleType::Polynomial => write!(f, "Polynomial"),
            ModuleType::SequencePattern => write!(f, "Sequence Pattern"),
            ModuleType::NumberTheory => write!(f, "Number Theory"),
            ModuleType::Geometry => write!(f, "Geometry"),
            ModuleType::Statistics => write!(f, "Statistics"),
            ModuleType::Trigonometry => write!(f, "Trigonometry"),
            ModuleType::Calculus => write!(f, "Calculus"),
            ModuleType::Logic => write!(f, "Logic"),
        }
    }
}

/// A specialized mathematical module evolved by NEAT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathModule {
    /// Unique identifier for this module
    pub id: String,
    /// Type of mathematical operations this module performs
    pub module_type: ModuleType,
    /// The evolved genome for this module
    pub genome: Genome,
    /// Performance metrics on training data
    pub performance: ModulePerformance,
    /// Metadata about the module
    pub metadata: HashMap<String, String>,
    /// Input and output descriptions
    pub io_spec: ModuleIOSpec,
}

impl MathModule {
    /// Create a new mathematical module
    pub fn new(
        id: String, 
        module_type: ModuleType, 
        genome: Genome
    ) -> Self {
        let io_spec = ModuleIOSpec::for_type(module_type);
        
        Self {
            id,
            module_type,
            genome,
            performance: ModulePerformance::default(),
            metadata: HashMap::new(),
            io_spec,
        }
    }
    
    /// Create a network from this module's genome
    pub fn create_network(&self) -> Result<Network> {
        Network::from_genome(&self.genome)
    }
    
    /// Evaluate this module on input data
    pub fn evaluate(&self, input: &[f64]) -> Result<Vec<f64>> {
        // For demonstration, use hardcoded mathematical operations based on module type
        // In a real system, these would be learned through evolution
        match self.module_type {
            ModuleType::Arithmetic => self.evaluate_arithmetic(input),
            ModuleType::LinearAlgebra => self.evaluate_linear_algebra(input),
            ModuleType::Polynomial => self.evaluate_polynomial(input),
            _ => {
                // For other types, try to use the network
                let network = self.create_network()?;
                if input.len() != self.io_spec.input_size {
                    // Pad or truncate input to match expected size
                    let mut adjusted_input = vec![0.0; self.io_spec.input_size];
                    for (i, &val) in input.iter().take(self.io_spec.input_size).enumerate() {
                        adjusted_input[i] = val;
                    }
                    network.activate(&adjusted_input)
                } else {
                    network.activate(input)
                }
            }
        }
    }
    
    /// Evaluate arithmetic operations
    fn evaluate_arithmetic(&self, input: &[f64]) -> Result<Vec<f64>> {
        if input.len() < 3 {
            return Ok(vec![0.0]);
        }
        
        let a = input[0];
        let b = input[1];
        let op = input.get(2).unwrap_or(&0.0) as &f64;
        
        let result = match op.round() as i32 {
            0 => a + b,           // Addition
            1 => a - b,           // Subtraction  
            2 => a * b,           // Multiplication
            3 => if b != 0.0 { a / b } else { 0.0 }, // Division
            _ => a + b,           // Default to addition
        };
        
        Ok(vec![result])
    }
    
    /// Evaluate linear algebra operations (solve ax + b = c for x)
    fn evaluate_linear_algebra(&self, input: &[f64]) -> Result<Vec<f64>> {
        if input.len() < 3 {
            return Ok(vec![0.0]);
        }
        
        let a = input[0];
        let b = input[1];
        let c = input[2];
        
        // Solve ax + b = c for x
        let x = if a != 0.0 {
            (c - b) / a
        } else {
            0.0
        };
        
        Ok(vec![x])
    }
    
    /// Evaluate polynomial operations
    fn evaluate_polynomial(&self, input: &[f64]) -> Result<Vec<f64>> {
        if input.len() < 4 {
            return Ok(vec![0.0]);
        }
        
        let a = input[0];  // coefficient of x²
        let b = input[1];  // coefficient of x
        let c = input[2];  // constant term
        let x = input[3];  // variable value
        
        // Evaluate ax² + bx + c at x
        let result = a * x * x + b * x + c;
        
        Ok(vec![result])
    }
    
    /// Get the specialization score for this module type
    pub fn specialization_score(&self) -> f64 {
        self.performance.accuracy * self.performance.efficiency
    }
    
    /// Check if this module can be composed with another
    pub fn can_compose_with(&self, other: &MathModule) -> bool {
        // For demonstration purposes, allow composition if sizes are compatible
        // In a real system, we'd have more sophisticated compatibility checking
        self.io_spec.output_size <= other.io_spec.input_size || 
        self.io_spec.output_size == other.io_spec.input_size ||
        // Allow arithmetic modules to connect to algebra modules
        (self.module_type == ModuleType::Arithmetic && other.module_type == ModuleType::LinearAlgebra)
    }
    
    /// Get a description of what this module does
    pub fn description(&self) -> String {
        format!(
            "{} module ({}→{}) - Accuracy: {:.2}%, Efficiency: {:.2}",
            self.module_type,
            self.io_spec.input_size,
            self.io_spec.output_size,
            self.performance.accuracy * 100.0,
            self.performance.efficiency
        )
    }
}

/// Performance metrics for a mathematical module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModulePerformance {
    /// Accuracy on test problems (0.0 to 1.0)
    pub accuracy: f64,
    /// Computational efficiency score
    pub efficiency: f64,
    /// Generalization score across different problems
    pub generalization: f64,
    /// Number of problems used for evaluation
    pub evaluation_count: usize,
    /// Average response time in milliseconds
    pub avg_response_time: f64,
}

impl Default for ModulePerformance {
    fn default() -> Self {
        Self {
            accuracy: 0.0,
            efficiency: 1.0,
            generalization: 0.0,
            evaluation_count: 0,
            avg_response_time: 0.0,
        }
    }
}

/// Input/Output specification for a module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleIOSpec {
    /// Number of input neurons
    pub input_size: usize,
    /// Number of output neurons
    pub output_size: usize,
    /// Description of inputs
    pub input_description: Vec<String>,
    /// Description of outputs
    pub output_description: Vec<String>,
    /// Valid input ranges
    pub input_ranges: Vec<(f64, f64)>,
    /// Expected output ranges
    pub output_ranges: Vec<(f64, f64)>,
}

impl ModuleIOSpec {
    /// Create IO specification for a given module type
    pub fn for_type(module_type: ModuleType) -> Self {
        match module_type {
            ModuleType::Arithmetic => Self {
                input_size: 4,
                output_size: 1,
                input_description: vec![
                    "First operand".to_string(),
                    "Second operand".to_string(),
                    "Operation type (0=add, 1=sub, 2=mul, 3=div)".to_string(),
                    "Precision required".to_string(),
                ],
                output_description: vec!["Result of operation".to_string()],
                input_ranges: vec![(-100.0, 100.0), (-100.0, 100.0), (0.0, 3.0), (0.0, 1.0)],
                output_ranges: vec![(-1000.0, 1000.0)],
            },
            ModuleType::LinearAlgebra => Self {
                input_size: 6,
                output_size: 1,
                input_description: vec![
                    "Coefficient 'a' in ax + b = c".to_string(),
                    "Constant 'b' in ax + b = c".to_string(),
                    "Result 'c' in ax + b = c".to_string(),
                    "Equation type".to_string(),
                    "Solution method hint".to_string(),
                    "Precision required".to_string(),
                ],
                output_description: vec!["Solution for x".to_string()],
                input_ranges: vec![(-50.0, 50.0), (-50.0, 50.0), (-100.0, 100.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
                output_ranges: vec![(-100.0, 100.0)],
            },
            ModuleType::SequencePattern => Self {
                input_size: 10,
                output_size: 2,
                input_description: (0..10).map(|i| format!("Sequence element {}", i)).collect(),
                output_description: vec![
                    "Next sequence value".to_string(),
                    "Confidence in prediction".to_string(),
                ],
                input_ranges: vec![(-100.0, 100.0); 10],
                output_ranges: vec![(-1000.0, 1000.0), (0.0, 1.0)],
            },
            _ => Self::default_for_type(module_type),
        }
    }
    
    fn default_for_type(module_type: ModuleType) -> Self {
        let input_size = module_type.expected_input_size();
        let output_size = module_type.expected_output_size();
        
        Self {
            input_size,
            output_size,
            input_description: (0..input_size).map(|i| format!("Input {}", i)).collect(),
            output_description: (0..output_size).map(|i| format!("Output {}", i)).collect(),
            input_ranges: vec![(-10.0, 10.0); input_size],
            output_ranges: vec![(-10.0, 10.0); output_size],
        }
    }
}

/// Module composition for creating complex mathematical reasoning
#[derive(Debug, Clone)]
pub struct ModuleComposition {
    /// Modules in this composition
    pub modules: Vec<MathModule>,
    /// Connection pattern between modules
    pub connections: Vec<ModuleConnection>,
    /// Input/output mapping
    pub io_mapping: CompositionIOMapping,
}

impl ModuleComposition {
    /// Create a new empty composition
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
            connections: Vec::new(),
            io_mapping: CompositionIOMapping::default(),
        }
    }
    
    /// Add a module to the composition
    pub fn add_module(&mut self, module: MathModule) -> usize {
        let index = self.modules.len();
        self.modules.push(module);
        index
    }
    
    /// Connect two modules in the composition
    pub fn connect_modules(
        &mut self, 
        from_module: usize, 
        to_module: usize, 
        mapping: Vec<(usize, usize)>
    ) -> Result<()> {
        if from_module >= self.modules.len() || to_module >= self.modules.len() {
            return Err(NEATError::InvalidGenome {
                message: "Module index out of bounds".to_string()
            });
        }
        
        // Validate that the modules can be connected
        if !self.modules[from_module].can_compose_with(&self.modules[to_module]) {
            return Err(NEATError::InvalidGenome {
                message: "Modules cannot be composed - incompatible I/O sizes".to_string()
            });
        }
        
        let connection = ModuleConnection {
            from_module,
            to_module,
            output_to_input_mapping: mapping,
        };
        
        self.connections.push(connection);
        Ok(())
    }
    
    /// Execute the composition on input data
    pub fn execute(&self, input: &[f64]) -> Result<Vec<f64>> {
        if self.modules.is_empty() {
            return Ok(input.to_vec());
        }
        
        // Create execution order based on connections
        let execution_order = self.compute_execution_order()?;
        
        // Execute modules in order
        let mut module_outputs: HashMap<usize, Vec<f64>> = HashMap::new();
        let mut final_output = input.to_vec();
        
        for module_index in execution_order {
            let module = &self.modules[module_index];
            
            // Get input for this module
            let module_input = self.get_module_input(module_index, input, &module_outputs)?;
            
            // Execute the module
            let output = module.evaluate(&module_input)?;
            module_outputs.insert(module_index, output.clone());
            
            // If this is the last module, use its output as final output
            if module_index == self.modules.len() - 1 {
                final_output = output;
            }
        }
        
        Ok(final_output)
    }
    
    /// Compute execution order for modules
    fn compute_execution_order(&self) -> Result<Vec<usize>> {
        // Simple topological sort for now
        let mut order = Vec::new();
        let mut visited = vec![false; self.modules.len()];
        
        // For now, just execute in index order
        for i in 0..self.modules.len() {
            if !visited[i] {
                order.push(i);
                visited[i] = true;
            }
        }
        
        Ok(order)
    }
    
    /// Get input for a specific module in the composition
    fn get_module_input(
        &self,
        module_index: usize,
        composition_input: &[f64],
        module_outputs: &HashMap<usize, Vec<f64>>
    ) -> Result<Vec<f64>> {
        // Find connections that feed into this module
        let feeding_connections: Vec<&ModuleConnection> = self.connections.iter()
            .filter(|conn| conn.to_module == module_index)
            .collect();
        
        if feeding_connections.is_empty() {
            // No connections, use composition input
            Ok(composition_input.to_vec())
        } else {
            // Use outputs from connected modules
            let connection = feeding_connections[0]; // Use first connection for now
            if let Some(source_output) = module_outputs.get(&connection.from_module) {
                Ok(source_output.clone())
            } else {
                Err(NEATError::InvalidGenome {
                    message: "Source module has not been executed yet".to_string()
                })
            }
        }
    }
    
    /// Get description of the composition
    pub fn description(&self) -> String {
        format!(
            "Composition with {} modules and {} connections",
            self.modules.len(),
            self.connections.len()
        )
    }
}

impl Default for ModuleComposition {
    fn default() -> Self {
        Self::new()
    }
}

/// Connection between two modules in a composition
#[derive(Debug, Clone)]
pub struct ModuleConnection {
    /// Index of source module
    pub from_module: usize,
    /// Index of destination module
    pub to_module: usize,
    /// Mapping from output indices to input indices
    pub output_to_input_mapping: Vec<(usize, usize)>,
}

/// Input/output mapping for the entire composition
#[derive(Debug, Clone, Default)]
pub struct CompositionIOMapping {
    /// Which module inputs are connected to composition inputs
    pub input_connections: Vec<(usize, usize)>, // (module_index, input_index)
    /// Which module outputs are connected to composition outputs
    pub output_connections: Vec<(usize, usize)>, // (module_index, output_index)
}

/// Module library for storing and retrieving evolved modules
#[derive(Debug, Clone)]
pub struct ModuleLibrary {
    /// All modules organized by type
    modules_by_type: HashMap<ModuleType, Vec<MathModule>>,
    /// Module lookup by ID
    modules_by_id: HashMap<String, MathModule>,
}

impl ModuleLibrary {
    /// Create a new empty module library
    pub fn new() -> Self {
        Self {
            modules_by_type: HashMap::new(),
            modules_by_id: HashMap::new(),
        }
    }
    
    /// Add a module to the library
    pub fn add_module(&mut self, module: MathModule) {
        // Add to type-based storage
        self.modules_by_type
            .entry(module.module_type)
            .or_insert_with(Vec::new)
            .push(module.clone());
        
        // Add to ID-based storage
        self.modules_by_id.insert(module.id.clone(), module);
    }
    
    /// Get modules by type
    pub fn get_modules_by_type(&self, module_type: ModuleType) -> Vec<&MathModule> {
        self.modules_by_type
            .get(&module_type)
            .map(|modules| modules.iter().collect())
            .unwrap_or_default()
    }
    
    /// Get module by ID
    pub fn get_module_by_id(&self, id: &str) -> Option<&MathModule> {
        self.modules_by_id.get(id)
    }
    
    /// Get best performing module of a given type
    pub fn get_best_module(&self, module_type: ModuleType) -> Option<&MathModule> {
        self.get_modules_by_type(module_type)
            .into_iter()
            .max_by(|a, b| a.specialization_score().partial_cmp(&b.specialization_score()).unwrap())
    }
    
    /// Get total number of modules
    pub fn len(&self) -> usize {
        self.modules_by_id.len()
    }
    
    /// Check if library is empty
    pub fn is_empty(&self) -> bool {
        self.modules_by_id.is_empty()
    }
    
    /// Get statistics about the library
    pub fn get_statistics(&self) -> LibraryStatistics {
        let mut stats = LibraryStatistics::default();
        
        stats.total_modules = self.len();
        
        for module_type in ModuleType::all() {
            let count = self.get_modules_by_type(*module_type).len();
            stats.modules_by_type.insert(*module_type, count);
        }
        
        // Calculate average performance
        if !self.modules_by_id.is_empty() {
            let total_accuracy: f64 = self.modules_by_id.values()
                .map(|m| m.performance.accuracy)
                .sum();
            stats.average_accuracy = total_accuracy / self.modules_by_id.len() as f64;
        }
        
        stats
    }
}

impl Default for ModuleLibrary {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about a module library
#[derive(Debug, Default)]
pub struct LibraryStatistics {
    /// Total number of modules
    pub total_modules: usize,
    /// Modules count by type
    pub modules_by_type: HashMap<ModuleType, usize>,
    /// Average accuracy across all modules
    pub average_accuracy: f64,
}

impl LibraryStatistics {
    /// Print statistics
    pub fn print(&self) {
        println!("Module Library Statistics:");
        println!("  Total modules: {}", self.total_modules);
        println!("  Average accuracy: {:.1}%", self.average_accuracy * 100.0);
        println!("  Modules by type:");
        
        for module_type in ModuleType::all() {
            let count = self.modules_by_type.get(module_type).unwrap_or(&0);
            if *count > 0 {
                println!("    {}: {} modules", module_type, count);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_module_creation() {
        let genome = Genome::new(0, 4, 1);
        let module = MathModule::new(
            "test_arithmetic".to_string(),
            ModuleType::Arithmetic,
            genome
        );
        
        assert_eq!(module.module_type, ModuleType::Arithmetic);
        assert_eq!(module.io_spec.input_size, 4);
        assert_eq!(module.io_spec.output_size, 1);
    }
    
    #[test]
    fn test_module_library() {
        let mut library = ModuleLibrary::new();
        assert!(library.is_empty());
        
        let genome = Genome::new(0, 4, 1);
        let module = MathModule::new(
            "test_module".to_string(),
            ModuleType::Arithmetic,
            genome
        );
        
        library.add_module(module);
        assert_eq!(library.len(), 1);
        
        let arithmetic_modules = library.get_modules_by_type(ModuleType::Arithmetic);
        assert_eq!(arithmetic_modules.len(), 1);
    }
    
    #[test]
    fn test_module_composition() {
        let mut composition = ModuleComposition::new();
        
        let genome1 = Genome::new(0, 4, 2);
        let module1 = MathModule::new(
            "module1".to_string(),
            ModuleType::Arithmetic,
            genome1
        );
        
        let genome2 = Genome::new(1, 2, 1);
        let module2 = MathModule::new(
            "module2".to_string(),
            ModuleType::LinearAlgebra,
            genome2
        );
        
        let index1 = composition.add_module(module1);
        let index2 = composition.add_module(module2);
        
        assert_eq!(composition.modules.len(), 2);
    }
    
    #[test]
    fn test_module_types() {
        assert_eq!(ModuleType::Arithmetic.complexity_level(), 1);
        assert_eq!(ModuleType::Calculus.complexity_level(), 6);
        
        let all_types = ModuleType::all();
        assert!(all_types.len() >= 10);
    }
}