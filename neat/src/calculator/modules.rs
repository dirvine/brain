//! Modular Mathematical Components for NEAT
//!
//! This module enables the evolution of specialized mathematical
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
    /// Discrete mathematics (combinatorics, graph theory)
    DiscreteMath,
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
            ModuleType::DiscreteMath,
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
            ModuleType::DiscreteMath => 7,
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
            ModuleType::DiscreteMath => 6,    // Operation type + parameters
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
            ModuleType::DiscreteMath => 1,    // Combinatorial/graph result
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
            ModuleType::DiscreteMath => write!(f, "Discrete Math"),
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
        // Adjust input to match expected size
        let adjusted_input = self.adjust_input_size(input);
        
        // For demonstration, use hardcoded mathematical operations based on module type
        // In a real system, these would be learned through evolution
        match self.module_type {
            ModuleType::Arithmetic => self.evaluate_arithmetic(&adjusted_input),
            ModuleType::LinearAlgebra => self.evaluate_linear_algebra(&adjusted_input),
            ModuleType::Polynomial => self.evaluate_polynomial(&adjusted_input),
            ModuleType::Calculus => self.evaluate_calculus(&adjusted_input),
            ModuleType::Trigonometry => self.evaluate_trigonometry(&adjusted_input),
            ModuleType::Statistics => self.evaluate_statistics(&adjusted_input),
            ModuleType::DiscreteMath => self.evaluate_discrete_math(&adjusted_input),
            ModuleType::NumberTheory => self.evaluate_number_theory(&adjusted_input),
            ModuleType::Geometry => self.evaluate_geometry(&adjusted_input),
            _ => {
                // For other types, provide simple fallback operations
                self.evaluate_fallback(&adjusted_input)
            }
        }
    }
    
    /// Adjust input size to match module requirements
    fn adjust_input_size(&self, input: &[f64]) -> Vec<f64> {
        let required_size = self.io_spec.input_size;
        let mut adjusted = vec![0.0; required_size];
        
        // Copy available inputs, padding with zeros if needed
        for (i, &val) in input.iter().take(required_size).enumerate() {
            adjusted[i] = val;
        }
        
        adjusted
    }
    
    /// Fallback evaluation for module types without specific implementations
    fn evaluate_fallback(&self, input: &[f64]) -> Result<Vec<f64>> {
        // Simple fallback: sum inputs and return expected output size
        let sum = input.iter().sum::<f64>();
        let result = match self.module_type {
            ModuleType::SequencePattern => vec![sum, 0.8], // Next value + confidence
            ModuleType::Statistics => vec![sum / input.len() as f64, sum], // Mean + sum
            _ => vec![sum], // Default to single output
        };
        
        // Ensure output size matches specification
        let mut output = vec![0.0; self.io_spec.output_size];
        for (i, &val) in result.iter().take(self.io_spec.output_size).enumerate() {
            output[i] = val;
        }
        
        Ok(output)
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
    
    /// Evaluate calculus operations
    fn evaluate_calculus(&self, input: &[f64]) -> Result<Vec<f64>> {
        if input.len() < 4 {
            return Ok(vec![0.0]);
        }
        
        let operation = input[0].round() as i32;
        let a = input[1];  // coefficient
        let b = input[2];  // power/parameter
        let x = input[3];  // evaluation point
        
        let result = match operation {
            0 => {
                // Derivative: d/dx(ax^b) = a*b*x^(b-1)
                if b == 0.0 {
                    0.0
                } else if b == 1.0 {
                    a
                } else {
                    a * b * x.powf(b - 1.0)
                }
            },
            1 => {
                // Simple integration: ∫ax^b dx = a*x^(b+1)/(b+1)
                if b == -1.0 {
                    a * x.ln()
                } else {
                    a * x.powf(b + 1.0) / (b + 1.0)
                }
            },
            2 => {
                // Function evaluation: f(x) = ax^b
                a * x.powf(b)
            },
            _ => {
                // Default: simple polynomial evaluation
                a * x + b
            }
        };
        
        Ok(vec![result])
    }
    
    /// Evaluate trigonometry operations
    fn evaluate_trigonometry(&self, input: &[f64]) -> Result<Vec<f64>> {
        if input.len() < 2 {
            return Ok(vec![0.0]);
        }
        
        let function_type = input[0].round() as i32;
        let angle = input[1];
        let amplitude = input.get(2).unwrap_or(&1.0);
        
        let result = match function_type {
            0 => amplitude * angle.sin(),        // sine
            1 => amplitude * angle.cos(),        // cosine
            2 => amplitude * angle.tan(),        // tangent
            3 => {                               // arcsine
                if angle.abs() <= 1.0 {
                    amplitude * angle.asin()
                } else {
                    0.0
                }
            },
            4 => {                               // arccosine
                if angle.abs() <= 1.0 {
                    amplitude * angle.acos()
                } else {
                    0.0
                }
            },
            5 => amplitude * angle.atan(),       // arctangent
            6 => amplitude * angle.sinh(),       // hyperbolic sine
            7 => amplitude * angle.cosh(),       // hyperbolic cosine
            8 => amplitude * angle.tanh(),       // hyperbolic tangent
            _ => amplitude * angle.sin(),        // default to sine
        };
        
        Ok(vec![result])
    }
    
    /// Evaluate statistics operations
    fn evaluate_statistics(&self, input: &[f64]) -> Result<Vec<f64>> {
        if input.is_empty() {
            return Ok(vec![0.0, 0.0]);
        }
        
        let operation = input[0].round() as i32;
        let data = &input[1..];
        
        if data.is_empty() {
            return Ok(vec![0.0, 0.0]);
        }
        
        let (stat1, stat2) = match operation {
            0 => {
                // Mean and standard deviation
                let mean = data.iter().sum::<f64>() / data.len() as f64;
                let variance = data.iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>() / (data.len() - 1).max(1) as f64;
                (mean, variance.sqrt())
            },
            1 => {
                // Min and max
                let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                (min, max)
            },
            2 => {
                // Median and IQR
                let mut sorted = data.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let n = sorted.len();
                let median = if n % 2 == 0 {
                    (sorted[n/2 - 1] + sorted[n/2]) / 2.0
                } else {
                    sorted[n/2]
                };
                let q1_idx = n / 4;
                let q3_idx = 3 * n / 4;
                let iqr = sorted[q3_idx] - sorted[q1_idx];
                (median, iqr)
            },
            _ => {
                // Default: sum and count
                let sum = data.iter().sum::<f64>();
                let count = data.len() as f64;
                (sum, count)
            }
        };
        
        Ok(vec![stat1, stat2])
    }
    
    /// Evaluate number theory operations
    fn evaluate_number_theory(&self, input: &[f64]) -> Result<Vec<f64>> {
        if input.len() < 2 {
            return Ok(vec![0.0]);
        }
        
        let operation = input[0].round() as i32;
        let a = input[1].abs() as u64;
        let b = input.get(2).unwrap_or(&0.0).abs() as u64;
        
        let result = match operation {
            0 => {
                // GCD (Greatest Common Divisor)
                self.gcd(a, b) as f64
            },
            1 => {
                // LCM (Least Common Multiple)
                if a == 0 || b == 0 {
                    0.0
                } else {
                    (a * b / self.gcd(a, b)) as f64
                }
            },
            2 => {
                // Prime check (1 if prime, 0 if not)
                if self.is_prime(a) { 1.0 } else { 0.0 }
            },
            3 => {
                // Factorial
                if a <= 20 { // Prevent overflow
                    self.factorial(a) as f64
                } else {
                    0.0
                }
            },
            4 => {
                // Fibonacci
                if a <= 93 { // Prevent overflow for u64
                    self.fibonacci(a) as f64
                } else {
                    0.0
                }
            },
            _ => {
                // Default: modulo operation
                if b > 0 {
                    (a % b) as f64
                } else {
                    a as f64
                }
            }
        };
        
        Ok(vec![result])
    }
    
    /// Evaluate geometry operations
    fn evaluate_geometry(&self, input: &[f64]) -> Result<Vec<f64>> {
        if input.len() < 3 {
            return Ok(vec![0.0]);
        }
        
        let shape_type = input[0].round() as i32;
        let param1 = input[1];
        let param2 = input[2];
        
        let result = match shape_type {
            0 => {
                // Circle: area = π * r²
                std::f64::consts::PI * param1 * param1
            },
            1 => {
                // Rectangle: area = width * height
                param1 * param2
            },
            2 => {
                // Triangle: area = 0.5 * base * height
                0.5 * param1 * param2
            },
            3 => {
                // Circle: circumference = 2 * π * r
                2.0 * std::f64::consts::PI * param1
            },
            4 => {
                // Rectangle: perimeter = 2 * (width + height)
                2.0 * (param1 + param2)
            },
            5 => {
                // Pythagorean theorem: c = √(a² + b²)
                (param1 * param1 + param2 * param2).sqrt()
            },
            _ => {
                // Default: distance between two points (assuming 2D)
                param1.hypot(param2)
            }
        };
        
        Ok(vec![result])
    }
    
    /// Evaluate discrete mathematics operations
    fn evaluate_discrete_math(&self, input: &[f64]) -> Result<Vec<f64>> {
        if input.len() < 2 {
            return Ok(vec![0.0]);
        }
        
        let operation = input[0].round() as i32;
        let n = input[1].abs() as u64;
        let r = input.get(2).unwrap_or(&0.0).abs() as u64;
        
        let result = match operation {
            0 => {
                // Factorial
                if n <= 20 { // Prevent overflow
                    self.factorial(n) as f64
                } else {
                    0.0
                }
            },
            1 => {
                // Permutation P(n,r)
                if r <= n && n <= 20 {
                    let mut perm = 1u64;
                    for i in 0..r {
                        perm *= n - i;
                    }
                    perm as f64
                } else {
                    0.0
                }
            },
            2 => {
                // Combination C(n,r)
                if r <= n && n <= 20 {
                    let r = r.min(n - r); // Use symmetry
                    let mut comb = 1u64;
                    for i in 0..r {
                        comb = comb * (n - i) / (i + 1);
                    }
                    comb as f64
                } else {
                    0.0
                }
            },
            3 => {
                // Catalan number C_n = C(2n,n)/(n+1)
                if n <= 10 { // Prevent overflow
                    let two_n = 2 * n;
                    let mut catalan = 1u64;
                    for i in 0..n {
                        catalan = catalan * (two_n - i) / (i + 1);
                    }
                    (catalan / (n + 1)) as f64
                } else {
                    0.0
                }
            },
            4 => {
                // Set cardinality (simple union)
                // Treat as |A ∪ B| = |A| + |B| - |A ∩ B|
                // Use input parameters as set sizes
                let set_a_size = n as f64;
                let set_b_size = r as f64;
                let intersection_size = input.get(3).unwrap_or(&0.0).abs();
                set_a_size + set_b_size - intersection_size
            },
            5 => {
                // Graph connectivity (simplified)
                // Treat as maximum number of edges in complete graph
                if n > 0 {
                    (n * (n - 1) / 2) as f64
                } else {
                    0.0
                }
            },
            6 => {
                // Modular arithmetic: a mod m
                let a = n;
                let m = r.max(1); // Prevent division by zero
                (a % m) as f64
            },
            _ => {
                // Default: simple counting operation
                n as f64
            }
        };
        
        Ok(vec![result])
    }
    
    // Helper methods for number theory
    fn gcd(&self, mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }
    
    fn is_prime(&self, n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }
        
        let sqrt_n = (n as f64).sqrt() as u64;
        for i in (3..=sqrt_n).step_by(2) {
            if n % i == 0 {
                return false;
            }
        }
        true
    }
    
    fn factorial(&self, n: u64) -> u64 {
        if n <= 1 {
            1
        } else {
            (1..=n).product()
        }
    }
    
    fn fibonacci(&self, n: u64) -> u64 {
        if n <= 1 {
            n
        } else {
            let mut prev = 0;
            let mut curr = 1;
            for _ in 2..=n {
                let next = prev + curr;
                prev = curr;
                curr = next;
            }
            curr
        }
    }
    
    /// Get the specialization score for this module type
    pub fn specialization_score(&self) -> f64 {
        self.performance.accuracy * self.performance.efficiency
    }
    
    /// Check if this module can be composed with another
    pub fn can_compose_with(&self, other: &MathModule) -> bool {
        // For now, allow all compositions - in a real system we'd use adapters
        // to handle size mismatches between modules
        true
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
        assert!(all_types.len() >= 11);
    }
}