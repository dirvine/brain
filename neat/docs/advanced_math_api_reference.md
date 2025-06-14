# Advanced Mathematics API Reference

## Overview

This document provides a comprehensive API reference for the advanced mathematical modules implemented in Phase 6 of the NEAT mathematical discovery platform.

## Core Modules

### Calculus Module

#### `CalculusEngine`

```rust
pub struct CalculusEngine {
    precision: f64,
    max_iterations: usize,
    integration_method: IntegrationMethod,
    optimization_method: OptimizationMethod,
}
```

**Methods:**

##### `new(precision: f64, max_iterations: usize) -> Self`
Creates a new calculus engine with specified precision and iteration limits.

##### `derivative(&self, function: &CalculusFunction, point: Option<f64>) -> Result<CalculusResult>`
Computes the derivative of a function, optionally evaluated at a specific point.

**Parameters:**
- `function`: The function to differentiate
- `point`: Optional evaluation point for numerical derivative

**Returns:** `CalculusResult` containing symbolic and/or numerical result

##### `definite_integral(&self, function: &CalculusFunction, a: f64, b: f64) -> Result<CalculusResult>`
Computes the definite integral of a function over the interval [a, b].

##### `indefinite_integral(&self, function: &CalculusFunction) -> Result<CalculusResult>`
Computes the indefinite integral (antiderivative) of a function.

##### `limit(&self, function: &CalculusFunction, point: f64, direction: LimitDirection) -> Result<CalculusResult>`
Computes the limit of a function as it approaches a point.

##### `optimize(&self, function: &CalculusFunction, initial_guess: f64) -> Result<CalculusResult>`
Finds local extrema of a function using optimization algorithms.

#### Data Types

```rust
pub enum CalculusOperation {
    Derivative,
    Integral,
    Limit,
    Optimization,
}

pub enum IntegrationMethod {
    Trapezoidal,
    Simpson,
    AdaptiveQuadrature,
    MonteCarlo,
}

pub enum LimitDirection {
    Left,
    Right,
    Both,
}

pub struct CalculusResult {
    pub operation: CalculusOperation,
    pub numerical_result: Option<f64>,
    pub symbolic_result: Option<Expression>,
    pub method: CalculusMethod,
    pub computation_steps: Vec<ComputationStep>,
    pub explanation: String,
    pub error_estimate: Option<f64>,
}
```

---

### Trigonometry Module

#### `TrigonometryEngine`

```rust
pub struct TrigonometryEngine {
    precision: f64,
    angle_tolerance: f64,
    max_iterations: usize,
}
```

**Methods:**

##### `new(precision: f64, angle_tolerance: f64) -> Self`
Creates a new trigonometry engine with specified precision and angle tolerance.

##### `evaluate(&self, function: TrigFunction, angle: f64, unit: AngleUnit) -> Result<TrigResult>`
Evaluates a trigonometric function at the given angle.

##### `solve_equation(&self, equation: &TrigEquation) -> Result<Vec<f64>>`
Solves trigonometric equations numerically.

##### `verify_identity(&self, identity: TrigIdentity, angle: f64) -> Result<bool>`
Verifies trigonometric identities at specific angles.

##### `analyze_wave(&self, amplitude: f64, frequency: f64, phase: f64, offset: f64) -> WaveProperties`
Analyzes wave properties for periodic functions.

#### Data Types

```rust
pub enum TrigFunction {
    Sin, Cos, Tan, Csc, Sec, Cot,
    Asin, Acos, Atan, Atan2,
    Sinh, Cosh, Tanh, Asinh, Acosh, Atanh,
}

pub enum AngleUnit {
    Radians,
    Degrees,
    Gradians,
}

pub enum TrigIdentity {
    Pythagorean,      // sin²θ + cos²θ = 1
    AngleSum,         // sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
    DoubleAngle,      // sin(2θ) = 2sin(θ)cos(θ)
    HalfAngle,        // sin(θ/2) = ±√((1-cos(θ))/2)
}

pub struct TrigResult {
    pub function: TrigFunction,
    pub angle: f64,
    pub angle_unit: AngleUnit,
    pub value: f64,
    pub quadrant: Option<u8>,
    pub reference_angle: Option<f64>,
    pub equivalent_angles: Vec<f64>,
    pub period: f64,
}

pub struct WaveProperties {
    pub amplitude: f64,
    pub period: f64,
    pub frequency: f64,
    pub phase_shift: f64,
    pub vertical_shift: f64,
    pub angular_frequency: f64,
}
```

---

### Statistics Module

#### `StatisticsEngine`

```rust
pub struct StatisticsEngine {
    precision: f64,
    default_alpha: f64,
    rng_seed: Option<u64>,
}
```

**Methods:**

##### `new(precision: f64, default_alpha: f64) -> Self`
Creates a new statistics engine with specified precision and default significance level.

##### `descriptive_statistics(&self, data: &[f64]) -> Result<DescriptiveStats>`
Computes comprehensive descriptive statistics for a dataset.

##### `one_sample_t_test(&self, data: &[f64], hypothesized_mean: f64, alpha: Option<f64>) -> Result<HypothesisTestResult>`
Performs a one-sample t-test.

##### `two_sample_t_test(&self, data1: &[f64], data2: &[f64], alpha: Option<f64>) -> Result<HypothesisTestResult>`
Performs a two-sample t-test (Welch's test for unequal variances).

##### `linear_regression(&self, x: &[f64], y: &[f64]) -> Result<RegressionResult>`
Performs simple linear regression analysis.

##### `confidence_interval_mean(&self, data: &[f64], confidence_level: f64) -> Result<ConfidenceInterval>`
Calculates confidence interval for the population mean.

##### `normal_probability(&self, x: f64, mean: f64, std_dev: f64) -> Result<f64>`
Calculates normal distribution probability density.

##### `normal_cdf(&self, x: f64, mean: f64, std_dev: f64) -> Result<f64>`
Calculates normal distribution cumulative probability.

#### Data Types

```rust
pub struct DescriptiveStats {
    pub n: usize,
    pub mean: f64,
    pub median: f64,
    pub mode: Option<f64>,
    pub std_dev: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
    pub range: f64,
    pub iqr: f64,
    pub q1: f64,
    pub q3: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub standard_error: f64,
}

pub struct HypothesisTestResult {
    pub test_statistic: f64,
    pub p_value: f64,
    pub critical_values: Vec<f64>,
    pub degrees_of_freedom: Option<usize>,
    pub alpha: f64,
    pub reject_null: bool,
    pub test_type: String,
    pub effect_size: Option<f64>,
}

pub struct RegressionResult {
    pub coefficients: Vec<f64>,        // [intercept, slope]
    pub r_squared: f64,
    pub adjusted_r_squared: f64,
    pub standard_errors: Vec<f64>,
    pub t_statistics: Vec<f64>,
    pub p_values: Vec<f64>,
    pub residual_ss: f64,
    pub total_ss: f64,
    pub f_statistic: f64,
    pub df_model: usize,
    pub df_residual: usize,
}

pub struct ConfidenceInterval {
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub confidence_level: f64,
    pub margin_of_error: f64,
    pub statistic: String,
}
```

---

### Discrete Mathematics Module

#### `DiscreteMathEngine`

```rust
pub struct DiscreteMathEngine {
    precision: f64,
    max_factorial: u64,
    factorial_cache: HashMap<u64, u64>,
    combination_cache: HashMap<(u64, u64), u64>,
}
```

**Methods:**

##### `new(max_factorial: u64) -> Self`
Creates a new discrete mathematics engine with specified factorial limit.

##### `combinatorics(&mut self, operation: CombinatorialType, n: u64, r: Option<u64>) -> Result<DiscreteResult>`
Performs combinatorial calculations.

##### `set_operations(&self, set1: &DiscreteSet, set2: &DiscreteSet, operation: &str) -> Result<DiscreteResult>`
Performs set theory operations.

##### `graph_operations(&self, graph: &DiscreteGraph, operation: &str, start: Option<usize>, end: Option<usize>) -> Result<DiscreteResult>`
Performs graph theory operations.

##### `discrete_probability(&self, total_outcomes: u64, favorable_outcomes: u64) -> Result<DiscreteResult>`
Calculates discrete probability.

##### `modular_arithmetic(&self, a: i64, b: i64, modulus: i64, operation: &str) -> Result<DiscreteResult>`
Performs modular arithmetic operations.

#### Data Types

```rust
pub enum CombinatorialType {
    Permutation,     // P(n,r)
    Combination,     // C(n,r)
    Factorial,       // n!
    Derangement,     // !n
    CatalanNumber,   // C_n
    StirlingNumber,  // S(n,k)
}

pub struct DiscreteGraph {
    pub vertices: usize,
    pub edges: HashMap<usize, Vec<usize>>,
    pub is_directed: bool,
    pub is_weighted: bool,
    pub weights: HashMap<(usize, usize), f64>,
}

pub struct DiscreteSet {
    pub elements: HashSet<i32>,
    pub name: String,
}

pub struct DiscreteResult {
    pub operation: DiscreteOperation,
    pub numerical_result: Option<f64>,
    pub set_result: Option<DiscreteSet>,
    pub boolean_result: Option<bool>,
    pub path_result: Option<Vec<usize>>,
    pub explanation: String,
    pub complexity: String,
}
```

---

## Module System

### `MathModule`

```rust
pub struct MathModule {
    pub id: String,
    pub module_type: ModuleType,
    pub genome: Genome,
    pub performance: ModulePerformance,
    pub metadata: HashMap<String, String>,
    pub io_spec: ModuleIOSpec,
}
```

**Methods:**

##### `new(id: String, module_type: ModuleType, genome: Genome) -> Self`
Creates a new mathematical module.

##### `evaluate(&self, input: &[f64]) -> Result<Vec<f64>>`
Evaluates the module on input data.

##### `create_network(&self) -> Result<Network>`
Creates a neural network from the module's genome.

##### `specialization_score(&self) -> f64`
Calculates the module's specialization score.

##### `can_compose_with(&self, other: &MathModule) -> bool`
Checks if the module can be composed with another module.

### `ModuleComposition`

```rust
pub struct ModuleComposition {
    pub modules: Vec<MathModule>,
    pub connections: Vec<ModuleConnection>,
    pub io_mapping: CompositionIOMapping,
}
```

**Methods:**

##### `new() -> Self`
Creates a new empty module composition.

##### `add_module(&mut self, module: MathModule) -> usize`
Adds a module to the composition and returns its index.

##### `connect_modules(&mut self, from_module: usize, to_module: usize, mapping: Vec<(usize, usize)>) -> Result<()>`
Connects two modules in the composition.

##### `execute(&self, input: &[f64]) -> Result<Vec<f64>>`
Executes the composition on input data.

### `ModuleLibrary`

```rust
pub struct ModuleLibrary {
    modules_by_type: HashMap<ModuleType, Vec<MathModule>>,
    modules_by_id: HashMap<String, MathModule>,
}
```

**Methods:**

##### `new() -> Self`
Creates a new empty module library.

##### `add_module(&mut self, module: MathModule)`
Adds a module to the library.

##### `get_modules_by_type(&self, module_type: ModuleType) -> Vec<&MathModule>`
Retrieves all modules of a specific type.

##### `get_module_by_id(&self, id: &str) -> Option<&MathModule>`
Retrieves a module by its ID.

##### `get_best_module(&self, module_type: ModuleType) -> Option<&MathModule>`
Gets the best performing module of a given type.

---

## Utility Functions

### Graph Creation (`discrete_math::structures`)

```rust
pub fn complete_graph(n: usize) -> DiscreteGraph
pub fn cycle_graph(n: usize) -> DiscreteGraph
pub fn path_graph(n: usize) -> DiscreteGraph
pub fn create_set(elements: Vec<i32>, name: String) -> DiscreteSet
```

### Function Builders (`calculus::functions`)

```rust
pub fn polynomial(a: f64, b: f64, c: f64) -> CalculusFunction
pub fn linear(slope: f64, intercept: f64) -> CalculusFunction
pub fn exponential(base: f64, coefficient: f64) -> CalculusFunction
pub fn trigonometric(function: TrigFunction, amplitude: f64, frequency: f64) -> CalculusFunction
```

---

## Error Handling

All functions return `Result<T, NEATError>` where `NEATError` includes:

```rust
pub enum NEATError {
    InvalidConfiguration { parameter: String, value: String },
    ComputationError { operation: String, details: String },
    NumericalInstability { method: String, suggestion: String },
    InvalidGenome { message: String },
    // ... other error types
}
```

---

## Usage Examples

### Basic Operations

```rust
use neat::calculator::*;

// Calculus
let calc_engine = CalculusEngine::default();
let poly = functions::polynomial(1.0, -2.0, 1.0); // x² - 2x + 1
let derivative = calc_engine.derivative(&poly, Some(1.0))?;

// Trigonometry
let trig_engine = TrigonometryEngine::default();
let sin_45 = trig_engine.evaluate(TrigFunction::Sin, 45.0, AngleUnit::Degrees)?;

// Statistics
let stats_engine = StatisticsEngine::default();
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let descriptive = stats_engine.descriptive_statistics(&data)?;

// Discrete Math
let mut discrete_engine = DiscreteMathEngine::default();
let factorial_5 = discrete_engine.combinatorics(CombinatorialType::Factorial, 5, None)?;
```

### Module Composition

```rust
// Create modules
let calc_module = MathModule::new("calc1".to_string(), ModuleType::Calculus, genome1);
let stat_module = MathModule::new("stat1".to_string(), ModuleType::Statistics, genome2);

// Compose modules
let mut composition = ModuleComposition::new();
let calc_idx = composition.add_module(calc_module);
let stat_idx = composition.add_module(stat_module);

// Execute composition
let result = composition.execute(&input_data)?;
```

---

## Performance Considerations

- **Caching**: Factorial and combination results are cached for efficiency
- **Precision Control**: All engines support configurable numerical precision
- **Iteration Limits**: Maximum iterations prevent infinite loops in numerical methods
- **Memory Management**: Large intermediate results are handled efficiently
- **Parallel Execution**: Module compositions can be executed in parallel where possible

---

## Testing

Each module includes comprehensive unit tests:

```bash
# Run all advanced math tests
cargo test advanced_math

# Run specific module tests
cargo test calculus
cargo test trigonometry
cargo test statistics
cargo test discrete_math

# Run integration tests
cargo test module_composition
```

---

## Dependencies

The advanced mathematics modules depend on:

- `serde` for serialization
- `std::collections` for data structures
- `std::f64::consts` for mathematical constants
- Internal NEAT modules for genome and network functionality