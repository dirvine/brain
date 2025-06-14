//! Calculus Module for Advanced Mathematical Operations
//!
//! This module implements comprehensive calculus operations including derivatives,
//! integrals, limits, and optimization using both symbolic and numerical methods.
//! It supports automatic differentiation and numerical integration techniques.

use crate::error::{NEATError, Result};
use crate::calculator::{algebra::Expression, Operation};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Types of calculus operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalculusOperation {
    /// Differentiation (derivative)
    Derivative,
    /// Integration (antiderivative)
    Integral,
    /// Definite integral with bounds
    DefiniteIntegral,
    /// Limit calculation
    Limit,
    /// Partial derivative
    PartialDerivative,
    /// Multiple integral
    MultipleIntegral,
    /// Optimization (finding extrema)
    Optimization,
    /// Series expansion
    SeriesExpansion,
}

/// Mathematical function representation for calculus operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalculusFunction {
    /// Function expression
    pub expression: Expression,
    /// Variable name (usually 'x')
    pub variable: String,
    /// Function domain
    pub domain: (f64, f64),
    /// Function metadata
    pub metadata: FunctionMetadata,
}

/// Metadata about a mathematical function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionMetadata {
    /// Whether function is continuous
    pub is_continuous: bool,
    /// Whether function is differentiable
    pub is_differentiable: bool,
    /// Known singularities
    pub singularities: Vec<f64>,
    /// Function type classification
    pub function_type: FunctionType,
}

/// Classification of function types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FunctionType {
    /// Polynomial function
    Polynomial,
    /// Rational function
    Rational,
    /// Exponential function
    Exponential,
    /// Logarithmic function
    Logarithmic,
    /// Trigonometric function
    Trigonometric,
    /// Hyperbolic function
    Hyperbolic,
    /// Composite function
    Composite,
    /// Piecewise function
    Piecewise,
}

/// Result of a calculus operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalculusResult {
    /// Operation that was performed
    pub operation: CalculusOperation,
    /// Result expression (for symbolic results)
    pub symbolic_result: Option<Expression>,
    /// Numerical result (for numerical computations)
    pub numerical_result: Option<f64>,
    /// Error estimate for numerical methods
    pub error_estimate: Option<f64>,
    /// Method used for computation
    pub method: ComputationMethod,
    /// Computation steps for educational purposes
    pub computation_steps: Vec<ComputationStep>,
}

/// Methods used for calculus computations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputationMethod {
    /// Symbolic differentiation
    SymbolicDifferentiation,
    /// Numerical differentiation
    NumericalDifferentiation,
    /// Symbolic integration
    SymbolicIntegration,
    /// Numerical integration (trapezoidal rule)
    TrapezoidalRule,
    /// Numerical integration (Simpson's rule)
    SimpsonsRule,
    /// Gaussian quadrature
    GaussianQuadrature,
    /// Monte Carlo integration
    MonteCarloIntegration,
    /// L'Hôpital's rule for limits
    LHopitalsRule,
    /// Newton's method for optimization
    NewtonsMethod,
    /// Gradient descent
    GradientDescent,
}

/// Individual computation step for educational explanations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationStep {
    /// Step number
    pub step_number: usize,
    /// Description of the step
    pub description: String,
    /// Mathematical expression before this step
    pub before_expression: String,
    /// Mathematical expression after this step
    pub after_expression: String,
    /// Rule or method applied
    pub rule_applied: String,
    /// Explanation of why this step is valid
    pub explanation: String,
}

/// Main calculus engine
pub struct CalculusEngine {
    /// Numerical precision for computations
    precision: f64,
    /// Maximum iterations for iterative methods
    max_iterations: usize,
    /// Step size for numerical differentiation
    numerical_step_size: f64,
    /// Integration tolerance
    integration_tolerance: f64,
}

impl Default for CalculusEngine {
    fn default() -> Self {
        Self {
            precision: 1e-10,
            max_iterations: 1000,
            numerical_step_size: 1e-8,
            integration_tolerance: 1e-6,
        }
    }
}

impl CalculusEngine {
    /// Create a new calculus engine with custom parameters
    pub fn new(precision: f64, max_iterations: usize) -> Self {
        Self {
            precision,
            max_iterations,
            numerical_step_size: precision.sqrt(),
            integration_tolerance: precision,
        }
    }

    /// Compute derivative of a function
    pub fn derivative(&self, function: &CalculusFunction, point: Option<f64>) -> Result<CalculusResult> {
        let mut steps = Vec::new();
        
        // Try symbolic differentiation first
        if let Ok(symbolic_derivative) = self.symbolic_derivative(&function.expression, &function.variable) {
            steps.push(ComputationStep {
                step_number: 1,
                description: "Apply symbolic differentiation rules".to_string(),
                before_expression: format!("f(x) = {:?}", function.expression),
                after_expression: format!("f'(x) = {:?}", symbolic_derivative),
                rule_applied: "Chain rule and power rule".to_string(),
                explanation: "Using standard differentiation rules for elementary functions".to_string(),
            });

            let numerical_result = if let Some(x) = point {
                Some(self.evaluate_expression_at_point(&symbolic_derivative, x)?)
            } else {
                None
            };

            return Ok(CalculusResult {
                operation: CalculusOperation::Derivative,
                symbolic_result: Some(symbolic_derivative),
                numerical_result,
                error_estimate: None,
                method: ComputationMethod::SymbolicDifferentiation,
                computation_steps: steps,
            });
        }

        // Fall back to numerical differentiation
        if let Some(x) = point {
            let numerical_derivative = self.numerical_derivative(function, x)?;
            
            steps.push(ComputationStep {
                step_number: 1,
                description: "Apply numerical differentiation using finite differences".to_string(),
                before_expression: format!("f({}) = ?", x),
                after_expression: format!("f'({}) ≈ {:.6}", x, numerical_derivative),
                rule_applied: "Central difference formula".to_string(),
                explanation: format!("Using h = {} for finite difference approximation", self.numerical_step_size),
            });

            Ok(CalculusResult {
                operation: CalculusOperation::Derivative,
                symbolic_result: None,
                numerical_result: Some(numerical_derivative),
                error_estimate: Some(self.numerical_step_size.powi(2)),
                method: ComputationMethod::NumericalDifferentiation,
                computation_steps: steps,
            })
        } else {
            Err(NEATError::InvalidConfiguration {
                parameter: "evaluation_point".to_string(),
                value: "None provided for numerical differentiation".to_string(),
            })
        }
    }

    /// Compute definite integral
    pub fn definite_integral(&self, function: &CalculusFunction, a: f64, b: f64) -> Result<CalculusResult> {
        let mut steps = Vec::new();
        
        // Use Simpson's rule for numerical integration
        let n = 1000; // Number of intervals
        let h = (b - a) / n as f64;
        let mut sum = self.evaluate_function_at_point(function, a)? + self.evaluate_function_at_point(function, b)?;
        
        steps.push(ComputationStep {
            step_number: 1,
            description: "Set up Simpson's rule integration".to_string(),
            before_expression: format!("∫[{}, {}] f(x) dx", a, b),
            after_expression: format!("Using {} intervals with h = {:.6}", n, h),
            rule_applied: "Simpson's 1/3 rule".to_string(),
            explanation: "Approximate the integral using parabolic segments".to_string(),
        });

        // Apply Simpson's rule
        for i in 1..n {
            let x = a + i as f64 * h;
            let factor = if i % 2 == 0 { 2.0 } else { 4.0 };
            sum += factor * self.evaluate_function_at_point(function, x)?;
        }
        
        let result = sum * h / 3.0;
        
        steps.push(ComputationStep {
            step_number: 2,
            description: "Compute weighted sum".to_string(),
            before_expression: "Apply Simpson's coefficients (1, 4, 2, 4, ..., 2, 4, 1)".to_string(),
            after_expression: format!("Result = {:.6}", result),
            rule_applied: "Simpson's rule formula".to_string(),
            explanation: "∫f(x)dx ≈ (h/3)[f(x₀) + 4f(x₁) + 2f(x₂) + ... + f(xₙ)]".to_string(),
        });

        Ok(CalculusResult {
            operation: CalculusOperation::DefiniteIntegral,
            symbolic_result: None,
            numerical_result: Some(result),
            error_estimate: Some(h.powi(4)), // Simpson's rule has O(h⁴) error
            method: ComputationMethod::SimpsonsRule,
            computation_steps: steps,
        })
    }

    /// Compute limit of a function as x approaches a value
    pub fn limit(&self, function: &CalculusFunction, approach_point: f64, direction: LimitDirection) -> Result<CalculusResult> {
        let mut steps = Vec::new();
        
        // Use numerical approach to estimate limit
        let delta_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6];
        let mut limit_estimates = Vec::new();
        
        steps.push(ComputationStep {
            step_number: 1,
            description: format!("Evaluate function approaching x = {} from the {}", approach_point, 
                match direction {
                    LimitDirection::Left => "left",
                    LimitDirection::Right => "right",
                    LimitDirection::Both => "both sides",
                }),
            before_expression: format!("lim(x→{}) f(x)", approach_point),
            after_expression: "Evaluating at points increasingly close to limit point".to_string(),
            rule_applied: "Sequential approximation".to_string(),
            explanation: "Compute function values at points approaching the limit".to_string(),
        });

        for &delta in &delta_values {
            let test_point = match direction {
                LimitDirection::Left => approach_point - delta,
                LimitDirection::Right => approach_point + delta,
                LimitDirection::Both => approach_point + delta, // Test from right for "both"
            };
            
            if let Ok(value) = self.evaluate_function_at_point(function, test_point) {
                limit_estimates.push(value);
            }
        }

        if limit_estimates.is_empty() {
            return Err(NEATError::InvalidConfiguration {
                parameter: "limit_computation".to_string(),
                value: "Cannot evaluate function near limit point".to_string(),
            });
        }

        // Check for convergence
        let estimated_limit = limit_estimates.last().unwrap();
        let convergence_check = limit_estimates.len() >= 3 && 
            (limit_estimates[limit_estimates.len()-1] - limit_estimates[limit_estimates.len()-2]).abs() < self.precision;

        steps.push(ComputationStep {
            step_number: 2,
            description: "Analyze convergence of function values".to_string(),
            before_expression: format!("Sequence: {:?}", limit_estimates.iter().map(|x| format!("{:.6}", x)).collect::<Vec<_>>()),
            after_expression: format!("Estimated limit: {:.6}", estimated_limit),
            rule_applied: "Convergence analysis".to_string(),
            explanation: if convergence_check {
                "Function values converge to a finite limit".to_string()
            } else {
                "Function values may not converge or limit may not exist".to_string()
            },
        });

        Ok(CalculusResult {
            operation: CalculusOperation::Limit,
            symbolic_result: None,
            numerical_result: Some(*estimated_limit),
            error_estimate: Some(delta_values.last().unwrap() * 10.0),
            method: ComputationMethod::NumericalDifferentiation, // Reusing method enum
            computation_steps: steps,
        })
    }

    /// Find critical points and optimize function
    pub fn optimize(&self, function: &CalculusFunction, initial_guess: f64) -> Result<CalculusResult> {
        let mut steps = Vec::new();
        let mut x = initial_guess;
        
        steps.push(ComputationStep {
            step_number: 1,
            description: "Initialize Newton's method for optimization".to_string(),
            before_expression: format!("Find critical points of f(x)"),
            after_expression: format!("Starting point: x₀ = {}", initial_guess),
            rule_applied: "Newton's method".to_string(),
            explanation: "Use Newton's method to find where f'(x) = 0".to_string(),
        });

        // Newton's method to find critical points (where derivative = 0)
        for iteration in 0..self.max_iterations {
            let fx_prime = self.numerical_derivative(function, x)?;
            let fx_double_prime = self.numerical_second_derivative(function, x)?;
            
            if fx_double_prime.abs() < self.precision {
                break; // Avoid division by zero
            }
            
            let x_new = x - fx_prime / fx_double_prime;
            
            if (x_new - x).abs() < self.precision {
                steps.push(ComputationStep {
                    step_number: iteration + 2,
                    description: format!("Convergence achieved at iteration {}", iteration + 1),
                    before_expression: format!("x_{} = {:.6}", iteration, x),
                    after_expression: format!("x_{} = {:.6}", iteration + 1, x_new),
                    rule_applied: "Newton convergence".to_string(),
                    explanation: format!("Change in x: {:.2e} < tolerance", (x_new - x).abs()),
                });
                x = x_new;
                break;
            }
            
            if iteration < 5 { // Limit step recording for readability
                steps.push(ComputationStep {
                    step_number: iteration + 2,
                    description: format!("Newton's method iteration {}", iteration + 1),
                    before_expression: format!("x_{} = {:.6}, f'(x) = {:.6}", iteration, x, fx_prime),
                    after_expression: format!("x_{} = {:.6}", iteration + 1, x_new),
                    rule_applied: "Newton update".to_string(),
                    explanation: "x_{n+1} = x_n - f'(x_n)/f''(x_n)".to_string(),
                });
            }
            
            x = x_new;
        }

        // Determine if it's a minimum, maximum, or saddle point
        let second_derivative = self.numerical_second_derivative(function, x)?;
        let critical_point_type = if second_derivative > 0.0 {
            "local minimum"
        } else if second_derivative < 0.0 {
            "local maximum"
        } else {
            "saddle point or inflection"
        };

        steps.push(ComputationStep {
            step_number: steps.len() + 1,
            description: "Classify critical point using second derivative test".to_string(),
            before_expression: format!("f''({:.6}) = {:.6}", x, second_derivative),
            after_expression: format!("Critical point is a {}", critical_point_type),
            rule_applied: "Second derivative test".to_string(),
            explanation: "f'' > 0: minimum, f'' < 0: maximum, f'' = 0: inconclusive".to_string(),
        });

        let function_value = self.evaluate_function_at_point(function, x)?;

        Ok(CalculusResult {
            operation: CalculusOperation::Optimization,
            symbolic_result: None,
            numerical_result: Some(function_value),
            error_estimate: Some(self.precision * 10.0),
            method: ComputationMethod::NewtonsMethod,
            computation_steps: steps,
        })
    }

    /// Symbolic differentiation for simple expressions
    fn symbolic_derivative(&self, expression: &Expression, _variable: &str) -> Result<Expression> {
        // Simplified symbolic differentiation for basic cases
        match expression {
            Expression::Variable(_) => Ok(Expression::Constant(1.0)),
            Expression::Constant(_) => Ok(Expression::Constant(0.0)),
            Expression::Binary { left, operation, right } => {
                let left_expr = left.as_ref();
                let right_expr = right.as_ref();
                
                match operation {
                    Operation::Add => {
                        let left_deriv = self.symbolic_derivative(left_expr, _variable)?;
                        let right_deriv = self.symbolic_derivative(right_expr, _variable)?;
                        Ok(Expression::Binary {
                            left: Box::new(left_deriv),
                            operation: Operation::Add,
                            right: Box::new(right_deriv),
                        })
                    },
                    Operation::Subtract => {
                        let left_deriv = self.symbolic_derivative(left_expr, _variable)?;
                        let right_deriv = self.symbolic_derivative(right_expr, _variable)?;
                        Ok(Expression::Binary {
                            left: Box::new(left_deriv),
                            operation: Operation::Subtract,
                            right: Box::new(right_deriv),
                        })
                    },
                    Operation::Multiply => {
                        // Product rule: (uv)' = u'v + uv'
                        let left_deriv = self.symbolic_derivative(left_expr, _variable)?;
                        let right_deriv = self.symbolic_derivative(right_expr, _variable)?;
                        
                        let term1 = Expression::Binary {
                            left: Box::new(left_deriv),
                            operation: Operation::Multiply,
                            right: Box::new(right_expr.clone()),
                        };
                        
                        let term2 = Expression::Binary {
                            left: Box::new(left_expr.clone()),
                            operation: Operation::Multiply,
                            right: Box::new(right_deriv),
                        };
                        
                        Ok(Expression::Binary {
                            left: Box::new(term1),
                            operation: Operation::Add,
                            right: Box::new(term2),
                        })
                    },
                    _ => Err(NEATError::InvalidConfiguration {
                        parameter: "symbolic_differentiation".to_string(),
                        value: "Unsupported operation for symbolic differentiation".to_string(),
                    }),
                }
            },
            _ => Err(NEATError::InvalidConfiguration {
                parameter: "symbolic_differentiation".to_string(),
                value: "Expression too complex for symbolic differentiation".to_string(),
            }),
        }
    }

    /// Numerical differentiation using central difference
    fn numerical_derivative(&self, function: &CalculusFunction, x: f64) -> Result<f64> {
        let h = self.numerical_step_size;
        let f_plus = self.evaluate_function_at_point(function, x + h)?;
        let f_minus = self.evaluate_function_at_point(function, x - h)?;
        Ok((f_plus - f_minus) / (2.0 * h))
    }

    /// Numerical second derivative
    fn numerical_second_derivative(&self, function: &CalculusFunction, x: f64) -> Result<f64> {
        let h = self.numerical_step_size;
        let f_center = self.evaluate_function_at_point(function, x)?;
        let f_plus = self.evaluate_function_at_point(function, x + h)?;
        let f_minus = self.evaluate_function_at_point(function, x - h)?;
        Ok((f_plus - 2.0 * f_center + f_minus) / (h * h))
    }

    /// Evaluate function at a specific point
    fn evaluate_function_at_point(&self, function: &CalculusFunction, x: f64) -> Result<f64> {
        self.evaluate_expression_at_point(&function.expression, x)
    }

    /// Evaluate mathematical expression at a point
    fn evaluate_expression_at_point(&self, expression: &Expression, x: f64) -> Result<f64> {
        match expression {
            Expression::Variable(_) => Ok(x),
            Expression::Constant(c) => Ok(*c),
            Expression::Binary { left, operation, right } => {
                let left_val = self.evaluate_expression_at_point(left, x)?;
                let right_val = self.evaluate_expression_at_point(right, x)?;
                
                match operation {
                    Operation::Add => Ok(left_val + right_val),
                    Operation::Subtract => Ok(left_val - right_val),
                    Operation::Multiply => Ok(left_val * right_val),
                    Operation::Divide => {
                        if right_val.abs() < self.precision {
                            Err(NEATError::InvalidConfiguration {
                                parameter: "division".to_string(),
                                value: "Division by zero".to_string(),
                            })
                        } else {
                            Ok(left_val / right_val)
                        }
                    },
                    _ => Err(NEATError::InvalidConfiguration {
                        parameter: "expression_evaluation".to_string(),
                        value: "Unsupported operation".to_string(),
                    }),
                }
            },
            _ => Err(NEATError::InvalidConfiguration {
                parameter: "expression_evaluation".to_string(),
                value: "Expression type not supported".to_string(),
            }),
        }
    }
}

/// Direction for limit computation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitDirection {
    /// Approach from the left
    Left,
    /// Approach from the right
    Right,
    /// Approach from both sides
    Both,
}

/// Create common calculus functions for testing and examples
pub mod functions {
    use super::*;

    /// Create a polynomial function: ax² + bx + c
    pub fn polynomial(a: f64, b: f64, c: f64) -> CalculusFunction {
        let expression = Expression::Binary {
            left: Box::new(Expression::Binary {
                left: Box::new(Expression::Binary {
                    left: Box::new(Expression::Constant(a)),
                    operation: Operation::Multiply,
                    right: Box::new(Expression::Binary {
                        left: Box::new(Expression::Variable),
                        operation: Operation::Multiply,
                        right: Box::new(Expression::Variable),
                    }),
                }),
                operation: Operation::Add,
                right: Box::new(Expression::Binary {
                    left: Box::new(Expression::Constant(b)),
                    operation: Operation::Multiply,
                    right: Box::new(Expression::Variable),
                }),
            }),
            operation: Operation::Add,
            right: Box::new(Expression::Constant(c)),
        };

        CalculusFunction {
            expression,
            variable: "x".to_string(),
            domain: (-f64::INFINITY, f64::INFINITY),
            metadata: FunctionMetadata {
                is_continuous: true,
                is_differentiable: true,
                singularities: vec![],
                function_type: FunctionType::Polynomial,
            },
        }
    }

    /// Create a simple linear function: ax + b
    pub fn linear(a: f64, b: f64) -> CalculusFunction {
        let expression = Expression::Binary {
            left: Box::new(Expression::Binary {
                left: Box::new(Expression::Constant(a)),
                operation: Operation::Multiply,
                right: Box::new(Expression::Variable),
            }),
            operation: Operation::Add,
            right: Box::new(Expression::Constant(b)),
        };

        CalculusFunction {
            expression,
            variable: "x".to_string(),
            domain: (-f64::INFINITY, f64::INFINITY),
            metadata: FunctionMetadata {
                is_continuous: true,
                is_differentiable: true,
                singularities: vec![],
                function_type: FunctionType::Polynomial,
            },
        }
    }

    /// Create a rational function: 1/x
    pub fn reciprocal() -> CalculusFunction {
        let expression = Expression::Binary {
            left: Box::new(Expression::Constant(1.0)),
            operation: Operation::Divide,
            right: Box::new(Expression::Variable),
        };

        CalculusFunction {
            expression,
            variable: "x".to_string(),
            domain: (-f64::INFINITY, f64::INFINITY),
            metadata: FunctionMetadata {
                is_continuous: false,
                is_differentiable: false,
                singularities: vec![0.0],
                function_type: FunctionType::Rational,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::functions::*;

    #[test]
    fn test_polynomial_derivative() -> Result<()> {
        let engine = CalculusEngine::default();
        let func = polynomial(1.0, 2.0, 3.0); // x² + 2x + 3
        
        let result = engine.derivative(&func, Some(2.0))?;
        
        // Derivative at x=2 should be 2x + 2 = 6
        assert!(result.numerical_result.is_some());
        let derivative_value = result.numerical_result.unwrap();
        assert!((derivative_value - 6.0).abs() < 1e-6);
        
        Ok(())
    }

    #[test]
    fn test_definite_integral() -> Result<()> {
        let engine = CalculusEngine::default();
        let func = linear(2.0, 1.0); // 2x + 1
        
        // Integral from 0 to 1 should be [x² + x] = 1 + 1 = 2
        let result = engine.definite_integral(&func, 0.0, 1.0)?;
        
        assert!(result.numerical_result.is_some());
        let integral_value = result.numerical_result.unwrap();
        assert!((integral_value - 2.0).abs() < 1e-3);
        
        Ok(())
    }

    #[test]
    fn test_optimization() -> Result<()> {
        let engine = CalculusEngine::default();
        let func = polynomial(1.0, -4.0, 5.0); // x² - 4x + 5
        
        // Minimum should be at x = 2, with value 1
        let result = engine.optimize(&func, 1.0)?;
        
        assert!(result.numerical_result.is_some());
        let min_value = result.numerical_result.unwrap();
        assert!((min_value - 1.0).abs() < 1e-3);
        
        Ok(())
    }

    #[test]
    fn test_limit() -> Result<()> {
        let engine = CalculusEngine::default();
        let func = linear(2.0, 3.0); // 2x + 3
        
        // Limit as x approaches 1 should be 5
        let result = engine.limit(&func, 1.0, LimitDirection::Both)?;
        
        assert!(result.numerical_result.is_some());
        let limit_value = result.numerical_result.unwrap();
        assert!((limit_value - 5.0).abs() < 1e-6);
        
        Ok(())
    }

    #[test]
    fn test_numerical_differentiation() -> Result<()> {
        let engine = CalculusEngine::default();
        let func = polynomial(2.0, 0.0, 0.0); // 2x²
        
        // Derivative at x=3 should be 4x = 12
        let derivative = engine.numerical_derivative(&func, 3.0)?;
        assert!((derivative - 12.0).abs() < 1e-6);
        
        Ok(())
    }

    #[test]
    fn test_function_evaluation() -> Result<()> {
        let engine = CalculusEngine::default();
        let func = polynomial(1.0, 2.0, 3.0); // x² + 2x + 3
        
        // At x=2: 4 + 4 + 3 = 11
        let value = engine.evaluate_function_at_point(&func, 2.0)?;
        assert!((value - 11.0).abs() < 1e-10);
        
        Ok(())
    }

    #[test]
    fn test_computation_steps() -> Result<()> {
        let engine = CalculusEngine::default();
        let func = linear(3.0, 2.0); // 3x + 2
        
        let result = engine.derivative(&func, Some(1.0))?;
        
        // Should have computation steps
        assert!(!result.computation_steps.is_empty());
        assert_eq!(result.method, ComputationMethod::NumericalDifferentiation);
        
        Ok(())
    }
}