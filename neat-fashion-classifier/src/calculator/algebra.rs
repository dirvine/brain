//! Algebraic expression handling and pattern recognition
//!
//! This module extends our calculator to handle algebraic expressions,
//! variables, and symbolic mathematics - a revolutionary step towards
//! mathematical reasoning in evolved neural networks.

use super::Operation;
use crate::error::{NEATError, Result};
use std::collections::HashMap;
use std::fmt;

/// Represents an algebraic expression that can contain variables
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    /// A constant number
    Constant(f64),
    /// A variable (e.g., x, y, z)
    Variable(String),
    /// Binary operation on two expressions
    BinaryOp {
        left: Box<Expression>,
        op: Operation,
        right: Box<Expression>,
    },
    /// Unary operation (negation, absolute value, etc.)
    UnaryOp {
        op: UnaryOperation,
        operand: Box<Expression>,
    },
    /// Power operation (e.g., x^2)
    Power {
        base: Box<Expression>,
        exponent: i32,
    },
}

/// Unary operations for algebraic expressions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperation {
    /// Negation (-x)
    Negate,
    /// Absolute value (|x|)
    Abs,
    /// Square root (√x)
    Sqrt,
}

impl Expression {
    /// Create a new constant expression
    pub fn constant(value: f64) -> Self {
        Expression::Constant(value)
    }
    
    /// Create a new variable expression
    pub fn variable(name: impl Into<String>) -> Self {
        Expression::Variable(name.into())
    }
    
    /// Create a binary operation
    pub fn binary(left: Expression, op: Operation, right: Expression) -> Self {
        Expression::BinaryOp {
            left: Box::new(left),
            op,
            right: Box::new(right),
        }
    }
    
    /// Create a power expression
    pub fn power(base: Expression, exponent: i32) -> Self {
        Expression::Power {
            base: Box::new(base),
            exponent,
        }
    }
    
    /// Evaluate the expression with given variable values
    pub fn evaluate(&self, variables: &HashMap<String, f64>) -> Result<f64> {
        match self {
            Expression::Constant(val) => Ok(*val),
            Expression::Variable(name) => {
                variables.get(name).copied().ok_or_else(|| {
                    NEATError::Other(anyhow::anyhow!("Variable {} not found", name))
                })
            }
            Expression::BinaryOp { left, op, right } => {
                let left_val = left.evaluate(variables)?;
                let right_val = right.evaluate(variables)?;
                
                match op {
                    Operation::Add => Ok(left_val + right_val),
                    Operation::Subtract => Ok(left_val - right_val),
                    Operation::Multiply => Ok(left_val * right_val),
                    Operation::Divide => {
                        if right_val != 0.0 {
                            Ok(left_val / right_val)
                        } else {
                            Err(NEATError::Other(anyhow::anyhow!("Division by zero")))
                        }
                    }
                }
            }
            Expression::UnaryOp { op, operand } => {
                let val = operand.evaluate(variables)?;
                match op {
                    UnaryOperation::Negate => Ok(-val),
                    UnaryOperation::Abs => Ok(val.abs()),
                    UnaryOperation::Sqrt => {
                        if val >= 0.0 {
                            Ok(val.sqrt())
                        } else {
                            Err(NEATError::Other(anyhow::anyhow!("Square root of negative")))
                        }
                    }
                }
            }
            Expression::Power { base, exponent } => {
                let base_val = base.evaluate(variables)?;
                Ok(base_val.powi(*exponent))
            }
        }
    }
    
    /// Simplify the expression
    pub fn simplify(&self) -> Expression {
        match self {
            Expression::BinaryOp { left, op, right } => {
                let left_simplified = left.simplify();
                let right_simplified = right.simplify();
                
                // Try constant folding
                if let (Expression::Constant(l), Expression::Constant(r)) = 
                    (&left_simplified, &right_simplified) {
                    if let Some(result) = op.apply(*l as i32, *r as i32) {
                        return Expression::Constant(result as f64);
                    }
                }
                
                // Identity simplifications
                match (op, &left_simplified, &right_simplified) {
                    (Operation::Add, expr, Expression::Constant(0.0)) |
                    (Operation::Add, Expression::Constant(0.0), expr) => expr.clone(),
                    (Operation::Multiply, expr, Expression::Constant(1.0)) |
                    (Operation::Multiply, Expression::Constant(1.0), expr) => expr.clone(),
                    (Operation::Multiply, _, Expression::Constant(0.0)) |
                    (Operation::Multiply, Expression::Constant(0.0), _) => Expression::Constant(0.0),
                    _ => Expression::binary(left_simplified, *op, right_simplified)
                }
            }
            Expression::Power { base, exponent } => {
                let base_simplified = base.simplify();
                match exponent {
                    0 => Expression::Constant(1.0),
                    1 => base_simplified,
                    _ => Expression::power(base_simplified, *exponent)
                }
            }
            _ => self.clone()
        }
    }
    
    /// Get all variables in the expression
    pub fn variables(&self) -> Vec<String> {
        let mut vars = Vec::new();
        self.collect_variables(&mut vars);
        vars.sort();
        vars.dedup();
        vars
    }
    
    fn collect_variables(&self, vars: &mut Vec<String>) {
        match self {
            Expression::Variable(name) => vars.push(name.clone()),
            Expression::BinaryOp { left, right, .. } => {
                left.collect_variables(vars);
                right.collect_variables(vars);
            }
            Expression::UnaryOp { operand, .. } => operand.collect_variables(vars),
            Expression::Power { base, .. } => base.collect_variables(vars),
            Expression::Constant(_) => {}
        }
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Constant(val) => write!(f, "{}", val),
            Expression::Variable(name) => write!(f, "{}", name),
            Expression::BinaryOp { left, op, right } => {
                write!(f, "({} {} {})", left, op.symbol(), right)
            }
            Expression::UnaryOp { op, operand } => {
                match op {
                    UnaryOperation::Negate => write!(f, "-{}", operand),
                    UnaryOperation::Abs => write!(f, "|{}|", operand),
                    UnaryOperation::Sqrt => write!(f, "√{}", operand),
                }
            }
            Expression::Power { base, exponent } => {
                write!(f, "{}^{}", base, exponent)
            }
        }
    }
}

/// Represents an algebraic equation (expression = expression)
#[derive(Debug, Clone)]
pub struct Equation {
    /// Left-hand side of the equation
    pub left: Expression,
    /// Right-hand side of the equation
    pub right: Expression,
}

impl Equation {
    /// Create a new equation
    pub fn new(left: Expression, right: Expression) -> Self {
        Self { left, right }
    }
    
    /// Check if a variable assignment satisfies the equation
    pub fn is_satisfied_by(&self, variables: &HashMap<String, f64>) -> Result<bool> {
        let left_val = self.left.evaluate(variables)?;
        let right_val = self.right.evaluate(variables)?;
        Ok((left_val - right_val).abs() < 1e-10)
    }
    
    /// Get all variables in the equation
    pub fn variables(&self) -> Vec<String> {
        let mut vars = self.left.variables();
        vars.extend(self.right.variables());
        vars.sort();
        vars.dedup();
        vars
    }
}

impl fmt::Display for Equation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} = {}", self.left, self.right)
    }
}

/// Algebraic problem types for NEAT to solve
#[derive(Debug, Clone)]
pub enum AlgebraProblem {
    /// Evaluate an expression given variable values
    Evaluation {
        expression: Expression,
        variables: HashMap<String, f64>,
        expected: f64,
    },
    /// Solve for a variable in an equation
    LinearEquation {
        equation: Equation,
        solve_for: String,
        expected: f64,
    },
    /// Factor a polynomial expression
    Factoring {
        expression: Expression,
        factors: Vec<Expression>,
    },
    /// Simplify an expression
    Simplification {
        expression: Expression,
        simplified: Expression,
    },
}

impl AlgebraProblem {
    /// Create an evaluation problem
    pub fn evaluation(expr: Expression, vars: HashMap<String, f64>) -> Result<Self> {
        let expected = expr.evaluate(&vars)?;
        Ok(AlgebraProblem::Evaluation {
            expression: expr,
            variables: vars,
            expected,
        })
    }
    
    /// Create a simple linear equation solving problem
    pub fn linear_equation(a: f64, b: f64, c: f64) -> Self {
        // ax + b = c, solve for x
        let left = Expression::binary(
            Expression::binary(
                Expression::constant(a),
                Operation::Multiply,
                Expression::variable("x")
            ),
            Operation::Add,
            Expression::constant(b)
        );
        let right = Expression::constant(c);
        let equation = Equation::new(left, right);
        let expected = (c - b) / a;
        
        AlgebraProblem::LinearEquation {
            equation,
            solve_for: "x".to_string(),
            expected,
        }
    }
    
    /// Get a description of the problem
    pub fn description(&self) -> String {
        match self {
            AlgebraProblem::Evaluation { expression, variables, .. } => {
                let var_str = variables.iter()
                    .map(|(k, v)| format!("{}={}", k, v))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("Evaluate {} where {}", expression, var_str)
            }
            AlgebraProblem::LinearEquation { equation, solve_for, .. } => {
                format!("Solve {} for {}", equation, solve_for)
            }
            AlgebraProblem::Factoring { expression, .. } => {
                format!("Factor {}", expression)
            }
            AlgebraProblem::Simplification { expression, .. } => {
                format!("Simplify {}", expression)
            }
        }
    }
}

/// Pattern recognition for algebraic expressions
pub struct AlgebraicPatternMatcher {
    patterns: Vec<(Expression, Expression)>,
}

impl AlgebraicPatternMatcher {
    /// Create a new pattern matcher with common algebraic patterns
    pub fn new() -> Self {
        let mut patterns = Vec::new();
        
        // Difference of squares: a² - b² = (a+b)(a-b)
        let a = Expression::variable("a");
        let b = Expression::variable("b");
        let pattern = Expression::binary(
            Expression::power(a.clone(), 2),
            Operation::Subtract,
            Expression::power(b.clone(), 2)
        );
        let factored = Expression::binary(
            Expression::binary(a.clone(), Operation::Add, b.clone()),
            Operation::Multiply,
            Expression::binary(a, Operation::Subtract, b)
        );
        patterns.push((pattern, factored));
        
        Self { patterns }
    }
    
    /// Try to match an expression against known patterns
    pub fn match_pattern(&self, expr: &Expression) -> Option<Expression> {
        // This is a simplified version - real pattern matching would be more complex
        for (pattern, result) in &self.patterns {
            if self.matches_pattern(expr, pattern) {
                return Some(result.clone());
            }
        }
        None
    }
    
    fn matches_pattern(&self, expr: &Expression, pattern: &Expression) -> bool {
        // Simplified pattern matching - would need proper unification
        match (expr, pattern) {
            (Expression::Constant(a), Expression::Constant(b)) => (a - b).abs() < 1e-10,
            (Expression::Variable(a), Expression::Variable(b)) => a == b,
            (Expression::BinaryOp { left: l1, op: op1, right: r1 },
             Expression::BinaryOp { left: l2, op: op2, right: r2 }) => {
                op1 == op2 && self.matches_pattern(l1, l2) && self.matches_pattern(r1, r2)
            }
            _ => false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_expression_evaluation() {
        // x + 2*y where x=3, y=4
        let expr = Expression::binary(
            Expression::variable("x"),
            Operation::Add,
            Expression::binary(
                Expression::constant(2.0),
                Operation::Multiply,
                Expression::variable("y")
            )
        );
        
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 3.0);
        vars.insert("y".to_string(), 4.0);
        
        assert_eq!(expr.evaluate(&vars).unwrap(), 11.0);
    }
    
    #[test]
    fn test_expression_simplification() {
        // x * 1 + 0 should simplify to x
        let expr = Expression::binary(
            Expression::binary(
                Expression::variable("x"),
                Operation::Multiply,
                Expression::constant(1.0)
            ),
            Operation::Add,
            Expression::constant(0.0)
        );
        
        let simplified = expr.simplify();
        assert_eq!(simplified, Expression::variable("x"));
    }
    
    #[test]
    fn test_equation_solving() {
        // 2x + 3 = 7, x should be 2
        let problem = AlgebraProblem::linear_equation(2.0, 3.0, 7.0);
        
        if let AlgebraProblem::LinearEquation { expected, .. } = problem {
            assert_eq!(expected, 2.0);
        }
    }
    
    #[test]
    fn test_power_expressions() {
        // x^2 where x=3 should be 9
        let expr = Expression::power(Expression::variable("x"), 2);
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 3.0);
        
        assert_eq!(expr.evaluate(&vars).unwrap(), 9.0);
    }
}