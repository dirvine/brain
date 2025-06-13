//! Encoding schemes for algebraic expressions in neural networks
//!
//! This module provides innovative ways to encode algebraic expressions
//! for NEAT networks to process, enabling symbolic mathematical reasoning.

use super::algebra::{Expression, AlgebraProblem, UnaryOperation};
use super::{Operation, EncodingConfig};
use crate::error::{NEATError, Result};
use std::collections::HashMap;

/// Configuration for algebraic expression encoding
#[derive(Debug, Clone)]
pub struct AlgebraEncodingConfig {
    /// Maximum expression depth
    pub max_depth: usize,
    /// Maximum number of variables
    pub max_variables: usize,
    /// Number encoding configuration
    pub number_config: EncodingConfig,
    /// Include expression structure encoding
    pub encode_structure: bool,
    /// Include operation precedence
    pub encode_precedence: bool,
}

impl Default for AlgebraEncodingConfig {
    fn default() -> Self {
        Self {
            max_depth: 4,
            max_variables: 3,
            number_config: EncodingConfig::default(),
            encode_structure: true,
            encode_precedence: false,
        }
    }
}

/// Encoder for algebraic expressions
pub struct AlgebraEncoder {
    config: AlgebraEncodingConfig,
    variable_map: HashMap<String, usize>,
}

impl AlgebraEncoder {
    /// Create a new algebra encoder
    pub fn new(config: AlgebraEncodingConfig) -> Self {
        Self {
            config,
            variable_map: HashMap::new(),
        }
    }
    
    /// Register variables for consistent encoding
    pub fn register_variables(&mut self, vars: &[String]) -> Result<()> {
        if vars.len() > self.config.max_variables {
            return Err(NEATError::InvalidConfiguration {
                parameter: "variables".to_string(),
                value: format!("{} > {}", vars.len(), self.config.max_variables),
            });
        }
        
        self.variable_map.clear();
        for (i, var) in vars.iter().enumerate() {
            self.variable_map.insert(var.clone(), i);
        }
        Ok(())
    }
    
    /// Encode an algebraic expression into neural network input
    pub fn encode_expression(&self, expr: &Expression) -> Result<Vec<f64>> {
        let mut encoding = vec![0.0; self.encoding_length()];
        
        // Simple fixed-size encoding
        self.encode_simple(expr, &mut encoding);
        
        Ok(encoding)
    }
    
    /// Simple fixed-size encoding for expressions
    fn encode_simple(&self, expr: &Expression, encoding: &mut [f64]) {
        // Extract coefficients and variables from simple expressions like ax + b
        match expr {
            Expression::BinaryOp { left, op: Operation::Add, right } => {
                // Try to extract a*x + b pattern
                let (a, b, var) = self.extract_linear_form(left, right);
                encoding[0] = 1.0; // Linear expression type
                encoding[1] = a / 10.0; // Coefficient
                encoding[2] = b / 10.0; // Constant
                if let Some(v) = var {
                    if let Some(&idx) = self.variable_map.get(&v) {
                        encoding[3 + idx] = 1.0; // Variable indicator
                    }
                }
            }
            Expression::Constant(val) => {
                encoding[0] = 0.0; // Constant type
                encoding[1] = val / 10.0;
            }
            Expression::Variable(name) => {
                encoding[0] = 0.0; // Variable type
                if let Some(&idx) = self.variable_map.get(name) {
                    encoding[3 + idx] = 1.0;
                }
            }
            _ => {
                // Default encoding for other expressions
                encoding[0] = 0.5;
            }
        }
    }
    
    /// Extract linear form ax + b from expression tree
    fn extract_linear_form(&self, left: &Expression, right: &Expression) -> (f64, f64, Option<String>) {
        match (left, right) {
            // a*x + b pattern
            (Expression::BinaryOp { left: coeff, op: Operation::Multiply, right: var }, 
             Expression::Constant(b)) => {
                if let (Expression::Constant(a), Expression::Variable(x)) = (coeff.as_ref(), var.as_ref()) {
                    (*a, *b, Some(x.clone()))
                } else {
                    (1.0, 0.0, None)
                }
            }
            // x*a + b pattern
            (Expression::BinaryOp { left: var, op: Operation::Multiply, right: coeff }, 
             Expression::Constant(b)) => {
                if let (Expression::Variable(x), Expression::Constant(a)) = (var.as_ref(), coeff.as_ref()) {
                    (*a, *b, Some(x.clone()))
                } else {
                    (1.0, 0.0, None)
                }
            }
            // x + b pattern
            (Expression::Variable(x), Expression::Constant(b)) => {
                (1.0, *b, Some(x.clone()))
            }
            // a + x pattern
            (Expression::Constant(a), Expression::Variable(x)) => {
                (1.0, *a, Some(x.clone()))
            }
            _ => (1.0, 0.0, None)
        }
    }
    
    /// Encode expression tree structure recursively
    fn encode_tree_structure(
        &self, 
        expr: &Expression, 
        encoding: &mut Vec<f64>,
        depth: usize
    ) -> Result<()> {
        if depth >= self.config.max_depth {
            return Ok(());
        }
        
        // Encode node type
        let node_type = match expr {
            Expression::Constant(_) => vec![1.0, 0.0, 0.0, 0.0, 0.0],
            Expression::Variable(_) => vec![0.0, 1.0, 0.0, 0.0, 0.0],
            Expression::BinaryOp { .. } => vec![0.0, 0.0, 1.0, 0.0, 0.0],
            Expression::UnaryOp { .. } => vec![0.0, 0.0, 0.0, 1.0, 0.0],
            Expression::Power { .. } => vec![0.0, 0.0, 0.0, 0.0, 1.0],
        };
        encoding.extend(node_type);
        
        // Encode depth information
        encoding.push(depth as f64 / self.config.max_depth as f64);
        
        // Encode node-specific information
        match expr {
            Expression::Constant(val) => {
                encoding.push(val / 100.0); // Normalize to reasonable range
            }
            Expression::Variable(name) => {
                if let Some(&idx) = self.variable_map.get(name) {
                    let mut var_encoding = vec![0.0; self.config.max_variables];
                    var_encoding[idx] = 1.0;
                    encoding.extend(var_encoding);
                }
            }
            Expression::BinaryOp { left, op, right } => {
                // Encode operation
                let op_encoding = match op {
                    Operation::Add => vec![1.0, 0.0, 0.0, 0.0],
                    Operation::Subtract => vec![0.0, 1.0, 0.0, 0.0],
                    Operation::Multiply => vec![0.0, 0.0, 1.0, 0.0],
                    Operation::Divide => vec![0.0, 0.0, 0.0, 1.0],
                };
                encoding.extend(op_encoding);
                
                // Recursively encode children
                self.encode_tree_structure(left, encoding, depth + 1)?;
                self.encode_tree_structure(right, encoding, depth + 1)?;
            }
            Expression::UnaryOp { op, operand } => {
                let op_encoding = match op {
                    UnaryOperation::Negate => vec![1.0, 0.0, 0.0],
                    UnaryOperation::Abs => vec![0.0, 1.0, 0.0],
                    UnaryOperation::Sqrt => vec![0.0, 0.0, 1.0],
                };
                encoding.extend(op_encoding);
                
                self.encode_tree_structure(operand, encoding, depth + 1)?;
            }
            Expression::Power { base, exponent } => {
                encoding.push(*exponent as f64 / 10.0); // Normalize exponent
                self.encode_tree_structure(base, encoding, depth + 1)?;
            }
        }
        
        Ok(())
    }
    
    /// Tokenize an expression into a sequence
    fn tokenize_expression(&self, expr: &Expression) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();
        self.tokenize_recursive(expr, &mut tokens)?;
        Ok(tokens)
    }
    
    fn tokenize_recursive(&self, expr: &Expression, tokens: &mut Vec<Token>) -> Result<()> {
        match expr {
            Expression::Constant(val) => tokens.push(Token::Number(*val)),
            Expression::Variable(name) => tokens.push(Token::Variable(name.clone())),
            Expression::BinaryOp { left, op, right } => {
                tokens.push(Token::OpenParen);
                self.tokenize_recursive(left, tokens)?;
                tokens.push(Token::Operation(*op));
                self.tokenize_recursive(right, tokens)?;
                tokens.push(Token::CloseParen);
            }
            Expression::UnaryOp { op, operand } => {
                tokens.push(Token::UnaryOp(*op));
                self.tokenize_recursive(operand, tokens)?;
            }
            Expression::Power { base, exponent } => {
                self.tokenize_recursive(base, tokens)?;
                tokens.push(Token::Power(*exponent));
            }
        }
        Ok(())
    }
    
    /// Encode tokens into neural network format
    fn encode_tokens(&self, tokens: &[Token]) -> Result<Vec<f64>> {
        let mut encoding = Vec::new();
        
        for token in tokens {
            let token_encoding = match token {
                Token::Number(val) => {
                    let mut enc = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                    enc.push(val / 100.0); // Normalized value
                    enc
                }
                Token::Variable(name) => {
                    let mut enc = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                    if let Some(&idx) = self.variable_map.get(name) {
                        let mut var_enc = vec![0.0; self.config.max_variables];
                        var_enc[idx] = 1.0;
                        enc.extend(var_enc);
                    }
                    enc
                }
                Token::Operation(op) => {
                    let mut enc = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
                    let op_enc = match op {
                        Operation::Add => vec![1.0, 0.0, 0.0, 0.0],
                        Operation::Subtract => vec![0.0, 1.0, 0.0, 0.0],
                        Operation::Multiply => vec![0.0, 0.0, 1.0, 0.0],
                        Operation::Divide => vec![0.0, 0.0, 0.0, 1.0],
                    };
                    enc.extend(op_enc);
                    enc
                }
                Token::UnaryOp(op) => {
                    let mut enc = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
                    let op_enc = match op {
                        UnaryOperation::Negate => vec![1.0, 0.0, 0.0],
                        UnaryOperation::Abs => vec![0.0, 1.0, 0.0],
                        UnaryOperation::Sqrt => vec![0.0, 0.0, 1.0],
                    };
                    enc.extend(op_enc);
                    enc
                }
                Token::Power(exp) => {
                    let mut enc = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
                    enc.push(*exp as f64 / 10.0);
                    enc
                }
                Token::OpenParen => vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                Token::CloseParen => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            };
            encoding.extend(token_encoding);
        }
        
        Ok(encoding)
    }
    
    /// Encode an algebraic problem
    pub fn encode_problem(&self, problem: &AlgebraProblem) -> Result<Vec<f64>> {
        match problem {
            AlgebraProblem::Evaluation { expression, variables, .. } => {
                let mut encoding = self.encode_expression(expression)?;
                
                // Add variable values in fixed positions
                for (var, val) in variables {
                    if let Some(&idx) = self.variable_map.get(var) {
                        let var_value_pos = self.encoding_length() - self.config.max_variables + idx;
                        if var_value_pos < encoding.len() {
                            encoding[var_value_pos] = val / 10.0; // Normalized value
                        }
                    }
                }
                
                Ok(encoding)
            }
            AlgebraProblem::LinearEquation { equation, solve_for, .. } => {
                // For linear equations, encode left side (should be of form ax + b)
                let encoding = self.encode_expression(&equation.left)?;
                Ok(encoding)
            }
            _ => Err(NEATError::Other(anyhow::anyhow!(
                "Encoding not implemented for this problem type"
            )))
        }
    }
    
    /// Get the expected encoding length
    pub fn encoding_length(&self) -> usize {
        // Fixed small encoding for simplicity
        // Expression type (5) + variable values (3 * 2) + constant values (3 * 2) + operation type (4)
        5 + (self.config.max_variables * 2) + (3 * 2) + 4
    }
}

/// Token types for expression parsing
#[derive(Debug, Clone)]
enum Token {
    Number(f64),
    Variable(String),
    Operation(Operation),
    UnaryOp(UnaryOperation),
    Power(i32),
    OpenParen,
    CloseParen,
}

/// Graph-based encoding for expressions
pub struct GraphEncoder {
    max_nodes: usize,
}

impl GraphEncoder {
    /// Create a new graph encoder
    pub fn new(max_nodes: usize) -> Self {
        Self { max_nodes }
    }
    
    /// Encode expression as a computation graph
    pub fn encode_expression(&self, expr: &Expression) -> Result<GraphEncoding> {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut node_counter = 0;
        
        self.build_graph(expr, &mut nodes, &mut edges, &mut node_counter)?;
        
        Ok(GraphEncoding {
            nodes,
            edges: edges.clone(),
            adjacency_matrix: self.build_adjacency_matrix(&edges, node_counter),
        })
    }
    
    fn build_graph(
        &self,
        expr: &Expression,
        nodes: &mut Vec<GraphNode>,
        edges: &mut Vec<(usize, usize)>,
        counter: &mut usize
    ) -> Result<usize> {
        let node_id = *counter;
        *counter += 1;
        
        if *counter > self.max_nodes {
            return Err(NEATError::Other(anyhow::anyhow!("Expression too complex")));
        }
        
        let node = match expr {
            Expression::Constant(val) => GraphNode::Constant(*val),
            Expression::Variable(name) => GraphNode::Variable(name.clone()),
            Expression::BinaryOp { left, op, right } => {
                let left_id = self.build_graph(left, nodes, edges, counter)?;
                let right_id = self.build_graph(right, nodes, edges, counter)?;
                edges.push((left_id, node_id));
                edges.push((right_id, node_id));
                GraphNode::Operation(*op)
            }
            Expression::UnaryOp { op, operand } => {
                let operand_id = self.build_graph(operand, nodes, edges, counter)?;
                edges.push((operand_id, node_id));
                GraphNode::UnaryOp(*op)
            }
            Expression::Power { base, exponent } => {
                let base_id = self.build_graph(base, nodes, edges, counter)?;
                edges.push((base_id, node_id));
                GraphNode::Power(*exponent)
            }
        };
        
        nodes.push(node);
        Ok(node_id)
    }
    
    fn build_adjacency_matrix(&self, edges: &[(usize, usize)], size: usize) -> Vec<Vec<f64>> {
        let mut matrix = vec![vec![0.0; size]; size];
        for &(from, to) in edges {
            matrix[from][to] = 1.0;
        }
        matrix
    }
}

/// Node types in computation graph
#[derive(Debug, Clone)]
enum GraphNode {
    Constant(f64),
    Variable(String),
    Operation(Operation),
    UnaryOp(UnaryOperation),
    Power(i32),
}

/// Graph encoding result
#[derive(Debug)]
pub struct GraphEncoding {
    nodes: Vec<GraphNode>,
    edges: Vec<(usize, usize)>,
    adjacency_matrix: Vec<Vec<f64>>,
}

impl GraphEncoding {
    /// Convert to flat vector for neural network
    pub fn to_vector(&self) -> Vec<f64> {
        let mut encoding = Vec::new();
        
        // Encode nodes
        for node in &self.nodes {
            let node_enc = match node {
                GraphNode::Constant(val) => vec![1.0, 0.0, 0.0, 0.0, 0.0, val / 100.0],
                GraphNode::Variable(_) => vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                GraphNode::Operation(_) => vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                GraphNode::UnaryOp(_) => vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                GraphNode::Power(exp) => vec![0.0, 0.0, 0.0, 0.0, 1.0, *exp as f64 / 10.0],
            };
            encoding.extend(node_enc);
        }
        
        // Flatten adjacency matrix
        for row in &self.adjacency_matrix {
            encoding.extend(row);
        }
        
        encoding
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_algebra_encoding() -> Result<()> {
        let config = AlgebraEncodingConfig::default();
        let mut encoder = AlgebraEncoder::new(config);
        
        // Register variables
        encoder.register_variables(&["x".to_string(), "y".to_string()])?;
        
        // Create expression: x + 2*y
        let expr = Expression::binary(
            Expression::variable("x"),
            Operation::Add,
            Expression::binary(
                Expression::constant(2.0),
                Operation::Multiply,
                Expression::variable("y")
            )
        );
        
        let encoding = encoder.encode_expression(&expr)?;
        assert!(!encoding.is_empty());
        
        Ok(())
    }
    
    #[test]
    fn test_graph_encoding() -> Result<()> {
        let encoder = GraphEncoder::new(10);
        
        // Simple expression: x + 1
        let expr = Expression::binary(
            Expression::variable("x"),
            Operation::Add,
            Expression::constant(1.0)
        );
        
        let graph = encoder.encode_expression(&expr)?;
        let vector = graph.to_vector();
        assert!(!vector.is_empty());
        
        Ok(())
    }
}