//! Discrete Mathematics Module
//!
//! This module implements discrete mathematical operations including
//! combinatorics, graph theory, set theory, and discrete probability
//! with efficient algorithms for finite mathematical structures.

use crate::error::{NEATError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Types of discrete mathematics operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscreteOperation {
    /// Combinatorics (permutations, combinations)
    Combinatorics,
    /// Graph theory operations
    GraphTheory,
    /// Set theory operations
    SetTheory,
    /// Discrete probability
    DiscreteProbability,
    /// Number sequences
    Sequences,
    /// Boolean algebra
    BooleanAlgebra,
    /// Modular arithmetic
    ModularArithmetic,
}

/// Combinatorial operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CombinatorialType {
    /// Permutations P(n,r) = n!/(n-r)!
    Permutation,
    /// Combinations C(n,r) = n!/(r!(n-r)!)
    Combination,
    /// Factorial n!
    Factorial,
    /// Derangements (permutations with no fixed points)
    Derangement,
    /// Catalan numbers
    CatalanNumber,
    /// Stirling numbers
    StirlingNumber,
}

/// Graph representation for discrete operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscreteGraph {
    /// Number of vertices
    pub vertices: usize,
    /// Adjacency list representation
    pub edges: HashMap<usize, Vec<usize>>,
    /// Whether the graph is directed
    pub is_directed: bool,
    /// Whether the graph is weighted
    pub is_weighted: bool,
    /// Edge weights (if weighted)
    pub weights: HashMap<(usize, usize), f64>,
}

/// Set operations for discrete mathematics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscreteSet {
    /// Elements in the set
    pub elements: HashSet<i32>,
    /// Set name/identifier
    pub name: String,
}

/// Result of discrete mathematics computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscreteResult {
    /// Type of operation performed
    pub operation: DiscreteOperation,
    /// Numerical result (if applicable)
    pub numerical_result: Option<f64>,
    /// Set result (for set operations)
    pub set_result: Option<DiscreteSet>,
    /// Boolean result (for logical operations)
    pub boolean_result: Option<bool>,
    /// Path result (for graph operations)
    pub path_result: Option<Vec<usize>>,
    /// Explanation of the computation
    pub explanation: String,
    /// Computational complexity
    pub complexity: String,
}

/// Main discrete mathematics engine
pub struct DiscreteMathEngine {
    /// Precision for floating point computations
    precision: f64,
    /// Maximum factorial to compute (prevent overflow)
    max_factorial: u64,
    /// Cache for computed factorials
    factorial_cache: HashMap<u64, u64>,
    /// Cache for computed combinations
    combination_cache: HashMap<(u64, u64), u64>,
}

impl Default for DiscreteMathEngine {
    fn default() -> Self {
        Self {
            precision: 1e-12,
            max_factorial: 20,
            factorial_cache: HashMap::new(),
            combination_cache: HashMap::new(),
        }
    }
}

impl DiscreteMathEngine {
    /// Create a new discrete mathematics engine
    pub fn new(max_factorial: u64) -> Self {
        Self {
            precision: 1e-12,
            max_factorial,
            factorial_cache: HashMap::new(),
            combination_cache: HashMap::new(),
        }
    }

    /// Compute combinatorial values
    pub fn combinatorics(&mut self, operation: CombinatorialType, n: u64, r: Option<u64>) -> Result<DiscreteResult> {
        let result = match operation {
            CombinatorialType::Factorial => {
                if n > self.max_factorial {
                    return Err(NEATError::InvalidConfiguration {
                        parameter: "factorial".to_string(),
                        value: format!("n = {} exceeds maximum {}", n, self.max_factorial),
                    });
                }
                
                let factorial_n = self.factorial(n);
                DiscreteResult {
                    operation: DiscreteOperation::Combinatorics,
                    numerical_result: Some(factorial_n as f64),
                    set_result: None,
                    boolean_result: None,
                    path_result: None,
                    explanation: format!("{}! = {}", n, factorial_n),
                    complexity: "O(n)".to_string(),
                }
            },
            
            CombinatorialType::Permutation => {
                let r = r.ok_or_else(|| NEATError::InvalidConfiguration {
                    parameter: "r_value".to_string(),
                    value: "r must be provided for permutations".to_string(),
                })?;
                
                if r > n {
                    return Err(NEATError::InvalidConfiguration {
                        parameter: "permutation".to_string(),
                        value: "r cannot be greater than n".to_string(),
                    });
                }
                
                let perm = self.permutation(n, r)?;
                DiscreteResult {
                    operation: DiscreteOperation::Combinatorics,
                    numerical_result: Some(perm as f64),
                    set_result: None,
                    boolean_result: None,
                    path_result: None,
                    explanation: format!("P({},{}) = {}!/({}−{})! = {}", n, r, n, n, r, perm),
                    complexity: "O(r)".to_string(),
                }
            },
            
            CombinatorialType::Combination => {
                let r = r.ok_or_else(|| NEATError::InvalidConfiguration {
                    parameter: "r_value".to_string(),
                    value: "r must be provided for combinations".to_string(),
                })?;
                
                if r > n {
                    return Err(NEATError::InvalidConfiguration {
                        parameter: "combination".to_string(),
                        value: "r cannot be greater than n".to_string(),
                    });
                }
                
                let comb = self.combination(n, r)?;
                DiscreteResult {
                    operation: DiscreteOperation::Combinatorics,
                    numerical_result: Some(comb as f64),
                    set_result: None,
                    boolean_result: None,
                    path_result: None,
                    explanation: format!("C({},{}) = {}!/({}!×{}!) = {}", n, r, n, r, n-r, comb),
                    complexity: "O(min(r, n-r))".to_string(),
                }
            },
            
            CombinatorialType::CatalanNumber => {
                let catalan = self.catalan_number(n as usize);
                DiscreteResult {
                    operation: DiscreteOperation::Combinatorics,
                    numerical_result: Some(catalan as f64),
                    set_result: None,
                    boolean_result: None,
                    path_result: None,
                    explanation: format!("Catalan({}) = C(2×{}, {})/({}+1) = {}", n, n, n, n, catalan),
                    complexity: "O(n)".to_string(),
                }
            },
            
            _ => {
                return Err(NEATError::InvalidConfiguration {
                    parameter: "operation".to_string(),
                    value: format!("Operation {:?} not implemented", operation),
                });
            }
        };

        Ok(result)
    }

    /// Perform set operations
    pub fn set_operations(&self, set1: &DiscreteSet, set2: &DiscreteSet, operation: &str) -> Result<DiscreteResult> {
        let result_set = match operation {
            "union" => {
                let mut union_set = set1.elements.clone();
                union_set.extend(&set2.elements);
                DiscreteSet {
                    elements: union_set,
                    name: format!("{} ∪ {}", set1.name, set2.name),
                }
            },
            
            "intersection" => {
                let intersection_set: HashSet<i32> = set1.elements.intersection(&set2.elements).cloned().collect();
                DiscreteSet {
                    elements: intersection_set,
                    name: format!("{} ∩ {}", set1.name, set2.name),
                }
            },
            
            "difference" => {
                let difference_set: HashSet<i32> = set1.elements.difference(&set2.elements).cloned().collect();
                DiscreteSet {
                    elements: difference_set,
                    name: format!("{} − {}", set1.name, set2.name),
                }
            },
            
            "symmetric_difference" => {
                let sym_diff_set: HashSet<i32> = set1.elements.symmetric_difference(&set2.elements).cloned().collect();
                DiscreteSet {
                    elements: sym_diff_set,
                    name: format!("{} △ {}", set1.name, set2.name),
                }
            },
            
            _ => {
                return Err(NEATError::InvalidConfiguration {
                    parameter: "set_operation".to_string(),
                    value: format!("Unknown operation: {}", operation),
                });
            }
        };

        Ok(DiscreteResult {
            operation: DiscreteOperation::SetTheory,
            numerical_result: Some(result_set.elements.len() as f64),
            set_result: Some(result_set),
            boolean_result: None,
            path_result: None,
            explanation: format!("Set {} on {} and {}", operation, set1.name, set2.name),
            complexity: "O(|A| + |B|)".to_string(),
        })
    }

    /// Graph theory operations
    pub fn graph_operations(&self, graph: &DiscreteGraph, operation: &str, start: Option<usize>, end: Option<usize>) -> Result<DiscreteResult> {
        match operation {
            "is_connected" => {
                let connected = self.is_connected(graph);
                Ok(DiscreteResult {
                    operation: DiscreteOperation::GraphTheory,
                    numerical_result: None,
                    set_result: None,
                    boolean_result: Some(connected),
                    path_result: None,
                    explanation: format!("Graph is {}", if connected { "connected" } else { "not connected" }),
                    complexity: "O(V + E)".to_string(),
                })
            },
            
            "shortest_path" => {
                let start = start.ok_or_else(|| NEATError::InvalidConfiguration {
                    parameter: "start_vertex".to_string(),
                    value: "Start vertex required for shortest path".to_string(),
                })?;
                
                let end = end.ok_or_else(|| NEATError::InvalidConfiguration {
                    parameter: "end_vertex".to_string(),
                    value: "End vertex required for shortest path".to_string(),
                })?;
                
                let path = self.bfs_shortest_path(graph, start, end);
                let path_length = if path.is_empty() { None } else { Some((path.len() - 1) as f64) };
                
                Ok(DiscreteResult {
                    operation: DiscreteOperation::GraphTheory,
                    numerical_result: path_length,
                    set_result: None,
                    boolean_result: None,
                    path_result: Some(path.clone()),
                    explanation: if path.is_empty() {
                        format!("No path from {} to {}", start, end)
                    } else {
                        format!("Shortest path from {} to {}: {:?} (length: {})", start, end, path, path.len() - 1)
                    },
                    complexity: "O(V + E)".to_string(),
                })
            },
            
            "vertex_count" => {
                Ok(DiscreteResult {
                    operation: DiscreteOperation::GraphTheory,
                    numerical_result: Some(graph.vertices as f64),
                    set_result: None,
                    boolean_result: None,
                    path_result: None,
                    explanation: format!("Graph has {} vertices", graph.vertices),
                    complexity: "O(1)".to_string(),
                })
            },
            
            "edge_count" => {
                let edge_count = if graph.is_directed {
                    graph.edges.values().map(|adj| adj.len()).sum::<usize>()
                } else {
                    graph.edges.values().map(|adj| adj.len()).sum::<usize>() / 2
                };
                
                Ok(DiscreteResult {
                    operation: DiscreteOperation::GraphTheory,
                    numerical_result: Some(edge_count as f64),
                    set_result: None,
                    boolean_result: None,
                    path_result: None,
                    explanation: format!("Graph has {} edges", edge_count),
                    complexity: "O(V)".to_string(),
                })
            },
            
            _ => Err(NEATError::InvalidConfiguration {
                parameter: "graph_operation".to_string(),
                value: format!("Unknown operation: {}", operation),
            })
        }
    }

    /// Discrete probability calculations
    pub fn discrete_probability(&self, total_outcomes: u64, favorable_outcomes: u64) -> Result<DiscreteResult> {
        if favorable_outcomes > total_outcomes {
            return Err(NEATError::InvalidConfiguration {
                parameter: "probability".to_string(),
                value: "Favorable outcomes cannot exceed total outcomes".to_string(),
            });
        }

        if total_outcomes == 0 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "total_outcomes".to_string(),
                value: "Total outcomes cannot be zero".to_string(),
            });
        }

        let probability = favorable_outcomes as f64 / total_outcomes as f64;

        Ok(DiscreteResult {
            operation: DiscreteOperation::DiscreteProbability,
            numerical_result: Some(probability),
            set_result: None,
            boolean_result: None,
            path_result: None,
            explanation: format!("P(Event) = {}/{} = {:.6}", favorable_outcomes, total_outcomes, probability),
            complexity: "O(1)".to_string(),
        })
    }

    /// Modular arithmetic operations
    pub fn modular_arithmetic(&self, a: i64, b: i64, modulus: i64, operation: &str) -> Result<DiscreteResult> {
        if modulus <= 0 {
            return Err(NEATError::InvalidConfiguration {
                parameter: "modulus".to_string(),
                value: "Modulus must be positive".to_string(),
            });
        }

        let result = match operation {
            "add" => (a + b) % modulus,
            "subtract" => ((a - b) % modulus + modulus) % modulus,
            "multiply" => (a * b) % modulus,
            "power" => self.mod_pow(a, b, modulus),
            "inverse" => {
                if let Some(inv) = self.mod_inverse(a, modulus) {
                    inv
                } else {
                    return Err(NEATError::InvalidConfiguration {
                        parameter: "modular_inverse".to_string(),
                        value: format!("{} has no inverse modulo {}", a, modulus),
                    });
                }
            },
            _ => {
                return Err(NEATError::InvalidConfiguration {
                    parameter: "modular_operation".to_string(),
                    value: format!("Unknown operation: {}", operation),
                });
            }
        };

        Ok(DiscreteResult {
            operation: DiscreteOperation::ModularArithmetic,
            numerical_result: Some(result as f64),
            set_result: None,
            boolean_result: None,
            path_result: None,
            explanation: format!("{} {} {} ≡ {} (mod {})", a, operation, b, result, modulus),
            complexity: "O(log n)".to_string(),
        })
    }

    // Helper methods

    /// Compute factorial with caching
    fn factorial(&mut self, n: u64) -> u64 {
        if let Some(&cached) = self.factorial_cache.get(&n) {
            return cached;
        }

        let result = if n <= 1 {
            1
        } else {
            n * self.factorial(n - 1)
        };

        self.factorial_cache.insert(n, result);
        result
    }

    /// Compute permutation P(n, r)
    fn permutation(&mut self, n: u64, r: u64) -> Result<u64> {
        if r > n {
            return Err(NEATError::InvalidConfiguration {
                parameter: "permutation".to_string(),
                value: "r cannot be greater than n".to_string(),
            });
        }

        let mut result = 1;
        for i in 0..r {
            result *= n - i;
        }
        Ok(result)
    }

    /// Compute combination C(n, r) with caching
    fn combination(&mut self, n: u64, r: u64) -> Result<u64> {
        let r = r.min(n - r); // Use symmetry C(n,r) = C(n,n-r)
        
        if let Some(&cached) = self.combination_cache.get(&(n, r)) {
            return Ok(cached);
        }

        if r == 0 || r == n {
            self.combination_cache.insert((n, r), 1);
            return Ok(1);
        }

        let mut result = 1;
        for i in 0..r {
            result = result * (n - i) / (i + 1);
        }

        self.combination_cache.insert((n, r), result);
        Ok(result)
    }

    /// Compute Catalan number
    fn catalan_number(&mut self, n: usize) -> u64 {
        if n == 0 {
            return 1;
        }

        // C_n = C(2n, n) / (n + 1)
        let binomial = self.combination(2 * n as u64, n as u64).unwrap_or(0);
        binomial / (n as u64 + 1)
    }

    /// Check if graph is connected using DFS
    fn is_connected(&self, graph: &DiscreteGraph) -> bool {
        if graph.vertices == 0 {
            return true;
        }

        let mut visited = vec![false; graph.vertices];
        let mut stack = Vec::new();
        
        // Start DFS from vertex 0
        stack.push(0);
        visited[0] = true;
        let mut visited_count = 1;

        while let Some(vertex) = stack.pop() {
            if let Some(neighbors) = graph.edges.get(&vertex) {
                for &neighbor in neighbors {
                    if neighbor < graph.vertices && !visited[neighbor] {
                        visited[neighbor] = true;
                        visited_count += 1;
                        stack.push(neighbor);
                    }
                }
            }
        }

        visited_count == graph.vertices
    }

    /// Find shortest path using BFS
    fn bfs_shortest_path(&self, graph: &DiscreteGraph, start: usize, end: usize) -> Vec<usize> {
        if start >= graph.vertices || end >= graph.vertices {
            return vec![];
        }

        if start == end {
            return vec![start];
        }

        let mut visited = vec![false; graph.vertices];
        let mut parent = vec![None; graph.vertices];
        let mut queue = VecDeque::new();

        queue.push_back(start);
        visited[start] = true;

        while let Some(vertex) = queue.pop_front() {
            if let Some(neighbors) = graph.edges.get(&vertex) {
                for &neighbor in neighbors {
                    if neighbor < graph.vertices && !visited[neighbor] {
                        visited[neighbor] = true;
                        parent[neighbor] = Some(vertex);
                        queue.push_back(neighbor);

                        if neighbor == end {
                            // Reconstruct path
                            let mut path = Vec::new();
                            let mut current = Some(end);
                            
                            while let Some(vertex) = current {
                                path.push(vertex);
                                current = parent[vertex];
                            }
                            
                            path.reverse();
                            return path;
                        }
                    }
                }
            }
        }

        vec![] // No path found
    }

    /// Modular exponentiation
    fn mod_pow(&self, mut base: i64, mut exp: i64, modulus: i64) -> i64 {
        if modulus == 1 {
            return 0;
        }

        let mut result = 1;
        base %= modulus;

        while exp > 0 {
            if exp % 2 == 1 {
                result = (result * base) % modulus;
            }
            exp >>= 1;
            base = (base * base) % modulus;
        }

        result
    }

    /// Modular multiplicative inverse using extended Euclidean algorithm
    fn mod_inverse(&self, a: i64, modulus: i64) -> Option<i64> {
        let (gcd, x, _) = self.extended_gcd(a, modulus);
        
        if gcd != 1 {
            None // Inverse doesn't exist
        } else {
            Some(((x % modulus) + modulus) % modulus)
        }
    }

    /// Extended Euclidean algorithm
    fn extended_gcd(&self, a: i64, b: i64) -> (i64, i64, i64) {
        if a == 0 {
            (b, 0, 1)
        } else {
            let (gcd, x1, y1) = self.extended_gcd(b % a, a);
            let x = y1 - (b / a) * x1;
            let y = x1;
            (gcd, x, y)
        }
    }
}

/// Helper functions for creating discrete structures
pub mod structures {
    use super::*;

    /// Create a complete graph K_n
    pub fn complete_graph(n: usize) -> DiscreteGraph {
        let mut edges = HashMap::new();
        
        for i in 0..n {
            let mut neighbors = Vec::new();
            for j in 0..n {
                if i != j {
                    neighbors.push(j);
                }
            }
            edges.insert(i, neighbors);
        }

        DiscreteGraph {
            vertices: n,
            edges,
            is_directed: false,
            is_weighted: false,
            weights: HashMap::new(),
        }
    }

    /// Create a cycle graph C_n
    pub fn cycle_graph(n: usize) -> DiscreteGraph {
        let mut edges = HashMap::new();
        
        for i in 0..n {
            let neighbors = vec![(i + 1) % n, (i + n - 1) % n];
            edges.insert(i, neighbors);
        }

        DiscreteGraph {
            vertices: n,
            edges,
            is_directed: false,
            is_weighted: false,
            weights: HashMap::new(),
        }
    }

    /// Create a path graph P_n
    pub fn path_graph(n: usize) -> DiscreteGraph {
        let mut edges = HashMap::new();
        
        for i in 0..n {
            let mut neighbors = Vec::new();
            if i > 0 {
                neighbors.push(i - 1);
            }
            if i < n - 1 {
                neighbors.push(i + 1);
            }
            edges.insert(i, neighbors);
        }

        DiscreteGraph {
            vertices: n,
            edges,
            is_directed: false,
            is_weighted: false,
            weights: HashMap::new(),
        }
    }

    /// Create a discrete set from a vector
    pub fn create_set(elements: Vec<i32>, name: String) -> DiscreteSet {
        DiscreteSet {
            elements: elements.into_iter().collect(),
            name,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::structures::*;

    #[test]
    fn test_factorial() -> Result<()> {
        let mut engine = DiscreteMathEngine::default();
        
        let result = engine.combinatorics(CombinatorialType::Factorial, 5, None)?;
        assert_eq!(result.numerical_result, Some(120.0));
        
        Ok(())
    }

    #[test]
    fn test_combinations() -> Result<()> {
        let mut engine = DiscreteMathEngine::default();
        
        let result = engine.combinatorics(CombinatorialType::Combination, 5, Some(2))?;
        assert_eq!(result.numerical_result, Some(10.0)); // C(5,2) = 10
        
        Ok(())
    }

    #[test]
    fn test_permutations() -> Result<()> {
        let mut engine = DiscreteMathEngine::default();
        
        let result = engine.combinatorics(CombinatorialType::Permutation, 5, Some(3))?;
        assert_eq!(result.numerical_result, Some(60.0)); // P(5,3) = 60
        
        Ok(())
    }

    #[test]
    fn test_set_operations() -> Result<()> {
        let engine = DiscreteMathEngine::default();
        
        let set1 = create_set(vec![1, 2, 3, 4], "A".to_string());
        let set2 = create_set(vec![3, 4, 5, 6], "B".to_string());
        
        let union_result = engine.set_operations(&set1, &set2, "union")?;
        assert_eq!(union_result.numerical_result, Some(6.0)); // |A ∪ B| = 6
        
        let intersection_result = engine.set_operations(&set1, &set2, "intersection")?;
        assert_eq!(intersection_result.numerical_result, Some(2.0)); // |A ∩ B| = 2
        
        Ok(())
    }

    #[test]
    fn test_graph_connectivity() -> Result<()> {
        let engine = DiscreteMathEngine::default();
        
        let connected_graph = complete_graph(4);
        let result = engine.graph_operations(&connected_graph, "is_connected", None, None)?;
        assert_eq!(result.boolean_result, Some(true));
        
        Ok(())
    }

    #[test]
    fn test_shortest_path() -> Result<()> {
        let engine = DiscreteMathEngine::default();
        
        let path_graph = path_graph(5); // 0-1-2-3-4
        let result = engine.graph_operations(&path_graph, "shortest_path", Some(0), Some(4))?;
        
        assert_eq!(result.numerical_result, Some(4.0)); // Path length = 4
        assert_eq!(result.path_result, Some(vec![0, 1, 2, 3, 4]));
        
        Ok(())
    }

    #[test]
    fn test_discrete_probability() -> Result<()> {
        let engine = DiscreteMathEngine::default();
        
        let result = engine.discrete_probability(6, 1)?; // Rolling a specific number on a die
        assert!((result.numerical_result.unwrap() - 1.0/6.0).abs() < 1e-10);
        
        Ok(())
    }

    #[test]
    fn test_modular_arithmetic() -> Result<()> {
        let engine = DiscreteMathEngine::default();
        
        let result = engine.modular_arithmetic(7, 5, 12, "add")?;
        assert_eq!(result.numerical_result, Some(0.0)); // (7 + 5) mod 12 = 0
        
        let mult_result = engine.modular_arithmetic(7, 5, 12, "multiply")?;
        assert_eq!(mult_result.numerical_result, Some(11.0)); // (7 * 5) mod 12 = 11
        
        Ok(())
    }

    #[test]
    fn test_catalan_numbers() -> Result<()> {
        let mut engine = DiscreteMathEngine::default();
        
        let result = engine.combinatorics(CombinatorialType::CatalanNumber, 3, None)?;
        assert_eq!(result.numerical_result, Some(5.0)); // C_3 = 5
        
        Ok(())
    }

    #[test]
    fn test_modular_inverse() -> Result<()> {
        let engine = DiscreteMathEngine::default();
        
        let result = engine.modular_arithmetic(3, 0, 7, "inverse")?;
        assert_eq!(result.numerical_result, Some(5.0)); // 3 * 5 ≡ 1 (mod 7)
        
        Ok(())
    }
}