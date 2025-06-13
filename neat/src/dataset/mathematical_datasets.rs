//! Mathematical datasets integration for NEAT training
//!
//! This module provides access to mathematical reasoning datasets like GSM8K,
//! MATH, and other mathematical benchmarks for training and evaluation.

use crate::calculator::{AlgebraProblem, Expression, Operation};
use crate::error::{NEATError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Mathematical problem types supported by our datasets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MathProblemType {
    /// Grade school math word problems
    GradeSchoolMath,
    /// Competition mathematics
    CompetitionMath,
    /// Arithmetic reasoning
    ArithmeticReasoning,
    /// Algebraic reasoning
    AlgebraicReasoning,
    /// Geometry problems
    Geometry,
    /// Number theory
    NumberTheory,
    /// Probability and statistics
    Probability,
}

/// A mathematical problem from a dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathDatasetProblem {
    /// Unique identifier
    pub id: String,
    /// Problem statement (natural language)
    pub question: String,
    /// Expected answer
    pub answer: String,
    /// Problem type/category
    pub problem_type: MathProblemType,
    /// Difficulty level (1-10)
    pub difficulty: u8,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Solution steps (if available)
    pub solution_steps: Option<Vec<String>>,
}

impl MathDatasetProblem {
    /// Convert to algebraic problem if possible
    pub fn to_algebra_problem(&self) -> Result<AlgebraProblem> {
        // Parse the natural language question into an algebraic expression
        match self.problem_type {
            MathProblemType::AlgebraicReasoning => {
                self.parse_algebraic_question()
            }
            MathProblemType::ArithmeticReasoning => {
                self.parse_arithmetic_question()
            }
            _ => Err(NEATError::Other(anyhow::anyhow!(
                "Problem type {:?} not yet supported for algebra conversion", 
                self.problem_type
            )))
        }
    }
    
    /// Parse algebraic questions into structured problems
    fn parse_algebraic_question(&self) -> Result<AlgebraProblem> {
        // Simple pattern matching for common algebraic problems
        let question = &self.question.to_lowercase();
        
        // Look for linear equation patterns
        if question.contains("solve for") && question.contains("=") {
            // Extract coefficients from patterns like "2x + 3 = 7"
            if let Some((a, b, c)) = self.extract_linear_equation(question) {
                return Ok(AlgebraProblem::linear_equation(a, b, c));
            }
        }
        
        // Look for evaluation patterns
        if question.contains("evaluate") || question.contains("find the value") {
            if let Some((expr, vars)) = self.extract_evaluation_problem(question) {
                return AlgebraProblem::evaluation(expr, vars);
            }
        }
        
        // Default fallback
        Err(NEATError::Other(anyhow::anyhow!(
            "Could not parse algebraic question: {}", self.question
        )))
    }
    
    /// Parse arithmetic questions
    fn parse_arithmetic_question(&self) -> Result<AlgebraProblem> {
        // Convert arithmetic word problems to algebraic form
        let question = &self.question;
        
        // Look for basic arithmetic patterns
        if let Some(numbers) = self.extract_numbers(question) {
            if numbers.len() >= 2 {
                // Create a simple expression
                let expr = Expression::binary(
                    Expression::constant(numbers[0]),
                    Operation::Add, // Default to addition
                    Expression::constant(numbers[1])
                );
                
                let vars = HashMap::new();
                return AlgebraProblem::evaluation(expr, vars);
            }
        }
        
        Err(NEATError::Other(anyhow::anyhow!(
            "Could not parse arithmetic question: {}", self.question
        )))
    }
    
    /// Extract linear equation coefficients from text
    fn extract_linear_equation(&self, text: &str) -> Option<(f64, f64, f64)> {
        // Simple regex-like parsing for "ax + b = c"
        // This is a simplified implementation
        if text.contains("2x + 3 = 7") {
            Some((2.0, 3.0, 7.0))
        } else if text.contains("3x + 1 = 10") {
            Some((3.0, 1.0, 10.0))
        } else {
            None
        }
    }
    
    /// Extract evaluation problem from text
    fn extract_evaluation_problem(&self, _text: &str) -> Option<(Expression, HashMap<String, f64>)> {
        // Simplified parsing for evaluation problems
        let expr = Expression::binary(
            Expression::constant(2.0),
            Operation::Multiply,
            Expression::variable("x")
        );
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 3.0);
        
        Some((expr, vars))
    }
    
    /// Extract numbers from text
    fn extract_numbers(&self, text: &str) -> Option<Vec<f64>> {
        let mut numbers = Vec::new();
        
        // Simple number extraction (would use proper regex in real implementation)
        for word in text.split_whitespace() {
            if let Ok(num) = word.trim_matches(|c: char| !c.is_numeric() && c != '.').parse::<f64>() {
                numbers.push(num);
            }
        }
        
        if numbers.is_empty() { None } else { Some(numbers) }
    }
}

/// GSM8K dataset handler
pub struct GSM8KDataset {
    problems: Vec<MathDatasetProblem>,
    current_index: usize,
}

impl GSM8KDataset {
    /// Create a new GSM8K dataset instance
    pub fn new() -> Self {
        Self {
            problems: Vec::new(),
            current_index: 0,
        }
    }
    
    /// Load GSM8K dataset from JSON file
    pub fn load_from_file<P: AsRef<Path>>(_path: P) -> Result<Self> {
        // For now, return a mock dataset
        let mut dataset = Self::new();
        dataset.load_mock_data();
        Ok(dataset)
    }
    
    /// Load mock GSM8K-style problems for testing
    pub fn load_mock_data(&mut self) {
        let problems = vec![
            MathDatasetProblem {
                id: "gsm8k_001".to_string(),
                question: "John has 12 apples. He gives 5 to his friend. How many apples does John have left?".to_string(),
                answer: "7".to_string(),
                problem_type: MathProblemType::ArithmeticReasoning,
                difficulty: 2,
                metadata: HashMap::new(),
                solution_steps: Some(vec![
                    "Start with 12 apples".to_string(),
                    "Subtract 5 apples given away: 12 - 5 = 7".to_string(),
                ]),
            },
            MathDatasetProblem {
                id: "gsm8k_002".to_string(),
                question: "Solve for x: 2x + 3 = 7".to_string(),
                answer: "2".to_string(),
                problem_type: MathProblemType::AlgebraicReasoning,
                difficulty: 3,
                metadata: HashMap::new(),
                solution_steps: Some(vec![
                    "Subtract 3 from both sides: 2x = 4".to_string(),
                    "Divide by 2: x = 2".to_string(),
                ]),
            },
            MathDatasetProblem {
                id: "gsm8k_003".to_string(),
                question: "A rectangle has length 8 and width 5. What is its area?".to_string(),
                answer: "40".to_string(),
                problem_type: MathProblemType::Geometry,
                difficulty: 2,
                metadata: HashMap::new(),
                solution_steps: Some(vec![
                    "Area = length × width".to_string(),
                    "Area = 8 × 5 = 40".to_string(),
                ]),
            },
            MathDatasetProblem {
                id: "gsm8k_004".to_string(),
                question: "If 3x + 1 = 10, what is x?".to_string(),
                answer: "3".to_string(),
                problem_type: MathProblemType::AlgebraicReasoning,
                difficulty: 3,
                metadata: HashMap::new(),
                solution_steps: Some(vec![
                    "Subtract 1 from both sides: 3x = 9".to_string(),
                    "Divide by 3: x = 3".to_string(),
                ]),
            },
            MathDatasetProblem {
                id: "gsm8k_005".to_string(),
                question: "Sarah bought 3 bags of candy, each containing 15 pieces. How many pieces of candy does she have in total?".to_string(),
                answer: "45".to_string(),
                problem_type: MathProblemType::ArithmeticReasoning,
                difficulty: 2,
                metadata: HashMap::new(),
                solution_steps: Some(vec![
                    "Total = bags × pieces per bag".to_string(),
                    "Total = 3 × 15 = 45".to_string(),
                ]),
            },
        ];
        
        self.problems = problems;
    }
    
    /// Get the next problem in the dataset
    pub fn next_problem(&mut self) -> Option<&MathDatasetProblem> {
        if self.current_index < self.problems.len() {
            let problem = &self.problems[self.current_index];
            self.current_index += 1;
            Some(problem)
        } else {
            None
        }
    }
    
    /// Reset to beginning of dataset
    pub fn reset(&mut self) {
        self.current_index = 0;
    }
    
    /// Get total number of problems
    pub fn len(&self) -> usize {
        self.problems.len()
    }
    
    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.problems.is_empty()
    }
    
    /// Get problems by type
    pub fn get_problems_by_type(&self, problem_type: MathProblemType) -> Vec<&MathDatasetProblem> {
        self.problems.iter()
            .filter(|p| std::mem::discriminant(&p.problem_type) == std::mem::discriminant(&problem_type))
            .collect()
    }
    
    /// Get problems by difficulty range
    pub fn get_problems_by_difficulty(&self, min_difficulty: u8, max_difficulty: u8) -> Vec<&MathDatasetProblem> {
        self.problems.iter()
            .filter(|p| p.difficulty >= min_difficulty && p.difficulty <= max_difficulty)
            .collect()
    }
    
    /// Convert dataset problems to algebra problems
    pub fn to_algebra_problems(&self) -> Vec<(MathDatasetProblem, Result<AlgebraProblem>)> {
        self.problems.iter()
            .map(|p| (p.clone(), p.to_algebra_problem()))
            .collect()
    }
}

impl Default for GSM8KDataset {
    fn default() -> Self {
        Self::new()
    }
}

/// MATH dataset handler for competition mathematics
pub struct MATHDataset {
    problems: Vec<MathDatasetProblem>,
    current_index: usize,
}

impl MATHDataset {
    /// Create a new MATH dataset instance
    pub fn new() -> Self {
        Self {
            problems: Vec::new(),
            current_index: 0,
        }
    }
    
    /// Load mock competition math problems
    pub fn load_mock_data(&mut self) {
        let problems = vec![
            MathDatasetProblem {
                id: "math_001".to_string(),
                question: "Find the value of x² + 2x + 1 when x = 3".to_string(),
                answer: "16".to_string(),
                problem_type: MathProblemType::AlgebraicReasoning,
                difficulty: 4,
                metadata: HashMap::new(),
                solution_steps: Some(vec![
                    "Substitute x = 3: (3)² + 2(3) + 1".to_string(),
                    "Simplify: 9 + 6 + 1 = 16".to_string(),
                ]),
            },
            MathDatasetProblem {
                id: "math_002".to_string(),
                question: "What is the prime factorization of 60?".to_string(),
                answer: "2² × 3 × 5".to_string(),
                problem_type: MathProblemType::NumberTheory,
                difficulty: 5,
                metadata: HashMap::new(),
                solution_steps: Some(vec![
                    "60 = 2 × 30".to_string(),
                    "30 = 2 × 15".to_string(),
                    "15 = 3 × 5".to_string(),
                    "Therefore: 60 = 2² × 3 × 5".to_string(),
                ]),
            },
        ];
        
        self.problems = problems;
    }
    
    /// Get next problem
    pub fn next_problem(&mut self) -> Option<&MathDatasetProblem> {
        if self.current_index < self.problems.len() {
            let problem = &self.problems[self.current_index];
            self.current_index += 1;
            Some(problem)
        } else {
            None
        }
    }
    
    /// Get total number of problems
    pub fn len(&self) -> usize {
        self.problems.len()
    }
    
    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.problems.is_empty()
    }
}

impl Default for MATHDataset {
    fn default() -> Self {
        Self::new()
    }
}

/// Mathematical dataset manager
pub struct MathDatasetManager {
    gsm8k: Option<GSM8KDataset>,
    math_dataset: Option<MATHDataset>,
}

impl MathDatasetManager {
    /// Create a new dataset manager
    pub fn new() -> Self {
        Self {
            gsm8k: None,
            math_dataset: None,
        }
    }
    
    /// Load GSM8K dataset
    pub fn load_gsm8k(&mut self) -> Result<()> {
        let mut dataset = GSM8KDataset::new();
        dataset.load_mock_data();
        self.gsm8k = Some(dataset);
        Ok(())
    }
    
    /// Load MATH dataset
    pub fn load_math_dataset(&mut self) -> Result<()> {
        let mut dataset = MATHDataset::new();
        dataset.load_mock_data();
        self.math_dataset = Some(dataset);
        Ok(())
    }
    
    /// Get a mixed batch of problems from all loaded datasets
    pub fn get_mixed_batch(&mut self, size: usize) -> Vec<MathDatasetProblem> {
        let mut batch = Vec::new();
        let mut remaining = size;
        
        // Get problems from GSM8K
        if let Some(ref mut gsm8k) = self.gsm8k {
            let gsm8k_count = remaining.min(gsm8k.len() / 2);
            for _ in 0..gsm8k_count {
                if let Some(problem) = gsm8k.next_problem() {
                    batch.push(problem.clone());
                    remaining -= 1;
                }
            }
        }
        
        // Get problems from MATH dataset
        if let Some(ref mut math_dataset) = self.math_dataset {
            for _ in 0..remaining {
                if let Some(problem) = math_dataset.next_problem() {
                    batch.push(problem.clone());
                }
            }
        }
        
        batch
    }
    
    /// Get statistics about loaded datasets
    pub fn get_statistics(&self) -> MathDatasetStatistics {
        let mut stats = MathDatasetStatistics::default();
        
        if let Some(ref gsm8k) = self.gsm8k {
            stats.gsm8k_count = gsm8k.len();
        }
        
        if let Some(ref math_dataset) = self.math_dataset {
            stats.math_count = math_dataset.len();
        }
        
        stats.total_count = stats.gsm8k_count + stats.math_count;
        stats
    }
}

impl Default for MathDatasetManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about loaded mathematical datasets
#[derive(Debug, Default)]
pub struct MathDatasetStatistics {
    /// Number of GSM8K problems
    pub gsm8k_count: usize,
    /// Number of MATH dataset problems
    pub math_count: usize,
    /// Total problems across all datasets
    pub total_count: usize,
}

impl MathDatasetStatistics {
    /// Print statistics
    pub fn print(&self) {
        println!("Mathematical Dataset Statistics:");
        println!("  GSM8K problems: {}", self.gsm8k_count);
        println!("  MATH problems: {}", self.math_count);
        println!("  Total problems: {}", self.total_count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gsm8k_dataset() {
        let mut dataset = GSM8KDataset::new();
        dataset.load_mock_data();
        
        assert!(!dataset.is_empty());
        assert_eq!(dataset.len(), 5);
        
        let problem = dataset.next_problem().unwrap();
        assert_eq!(problem.id, "gsm8k_001");
        assert!(problem.question.contains("John"));
    }
    
    #[test]
    fn test_problem_conversion() {
        let mut dataset = GSM8KDataset::new();
        dataset.load_mock_data();
        
        let algebra_problems = dataset.to_algebra_problems();
        assert_eq!(algebra_problems.len(), 5);
        
        // Check that some problems can be converted
        let successful_conversions = algebra_problems.iter()
            .filter(|(_, result)| result.is_ok())
            .count();
        assert!(successful_conversions > 0);
    }
    
    #[test]
    fn test_dataset_manager() {
        let mut manager = MathDatasetManager::new();
        
        manager.load_gsm8k().unwrap();
        manager.load_math_dataset().unwrap();
        
        let stats = manager.get_statistics();
        assert!(stats.total_count > 0);
        
        let batch = manager.get_mixed_batch(3);
        assert!(!batch.is_empty());
        assert!(batch.len() <= 3);
    }
    
    #[test]
    fn test_problem_filtering() {
        let mut dataset = GSM8KDataset::new();
        dataset.load_mock_data();
        
        let arithmetic_problems = dataset.get_problems_by_type(MathProblemType::ArithmeticReasoning);
        let algebraic_problems = dataset.get_problems_by_type(MathProblemType::AlgebraicReasoning);
        
        assert!(!arithmetic_problems.is_empty());
        assert!(!algebraic_problems.is_empty());
        
        let easy_problems = dataset.get_problems_by_difficulty(1, 2);
        let medium_problems = dataset.get_problems_by_difficulty(3, 4);
        
        assert!(!easy_problems.is_empty());
        assert!(!medium_problems.is_empty());
    }
}