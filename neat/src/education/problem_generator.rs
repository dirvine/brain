//! Educational Problem Generator
//!
//! This module generates diverse mathematical problems tailored to student
//! skill levels, learning objectives, and educational contexts, with support
//! for multiple problem types and adaptive difficulty.

use super::{StudentModel, EducationConfig, DifficultyLevel};
use crate::error::{NEATError, Result};
use crate::calculator::{algebra::{AlgebraProblem, Expression}, Operation};
use crate::calculator::modules::ModuleType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rand::{thread_rng, Rng};
use chrono::{DateTime, Utc};

/// Types of mathematical problems that can be generated
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProblemType {
    /// Basic arithmetic operations
    BasicArithmetic,
    /// Linear equations and expressions
    LinearEquations,
    /// Polynomial operations
    PolynomialOperations,
    /// Fraction and decimal operations
    FractionDecimals,
    /// Word problems with context
    WordProblems,
    /// Pattern recognition and sequences
    PatternRecognition,
    /// Geometry and spatial reasoning
    GeometryProblems,
    /// Statistics and probability
    StatisticsProblems,
    /// Multi-step problem solving
    MultiStepProblems,
}

/// Problem difficulty levels for fine-grained control
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ProblemDifficulty {
    /// Very basic concepts
    VeryEasy = 1,
    /// Elementary level
    Easy = 2,
    /// Standard level
    Medium = 3,
    /// Challenging level
    Hard = 4,
    /// Advanced level
    VeryHard = 5,
}

impl From<u8> for ProblemDifficulty {
    fn from(value: u8) -> Self {
        match value {
            1 => ProblemDifficulty::VeryEasy,
            2 => ProblemDifficulty::Easy,
            3 => ProblemDifficulty::Medium,
            4 => ProblemDifficulty::Hard,
            _ => ProblemDifficulty::VeryHard,
        }
    }
}

impl From<ProblemDifficulty> for u8 {
    fn from(difficulty: ProblemDifficulty) -> Self {
        difficulty as u8
    }
}

/// Context for word problems and real-world applications
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProblemContext {
    /// Abstract mathematical context
    Abstract,
    /// Money and financial scenarios
    Financial,
    /// Time and scheduling
    TimeScheduling,
    /// Measurement and units
    Measurement,
    /// Sports and games
    Sports,
    /// Science and nature
    Science,
    /// Shopping and commerce
    Shopping,
    /// Travel and transportation
    Travel,
}

/// Educational problem with metadata and context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EducationalProblem {
    /// Problem identifier
    pub problem_id: String,
    /// Type of mathematical problem
    pub problem_type: ProblemType,
    /// Difficulty level
    pub difficulty: ProblemDifficulty,
    /// Mathematical content
    pub algebra_problem: AlgebraProblem,
    /// Human-readable problem statement
    pub problem_statement: String,
    /// Context for word problems
    pub context: ProblemContext,
    /// Expected solution
    pub expected_solution: f64,
    /// Learning objectives addressed
    pub learning_objectives: Vec<String>,
    /// Hints available for this problem
    pub hints: Vec<String>,
    /// Alternative solution methods
    pub solution_methods: Vec<String>,
    /// Estimated time to solve (minutes)
    pub estimated_time: u32,
    /// Prerequisites needed
    pub prerequisites: Vec<String>,
    /// When problem was generated
    pub generated_at: DateTime<Utc>,
}

/// Problem generation parameters for customization
#[derive(Debug, Clone)]
pub struct GenerationParameters {
    /// Target difficulty distribution
    pub difficulty_weights: HashMap<ProblemDifficulty, f64>,
    /// Problem type preferences
    pub type_preferences: HashMap<ProblemType, f64>,
    /// Context preferences for word problems
    pub context_preferences: HashMap<ProblemContext, f64>,
    /// Include step-by-step hints
    pub include_hints: bool,
    /// Maximum problem complexity
    pub max_complexity: u8,
    /// Prefer conceptual over computational problems
    pub conceptual_focus: bool,
}

impl Default for GenerationParameters {
    fn default() -> Self {
        let mut difficulty_weights = HashMap::new();
        difficulty_weights.insert(ProblemDifficulty::VeryEasy, 0.1);
        difficulty_weights.insert(ProblemDifficulty::Easy, 0.3);
        difficulty_weights.insert(ProblemDifficulty::Medium, 0.4);
        difficulty_weights.insert(ProblemDifficulty::Hard, 0.15);
        difficulty_weights.insert(ProblemDifficulty::VeryHard, 0.05);

        let mut type_preferences = HashMap::new();
        type_preferences.insert(ProblemType::BasicArithmetic, 0.2);
        type_preferences.insert(ProblemType::LinearEquations, 0.3);
        type_preferences.insert(ProblemType::PolynomialOperations, 0.15);
        type_preferences.insert(ProblemType::WordProblems, 0.2);
        type_preferences.insert(ProblemType::PatternRecognition, 0.1);
        type_preferences.insert(ProblemType::MultiStepProblems, 0.05);

        let mut context_preferences = HashMap::new();
        context_preferences.insert(ProblemContext::Abstract, 0.3);
        context_preferences.insert(ProblemContext::Financial, 0.2);
        context_preferences.insert(ProblemContext::Shopping, 0.2);
        context_preferences.insert(ProblemContext::TimeScheduling, 0.15);
        context_preferences.insert(ProblemContext::Sports, 0.15);

        Self {
            difficulty_weights,
            type_preferences,
            context_preferences,
            include_hints: true,
            max_complexity: 5,
            conceptual_focus: false,
        }
    }
}

/// Educational problem generation system
pub struct EducationalProblemGenerator {
    /// Educational configuration
    config: EducationConfig,
    /// Generation parameters
    parameters: GenerationParameters,
    /// Problem templates for different types
    templates: ProblemTemplates,
    /// Generated problem history
    problem_history: HashMap<String, Vec<EducationalProblem>>,
}

/// Templates for generating different types of problems
#[derive(Debug, Clone, Default)]
struct ProblemTemplates {
    /// Arithmetic problem templates
    arithmetic_templates: Vec<ArithmeticTemplate>,
    /// Linear equation templates
    linear_templates: Vec<LinearTemplate>,
    /// Word problem templates
    word_templates: Vec<WordTemplate>,
}

/// Template for arithmetic problems
#[derive(Debug, Clone)]
struct ArithmeticTemplate {
    /// Operation type
    operation: String,
    /// Number range for operands
    number_range: (i32, i32),
    /// Problem statement format
    statement_format: String,
}

/// Template for linear equation problems
#[derive(Debug, Clone)]
struct LinearTemplate {
    /// Coefficient ranges
    coefficient_range: (f64, f64),
    /// Constant ranges
    constant_range: (f64, f64),
    /// Problem statement format
    statement_format: String,
}

/// Template for word problems
#[derive(Debug, Clone)]
struct WordTemplate {
    /// Problem context
    context: ProblemContext,
    /// Story template
    story_template: String,
    /// Variable descriptions
    variable_descriptions: Vec<String>,
}

impl EducationalProblemGenerator {
    /// Create a new educational problem generator
    pub fn new(config: &EducationConfig) -> Self {
        let mut generator = Self {
            config: config.clone(),
            parameters: GenerationParameters::default(),
            templates: ProblemTemplates::default(),
            problem_history: HashMap::new(),
        };

        // Initialize problem templates
        generator.initialize_templates();

        generator
    }

    /// Generate a problem tailored to a specific student
    pub fn generate_for_student(&mut self, student_model: &StudentModel, topic: &str) -> Result<EducationalProblem> {
        // Determine appropriate difficulty based on student model
        let difficulty = self.determine_difficulty(student_model, topic);
        
        // Select problem type based on topic and preferences
        let problem_type = self.select_problem_type(topic);
        
        // Generate the actual problem
        let mut problem = self.generate_problem(problem_type, difficulty, topic)?;
        
        // Customize based on student preferences
        self.customize_for_student(&mut problem, student_model);
        
        // Record in history
        self.problem_history
            .entry(student_model.student_id.clone())
            .or_insert_with(Vec::new)
            .push(problem.clone());

        Ok(problem)
    }

    /// Generate a problem of specific type and difficulty
    pub fn generate_problem(&self, problem_type: ProblemType, difficulty: ProblemDifficulty, topic: &str) -> Result<EducationalProblem> {
        let problem_id = format!("prob_{}_{}", topic, Utc::now().timestamp_nanos());
        
        match problem_type {
            ProblemType::BasicArithmetic => self.generate_arithmetic_problem(problem_id, difficulty),
            ProblemType::LinearEquations => self.generate_linear_equation_problem(problem_id, difficulty),
            ProblemType::WordProblems => self.generate_word_problem(problem_id, difficulty, topic),
            ProblemType::PolynomialOperations => self.generate_polynomial_problem(problem_id, difficulty),
            ProblemType::PatternRecognition => self.generate_pattern_problem(problem_id, difficulty),
            _ => self.generate_linear_equation_problem(problem_id, difficulty), // Default fallback
        }
    }

    /// Generate basic arithmetic problem
    fn generate_arithmetic_problem(&self, problem_id: String, difficulty: ProblemDifficulty) -> Result<EducationalProblem> {
        let mut rng = thread_rng();
        
        let (min_val, max_val) = match difficulty {
            ProblemDifficulty::VeryEasy => (1, 10),
            ProblemDifficulty::Easy => (1, 50),
            ProblemDifficulty::Medium => (10, 100),
            ProblemDifficulty::Hard => (50, 500),
            ProblemDifficulty::VeryHard => (100, 1000),
        };

        let a = rng.gen_range(min_val..=max_val) as f64;
        let b = rng.gen_range(min_val..=max_val) as f64;
        let operation = if difficulty <= ProblemDifficulty::Easy { "+" } else { 
            match rng.gen_range(0..4) {
                0 => "+",
                1 => "-",
                2 => "×",
                _ => "÷",
            }
        };

        let (algebra_problem, expected_solution, statement) = match operation {
            "+" => {
                let sum = a + b;
                (AlgebraProblem::linear_equation(1.0, 0.0, sum), sum, 
                 format!("What is {} + {}?", a, b))
            },
            "-" => {
                let diff = a - b;
                (AlgebraProblem::linear_equation(1.0, b, a), a, 
                 format!("What is {} - {}?", a, b))
            },
            "×" => {
                let product = a * b;
                (AlgebraProblem::linear_equation(a, 0.0, product), b,
                 format!("What is {} × {}?", a, b))
            },
            "÷" => {
                let quotient = a / b;
                (AlgebraProblem::linear_equation(b, 0.0, a), quotient,
                 format!("What is {} ÷ {}?", a, b))
            },
            _ => unreachable!(),
        };

        Ok(EducationalProblem {
            problem_id,
            problem_type: ProblemType::BasicArithmetic,
            difficulty,
            algebra_problem,
            problem_statement: statement,
            context: ProblemContext::Abstract,
            expected_solution,
            learning_objectives: vec![format!("Perform {} operations", operation)],
            hints: self.generate_arithmetic_hints(operation, a, b),
            solution_methods: vec!["Direct calculation".to_string()],
            estimated_time: match difficulty {
                ProblemDifficulty::VeryEasy => 1,
                ProblemDifficulty::Easy => 2,
                ProblemDifficulty::Medium => 3,
                ProblemDifficulty::Hard => 5,
                ProblemDifficulty::VeryHard => 8,
            },
            prerequisites: if difficulty == ProblemDifficulty::VeryEasy { 
                vec![] 
            } else { 
                vec!["basic arithmetic".to_string()] 
            },
            generated_at: Utc::now(),
        })
    }

    /// Generate linear equation problem
    fn generate_linear_equation_problem(&self, problem_id: String, difficulty: ProblemDifficulty) -> Result<EducationalProblem> {
        let mut rng = thread_rng();
        
        let (a, b, c) = match difficulty {
            ProblemDifficulty::VeryEasy => (1.0, 0.0, rng.gen_range(1..=10) as f64),
            ProblemDifficulty::Easy => (rng.gen_range(1..=5) as f64, 0.0, rng.gen_range(5..=25) as f64),
            ProblemDifficulty::Medium => (
                rng.gen_range(2..=10) as f64,
                rng.gen_range(-10..=10) as f64,
                rng.gen_range(10..=50) as f64
            ),
            ProblemDifficulty::Hard => (
                rng.gen_range(3..=15) as f64,
                rng.gen_range(-20..=20) as f64,
                rng.gen_range(20..=100) as f64
            ),
            ProblemDifficulty::VeryHard => (
                rng.gen_range(5..=25) as f64,
                rng.gen_range(-50..=50) as f64,
                rng.gen_range(50..=200) as f64
            ),
        };

        let algebra_problem = AlgebraProblem::linear_equation(a, b, c);
        let expected_solution = (c - b) / a;

        let statement = if b == 0.0 {
            format!("Solve for x: {}x = {}", a, c)
        } else if b > 0.0 {
            format!("Solve for x: {}x + {} = {}", a, b, c)
        } else {
            format!("Solve for x: {}x - {} = {}", a, -b, c)
        };

        Ok(EducationalProblem {
            problem_id,
            problem_type: ProblemType::LinearEquations,
            difficulty,
            algebra_problem,
            problem_statement: statement,
            context: ProblemContext::Abstract,
            expected_solution,
            learning_objectives: vec!["Solve linear equations".to_string()],
            hints: self.generate_linear_equation_hints(a, b, c),
            solution_methods: vec!["Algebraic manipulation".to_string()],
            estimated_time: match difficulty {
                ProblemDifficulty::VeryEasy => 2,
                ProblemDifficulty::Easy => 3,
                ProblemDifficulty::Medium => 5,
                ProblemDifficulty::Hard => 7,
                ProblemDifficulty::VeryHard => 10,
            },
            prerequisites: vec!["algebraic expressions".to_string()],
            generated_at: Utc::now(),
        })
    }

    /// Generate word problem
    fn generate_word_problem(&self, problem_id: String, difficulty: ProblemDifficulty, topic: &str) -> Result<EducationalProblem> {
        let context = self.select_word_problem_context();
        let (statement, algebra_problem, expected_solution) = self.generate_word_problem_content(context.clone(), difficulty)?;

        Ok(EducationalProblem {
            problem_id,
            problem_type: ProblemType::WordProblems,
            difficulty,
            algebra_problem,
            problem_statement: statement,
            context: context.clone(),
            expected_solution,
            learning_objectives: vec![
                "Apply math to real-world scenarios".to_string(),
                "Translate word problems to equations".to_string(),
            ],
            hints: self.generate_word_problem_hints(context),
            solution_methods: vec!["Problem analysis and equation setup".to_string()],
            estimated_time: match difficulty {
                ProblemDifficulty::VeryEasy => 3,
                ProblemDifficulty::Easy => 5,
                ProblemDifficulty::Medium => 8,
                ProblemDifficulty::Hard => 12,
                ProblemDifficulty::VeryHard => 15,
            },
            prerequisites: vec!["linear equations".to_string(), "problem solving".to_string()],
            generated_at: Utc::now(),
        })
    }

    /// Generate polynomial problem
    fn generate_polynomial_problem(&self, problem_id: String, difficulty: ProblemDifficulty) -> Result<EducationalProblem> {
        // For now, generate a simple linear equation as placeholder
        // In a full implementation, this would generate actual polynomial problems
        self.generate_linear_equation_problem(problem_id, difficulty)
    }

    /// Generate pattern recognition problem
    fn generate_pattern_problem(&self, problem_id: String, difficulty: ProblemDifficulty) -> Result<EducationalProblem> {
        let mut rng = thread_rng();
        
        // Generate arithmetic sequence
        let first_term = rng.gen_range(1..=20);
        let common_diff = rng.gen_range(1..=10);
        let sequence_length = match difficulty {
            ProblemDifficulty::VeryEasy => 4,
            ProblemDifficulty::Easy => 5,
            ProblemDifficulty::Medium => 6,
            ProblemDifficulty::Hard => 7,
            ProblemDifficulty::VeryHard => 8,
        };

        let sequence: Vec<i32> = (0..sequence_length)
            .map(|i| first_term + i * common_diff)
            .collect();

        let next_term = first_term + sequence_length * common_diff;
        let statement = format!(
            "What is the next number in this sequence? {}, ?, ...",
            sequence.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(", ")
        );

        // Create a linear equation representing the pattern
        let algebra_problem = AlgebraProblem::linear_equation(common_diff as f64, first_term as f64, next_term as f64);

        Ok(EducationalProblem {
            problem_id,
            problem_type: ProblemType::PatternRecognition,
            difficulty,
            algebra_problem,
            problem_statement: statement,
            context: ProblemContext::Abstract,
            expected_solution: next_term as f64,
            learning_objectives: vec!["Identify mathematical patterns".to_string()],
            hints: vec![
                "Look for the difference between consecutive numbers".to_string(),
                "Check if the sequence increases by the same amount each time".to_string(),
            ],
            solution_methods: vec!["Pattern analysis".to_string()],
            estimated_time: match difficulty {
                ProblemDifficulty::VeryEasy => 2,
                ProblemDifficulty::Easy => 3,
                ProblemDifficulty::Medium => 4,
                ProblemDifficulty::Hard => 6,
                ProblemDifficulty::VeryHard => 8,
            },
            prerequisites: vec!["number sequences".to_string()],
            generated_at: Utc::now(),
        })
    }

    /// Determine appropriate difficulty for a student and topic
    fn determine_difficulty(&self, student_model: &StudentModel, topic: &str) -> ProblemDifficulty {
        let recommended_level = student_model.get_recommended_difficulty(topic);
        ProblemDifficulty::from(recommended_level)
    }

    /// Select problem type based on topic
    fn select_problem_type(&self, topic: &str) -> ProblemType {
        match topic {
            t if t.contains("arithmetic") => ProblemType::BasicArithmetic,
            t if t.contains("equation") || t.contains("algebra") => ProblemType::LinearEquations,
            t if t.contains("polynomial") => ProblemType::PolynomialOperations,
            t if t.contains("word") || t.contains("application") => ProblemType::WordProblems,
            t if t.contains("pattern") || t.contains("sequence") => ProblemType::PatternRecognition,
            _ => ProblemType::LinearEquations, // Default
        }
    }

    /// Customize problem for specific student
    fn customize_for_student(&self, problem: &mut EducationalProblem, student_model: &StudentModel) {
        // Adjust based on learning style
        match student_model.learning_style {
            crate::education::student_model::LearningStyle::Visual => {
                problem.hints.insert(0, "Try drawing a diagram or visual representation".to_string());
            },
            crate::education::student_model::LearningStyle::Kinesthetic => {
                problem.hints.insert(0, "Consider using physical objects or manipulatives".to_string());
            },
            _ => {},
        }

        // Adjust estimated time based on student performance
        if student_model.overall_performance.average_mastery > 0.8 {
            problem.estimated_time = (problem.estimated_time as f64 * 0.8) as u32;
        } else if student_model.overall_performance.average_mastery < 0.4 {
            problem.estimated_time = (problem.estimated_time as f64 * 1.3) as u32;
        }
    }

    /// Generate hints for arithmetic problems
    fn generate_arithmetic_hints(&self, operation: &str, a: f64, b: f64) -> Vec<String> {
        match operation {
            "+" => vec![
                "Add the two numbers together".to_string(),
                format!("Start with {} and add {}", a, b),
            ],
            "-" => vec![
                "Subtract the second number from the first".to_string(),
                format!("Start with {} and subtract {}", a, b),
            ],
            "×" => vec![
                "Multiply the two numbers".to_string(),
                format!("Think of {} groups of {}", a, b),
            ],
            "÷" => vec![
                "Divide the first number by the second".to_string(),
                format!("How many times does {} go into {}?", b, a),
            ],
            _ => vec!["Follow the order of operations".to_string()],
        }
    }

    /// Generate hints for linear equation problems
    fn generate_linear_equation_hints(&self, a: f64, b: f64, _c: f64) -> Vec<String> {
        let mut hints = vec![
            "Isolate the variable x on one side of the equation".to_string(),
        ];

        if b != 0.0 {
            let operation = if b > 0.0 { "subtract" } else { "add" };
            let abs_b = b.abs();
            hints.push(format!("First {} {} from both sides", operation, abs_b));
        }

        hints.push(format!("Then divide both sides by {}", a));
        hints.push("Check your answer by substituting back into the original equation".to_string());

        hints
    }

    /// Generate hints for word problems
    fn generate_word_problem_hints(&self, context: ProblemContext) -> Vec<String> {
        let mut hints = vec![
            "Read the problem carefully and identify what you're solving for".to_string(),
            "Write down the known information".to_string(),
            "Set up an equation using the relationships described".to_string(),
        ];

        match context {
            ProblemContext::Financial => {
                hints.push("Think about income, expenses, and profit relationships".to_string());
            },
            ProblemContext::TimeScheduling => {
                hints.push("Consider relationships between time, rate, and distance or work".to_string());
            },
            ProblemContext::Shopping => {
                hints.push("Think about quantities, prices, and totals".to_string());
            },
            _ => {},
        }

        hints
    }

    /// Select context for word problems
    fn select_word_problem_context(&self) -> ProblemContext {
        let contexts = vec![
            ProblemContext::Financial,
            ProblemContext::Shopping,
            ProblemContext::TimeScheduling,
            ProblemContext::Sports,
        ];
        
        let mut rng = thread_rng();
        contexts[rng.gen_range(0..contexts.len())].clone()
    }

    /// Generate word problem content
    fn generate_word_problem_content(&self, context: ProblemContext, _difficulty: ProblemDifficulty) -> Result<(String, AlgebraProblem, f64)> {
        let mut rng = thread_rng();
        
        match context {
            ProblemContext::Shopping => {
                let item_price = rng.gen_range(5..=25) as f64;
                let quantity = rng.gen_range(2..=10) as f64;
                let total = item_price * quantity;
                
                let statement = format!(
                    "Sarah bought {} items at ${:.2} each. How much did she spend in total?",
                    quantity, item_price
                );
                
                let algebra_problem = AlgebraProblem::linear_equation(1.0, 0.0, total);
                Ok((statement, algebra_problem, total))
            },
            ProblemContext::Financial => {
                let initial_amount = rng.gen_range(100..=500) as f64;
                let spent = rng.gen_range(20..=initial_amount as i32 - 10) as f64;
                let remaining = initial_amount - spent;
                
                let statement = format!(
                    "Mike had ${:.2} and spent ${:.2}. How much money does he have left?",
                    initial_amount, spent
                );
                
                let algebra_problem = AlgebraProblem::linear_equation(1.0, spent, initial_amount);
                Ok((statement, algebra_problem, remaining))
            },
            _ => {
                // Default to simple linear equation
                let a = 2.0;
                let b = 5.0;
                let c = 15.0;
                let solution = (c - b) / a;
                
                let statement = format!("Find x when 2x + 5 = 15");
                let algebra_problem = AlgebraProblem::linear_equation(a, b, c);
                Ok((statement, algebra_problem, solution))
            }
        }
    }

    /// Initialize problem templates
    fn initialize_templates(&mut self) {
        // Initialize arithmetic templates
        self.templates.arithmetic_templates = vec![
            ArithmeticTemplate {
                operation: "addition".to_string(),
                number_range: (1, 100),
                statement_format: "What is {} + {}?".to_string(),
            },
            ArithmeticTemplate {
                operation: "subtraction".to_string(),
                number_range: (1, 100),
                statement_format: "What is {} - {}?".to_string(),
            },
        ];

        // Initialize linear templates
        self.templates.linear_templates = vec![
            LinearTemplate {
                coefficient_range: (1.0, 10.0),
                constant_range: (-20.0, 20.0),
                statement_format: "Solve for x: {}x + {} = {}".to_string(),
            },
        ];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::education::student_model::LearningStyle;

    #[test]
    fn test_problem_generator_creation() {
        let config = EducationConfig::default();
        let generator = EducationalProblemGenerator::new(&config);
        
        assert!(generator.problem_history.is_empty());
    }

    #[test]
    fn test_difficulty_conversion() {
        assert_eq!(ProblemDifficulty::from(1), ProblemDifficulty::VeryEasy);
        assert_eq!(ProblemDifficulty::from(3), ProblemDifficulty::Medium);
        assert_eq!(u8::from(ProblemDifficulty::Hard), 4);
    }

    #[test]
    fn test_arithmetic_problem_generation() -> Result<()> {
        let config = EducationConfig::default();
        let generator = EducationalProblemGenerator::new(&config);
        
        let problem = generator.generate_arithmetic_problem(
            "test_prob".to_string(),
            ProblemDifficulty::Easy
        )?;
        
        assert_eq!(problem.problem_type, ProblemType::BasicArithmetic);
        assert_eq!(problem.difficulty, ProblemDifficulty::Easy);
        assert!(!problem.hints.is_empty());
        
        Ok(())
    }

    #[test]
    fn test_linear_equation_generation() -> Result<()> {
        let config = EducationConfig::default();
        let generator = EducationalProblemGenerator::new(&config);
        
        let problem = generator.generate_linear_equation_problem(
            "test_linear".to_string(),
            ProblemDifficulty::Medium
        )?;
        
        assert_eq!(problem.problem_type, ProblemType::LinearEquations);
        assert_eq!(problem.difficulty, ProblemDifficulty::Medium);
        assert!(problem.estimated_time > 0);
        
        Ok(())
    }

    #[test]
    fn test_pattern_problem_generation() -> Result<()> {
        let config = EducationConfig::default();
        let generator = EducationalProblemGenerator::new(&config);
        
        let problem = generator.generate_pattern_problem(
            "test_pattern".to_string(),
            ProblemDifficulty::Easy
        )?;
        
        assert_eq!(problem.problem_type, ProblemType::PatternRecognition);
        assert!(problem.problem_statement.contains("sequence"));
        
        Ok(())
    }

    #[test]
    fn test_student_customization() -> Result<()> {
        let config = EducationConfig::default();
        let mut generator = EducationalProblemGenerator::new(&config);
        let student_model = StudentModel::new(
            "test_student".to_string(),
            15,
            LearningStyle::Visual
        );

        let problem = generator.generate_for_student(&student_model, "algebra")?;
        
        assert_eq!(generator.problem_history.get("test_student").unwrap().len(), 1);
        assert!(problem.hints.iter().any(|h| h.contains("visual") || h.contains("diagram")));
        
        Ok(())
    }
}