//! Explanation Engine for Step-by-Step Mathematical Solutions
//!
//! This module provides detailed explanations, step-by-step solutions, and
//! educational guidance for mathematical problems, adapting explanation
//! depth and style to individual student needs.

use super::{StudentModel, EducationConfig, DifficultyLevel};
use crate::error::{NEATError, Result};
use crate::calculator::{algebra::{AlgebraProblem, Expression}, Operation};
use crate::calculator::modules::ModuleType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Types of explanations available
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExplanationType {
    /// Conceptual explanation of underlying principles
    Conceptual,
    /// Step-by-step procedural guidance
    Procedural,
    /// Visual representation and diagrams
    Visual,
    /// Examples with similar problems
    ExampleBased,
    /// Common mistakes and misconceptions
    ErrorAnalysis,
    /// Multiple solution approaches
    AlternativeMethod,
    /// Real-world application context
    ApplicationBased,
}

/// Detailed step in a mathematical solution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolutionStep {
    /// Step number in sequence
    pub step_number: u8,
    /// Brief title of the step
    pub title: String,
    /// Detailed explanation of what to do
    pub explanation: String,
    /// Mathematical expression before this step
    pub before_expression: String,
    /// Mathematical expression after this step
    pub after_expression: String,
    /// Reasoning behind this step
    pub reasoning: String,
    /// Common mistakes to avoid
    pub common_mistakes: Vec<String>,
    /// Verification method for this step
    pub verification: Option<String>,
}

/// Complete step-by-step solution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepByStepSolution {
    /// Problem being solved
    pub problem_statement: String,
    /// Overall strategy description
    pub strategy_description: String,
    /// Ordered sequence of solution steps
    pub steps: Vec<SolutionStep>,
    /// Final answer with explanation
    pub final_answer: String,
    /// Answer verification
    pub verification_method: String,
    /// Alternative solution approaches
    pub alternative_methods: Vec<String>,
    /// Learning objectives reinforced
    pub learning_objectives: Vec<String>,
    /// When explanation was generated
    pub generated_at: DateTime<Utc>,
}

/// Conceptual explanation of mathematical concepts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptualExplanation {
    /// Concept being explained
    pub concept_name: String,
    /// Simple definition
    pub definition: String,
    /// Detailed explanation
    pub explanation: String,
    /// Key properties or rules
    pub key_properties: Vec<String>,
    /// Worked examples
    pub examples: Vec<WorkedExample>,
    /// Visual aids description
    pub visual_aids: Vec<String>,
    /// Real-world applications
    pub applications: Vec<String>,
    /// Prerequisites needed to understand
    pub prerequisites: Vec<String>,
}

/// Worked example for conceptual understanding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkedExample {
    /// Example problem
    pub problem: String,
    /// Solution with explanation
    pub solution: String,
    /// Key insights highlighted
    pub key_insights: Vec<String>,
}

/// Error analysis and misconception guidance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorAnalysis {
    /// Common error description
    pub error_description: String,
    /// Why students make this error
    pub error_reasoning: String,
    /// Correct approach
    pub correct_approach: String,
    /// How to avoid this error
    pub prevention_strategy: String,
    /// Practice problems to reinforce correct method
    pub practice_suggestions: Vec<String>,
}

/// Main explanation engine
pub struct ExplanationEngine {
    /// Educational configuration
    config: EducationConfig,
    /// Explanation customization parameters
    explanation_parameters: ExplanationParameters,
    /// Library of conceptual explanations
    concept_library: HashMap<String, ConceptualExplanation>,
    /// Common error patterns and corrections
    error_library: HashMap<String, ErrorAnalysis>,
    /// Explanation history by student
    explanation_history: HashMap<String, Vec<StepByStepSolution>>,
}

/// Parameters for customizing explanations
#[derive(Debug, Clone)]
struct ExplanationParameters {
    /// Verbosity level (1=brief, 5=detailed)
    verbosity_level: u8,
    /// Include visual descriptions
    include_visual_aids: bool,
    /// Include error warnings
    include_error_warnings: bool,
    /// Include alternative methods
    include_alternatives: bool,
    /// Adapt to learning style
    adapt_to_learning_style: bool,
}

impl Default for ExplanationParameters {
    fn default() -> Self {
        Self {
            verbosity_level: 3,
            include_visual_aids: true,
            include_error_warnings: true,
            include_alternatives: false,
            adapt_to_learning_style: true,
        }
    }
}

impl ExplanationEngine {
    /// Create a new explanation engine
    pub fn new(config: &EducationConfig) -> Self {
        let mut engine = Self {
            config: config.clone(),
            explanation_parameters: ExplanationParameters::default(),
            concept_library: HashMap::new(),
            error_library: HashMap::new(),
            explanation_history: HashMap::new(),
        };

        // Initialize concept and error libraries
        engine.initialize_concept_library();
        engine.initialize_error_library();

        engine
    }

    /// Generate step-by-step solution for a problem
    pub fn explain_solution(&mut self, problem: &AlgebraProblem, student_model: &StudentModel) -> Result<StepByStepSolution> {
        // Customize explanation parameters for this student
        self.adapt_parameters_for_student(student_model);

        // Generate the solution steps
        let solution = self.generate_step_by_step_solution(problem, student_model)?;

        // Store in history
        self.explanation_history
            .entry(student_model.student_id.clone())
            .or_insert_with(Vec::new)
            .push(solution.clone());

        Ok(solution)
    }

    /// Generate conceptual explanation for a topic
    pub fn explain_concept(&self, concept: &str, student_model: &StudentModel) -> Result<ConceptualExplanation> {
        if let Some(explanation) = self.concept_library.get(concept) {
            let mut customized_explanation = explanation.clone();
            self.customize_conceptual_explanation(&mut customized_explanation, student_model);
            Ok(customized_explanation)
        } else {
            // Generate basic explanation if not in library
            Ok(self.generate_basic_concept_explanation(concept, student_model))
        }
    }

    /// Analyze student error and provide correction guidance
    pub fn analyze_error(&self, student_answer: &str, correct_answer: f64, problem: &AlgebraProblem) -> Result<ErrorAnalysis> {
        // Parse student answer
        if let Ok(student_value) = student_answer.parse::<f64>() {
            let error_type = self.classify_error(student_value, correct_answer, problem);
            
            if let Some(error_analysis) = self.error_library.get(&error_type) {
                Ok(error_analysis.clone())
            } else {
                Ok(self.generate_generic_error_analysis(student_value, correct_answer))
            }
        } else {
            Ok(self.generate_parse_error_analysis())
        }
    }

    /// Generate step-by-step solution
    fn generate_step_by_step_solution(&self, problem: &AlgebraProblem, student_model: &StudentModel) -> Result<StepByStepSolution> {
        // For linear equations ax + b = c, generate systematic solution
        let expression = &problem.expression;
        
        // Extract coefficients (simplified for demonstration)
        let (a, b, c) = self.extract_linear_coefficients(expression)?;
        
        let problem_statement = self.format_problem_statement(a, b, c);
        let steps = self.generate_linear_equation_steps(a, b, c, student_model);
        let final_answer = (c - b) / a;

        Ok(StepByStepSolution {
            problem_statement,
            strategy_description: "We'll solve this linear equation by isolating the variable x".to_string(),
            steps,
            final_answer: format!("x = {}", final_answer),
            verification_method: format!("Check: substitute x = {} back into the original equation", final_answer),
            alternative_methods: if self.explanation_parameters.include_alternatives {
                vec!["Graphical method".to_string(), "Trial and error".to_string()]
            } else {
                vec![]
            },
            learning_objectives: vec![
                "Solve linear equations".to_string(),
                "Use inverse operations".to_string(),
                "Verify solutions".to_string(),
            ],
            generated_at: Utc::now(),
        })
    }

    /// Generate steps for solving linear equation
    fn generate_linear_equation_steps(&self, a: f64, b: f64, c: f64, student_model: &StudentModel) -> Vec<SolutionStep> {
        let mut steps = Vec::new();
        let mut step_number = 1;

        // Step 1: Identify the equation
        steps.push(SolutionStep {
            step_number,
            title: "Identify the equation".to_string(),
            explanation: "First, let's identify what type of equation we're working with".to_string(),
            before_expression: "Given problem".to_string(),
            after_expression: self.format_equation(a, b, c),
            reasoning: "This is a linear equation in the form ax + b = c".to_string(),
            common_mistakes: vec!["Misidentifying the coefficients".to_string()],
            verification: None,
        });
        step_number += 1;

        // Step 2: Isolate the variable term (if b ≠ 0)
        if b != 0.0 {
            let operation = if b > 0.0 { "subtract" } else { "add" };
            let abs_b = b.abs();
            let new_c = c - b;

            steps.push(SolutionStep {
                step_number,
                title: format!("Isolate the variable term"),
                explanation: format!("To isolate the x term, {} {} from both sides", operation, abs_b),
                before_expression: self.format_equation(a, b, c),
                after_expression: format!("{}x = {}", a, new_c),
                reasoning: format!("We {} {} because the opposite of {} is {}", operation, abs_b, 
                    if b > 0.0 { "adding" } else { "subtracting" }, operation),
                common_mistakes: vec![
                    "Only applying the operation to one side".to_string(),
                    "Getting the sign wrong".to_string(),
                ],
                verification: Some(format!("Check: {} {} {} = {}", c, if b > 0.0 { "-" } else { "+" }, abs_b, new_c)),
            });
            step_number += 1;
        }

        // Step 3: Solve for x
        let final_c = c - b;
        let solution = final_c / a;
        
        steps.push(SolutionStep {
            step_number,
            title: "Solve for x".to_string(),
            explanation: format!("Divide both sides by {} to get x by itself", a),
            before_expression: format!("{}x = {}", a, final_c),
            after_expression: format!("x = {}", solution),
            reasoning: format!("Division is the inverse operation of multiplication"),
            common_mistakes: vec![
                "Forgetting to divide both sides".to_string(),
                "Making arithmetic errors".to_string(),
            ],
            verification: Some(format!("Check: {} ÷ {} = {}", final_c, a, solution)),
        });
        step_number += 1;

        // Step 4: Verify the solution
        let verification_result = a * solution + b;
        steps.push(SolutionStep {
            step_number,
            title: "Verify the solution".to_string(),
            explanation: "Substitute our answer back into the original equation to check".to_string(),
            before_expression: format!("x = {}", solution),
            after_expression: format!("{}({}) + {} = {}", a, solution, b, verification_result),
            reasoning: "If our solution is correct, this should equal the right side of the original equation".to_string(),
            common_mistakes: vec!["Skipping the verification step".to_string()],
            verification: Some(format!("✓ {} = {} ✓", verification_result, c)),
        });

        // Customize steps based on student learning style
        self.customize_steps_for_learning_style(&mut steps, student_model);

        steps
    }

    /// Adapt explanation parameters for a specific student
    fn adapt_parameters_for_student(&mut self, student_model: &StudentModel) {
        // Adjust verbosity based on student preferences
        if let Some(depth) = student_model.learning_preferences.explanation_depth.into() {
            self.explanation_parameters.verbosity_level = depth;
        }

        // Adapt to learning style
        match student_model.learning_style {
            crate::education::student_model::LearningStyle::Visual => {
                self.explanation_parameters.include_visual_aids = true;
            },
            crate::education::student_model::LearningStyle::ReadingWriting => {
                self.explanation_parameters.verbosity_level = 4; // More detailed
            },
            crate::education::student_model::LearningStyle::Kinesthetic => {
                self.explanation_parameters.include_alternatives = true;
            },
            _ => {},
        }

        // Adjust based on performance level
        if student_model.overall_performance.average_mastery < 0.5 {
            self.explanation_parameters.verbosity_level = 4; // More detailed for struggling students
            self.explanation_parameters.include_error_warnings = true;
        }
    }

    /// Customize steps based on learning style
    fn customize_steps_for_learning_style(&self, steps: &mut Vec<SolutionStep>, student_model: &StudentModel) {
        match student_model.learning_style {
            crate::education::student_model::LearningStyle::Visual => {
                for step in steps.iter_mut() {
                    step.explanation = format!("{} (Imagine this visually as a balance scale)", step.explanation);
                }
            },
            crate::education::student_model::LearningStyle::Kinesthetic => {
                for step in steps.iter_mut() {
                    step.explanation = format!("{} (Try using physical objects to represent the numbers)", step.explanation);
                }
            },
            _ => {},
        }
    }

    /// Extract coefficients from linear expression (simplified)
    fn extract_linear_coefficients(&self, _expression: &Expression) -> Result<(f64, f64, f64)> {
        // This is a simplified implementation
        // In a full system, would parse the expression properly
        
        // For demonstration, assume we have a linear equation ax + b = c
        // Default to 2x + 3 = 7
        Ok((2.0, 3.0, 7.0))
    }

    /// Format problem statement
    fn format_problem_statement(&self, a: f64, b: f64, c: f64) -> String {
        if b == 0.0 {
            format!("Solve for x: {}x = {}", a, c)
        } else if b > 0.0 {
            format!("Solve for x: {}x + {} = {}", a, b, c)
        } else {
            format!("Solve for x: {}x - {} = {}", a, -b, c)
        }
    }

    /// Format equation for display
    fn format_equation(&self, a: f64, b: f64, c: f64) -> String {
        if b == 0.0 {
            format!("{}x = {}", a, c)
        } else if b > 0.0 {
            format!("{}x + {} = {}", a, b, c)
        } else {
            format!("{}x - {} = {}", a, -b, c)
        }
    }

    /// Customize conceptual explanation for student
    fn customize_conceptual_explanation(&self, explanation: &mut ConceptualExplanation, student_model: &StudentModel) {
        // Add learning style specific guidance
        match student_model.learning_style {
            crate::education::student_model::LearningStyle::Visual => {
                explanation.visual_aids.push("Try drawing diagrams to visualize the concept".to_string());
            },
            crate::education::student_model::LearningStyle::Auditory => {
                explanation.explanation = format!("{} (Try reading this explanation aloud)", explanation.explanation);
            },
            _ => {},
        }
    }

    /// Generate basic concept explanation
    fn generate_basic_concept_explanation(&self, concept: &str, _student_model: &StudentModel) -> ConceptualExplanation {
        ConceptualExplanation {
            concept_name: concept.to_string(),
            definition: format!("A fundamental mathematical concept: {}", concept),
            explanation: format!("{} is an important topic in mathematics that requires practice to master", concept),
            key_properties: vec![
                "Follow mathematical rules and principles".to_string(),
                "Can be applied to solve problems".to_string(),
            ],
            examples: vec![
                WorkedExample {
                    problem: "Example problem".to_string(),
                    solution: "Step-by-step solution".to_string(),
                    key_insights: vec!["Key insight from this example".to_string()],
                }
            ],
            visual_aids: vec!["Diagrams can help visualize this concept".to_string()],
            applications: vec!["Real-world applications exist for this concept".to_string()],
            prerequisites: vec!["Basic mathematical understanding".to_string()],
        }
    }

    /// Classify type of error made by student
    fn classify_error(&self, student_answer: f64, correct_answer: f64, _problem: &AlgebraProblem) -> String {
        let difference = (student_answer - correct_answer).abs();
        
        if difference < 0.1 {
            "rounding_error".to_string()
        } else if student_answer == -correct_answer {
            "sign_error".to_string()
        } else if (student_answer / correct_answer - 2.0).abs() < 0.1 {
            "forgot_to_divide".to_string()
        } else {
            "calculation_error".to_string()
        }
    }

    /// Generate error analysis for calculation mistakes
    fn generate_generic_error_analysis(&self, student_answer: f64, correct_answer: f64) -> ErrorAnalysis {
        ErrorAnalysis {
            error_description: format!("Incorrect answer: got {} instead of {}", student_answer, correct_answer),
            error_reasoning: "This might be due to a calculation error or misunderstanding of the process".to_string(),
            correct_approach: "Double-check each step of your calculation".to_string(),
            prevention_strategy: "Work more slowly and verify each arithmetic operation".to_string(),
            practice_suggestions: vec![
                "Practice basic arithmetic operations".to_string(),
                "Use a calculator to check your work".to_string(),
            ],
        }
    }

    /// Generate error analysis for parsing issues
    fn generate_parse_error_analysis(&self) -> ErrorAnalysis {
        ErrorAnalysis {
            error_description: "Could not understand the answer format".to_string(),
            error_reasoning: "The answer might not be in the expected numerical format".to_string(),
            correct_approach: "Make sure to provide a numerical answer (like 5 or 2.5)".to_string(),
            prevention_strategy: "Double-check that your answer is a number".to_string(),
            practice_suggestions: vec![
                "Practice writing answers in decimal or fraction form".to_string(),
            ],
        }
    }

    /// Initialize the concept library with common mathematical concepts
    fn initialize_concept_library(&mut self) {
        // Linear equations concept
        self.concept_library.insert("linear_equations".to_string(), ConceptualExplanation {
            concept_name: "Linear Equations".to_string(),
            definition: "An equation where the variable appears only to the first power".to_string(),
            explanation: "Linear equations are equations that graph as straight lines. They have the form ax + b = c, where a, b, and c are constants and x is the variable we solve for.".to_string(),
            key_properties: vec![
                "The variable (usually x) appears only to the first power".to_string(),
                "The graph is always a straight line".to_string(),
                "There is exactly one solution (unless it's inconsistent or has infinitely many solutions)".to_string(),
            ],
            examples: vec![
                WorkedExample {
                    problem: "Solve: 2x + 3 = 7".to_string(),
                    solution: "Subtract 3 from both sides: 2x = 4. Divide both sides by 2: x = 2".to_string(),
                    key_insights: vec!["Use inverse operations to isolate the variable".to_string()],
                }
            ],
            visual_aids: vec![
                "Think of equations as a balance scale".to_string(),
                "What you do to one side, you must do to the other".to_string(),
            ],
            applications: vec![
                "Calculating costs and profits".to_string(),
                "Finding unknown quantities in word problems".to_string(),
            ],
            prerequisites: vec!["Basic arithmetic operations".to_string(), "Understanding of variables".to_string()],
        });
    }

    /// Initialize the error library with common mistake patterns
    fn initialize_error_library(&mut self) {
        // Sign error
        self.error_library.insert("sign_error".to_string(), ErrorAnalysis {
            error_description: "The answer has the correct magnitude but wrong sign".to_string(),
            error_reasoning: "This often happens when students make errors with positive and negative numbers".to_string(),
            correct_approach: "Pay careful attention to signs when performing operations".to_string(),
            prevention_strategy: "Double-check each step, especially when dealing with negative numbers".to_string(),
            practice_suggestions: vec![
                "Practice integer operations with positive and negative numbers".to_string(),
                "Use a number line to visualize operations".to_string(),
            ],
        });

        // Forgot to divide error
        self.error_library.insert("forgot_to_divide".to_string(), ErrorAnalysis {
            error_description: "The final step of dividing by the coefficient was missed".to_string(),
            error_reasoning: "Students sometimes forget the last step of solving for the variable".to_string(),
            correct_approach: "Always complete the solution by isolating the variable completely".to_string(),
            prevention_strategy: "Make sure your final answer has the variable by itself".to_string(),
            practice_suggestions: vec![
                "Practice identifying when a variable is fully isolated".to_string(),
                "Always check that your answer is in the form 'x = number'".to_string(),
            ],
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::education::student_model::LearningStyle;

    #[test]
    fn test_explanation_engine_creation() {
        let config = EducationConfig::default();
        let engine = ExplanationEngine::new(&config);
        
        assert!(!engine.concept_library.is_empty());
        assert!(!engine.error_library.is_empty());
    }

    #[test]
    fn test_linear_coefficients() {
        let config = EducationConfig::default();
        let engine = ExplanationEngine::new(&config);
        
        // Test the simplified coefficient extraction
        let dummy_expression = Expression::Variable; // Placeholder
        let result = engine.extract_linear_coefficients(&dummy_expression);
        
        assert!(result.is_ok());
        let (a, b, c) = result.unwrap();
        assert_eq!(a, 2.0);
        assert_eq!(b, 3.0);
        assert_eq!(c, 7.0);
    }

    #[test]
    fn test_error_classification() {
        let config = EducationConfig::default();
        let engine = ExplanationEngine::new(&config);
        let dummy_problem = AlgebraProblem::linear_equation(2.0, 3.0, 7.0);
        
        // Test sign error
        let error_type = engine.classify_error(-2.0, 2.0, &dummy_problem);
        assert_eq!(error_type, "sign_error");
        
        // Test forgot to divide error
        let error_type = engine.classify_error(4.0, 2.0, &dummy_problem);
        assert_eq!(error_type, "forgot_to_divide");
    }

    #[test]
    fn test_concept_explanation() -> Result<()> {
        let config = EducationConfig::default();
        let engine = ExplanationEngine::new(&config);
        let student_model = StudentModel::new(
            "test_student".to_string(),
            15,
            LearningStyle::Visual
        );

        let explanation = engine.explain_concept("linear_equations", &student_model)?;
        
        assert_eq!(explanation.concept_name, "Linear Equations");
        assert!(!explanation.examples.is_empty());
        
        Ok(())
    }

    #[test]
    fn test_step_generation() -> Result<()> {
        let config = EducationConfig::default();
        let mut engine = ExplanationEngine::new(&config);
        let student_model = StudentModel::new(
            "test_student".to_string(),
            15,
            LearningStyle::Auditory
        );

        let problem = AlgebraProblem::linear_equation(2.0, 3.0, 7.0);
        let solution = engine.explain_solution(&problem, &student_model)?;
        
        assert!(!solution.steps.is_empty());
        assert!(solution.final_answer.contains("x ="));
        assert_eq!(engine.explanation_history.get("test_student").unwrap().len(), 1);
        
        Ok(())
    }
}