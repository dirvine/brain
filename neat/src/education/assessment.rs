//! Assessment Engine for Educational Platform
//!
//! This module provides comprehensive assessment capabilities including
//! adaptive testing, performance evaluation, and difficulty calibration.

use super::{StudentModel, EducationConfig};
use crate::error::{NEATError, Result};
use crate::calculator::{algebra::{AlgebraProblem, Expression}, Operation};
use crate::calculator::modules::ModuleType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Difficulty levels for problems and assessments
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DifficultyLevel {
    /// Very basic concepts
    Beginner = 1,
    /// Elementary level
    Elementary = 2,
    /// Intermediate level
    Intermediate = 3,
    /// Advanced level
    Advanced = 4,
    /// Expert level
    Expert = 5,
}

impl From<u8> for DifficultyLevel {
    fn from(value: u8) -> Self {
        match value {
            1 => DifficultyLevel::Beginner,
            2 => DifficultyLevel::Elementary,
            3 => DifficultyLevel::Intermediate,
            4 => DifficultyLevel::Advanced,
            _ => DifficultyLevel::Expert,
        }
    }
}

impl From<DifficultyLevel> for u8 {
    fn from(level: DifficultyLevel) -> Self {
        level as u8
    }
}

/// Assessment question with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessmentQuestion {
    /// Question identifier
    pub question_id: String,
    /// The mathematical problem
    pub problem: AlgebraProblem,
    /// Difficulty level
    pub difficulty: DifficultyLevel,
    /// Mathematical topic/concept being tested
    pub topic: String,
    /// Module type this question assesses
    pub module_type: ModuleType,
    /// Expected solution
    pub expected_answer: f64,
    /// Time limit in seconds
    pub time_limit: Option<u32>,
    /// Point value for scoring
    pub points: u32,
    /// Cognitive level being assessed
    pub cognitive_level: CognitiveLevel,
}

/// Cognitive complexity levels based on Bloom's taxonomy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CognitiveLevel {
    /// Remember basic facts
    Remember,
    /// Understand concepts
    Understand,
    /// Apply knowledge to new situations
    Apply,
    /// Analyze and break down problems
    Analyze,
    /// Evaluate and make judgments
    Evaluate,
    /// Create new solutions
    Create,
}

/// Student's response to an assessment question
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessmentResponse {
    /// Question this response is for
    pub question_id: String,
    /// Student's answer
    pub answer: String,
    /// Parsed numerical answer (if applicable)
    pub numerical_answer: Option<f64>,
    /// Whether the answer is correct
    pub is_correct: bool,
    /// Partial credit earned (0.0 to 1.0)
    pub partial_credit: f64,
    /// Time taken to answer (seconds)
    pub response_time: u32,
    /// Work shown or steps taken
    pub work_shown: Option<String>,
    /// Response timestamp
    pub responded_at: DateTime<Utc>,
}

/// Complete assessment result and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessmentResult {
    /// Assessment identifier
    pub assessment_id: String,
    /// Student who took the assessment
    pub student_id: String,
    /// Assessment start time
    pub start_time: DateTime<Utc>,
    /// Assessment completion time
    pub end_time: DateTime<Utc>,
    /// Questions presented
    pub questions: Vec<AssessmentQuestion>,
    /// Student responses
    pub responses: Vec<AssessmentResponse>,
    /// Overall score (0.0 to 1.0)
    pub overall_score: f64,
    /// Total points earned
    pub points_earned: u32,
    /// Total possible points
    pub total_points: u32,
    /// Performance by topic
    pub topic_scores: HashMap<String, f64>,
    /// Performance by difficulty level
    pub difficulty_scores: HashMap<DifficultyLevel, f64>,
    /// Performance by module type
    pub module_scores: HashMap<ModuleType, f64>,
    /// Recommended next difficulty level
    pub recommended_difficulty: DifficultyLevel,
    /// Areas needing improvement
    pub improvement_areas: Vec<String>,
    /// Strengths identified
    pub strengths: Vec<String>,
    /// Detailed feedback
    pub feedback: String,
}

/// Assessment configuration and parameters
#[derive(Debug, Clone)]
pub struct AssessmentConfig {
    /// Number of questions in assessment
    pub question_count: usize,
    /// Time limit for entire assessment (minutes)
    pub time_limit: Option<u32>,
    /// Whether to use adaptive difficulty
    pub adaptive_difficulty: bool,
    /// Minimum score to pass topic
    pub passing_threshold: f64,
    /// Topics to include in assessment
    pub topics: Vec<String>,
    /// Difficulty distribution (beginner to expert percentages)
    pub difficulty_distribution: [f64; 5],
    /// Whether to provide immediate feedback
    pub immediate_feedback: bool,
}

impl Default for AssessmentConfig {
    fn default() -> Self {
        Self {
            question_count: 10,
            time_limit: Some(30), // 30 minutes
            adaptive_difficulty: true,
            passing_threshold: 0.7,
            topics: vec![
                "basic_arithmetic".to_string(),
                "linear_equations".to_string(),
                "algebraic_expressions".to_string(),
            ],
            difficulty_distribution: [0.2, 0.3, 0.3, 0.15, 0.05], // Weighted toward easier problems
            immediate_feedback: false,
        }
    }
}

/// Comprehensive assessment engine
pub struct AssessmentEngine {
    /// Educational configuration
    config: EducationConfig,
    /// Assessment configurations by type
    assessment_configs: HashMap<String, AssessmentConfig>,
    /// Question bank organized by topic and difficulty
    question_bank: QuestionBank,
    /// Assessment history
    assessment_history: Vec<AssessmentResult>,
}

/// Question bank for organizing assessment questions
#[derive(Debug, Clone, Default)]
struct QuestionBank {
    /// Questions organized by topic
    by_topic: HashMap<String, Vec<AssessmentQuestion>>,
    /// Questions organized by difficulty
    by_difficulty: HashMap<DifficultyLevel, Vec<AssessmentQuestion>>,
    /// Questions organized by module type
    by_module: HashMap<ModuleType, Vec<AssessmentQuestion>>,
}

impl AssessmentEngine {
    /// Create a new assessment engine
    pub fn new(config: &EducationConfig) -> Self {
        let mut engine = Self {
            config: config.clone(),
            assessment_configs: HashMap::new(),
            question_bank: QuestionBank::default(),
            assessment_history: Vec::new(),
        };

        // Initialize with default assessment configurations
        engine.assessment_configs.insert(
            "diagnostic".to_string(),
            AssessmentConfig::default()
        );

        // Build initial question bank
        engine.build_question_bank();

        engine
    }

    /// Conduct an assessment for a student
    pub fn assess_student(&mut self, student_model: &StudentModel, topic: &str) -> Result<AssessmentResult> {
        let assessment_id = format!("assessment_{}_{}", student_model.student_id, Utc::now().timestamp());
        let config = self.assessment_configs.get("diagnostic")
            .ok_or_else(|| NEATError::InvalidConfiguration {
                parameter: "assessment_type".to_string(),
                value: "diagnostic".to_string(),
            })?;

        // Select questions based on student model and config
        let questions = self.select_assessment_questions(student_model, topic, config)?;

        // Create assessment result structure
        let mut result = AssessmentResult {
            assessment_id: assessment_id.clone(),
            student_id: student_model.student_id.clone(),
            start_time: Utc::now(),
            end_time: Utc::now(), // Will be updated when assessment completes
            questions: questions.clone(),
            responses: Vec::new(),
            overall_score: 0.0,
            points_earned: 0,
            total_points: questions.iter().map(|q| q.points).sum(),
            topic_scores: HashMap::new(),
            difficulty_scores: HashMap::new(),
            module_scores: HashMap::new(),
            recommended_difficulty: DifficultyLevel::Intermediate,
            improvement_areas: Vec::new(),
            strengths: Vec::new(),
            feedback: String::new(),
        };

        // Simulate assessment responses (in real system, this would be interactive)
        self.simulate_assessment_responses(&mut result, student_model);

        // Analyze results and generate feedback
        self.analyze_assessment_results(&mut result, student_model);

        // Store assessment history
        self.assessment_history.push(result.clone());

        Ok(result)
    }

    /// Select appropriate questions for assessment
    fn select_assessment_questions(
        &self,
        student_model: &StudentModel,
        topic: &str,
        config: &AssessmentConfig
    ) -> Result<Vec<AssessmentQuestion>> {
        let mut questions = Vec::new();
        let target_count = config.question_count;

        // Get questions for the specific topic
        let available_questions = self.question_bank.by_topic.get(topic)
            .ok_or_else(|| NEATError::InvalidConfiguration {
                parameter: "topic".to_string(),
                value: topic.to_string(),
            })?;

        if available_questions.is_empty() {
            return Err(NEATError::InvalidConfiguration {
                parameter: "question_bank".to_string(),
                value: format!("No questions available for topic: {}", topic),
            });
        }

        // Determine difficulty distribution based on student model
        let difficulty_weights = self.calculate_difficulty_weights(student_model, config);

        // Select questions according to difficulty distribution
        for (difficulty, weight) in difficulty_weights {
            let questions_for_difficulty = (target_count as f64 * weight).round() as usize;
            
            let difficulty_questions: Vec<_> = available_questions.iter()
                .filter(|q| q.difficulty == difficulty)
                .take(questions_for_difficulty)
                .cloned()
                .collect();

            questions.extend(difficulty_questions);
        }

        // Ensure we have enough questions
        while questions.len() < target_count && questions.len() < available_questions.len() {
            for question in available_questions {
                if !questions.iter().any(|q| q.question_id == question.question_id) {
                    questions.push(question.clone());
                    if questions.len() >= target_count {
                        break;
                    }
                }
            }
        }

        Ok(questions.into_iter().take(target_count).collect())
    }

    /// Calculate difficulty weights based on student model
    fn calculate_difficulty_weights(
        &self,
        student_model: &StudentModel,
        config: &AssessmentConfig
    ) -> Vec<(DifficultyLevel, f64)> {
        let base_distribution = config.difficulty_distribution;
        let student_level = student_model.overall_performance.average_mastery;

        // Adjust distribution based on student performance
        let mut weights = Vec::new();
        
        if student_level < 0.3 {
            // Struggling student - more easier questions
            weights.push((DifficultyLevel::Beginner, 0.4));
            weights.push((DifficultyLevel::Elementary, 0.4));
            weights.push((DifficultyLevel::Intermediate, 0.2));
            weights.push((DifficultyLevel::Advanced, 0.0));
            weights.push((DifficultyLevel::Expert, 0.0));
        } else if student_level < 0.6 {
            // Average student - balanced distribution
            weights.push((DifficultyLevel::Beginner, 0.2));
            weights.push((DifficultyLevel::Elementary, 0.3));
            weights.push((DifficultyLevel::Intermediate, 0.3));
            weights.push((DifficultyLevel::Advanced, 0.15));
            weights.push((DifficultyLevel::Expert, 0.05));
        } else {
            // Advanced student - more challenging questions
            weights.push((DifficultyLevel::Beginner, 0.1));
            weights.push((DifficultyLevel::Elementary, 0.2));
            weights.push((DifficultyLevel::Intermediate, 0.3));
            weights.push((DifficultyLevel::Advanced, 0.3));
            weights.push((DifficultyLevel::Expert, 0.1));
        }

        weights
    }

    /// Simulate assessment responses for demonstration
    fn simulate_assessment_responses(&self, result: &mut AssessmentResult, student_model: &StudentModel) {
        for question in &result.questions {
            // Simulate response based on student model and question difficulty
            let success_probability = self.calculate_success_probability(student_model, question);
            let is_correct = rand::random::<f64>() < success_probability;
            
            let response = AssessmentResponse {
                question_id: question.question_id.clone(),
                answer: if is_correct { 
                    question.expected_answer.to_string() 
                } else { 
                    (question.expected_answer + 1.0).to_string() 
                },
                numerical_answer: Some(if is_correct { 
                    question.expected_answer 
                } else { 
                    question.expected_answer + 1.0 
                }),
                is_correct,
                partial_credit: if is_correct { 1.0 } else { 0.0 },
                response_time: 30 + (rand::random::<u32>() % 60), // 30-90 seconds
                work_shown: None,
                responded_at: Utc::now(),
            };

            result.responses.push(response);
        }

        result.end_time = Utc::now();
    }

    /// Calculate probability of success based on student model and question
    fn calculate_success_probability(&self, student_model: &StudentModel, question: &AssessmentQuestion) -> f64 {
        // Base probability from student's overall performance
        let base_prob = student_model.overall_performance.average_mastery;
        
        // Adjust for topic-specific knowledge
        let topic_adjustment = if let Some(knowledge_state) = student_model.get_knowledge_state(&question.topic) {
            knowledge_state.mastery_level - base_prob
        } else {
            -0.2 // Penalty for unknown topic
        };

        // Adjust for difficulty
        let difficulty_adjustment = match question.difficulty {
            DifficultyLevel::Beginner => 0.2,
            DifficultyLevel::Elementary => 0.1,
            DifficultyLevel::Intermediate => 0.0,
            DifficultyLevel::Advanced => -0.15,
            DifficultyLevel::Expert => -0.3,
        };

        // Combine adjustments
        let probability = base_prob + topic_adjustment + difficulty_adjustment;
        probability.max(0.1).min(0.95) // Clamp between 10% and 95%
    }

    /// Analyze assessment results and generate detailed feedback
    fn analyze_assessment_results(&self, result: &mut AssessmentResult, student_model: &StudentModel) {
        // Calculate overall score
        let correct_responses = result.responses.iter().filter(|r| r.is_correct).count();
        result.overall_score = correct_responses as f64 / result.responses.len() as f64;
        result.points_earned = result.responses.iter()
            .zip(&result.questions)
            .map(|(response, question)| {
                if response.is_correct { question.points } else { 0 }
            })
            .sum();

        // Calculate topic-specific scores
        self.calculate_topic_scores(result);

        // Calculate difficulty-level scores
        self.calculate_difficulty_scores(result);

        // Calculate module-type scores
        self.calculate_module_scores(result);

        // Determine recommended difficulty
        result.recommended_difficulty = self.determine_recommended_difficulty(result);

        // Identify strengths and improvement areas
        self.identify_strengths_and_weaknesses(result);

        // Generate comprehensive feedback
        result.feedback = self.generate_assessment_feedback(result, student_model);
    }

    /// Calculate scores by topic
    fn calculate_topic_scores(&self, result: &mut AssessmentResult) {
        let mut topic_totals: HashMap<String, (u32, u32)> = HashMap::new();

        for (response, question) in result.responses.iter().zip(&result.questions) {
            let (correct, total) = topic_totals.entry(question.topic.clone()).or_insert((0, 0));
            *total += 1;
            if response.is_correct {
                *correct += 1;
            }
        }

        for (topic, (correct, total)) in topic_totals {
            result.topic_scores.insert(topic, correct as f64 / total as f64);
        }
    }

    /// Calculate scores by difficulty level
    fn calculate_difficulty_scores(&self, result: &mut AssessmentResult) {
        let mut difficulty_totals: HashMap<DifficultyLevel, (u32, u32)> = HashMap::new();

        for (response, question) in result.responses.iter().zip(&result.questions) {
            let (correct, total) = difficulty_totals.entry(question.difficulty).or_insert((0, 0));
            *total += 1;
            if response.is_correct {
                *correct += 1;
            }
        }

        for (difficulty, (correct, total)) in difficulty_totals {
            result.difficulty_scores.insert(difficulty, correct as f64 / total as f64);
        }
    }

    /// Calculate scores by module type
    fn calculate_module_scores(&self, result: &mut AssessmentResult) {
        let mut module_totals: HashMap<ModuleType, (u32, u32)> = HashMap::new();

        for (response, question) in result.responses.iter().zip(&result.questions) {
            let (correct, total) = module_totals.entry(question.module_type).or_insert((0, 0));
            *total += 1;
            if response.is_correct {
                *correct += 1;
            }
        }

        for (module_type, (correct, total)) in module_totals {
            result.module_scores.insert(module_type, correct as f64 / total as f64);
        }
    }

    /// Determine recommended difficulty level for future work
    fn determine_recommended_difficulty(&self, result: &AssessmentResult) -> DifficultyLevel {
        // Find the highest difficulty level where student scored >= 70%
        for difficulty in [DifficultyLevel::Expert, DifficultyLevel::Advanced, 
                          DifficultyLevel::Intermediate, DifficultyLevel::Elementary, 
                          DifficultyLevel::Beginner] {
            if let Some(&score) = result.difficulty_scores.get(&difficulty) {
                if score >= 0.7 {
                    return difficulty;
                }
            }
        }
        DifficultyLevel::Beginner
    }

    /// Identify strengths and areas for improvement
    fn identify_strengths_and_weaknesses(&self, result: &mut AssessmentResult) {
        // Identify strong topics (>= 80% correct)
        for (topic, &score) in &result.topic_scores {
            if score >= 0.8 {
                result.strengths.push(format!("Strong performance in {}", topic));
            } else if score < 0.6 {
                result.improvement_areas.push(format!("Needs practice with {}", topic));
            }
        }

        // Identify strong/weak module types
        for (module_type, &score) in &result.module_scores {
            if score >= 0.8 {
                result.strengths.push(format!("Excellent {:?} skills", module_type));
            } else if score < 0.6 {
                result.improvement_areas.push(format!("{:?} concepts need reinforcement", module_type));
            }
        }
    }

    /// Generate comprehensive assessment feedback
    fn generate_assessment_feedback(&self, result: &AssessmentResult, _student_model: &StudentModel) -> String {
        let mut feedback = String::new();

        feedback.push_str(&format!(
            "Assessment completed with an overall score of {:.1}% ({}/{} points).\n\n",
            result.overall_score * 100.0,
            result.points_earned,
            result.total_points
        ));

        // Performance level feedback
        if result.overall_score >= 0.9 {
            feedback.push_str("Excellent performance! You demonstrate strong mastery of the material.\n");
        } else if result.overall_score >= 0.8 {
            feedback.push_str("Very good work! You show solid understanding with room for minor improvements.\n");
        } else if result.overall_score >= 0.7 {
            feedback.push_str("Good progress! You're on track but could benefit from additional practice.\n");
        } else if result.overall_score >= 0.6 {
            feedback.push_str("You're making progress, but need more practice to strengthen your understanding.\n");
        } else {
            feedback.push_str("This material is challenging for you. Let's focus on building foundational skills.\n");
        }

        // Strengths
        if !result.strengths.is_empty() {
            feedback.push_str("\nStrengths identified:\n");
            for strength in &result.strengths {
                feedback.push_str(&format!("• {}\n", strength));
            }
        }

        // Improvement areas
        if !result.improvement_areas.is_empty() {
            feedback.push_str("\nAreas for improvement:\n");
            for area in &result.improvement_areas {
                feedback.push_str(&format!("• {}\n", area));
            }
        }

        // Recommendations
        feedback.push_str(&format!(
            "\nRecommended next difficulty level: {:?}\n",
            result.recommended_difficulty
        ));

        feedback
    }

    /// Build initial question bank with sample questions
    fn build_question_bank(&mut self) {
        // Create sample questions for basic arithmetic
        self.add_arithmetic_questions();
        
        // Create sample questions for linear equations
        self.add_linear_equation_questions();
        
        // Create sample questions for algebraic expressions
        self.add_algebraic_expression_questions();
    }

    /// Add arithmetic questions to the bank
    fn add_arithmetic_questions(&mut self) {
        let questions = vec![
            // Beginner level
            self.create_question("arith_1", AlgebraProblem::linear_equation(1.0, 0.0, 5.0), 
                                DifficultyLevel::Beginner, "basic_arithmetic", ModuleType::Arithmetic, 5.0),
            self.create_question("arith_2", AlgebraProblem::linear_equation(1.0, 3.0, 8.0), 
                                DifficultyLevel::Elementary, "basic_arithmetic", ModuleType::Arithmetic, 5.0),
            // Add more questions...
        ];

        for question in questions {
            self.add_question_to_bank(question);
        }
    }

    /// Add linear equation questions
    fn add_linear_equation_questions(&mut self) {
        let questions = vec![
            self.create_question("linear_1", AlgebraProblem::linear_equation(2.0, 3.0, 7.0),
                                DifficultyLevel::Intermediate, "linear_equations", ModuleType::LinearAlgebra, 2.0),
            self.create_question("linear_2", AlgebraProblem::linear_equation(3.0, -1.0, 8.0),
                                DifficultyLevel::Advanced, "linear_equations", ModuleType::LinearAlgebra, 3.0),
        ];

        for question in questions {
            self.add_question_to_bank(question);
        }
    }

    /// Add algebraic expression questions
    fn add_algebraic_expression_questions(&mut self) {
        // For now, use linear equations as placeholders
        // In a real system, these would be more diverse algebraic expressions
        let questions = vec![
            self.create_question("alg_1", AlgebraProblem::linear_equation(4.0, 2.0, 14.0),
                                DifficultyLevel::Intermediate, "algebraic_expressions", ModuleType::Polynomial, 3.0),
        ];

        for question in questions {
            self.add_question_to_bank(question);
        }
    }

    /// Helper to create assessment questions
    fn create_question(
        &self,
        id: &str,
        problem: AlgebraProblem,
        difficulty: DifficultyLevel,
        topic: &str,
        module_type: ModuleType,
        expected_answer: f64
    ) -> AssessmentQuestion {
        AssessmentQuestion {
            question_id: id.to_string(),
            problem,
            difficulty,
            topic: topic.to_string(),
            module_type,
            expected_answer,
            time_limit: Some(60), // 1 minute per question
            points: match difficulty {
                DifficultyLevel::Beginner => 1,
                DifficultyLevel::Elementary => 2,
                DifficultyLevel::Intermediate => 3,
                DifficultyLevel::Advanced => 4,
                DifficultyLevel::Expert => 5,
            },
            cognitive_level: CognitiveLevel::Apply,
        }
    }

    /// Add question to organized question bank
    fn add_question_to_bank(&mut self, question: AssessmentQuestion) {
        // Add to topic index
        self.question_bank.by_topic
            .entry(question.topic.clone())
            .or_insert_with(Vec::new)
            .push(question.clone());

        // Add to difficulty index
        self.question_bank.by_difficulty
            .entry(question.difficulty)
            .or_insert_with(Vec::new)
            .push(question.clone());

        // Add to module type index
        self.question_bank.by_module
            .entry(question.module_type)
            .or_insert_with(Vec::new)
            .push(question);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::education::student_model::LearningStyle;

    #[test]
    fn test_assessment_engine_creation() {
        let config = EducationConfig::default();
        let engine = AssessmentEngine::new(&config);
        
        assert!(!engine.assessment_configs.is_empty());
        assert!(!engine.question_bank.by_topic.is_empty());
    }

    #[test]
    fn test_difficulty_level_conversion() {
        assert_eq!(DifficultyLevel::from(1), DifficultyLevel::Beginner);
        assert_eq!(DifficultyLevel::from(3), DifficultyLevel::Intermediate);
        assert_eq!(u8::from(DifficultyLevel::Advanced), 4);
    }

    #[test]
    fn test_assessment_conduct() -> Result<()> {
        let config = EducationConfig::default();
        let mut engine = AssessmentEngine::new(&config);
        let student_model = StudentModel::new(
            "test_student".to_string(),
            15,
            LearningStyle::Visual
        );

        let result = engine.assess_student(&student_model, "basic_arithmetic")?;
        
        assert_eq!(result.student_id, "test_student");
        assert!(!result.questions.is_empty());
        assert_eq!(result.questions.len(), result.responses.len());
        assert!(result.overall_score >= 0.0 && result.overall_score <= 1.0);

        Ok(())
    }
}