//! Adaptive Tutoring System
//!
//! This module provides an intelligent tutoring system that adapts its teaching
//! strategies, problem selection, and explanation style based on individual
//! student learning patterns and performance.

use super::{StudentModel, LearningStyle, EducationConfig};
use crate::error::{NEATError, Result};
use crate::calculator::modules::{MathModule, ModuleType};
use crate::calculator::algebra::{Expression, AlgebraProblem};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Different tutoring strategies based on learning research
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TutoringStrategy {
    /// Direct instruction with clear explanations
    DirectInstruction,
    /// Guided discovery through questioning
    GuidedDiscovery,
    /// Problem-based learning approach
    ProblemBased,
    /// Scaffolded support with gradual release
    Scaffolded,
    /// Spaced repetition for reinforcement
    SpacedRepetition,
    /// Mastery-based progression
    MasteryBased,
}

/// Current tutoring session state and context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TutoringSession {
    /// Session identifier
    pub session_id: String,
    /// Student being tutored
    pub student_id: String,
    /// Current topic being taught
    pub current_topic: String,
    /// Active tutoring strategy
    pub strategy: TutoringStrategy,
    /// Session start time
    pub start_time: DateTime<Utc>,
    /// Problems presented in this session
    pub problems_presented: Vec<TutoringProblem>,
    /// Student responses and performance
    pub responses: Vec<StudentResponse>,
    /// Current difficulty level
    pub current_difficulty: u8,
    /// Session goals and objectives
    pub session_goals: Vec<String>,
    /// Adaptive adjustments made
    pub adaptations: Vec<TutoringAdaptation>,
}

/// Problem presented to student with tutoring context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TutoringProblem {
    /// Problem identifier
    pub problem_id: String,
    /// The mathematical problem
    pub problem: AlgebraProblem,
    /// Difficulty level (1-5)
    pub difficulty: u8,
    /// Topic/skill being assessed
    pub topic: String,
    /// Hints available for this problem
    pub hints: Vec<String>,
    /// Step-by-step solution breakdown
    pub solution_steps: Vec<SolutionStep>,
    /// Expected learning objective
    pub learning_objective: String,
    /// Time when problem was presented
    pub presented_at: DateTime<Utc>,
}

/// Solution step for guided problem solving
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolutionStep {
    /// Step number in sequence
    pub step_number: u8,
    /// Description of this step
    pub description: String,
    /// Mathematical operation or concept
    pub operation: String,
    /// Result after this step
    pub result: String,
    /// Explanation of why this step is needed
    pub explanation: String,
}

/// Student response to a tutoring problem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudentResponse {
    /// Problem this response is for
    pub problem_id: String,
    /// Student's answer
    pub answer: String,
    /// Whether answer was correct
    pub is_correct: bool,
    /// Time taken to respond (seconds)
    pub response_time: u32,
    /// Number of hints used
    pub hints_used: u8,
    /// Number of attempts made
    pub attempts: u8,
    /// Confidence level expressed by student
    pub confidence: Option<f64>,
    /// Response timestamp
    pub responded_at: DateTime<Utc>,
}

/// Adaptive adjustment made during tutoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TutoringAdaptation {
    /// Type of adaptation made
    pub adaptation_type: AdaptationType,
    /// Reason for the adaptation
    pub reason: String,
    /// Previous value (if applicable)
    pub previous_value: Option<String>,
    /// New value
    pub new_value: String,
    /// Timestamp of adaptation
    pub adapted_at: DateTime<Utc>,
}

/// Types of adaptive adjustments
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptationType {
    /// Changed difficulty level
    DifficultyAdjustment,
    /// Changed tutoring strategy
    StrategyChange,
    /// Added more scaffolding
    ScaffoldingIncrease,
    /// Reduced scaffolding
    ScaffoldingDecrease,
    /// Changed explanation style
    ExplanationStyleChange,
    /// Adjusted pacing
    PacingAdjustment,
}

/// Intelligent adaptive tutoring system
pub struct AdaptiveTutor {
    /// Educational configuration
    config: EducationConfig,
    /// Active tutoring sessions
    active_sessions: HashMap<String, TutoringSession>,
    /// Strategy effectiveness tracking
    strategy_effectiveness: HashMap<String, StrategyMetrics>,
    /// Problem generation parameters
    problem_parameters: ProblemParameters,
}

/// Metrics for tracking strategy effectiveness
#[derive(Debug, Clone, Default)]
struct StrategyMetrics {
    /// Times this strategy was used
    usage_count: u32,
    /// Average success rate
    success_rate: f64,
    /// Average engagement level
    engagement_level: f64,
    /// Average time to mastery
    time_to_mastery: f64,
}

/// Parameters for generating educational problems
#[derive(Debug, Clone)]
struct ProblemParameters {
    /// Base difficulty adjustment
    difficulty_adjustment: f64,
    /// Hint availability settings
    hint_settings: HintSettings,
    /// Problem type preferences
    problem_type_weights: HashMap<String, f64>,
}

/// Settings for hint system
#[derive(Debug, Clone)]
struct HintSettings {
    /// Maximum hints per problem
    max_hints: u8,
    /// Hint reveal strategy
    reveal_strategy: HintRevealStrategy,
    /// Adaptive hint difficulty
    adaptive_difficulty: bool,
}

/// Strategy for revealing hints
#[derive(Debug, Clone, Copy)]
enum HintRevealStrategy {
    /// Reveal hints immediately when requested
    Immediate,
    /// Reveal hints after time delay
    Delayed,
    /// Reveal hints based on number of attempts
    AttemptBased,
    /// Adaptive based on student model
    Adaptive,
}

impl Default for ProblemParameters {
    fn default() -> Self {
        Self {
            difficulty_adjustment: 0.0,
            hint_settings: HintSettings {
                max_hints: 3,
                reveal_strategy: HintRevealStrategy::Adaptive,
                adaptive_difficulty: true,
            },
            problem_type_weights: HashMap::new(),
        }
    }
}

impl AdaptiveTutor {
    /// Create a new adaptive tutoring system
    pub fn new(config: &EducationConfig) -> Self {
        Self {
            config: config.clone(),
            active_sessions: HashMap::new(),
            strategy_effectiveness: HashMap::new(),
            problem_parameters: ProblemParameters::default(),
        }
    }

    /// Start a new tutoring session for a student
    pub fn start_session(&mut self, student_model: &StudentModel) -> Result<TutoringSession> {
        let session_id = format!("session_{}_{}", student_model.student_id, Utc::now().timestamp());
        
        // Determine optimal tutoring strategy
        let strategy = self.select_optimal_strategy(student_model);
        
        // Identify priority topic for this session
        let current_topic = self.select_session_topic(student_model)?;
        
        // Set session goals
        let session_goals = self.generate_session_goals(student_model, &current_topic);
        
        let session = TutoringSession {
            session_id: session_id.clone(),
            student_id: student_model.student_id.clone(),
            current_topic,
            strategy,
            start_time: Utc::now(),
            problems_presented: Vec::new(),
            responses: Vec::new(),
            current_difficulty: student_model.overall_performance.preferred_difficulty,
            session_goals,
            adaptations: Vec::new(),
        };

        self.active_sessions.insert(session_id.clone(), session);
        Ok(self.active_sessions[&session_id].clone())
    }

    /// Get next problem for student in current session
    pub fn get_next_problem(&mut self, session_id: &str, student_model: &StudentModel) -> Result<TutoringProblem> {
        let session = self.active_sessions.get_mut(session_id)
            .ok_or_else(|| NEATError::InvalidConfiguration {
                parameter: "session_id".to_string(),
                value: session_id.to_string(),
            })?;

        // Analyze recent performance for adaptations
        self.analyze_and_adapt(session, student_model)?;

        // Generate appropriate problem based on current state
        let problem = self.generate_adaptive_problem(session, student_model)?;
        
        session.problems_presented.push(problem.clone());
        Ok(problem)
    }

    /// Process student response and provide feedback
    pub fn process_response(&mut self, session_id: &str, response: StudentResponse) -> Result<TutoringFeedback> {
        let session = self.active_sessions.get_mut(session_id)
            .ok_or_else(|| NEATError::InvalidConfiguration {
                parameter: "session_id".to_string(),
                value: session_id.to_string(),
            })?;

        session.responses.push(response.clone());

        // Generate appropriate feedback based on strategy
        let feedback = self.generate_feedback(&response, session)?;
        
        Ok(feedback)
    }

    /// Select optimal tutoring strategy for student
    fn select_optimal_strategy(&self, student_model: &StudentModel) -> TutoringStrategy {
        // Consider learning style
        let style_preference = match student_model.learning_style {
            LearningStyle::Visual => TutoringStrategy::Scaffolded,
            LearningStyle::Auditory => TutoringStrategy::DirectInstruction,
            LearningStyle::Kinesthetic => TutoringStrategy::ProblemBased,
            LearningStyle::ReadingWriting => TutoringStrategy::GuidedDiscovery,
            LearningStyle::Multimodal => TutoringStrategy::MasteryBased,
        };

        // Consider performance level
        if student_model.overall_performance.average_mastery < 0.4 {
            TutoringStrategy::DirectInstruction
        } else if student_model.overall_performance.average_confidence < 0.5 {
            TutoringStrategy::Scaffolded
        } else {
            style_preference
        }
    }

    /// Select topic for current session
    fn select_session_topic(&self, student_model: &StudentModel) -> Result<String> {
        // Priority: topics that need review
        let needs_review = student_model.get_topics_needing_review();
        if !needs_review.is_empty() {
            return Ok(needs_review[0].clone());
        }

        // Next: continue with partially mastered topics
        for (topic, state) in &student_model.knowledge_states {
            if state.mastery_level > 0.3 && state.mastery_level < 0.8 {
                return Ok(topic.clone());
            }
        }

        // Default: start with basic arithmetic
        Ok("basic_arithmetic".to_string())
    }

    /// Generate session goals based on student needs
    fn generate_session_goals(&self, student_model: &StudentModel, topic: &str) -> Vec<String> {
        let mut goals = Vec::new();

        if let Some(knowledge_state) = student_model.get_knowledge_state(topic) {
            if knowledge_state.mastery_level < 0.5 {
                goals.push(format!("Build foundational understanding of {}", topic));
            } else if knowledge_state.confidence < 0.7 {
                goals.push(format!("Increase confidence in {}", topic));
            } else {
                goals.push(format!("Advance to higher difficulty in {}", topic));
            }
        } else {
            goals.push(format!("Introduce basic concepts of {}", topic));
        }

        // Add engagement and time goals
        goals.push("Maintain high engagement throughout session".to_string());
        goals.push(format!("Complete session within {} minutes", self.config.max_session_duration));

        goals
    }

    /// Analyze performance and make adaptive adjustments
    fn analyze_and_adapt(&mut self, session: &mut TutoringSession, student_model: &StudentModel) -> Result<()> {
        if session.responses.len() < 2 {
            return Ok(()); // Need more data
        }

        let recent_responses = &session.responses[session.responses.len().saturating_sub(3)..];
        let success_rate = recent_responses.iter()
            .map(|r| if r.is_correct { 1.0 } else { 0.0 })
            .sum::<f64>() / recent_responses.len() as f64;

        // Adapt difficulty based on performance
        if success_rate < 0.4 && session.current_difficulty > 1 {
            session.current_difficulty -= 1;
            session.adaptations.push(TutoringAdaptation {
                adaptation_type: AdaptationType::DifficultyAdjustment,
                reason: "Low success rate, reducing difficulty".to_string(),
                previous_value: Some((session.current_difficulty + 1).to_string()),
                new_value: session.current_difficulty.to_string(),
                adapted_at: Utc::now(),
            });
        } else if success_rate > 0.8 && session.current_difficulty < 5 {
            session.current_difficulty += 1;
            session.adaptations.push(TutoringAdaptation {
                adaptation_type: AdaptationType::DifficultyAdjustment,
                reason: "High success rate, increasing difficulty".to_string(),
                previous_value: Some((session.current_difficulty - 1).to_string()),
                new_value: session.current_difficulty.to_string(),
                adapted_at: Utc::now(),
            });
        }

        // Adapt strategy if needed
        let avg_response_time = recent_responses.iter()
            .map(|r| r.response_time)
            .sum::<u32>() / recent_responses.len() as u32;

        if avg_response_time > 120 && session.strategy != TutoringStrategy::Scaffolded {
            session.strategy = TutoringStrategy::Scaffolded;
            session.adaptations.push(TutoringAdaptation {
                adaptation_type: AdaptationType::StrategyChange,
                reason: "Slow response times, adding more scaffolding".to_string(),
                previous_value: None,
                new_value: "Scaffolded".to_string(),
                adapted_at: Utc::now(),
            });
        }

        Ok(())
    }

    /// Generate adaptive problem based on current session state
    fn generate_adaptive_problem(&self, session: &TutoringSession, student_model: &StudentModel) -> Result<TutoringProblem> {
        let problem_id = format!("problem_{}_{}", session.session_id, session.problems_presented.len());
        
        // Create a simple linear equation problem based on difficulty
        let (a, b, c) = match session.current_difficulty {
            1 => (1.0, 0.0, 5.0),    // x = 5
            2 => (2.0, 0.0, 10.0),   // 2x = 10
            3 => (3.0, 1.0, 10.0),   // 3x + 1 = 10
            4 => (5.0, -2.0, 18.0),  // 5x - 2 = 18
            _ => (7.0, -3.0, 25.0),  // 7x - 3 = 25
        };

        let problem = AlgebraProblem::linear_equation(a, b, c);
        
        // Generate hints based on student needs
        let hints = self.generate_hints_for_problem(&problem, student_model);
        
        // Generate solution steps
        let solution_steps = self.generate_solution_steps(a, b, c);

        Ok(TutoringProblem {
            problem_id,
            problem,
            difficulty: session.current_difficulty,
            topic: session.current_topic.clone(),
            hints,
            solution_steps,
            learning_objective: format!("Solve linear equation with difficulty {}", session.current_difficulty),
            presented_at: Utc::now(),
        })
    }

    /// Generate appropriate hints for a problem
    fn generate_hints_for_problem(&self, _problem: &AlgebraProblem, student_model: &StudentModel) -> Vec<String> {
        let mut hints = Vec::new();
        
        // Customize hints based on learning style
        match student_model.learning_style {
            LearningStyle::Visual => {
                hints.push("Try visualizing the equation as a balance scale".to_string());
                hints.push("Draw a picture or diagram to represent the problem".to_string());
            },
            LearningStyle::Auditory => {
                hints.push("Read the equation out loud step by step".to_string());
                hints.push("Think about what the equation is 'saying' in words".to_string());
            },
            LearningStyle::Kinesthetic => {
                hints.push("Use physical objects to represent the variables".to_string());
                hints.push("Work through the problem with your hands or manipulatives".to_string());
            },
            _ => {
                hints.push("Break the problem down into smaller steps".to_string());
                hints.push("Remember to perform the same operation on both sides".to_string());
            }
        }

        hints.push("Check your answer by substituting back into the original equation".to_string());
        hints
    }

    /// Generate step-by-step solution
    fn generate_solution_steps(&self, a: f64, b: f64, c: f64) -> Vec<SolutionStep> {
        let mut steps = Vec::new();

        steps.push(SolutionStep {
            step_number: 1,
            description: "Identify the equation form".to_string(),
            operation: "analysis".to_string(),
            result: format!("{}x + {} = {}", a, b, c),
            explanation: "This is a linear equation in the form ax + b = c".to_string(),
        });

        if b != 0.0 {
            let sign = if b > 0.0 { "subtract" } else { "add" };
            let abs_b = b.abs();
            steps.push(SolutionStep {
                step_number: 2,
                description: format!("Isolate the variable term by {}ing {} from both sides", sign, abs_b),
                operation: format!("{} {}", sign, abs_b),
                result: format!("{}x = {}", a, c - b),
                explanation: "We need to get the x term by itself".to_string(),
            });
        }

        steps.push(SolutionStep {
            step_number: if b != 0.0 { 3 } else { 2 },
            description: format!("Divide both sides by {}", a),
            operation: format!("÷ {}", a),
            result: format!("x = {}", (c - b) / a),
            explanation: "This gives us the value of x".to_string(),
        });

        steps
    }

    /// Generate feedback based on student response
    fn generate_feedback(&self, response: &StudentResponse, session: &TutoringSession) -> Result<TutoringFeedback> {
        let feedback_type = if response.is_correct {
            if response.attempts == 1 && response.hints_used == 0 {
                FeedbackType::Excellent
            } else {
                FeedbackType::Correct
            }
        } else {
            if response.attempts >= 3 {
                FeedbackType::NeedsHelp
            } else {
                FeedbackType::TryAgain
            }
        };

        let message = match feedback_type {
            FeedbackType::Excellent => "Excellent work! You solved that perfectly!".to_string(),
            FeedbackType::Correct => "Great job! You got the right answer.".to_string(),
            FeedbackType::TryAgain => "Not quite right. Try thinking about the problem step by step.".to_string(),
            FeedbackType::NeedsHelp => "Let me help you work through this step by step.".to_string(),
        };

        Ok(TutoringFeedback {
            feedback_type,
            message,
            encouragement: self.generate_encouragement(response, session),
            next_hint: self.get_next_hint(response, session),
            suggested_action: self.suggest_next_action(response, session),
        })
    }

    /// Generate encouraging message
    fn generate_encouragement(&self, response: &StudentResponse, _session: &TutoringSession) -> String {
        if response.is_correct {
            "Keep up the great work!".to_string()
        } else if response.attempts == 1 {
            "Good effort! Let's try a different approach.".to_string()
        } else {
            "Don't give up! You're learning with each attempt.".to_string()
        }
    }

    /// Get next hint if needed
    fn get_next_hint(&self, response: &StudentResponse, session: &TutoringSession) -> Option<String> {
        if !response.is_correct && response.hints_used < 3 {
            if let Some(problem) = session.problems_presented.last() {
                problem.hints.get(response.hints_used as usize).cloned()
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Suggest next action for student
    fn suggest_next_action(&self, response: &StudentResponse, _session: &TutoringSession) -> String {
        if response.is_correct {
            "Ready for the next problem?".to_string()
        } else if response.attempts >= 3 {
            "Let's work through this together step by step.".to_string()
        } else {
            "Try again, or ask for a hint if you need help.".to_string()
        }
    }
}

/// Feedback provided to student after response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TutoringFeedback {
    /// Type of feedback
    pub feedback_type: FeedbackType,
    /// Main feedback message
    pub message: String,
    /// Encouraging words
    pub encouragement: String,
    /// Next hint if available
    pub next_hint: Option<String>,
    /// Suggested next action
    pub suggested_action: String,
}

/// Types of feedback
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedbackType {
    /// Perfect performance
    Excellent,
    /// Correct answer
    Correct,
    /// Incorrect, try again
    TryAgain,
    /// Student needs additional help
    NeedsHelp,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::education::student_model::LearningStyle;

    #[test]
    fn test_adaptive_tutor_creation() {
        let config = EducationConfig::default();
        let tutor = AdaptiveTutor::new(&config);
        
        assert!(tutor.active_sessions.is_empty());
    }

    #[test]
    fn test_strategy_selection() {
        let config = EducationConfig::default();
        let tutor = AdaptiveTutor::new(&config);
        let student_model = StudentModel::new(
            "test".to_string(),
            15,
            LearningStyle::Visual
        );

        let strategy = tutor.select_optimal_strategy(&student_model);
        assert_eq!(strategy, TutoringStrategy::DirectInstruction); // Low mastery
    }

    #[test]
    fn test_session_start() -> Result<()> {
        let config = EducationConfig::default();
        let mut tutor = AdaptiveTutor::new(&config);
        let student_model = StudentModel::new(
            "test".to_string(),
            15,
            LearningStyle::Visual
        );

        let session = tutor.start_session(&student_model)?;
        
        assert_eq!(session.student_id, "test");
        assert!(!session.session_goals.is_empty());
        assert!(tutor.active_sessions.contains_key(&session.session_id));

        Ok(())
    }
}