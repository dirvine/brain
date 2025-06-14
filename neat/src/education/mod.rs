//! Educational Technology Integration for NEAT Mathematical Platform
//!
//! This module transforms the NEAT mathematical research platform into a comprehensive
//! educational tool featuring adaptive tutoring, personalized learning paths, automated
//! assessment, and real-time learning analytics.

pub mod adaptive_tutor;
pub mod assessment;
pub mod curriculum;
pub mod learning_analytics;
pub mod problem_generator;
pub mod student_model;
pub mod explanation_engine;

pub use adaptive_tutor::{AdaptiveTutor, TutoringSession, TutoringStrategy};
pub use assessment::{AssessmentEngine, AssessmentResult, DifficultyLevel};
pub use curriculum::{CurriculumGenerator, LearningPath, LearningObjective};
pub use learning_analytics::{LearningAnalytics, StudentProgress, LearningInsight};
pub use problem_generator::{EducationalProblemGenerator, ProblemType, ProblemDifficulty};
pub use student_model::{StudentModel, LearningStyle, KnowledgeState};
pub use explanation_engine::{ExplanationEngine, ExplanationType, StepByStepSolution};

use crate::error::Result;
use crate::calculator::modules::{MathModule, ModuleType};
use std::collections::HashMap;

/// Educational configuration for the tutoring system
#[derive(Debug, Clone)]
pub struct EducationConfig {
    /// Target age group (in years)
    pub target_age_range: (u8, u8),
    /// Maximum session duration in minutes
    pub max_session_duration: u32,
    /// Adaptive difficulty adjustment rate
    pub adaptation_rate: f64,
    /// Minimum problems per assessment
    pub min_assessment_problems: usize,
    /// Enable detailed explanations
    pub enable_explanations: bool,
    /// Enable progress tracking
    pub enable_analytics: bool,
}

impl Default for EducationConfig {
    fn default() -> Self {
        Self {
            target_age_range: (10, 18),
            max_session_duration: 45,
            adaptation_rate: 0.1,
            min_assessment_problems: 5,
            enable_explanations: true,
            enable_analytics: true,
        }
    }
}

/// Main educational platform that orchestrates all components
pub struct EducationalPlatform {
    /// Adaptive tutoring system
    adaptive_tutor: AdaptiveTutor,
    /// Assessment and testing engine
    assessment_engine: AssessmentEngine,
    /// Curriculum and learning path generator
    curriculum_generator: CurriculumGenerator,
    /// Learning analytics system
    learning_analytics: LearningAnalytics,
    /// Problem generation system
    problem_generator: EducationalProblemGenerator,
    /// Student models for all users
    student_models: HashMap<String, StudentModel>,
    /// Educational configuration
    config: EducationConfig,
}

impl EducationalPlatform {
    /// Create a new educational platform
    pub fn new(config: EducationConfig) -> Self {
        Self {
            adaptive_tutor: AdaptiveTutor::new(&config),
            assessment_engine: AssessmentEngine::new(&config),
            curriculum_generator: CurriculumGenerator::new(&config),
            learning_analytics: LearningAnalytics::new(&config),
            problem_generator: EducationalProblemGenerator::new(&config),
            student_models: HashMap::new(),
            config,
        }
    }

    /// Register a new student in the system
    pub fn register_student(&mut self, student_id: String, age: u8, learning_style: LearningStyle) -> Result<()> {
        let student_model = StudentModel::new(student_id.clone(), age, learning_style);
        self.student_models.insert(student_id, student_model);
        Ok(())
    }

    /// Start a new tutoring session for a student
    pub fn start_tutoring_session(&mut self, student_id: &str) -> Result<TutoringSession> {
        let student_model = self.student_models.get(student_id)
            .ok_or_else(|| crate::error::NEATError::InvalidConfiguration {
                parameter: "student_id".to_string(),
                value: student_id.to_string(),
            })?;

        self.adaptive_tutor.start_session(student_model)
    }

    /// Conduct an assessment for a student
    pub fn conduct_assessment(&mut self, student_id: &str, topic: &str) -> Result<AssessmentResult> {
        let student_model = self.student_models.get(student_id)
            .ok_or_else(|| crate::error::NEATError::InvalidConfiguration {
                parameter: "student_id".to_string(),
                value: student_id.to_string(),
            })?;

        self.assessment_engine.assess_student(student_model, topic)
    }

    /// Generate a personalized learning path for a student
    pub fn generate_learning_path(&self, student_id: &str) -> Result<LearningPath> {
        let student_model = self.student_models.get(student_id)
            .ok_or_else(|| crate::error::NEATError::InvalidConfiguration {
                parameter: "student_id".to_string(),
                value: student_id.to_string(),
            })?;

        self.curriculum_generator.generate_path(student_model)
    }

    /// Get learning analytics for a student
    pub fn get_student_analytics(&self, student_id: &str) -> Result<Vec<LearningInsight>> {
        let student_model = self.student_models.get(student_id)
            .ok_or_else(|| crate::error::NEATError::InvalidConfiguration {
                parameter: "student_id".to_string(),
                value: student_id.to_string(),
            })?;

        Ok(self.learning_analytics.analyze_student_progress(student_model))
    }

    /// Update student model based on performance
    pub fn update_student_progress(&mut self, student_id: &str, performance_data: &StudentProgress) -> Result<()> {
        if let Some(student_model) = self.student_models.get_mut(student_id) {
            student_model.update_progress(performance_data)?;
            
            // Update analytics
            if self.config.enable_analytics {
                self.learning_analytics.record_progress(student_id, performance_data);
            }
        }
        Ok(())
    }

    /// Get recommended next activity for a student
    pub fn get_next_activity(&self, student_id: &str) -> Result<LearningObjective> {
        let student_model = self.student_models.get(student_id)
            .ok_or_else(|| crate::error::NEATError::InvalidConfiguration {
                parameter: "student_id".to_string(),
                value: student_id.to_string(),
            })?;

        self.curriculum_generator.recommend_next_activity(student_model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_educational_platform_creation() {
        let config = EducationConfig::default();
        let platform = EducationalPlatform::new(config);
        
        assert_eq!(platform.student_models.len(), 0);
    }

    #[test]
    fn test_student_registration() -> Result<()> {
        let mut platform = EducationalPlatform::new(EducationConfig::default());
        
        platform.register_student(
            "student1".to_string(),
            14,
            LearningStyle::Visual
        )?;
        
        assert_eq!(platform.student_models.len(), 1);
        assert!(platform.student_models.contains_key("student1"));
        
        Ok(())
    }
}