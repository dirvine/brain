//! Student Model and Learning State Tracking
//!
//! This module provides sophisticated modeling of individual students' learning
//! states, preferences, strengths, weaknesses, and progress over time.

use crate::error::{NEATError, Result};
use crate::calculator::modules::ModuleType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Learning style preferences for personalized instruction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LearningStyle {
    /// Learns best through visual representations
    Visual,
    /// Learns best through auditory instruction
    Auditory,
    /// Learns best through hands-on practice
    Kinesthetic,
    /// Learns best through reading and writing
    ReadingWriting,
    /// Mixed learning style preferences
    Multimodal,
}

/// Current knowledge state for a mathematical topic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeState {
    /// Topic or skill identifier
    pub topic: String,
    /// Mastery level (0.0 to 1.0)
    pub mastery_level: f64,
    /// Confidence in knowledge (0.0 to 1.0)
    pub confidence: f64,
    /// Number of practice sessions
    pub practice_count: u32,
    /// Last assessment score
    pub last_assessment_score: Option<f64>,
    /// Time spent learning this topic (in minutes)
    pub time_spent: u32,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

impl KnowledgeState {
    /// Create new knowledge state for a topic
    pub fn new(topic: String) -> Self {
        Self {
            topic,
            mastery_level: 0.0,
            confidence: 0.0,
            practice_count: 0,
            last_assessment_score: None,
            time_spent: 0,
            last_updated: Utc::now(),
        }
    }

    /// Update knowledge state based on performance
    pub fn update(&mut self, score: f64, time_spent: u32) {
        self.practice_count += 1;
        self.time_spent += time_spent;
        self.last_assessment_score = Some(score);
        self.last_updated = Utc::now();

        // Update mastery level with weighted average
        let learning_rate = 0.1;
        self.mastery_level = self.mastery_level * (1.0 - learning_rate) + score * learning_rate;
        
        // Update confidence based on consistency
        let consistency = if self.practice_count > 1 {
            1.0 - (score - self.mastery_level).abs()
        } else {
            score
        };
        self.confidence = self.confidence * 0.8 + consistency * 0.2;
    }

    /// Check if topic is mastered
    pub fn is_mastered(&self) -> bool {
        self.mastery_level >= 0.8 && self.confidence >= 0.7
    }

    /// Check if topic needs review
    pub fn needs_review(&self) -> bool {
        self.mastery_level < 0.6 || self.confidence < 0.5
    }
}

/// Progress tracking data for analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudentProgress {
    /// Session timestamp
    pub timestamp: DateTime<Utc>,
    /// Topic or skill practiced
    pub topic: String,
    /// Performance score (0.0 to 1.0)
    pub score: f64,
    /// Time spent in minutes
    pub time_spent: u32,
    /// Number of problems attempted
    pub problems_attempted: u32,
    /// Number of problems correct
    pub problems_correct: u32,
    /// Difficulty level of problems
    pub difficulty_level: u8,
    /// Learning strategy used
    pub strategy_used: String,
    /// Student engagement level (0.0 to 1.0)
    pub engagement_level: f64,
}

/// Comprehensive student model tracking learning state and preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudentModel {
    /// Unique student identifier
    pub student_id: String,
    /// Student age in years
    pub age: u8,
    /// Preferred learning style
    pub learning_style: LearningStyle,
    /// Knowledge states for different topics
    pub knowledge_states: HashMap<String, KnowledgeState>,
    /// Overall performance metrics
    pub overall_performance: PerformanceMetrics,
    /// Learning preferences and patterns
    pub learning_preferences: LearningPreferences,
    /// Progress history
    pub progress_history: Vec<StudentProgress>,
    /// Registration timestamp
    pub created_at: DateTime<Utc>,
    /// Last activity timestamp
    pub last_active: DateTime<Utc>,
}

/// Overall performance metrics for the student
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average mastery across all topics
    pub average_mastery: f64,
    /// Average confidence across all topics
    pub average_confidence: f64,
    /// Total time spent learning (minutes)
    pub total_time_spent: u32,
    /// Total sessions completed
    pub total_sessions: u32,
    /// Current learning streak (consecutive days)
    pub learning_streak: u32,
    /// Preferred difficulty level
    pub preferred_difficulty: u8,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            average_mastery: 0.0,
            average_confidence: 0.0,
            total_time_spent: 0,
            total_sessions: 0,
            learning_streak: 0,
            preferred_difficulty: 3, // Medium difficulty
        }
    }
}

/// Learning preferences discovered through interaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningPreferences {
    /// Preferred session length (minutes)
    pub preferred_session_length: u32,
    /// Preferred number of problems per session
    pub preferred_problems_per_session: u32,
    /// Preferred explanation depth (1=brief, 5=detailed)
    pub explanation_depth: u8,
    /// Prefers visual aids
    pub prefers_visual_aids: bool,
    /// Prefers step-by-step guidance
    pub prefers_step_by_step: bool,
    /// Prefers immediate feedback
    pub prefers_immediate_feedback: bool,
    /// Module types the student excels at
    pub strong_modules: Vec<ModuleType>,
    /// Module types the student struggles with
    pub weak_modules: Vec<ModuleType>,
}

impl Default for LearningPreferences {
    fn default() -> Self {
        Self {
            preferred_session_length: 30,
            preferred_problems_per_session: 10,
            explanation_depth: 3,
            prefers_visual_aids: true,
            prefers_step_by_step: true,
            prefers_immediate_feedback: true,
            strong_modules: Vec::new(),
            weak_modules: Vec::new(),
        }
    }
}

impl StudentModel {
    /// Create a new student model
    pub fn new(student_id: String, age: u8, learning_style: LearningStyle) -> Self {
        Self {
            student_id,
            age,
            learning_style,
            knowledge_states: HashMap::new(),
            overall_performance: PerformanceMetrics::default(),
            learning_preferences: LearningPreferences::default(),
            progress_history: Vec::new(),
            created_at: Utc::now(),
            last_active: Utc::now(),
        }
    }

    /// Update student progress with new data
    pub fn update_progress(&mut self, progress: &StudentProgress) -> Result<()> {
        // Update or create knowledge state for the topic
        let knowledge_state = self.knowledge_states
            .entry(progress.topic.clone())
            .or_insert_with(|| KnowledgeState::new(progress.topic.clone()));
        
        knowledge_state.update(progress.score, progress.time_spent);

        // Update overall performance metrics
        self.update_performance_metrics();

        // Add to progress history
        self.progress_history.push(progress.clone());

        // Update last active timestamp
        self.last_active = Utc::now();

        // Update learning preferences based on patterns
        self.update_learning_preferences();

        Ok(())
    }

    /// Get knowledge state for a specific topic
    pub fn get_knowledge_state(&self, topic: &str) -> Option<&KnowledgeState> {
        self.knowledge_states.get(topic)
    }

    /// Get topics that need review
    pub fn get_topics_needing_review(&self) -> Vec<String> {
        self.knowledge_states
            .values()
            .filter(|state| state.needs_review())
            .map(|state| state.topic.clone())
            .collect()
    }

    /// Get mastered topics
    pub fn get_mastered_topics(&self) -> Vec<String> {
        self.knowledge_states
            .values()
            .filter(|state| state.is_mastered())
            .map(|state| state.topic.clone())
            .collect()
    }

    /// Get recommended difficulty level for a topic
    pub fn get_recommended_difficulty(&self, topic: &str) -> u8 {
        if let Some(knowledge_state) = self.get_knowledge_state(topic) {
            let base_difficulty = if knowledge_state.mastery_level >= 0.8 {
                5 // Advanced
            } else if knowledge_state.mastery_level >= 0.6 {
                4 // Intermediate-Advanced
            } else if knowledge_state.mastery_level >= 0.4 {
                3 // Intermediate
            } else if knowledge_state.mastery_level >= 0.2 {
                2 // Basic-Intermediate
            } else {
                1 // Basic
            };

            // Adjust based on confidence
            if knowledge_state.confidence < 0.5 {
                (base_difficulty.saturating_sub(1)).max(1)
            } else if knowledge_state.confidence > 0.8 {
                (base_difficulty + 1).min(5)
            } else {
                base_difficulty
            }
        } else {
            // No previous knowledge, start with basic
            1
        }
    }

    /// Update overall performance metrics
    fn update_performance_metrics(&mut self) {
        if self.knowledge_states.is_empty() {
            return;
        }

        // Calculate averages
        let total_states = self.knowledge_states.len() as f64;
        self.overall_performance.average_mastery = self.knowledge_states
            .values()
            .map(|state| state.mastery_level)
            .sum::<f64>() / total_states;

        self.overall_performance.average_confidence = self.knowledge_states
            .values()
            .map(|state| state.confidence)
            .sum::<f64>() / total_states;

        self.overall_performance.total_time_spent = self.knowledge_states
            .values()
            .map(|state| state.time_spent)
            .sum();

        self.overall_performance.total_sessions = self.progress_history.len() as u32;

        // Update preferred difficulty based on performance
        if self.overall_performance.average_mastery >= 0.8 {
            self.overall_performance.preferred_difficulty = 4;
        } else if self.overall_performance.average_mastery >= 0.6 {
            self.overall_performance.preferred_difficulty = 3;
        } else {
            self.overall_performance.preferred_difficulty = 2;
        }
    }

    /// Update learning preferences based on observed patterns
    fn update_learning_preferences(&mut self) {
        if self.progress_history.len() < 3 {
            return; // Need sufficient data
        }

        // Analyze session lengths
        let avg_session_time = self.progress_history
            .iter()
            .map(|p| p.time_spent)
            .sum::<u32>() / self.progress_history.len() as u32;
        
        self.learning_preferences.preferred_session_length = avg_session_time;

        // Analyze problems per session
        let avg_problems = self.progress_history
            .iter()
            .map(|p| p.problems_attempted)
            .sum::<u32>() / self.progress_history.len() as u32;
        
        self.learning_preferences.preferred_problems_per_session = avg_problems;

        // Analyze engagement patterns
        let high_engagement_sessions: Vec<_> = self.progress_history
            .iter()
            .filter(|p| p.engagement_level > 0.7)
            .collect();

        if !high_engagement_sessions.is_empty() {
            // Update preferences based on high-engagement sessions
            self.learning_preferences.prefers_immediate_feedback = 
                high_engagement_sessions.len() as f64 / self.progress_history.len() as f64 > 0.6;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_student_model_creation() {
        let model = StudentModel::new(
            "test_student".to_string(),
            15,
            LearningStyle::Visual
        );

        assert_eq!(model.student_id, "test_student");
        assert_eq!(model.age, 15);
        assert_eq!(model.learning_style, LearningStyle::Visual);
        assert!(model.knowledge_states.is_empty());
    }

    #[test]
    fn test_knowledge_state_update() {
        let mut state = KnowledgeState::new("algebra".to_string());
        
        assert_eq!(state.mastery_level, 0.0);
        assert_eq!(state.practice_count, 0);
        
        state.update(0.8, 30);
        
        assert!(state.mastery_level > 0.0);
        assert_eq!(state.practice_count, 1);
        assert_eq!(state.time_spent, 30);
    }

    #[test]
    fn test_progress_update() -> Result<()> {
        let mut model = StudentModel::new(
            "test_student".to_string(),
            15,
            LearningStyle::Visual
        );

        let progress = StudentProgress {
            timestamp: Utc::now(),
            topic: "algebra".to_string(),
            score: 0.85,
            time_spent: 25,
            problems_attempted: 10,
            problems_correct: 8,
            difficulty_level: 3,
            strategy_used: "guided_practice".to_string(),
            engagement_level: 0.9,
        };

        model.update_progress(&progress)?;

        assert!(model.knowledge_states.contains_key("algebra"));
        assert_eq!(model.progress_history.len(), 1);
        assert_eq!(model.overall_performance.total_sessions, 1);

        Ok(())
    }

    #[test]
    fn test_difficulty_recommendation() {
        let mut model = StudentModel::new(
            "test_student".to_string(),
            15,
            LearningStyle::Visual
        );

        // Test with no prior knowledge
        assert_eq!(model.get_recommended_difficulty("new_topic"), 1);

        // Add some knowledge
        let mut state = KnowledgeState::new("algebra".to_string());
        state.mastery_level = 0.7;
        state.confidence = 0.8;
        model.knowledge_states.insert("algebra".to_string(), state);

        assert_eq!(model.get_recommended_difficulty("algebra"), 4);
    }
}