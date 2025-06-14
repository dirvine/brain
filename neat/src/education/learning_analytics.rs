//! Learning Analytics and Progress Tracking
//!
//! This module provides comprehensive analytics capabilities for tracking
//! student learning progress, identifying patterns, and generating actionable
//! insights for both students and educators.

use super::{StudentModel, EducationConfig, DifficultyLevel};
use crate::error::{NEATError, Result};
use crate::calculator::modules::ModuleType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc, Duration};

/// Learning insight types for different analytical perspectives
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InsightType {
    /// Performance trend analysis
    PerformanceTrend,
    /// Learning velocity insights
    LearningVelocity,
    /// Strength identification
    StrengthAnalysis,
    /// Weakness identification
    WeaknessAnalysis,
    /// Engagement pattern analysis
    EngagementPattern,
    /// Time management insights
    TimeManagement,
    /// Difficulty progression
    DifficultyProgression,
}

/// Actionable learning insight with recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningInsight {
    /// Type of insight
    pub insight_type: InsightType,
    /// Brief title of the insight
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Confidence level of this insight (0.0 to 1.0)
    pub confidence: f64,
    /// Specific recommendations
    pub recommendations: Vec<String>,
    /// Metrics supporting this insight
    pub supporting_metrics: HashMap<String, f64>,
    /// When this insight was generated
    pub generated_at: DateTime<Utc>,
    /// Priority level for addressing this insight
    pub priority: InsightPriority,
}

/// Priority levels for insights
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InsightPriority {
    /// Critical issue requiring immediate attention
    Critical,
    /// High priority for near-term focus
    High,
    /// Medium priority for regular attention
    Medium,
    /// Low priority for long-term consideration
    Low,
    /// Informational insight
    Info,
}

/// Comprehensive student progress data
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

/// Analytics session for tracking learning patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsSession {
    /// Session identifier
    pub session_id: String,
    /// Student identifier
    pub student_id: String,
    /// Session start time
    pub start_time: DateTime<Utc>,
    /// Session end time
    pub end_time: Option<DateTime<Utc>>,
    /// Topics covered in session
    pub topics_covered: Vec<String>,
    /// Overall session performance
    pub session_performance: f64,
    /// Session engagement metrics
    pub engagement_metrics: EngagementMetrics,
    /// Learning goals achieved
    pub goals_achieved: Vec<String>,
    /// Challenges encountered
    pub challenges: Vec<String>,
}

/// Engagement tracking metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngagementMetrics {
    /// Average time per problem (seconds)
    pub avg_time_per_problem: f64,
    /// Number of hint requests
    pub hint_requests: u32,
    /// Number of retry attempts
    pub retry_attempts: u32,
    /// Session completion rate
    pub completion_rate: f64,
    /// Focus level indicators
    pub focus_indicators: FocusIndicators,
}

/// Indicators of student focus and attention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocusIndicators {
    /// Consistent response timing
    pub timing_consistency: f64,
    /// Problem-solving persistence
    pub persistence_level: f64,
    /// Error pattern analysis
    pub error_pattern_score: f64,
}

impl Default for FocusIndicators {
    fn default() -> Self {
        Self {
            timing_consistency: 0.5,
            persistence_level: 0.5,
            error_pattern_score: 0.5,
        }
    }
}

/// Main learning analytics engine
pub struct LearningAnalytics {
    /// Educational configuration
    config: EducationConfig,
    /// Active analytics sessions
    active_sessions: HashMap<String, AnalyticsSession>,
    /// Progress history by student
    progress_history: HashMap<String, Vec<StudentProgress>>,
    /// Generated insights cache
    insights_cache: HashMap<String, Vec<LearningInsight>>,
    /// Analytics parameters
    analytics_parameters: AnalyticsParameters,
}

/// Parameters for analytics calculations
#[derive(Debug, Clone)]
struct AnalyticsParameters {
    /// Minimum data points for trend analysis
    min_trend_points: usize,
    /// Performance trend analysis window (days)
    trend_window_days: i64,
    /// Engagement threshold for concern
    low_engagement_threshold: f64,
    /// Performance drop threshold
    performance_drop_threshold: f64,
}

impl Default for AnalyticsParameters {
    fn default() -> Self {
        Self {
            min_trend_points: 5,
            trend_window_days: 7,
            low_engagement_threshold: 0.4,
            performance_drop_threshold: 0.2,
        }
    }
}

impl LearningAnalytics {
    /// Create a new learning analytics system
    pub fn new(config: &EducationConfig) -> Self {
        Self {
            config: config.clone(),
            active_sessions: HashMap::new(),
            progress_history: HashMap::new(),
            insights_cache: HashMap::new(),
            analytics_parameters: AnalyticsParameters::default(),
        }
    }

    /// Record student progress data
    pub fn record_progress(&mut self, student_id: &str, progress: &StudentProgress) {
        self.progress_history
            .entry(student_id.to_string())
            .or_insert_with(Vec::new)
            .push(progress.clone());

        // Clear insights cache to force regeneration
        self.insights_cache.remove(student_id);
    }

    /// Analyze student progress and generate insights
    pub fn analyze_student_progress(&self, student_model: &StudentModel) -> Vec<LearningInsight> {
        let mut insights = Vec::new();

        // Check cache first
        if let Some(cached_insights) = self.insights_cache.get(&student_model.student_id) {
            return cached_insights.clone();
        }

        // Get student's progress history
        let empty_vec = Vec::new();
        let progress_history = self.progress_history
            .get(&student_model.student_id)
            .unwrap_or(&empty_vec);

        if progress_history.len() < 3 {
            return vec![self.create_insufficient_data_insight()];
        }

        // Analyze different aspects
        insights.extend(self.analyze_performance_trends(progress_history));
        insights.extend(self.analyze_learning_velocity(progress_history));
        insights.extend(self.analyze_strengths_weaknesses(student_model, progress_history));
        insights.extend(self.analyze_engagement_patterns(progress_history));
        insights.extend(self.analyze_time_management(progress_history));
        insights.extend(self.analyze_difficulty_progression(progress_history));

        insights
    }

    /// Analyze performance trends over time
    fn analyze_performance_trends(&self, progress_history: &[StudentProgress]) -> Vec<LearningInsight> {
        let mut insights = Vec::new();
        
        if progress_history.len() < self.analytics_parameters.min_trend_points {
            return insights;
        }

        // Calculate trend over recent sessions
        let recent_sessions = &progress_history[progress_history.len().saturating_sub(10)..];
        let scores: Vec<f64> = recent_sessions.iter().map(|p| p.score).collect();

        if scores.len() < 3 {
            return insights;
        }

        // Simple linear trend calculation
        let trend = self.calculate_linear_trend(&scores);

        if trend > 0.1 {
            insights.push(LearningInsight {
                insight_type: InsightType::PerformanceTrend,
                title: "Strong Upward Performance Trend".to_string(),
                description: format!(
                    "Performance has improved by {:.1}% over recent sessions, showing excellent learning progress.",
                    trend * 100.0
                ),
                confidence: 0.85,
                recommendations: vec![
                    "Continue current learning approach".to_string(),
                    "Consider advancing to more challenging material".to_string(),
                    "Celebrate this progress to maintain motivation".to_string(),
                ],
                supporting_metrics: [
                    ("trend_slope".to_string(), trend),
                    ("recent_avg_score".to_string(), scores.iter().sum::<f64>() / scores.len() as f64),
                ].iter().cloned().collect(),
                generated_at: Utc::now(),
                priority: InsightPriority::Medium,
            });
        } else if trend < -0.1 {
            insights.push(LearningInsight {
                insight_type: InsightType::PerformanceTrend,
                title: "Performance Decline Detected".to_string(),
                description: format!(
                    "Performance has declined by {:.1}% over recent sessions, requiring attention.",
                    -trend * 100.0
                ),
                confidence: 0.80,
                recommendations: vec![
                    "Review recent learning materials".to_string(),
                    "Consider reducing difficulty temporarily".to_string(),
                    "Schedule review sessions for problem areas".to_string(),
                    "Check for external factors affecting learning".to_string(),
                ],
                supporting_metrics: [
                    ("trend_slope".to_string(), trend),
                    ("recent_avg_score".to_string(), scores.iter().sum::<f64>() / scores.len() as f64),
                ].iter().cloned().collect(),
                generated_at: Utc::now(),
                priority: InsightPriority::High,
            });
        }

        insights
    }

    /// Analyze learning velocity and efficiency
    fn analyze_learning_velocity(&self, progress_history: &[StudentProgress]) -> Vec<LearningInsight> {
        let mut insights = Vec::new();

        // Calculate problems per minute for recent sessions
        let recent_sessions = &progress_history[progress_history.len().saturating_sub(5)..];
        let velocities: Vec<f64> = recent_sessions.iter()
            .filter(|p| p.time_spent > 0)
            .map(|p| p.problems_attempted as f64 / p.time_spent as f64)
            .collect();

        if velocities.is_empty() {
            return insights;
        }

        let avg_velocity = velocities.iter().sum::<f64>() / velocities.len() as f64;

        if avg_velocity > 0.5 {
            insights.push(LearningInsight {
                insight_type: InsightType::LearningVelocity,
                title: "High Learning Efficiency".to_string(),
                description: format!(
                    "Solving problems at {:.2} problems per minute shows excellent efficiency.",
                    avg_velocity
                ),
                confidence: 0.75,
                recommendations: vec![
                    "Maintain current pace".to_string(),
                    "Consider more challenging problems".to_string(),
                ],
                supporting_metrics: [
                    ("avg_velocity".to_string(), avg_velocity),
                ].iter().cloned().collect(),
                generated_at: Utc::now(),
                priority: InsightPriority::Low,
            });
        } else if avg_velocity < 0.2 {
            insights.push(LearningInsight {
                insight_type: InsightType::LearningVelocity,
                title: "Low Learning Velocity".to_string(),
                description: format!(
                    "Solving problems at {:.2} problems per minute suggests need for efficiency improvement.",
                    avg_velocity
                ),
                confidence: 0.70,
                recommendations: vec![
                    "Focus on problem-solving strategies".to_string(),
                    "Practice time management techniques".to_string(),
                    "Consider additional guided practice".to_string(),
                ],
                supporting_metrics: [
                    ("avg_velocity".to_string(), avg_velocity),
                ].iter().cloned().collect(),
                generated_at: Utc::now(),
                priority: InsightPriority::Medium,
            });
        }

        insights
    }

    /// Analyze student strengths and weaknesses
    fn analyze_strengths_weaknesses(&self, student_model: &StudentModel, progress_history: &[StudentProgress]) -> Vec<LearningInsight> {
        let mut insights = Vec::new();

        // Group progress by topic
        let mut topic_performance: HashMap<String, Vec<f64>> = HashMap::new();
        for progress in progress_history {
            topic_performance
                .entry(progress.topic.clone())
                .or_insert_with(Vec::new)
                .push(progress.score);
        }

        // Identify strengths (topics with consistently high performance)
        let mut strengths = Vec::new();
        let mut weaknesses = Vec::new();

        for (topic, scores) in &topic_performance {
            if scores.len() < 3 {
                continue;
            }

            let avg_score = scores.iter().sum::<f64>() / scores.len() as f64;
            if avg_score >= 0.8 {
                strengths.push((topic.clone(), avg_score));
            } else if avg_score < 0.6 {
                weaknesses.push((topic.clone(), avg_score));
            }
        }

        // Generate strength insights
        if !strengths.is_empty() {
            strengths.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top_strength = &strengths[0];
            
            insights.push(LearningInsight {
                insight_type: InsightType::StrengthAnalysis,
                title: format!("Strong Performance in {}", top_strength.0),
                description: format!(
                    "Consistently excellent performance in {} with {:.1}% average score.",
                    top_strength.0, top_strength.1 * 100.0
                ),
                confidence: 0.90,
                recommendations: vec![
                    format!("Use {} skills to support learning in other areas", top_strength.0),
                    "Consider peer tutoring opportunities".to_string(),
                    "Explore advanced topics in this strength area".to_string(),
                ],
                supporting_metrics: [
                    ("avg_score".to_string(), top_strength.1),
                    ("sessions_count".to_string(), topic_performance[&top_strength.0].len() as f64),
                ].iter().cloned().collect(),
                generated_at: Utc::now(),
                priority: InsightPriority::Info,
            });
        }

        // Generate weakness insights
        if !weaknesses.is_empty() {
            weaknesses.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let top_weakness = &weaknesses[0];
            
            insights.push(LearningInsight {
                insight_type: InsightType::WeaknessAnalysis,
                title: format!("Improvement Needed in {}", top_weakness.0),
                description: format!(
                    "Performance in {} needs attention with {:.1}% average score.",
                    top_weakness.0, top_weakness.1 * 100.0
                ),
                confidence: 0.85,
                recommendations: vec![
                    format!("Schedule focused practice sessions for {}", top_weakness.0),
                    "Break down concepts into smaller steps".to_string(),
                    "Seek additional support or tutoring".to_string(),
                    "Review prerequisite skills".to_string(),
                ],
                supporting_metrics: [
                    ("avg_score".to_string(), top_weakness.1),
                    ("sessions_count".to_string(), topic_performance[&top_weakness.0].len() as f64),
                ].iter().cloned().collect(),
                generated_at: Utc::now(),
                priority: InsightPriority::High,
            });
        }

        insights
    }

    /// Analyze engagement patterns
    fn analyze_engagement_patterns(&self, progress_history: &[StudentProgress]) -> Vec<LearningInsight> {
        let mut insights = Vec::new();

        let recent_sessions = &progress_history[progress_history.len().saturating_sub(10)..];
        let avg_engagement = recent_sessions.iter()
            .map(|p| p.engagement_level)
            .sum::<f64>() / recent_sessions.len() as f64;

        if avg_engagement < self.analytics_parameters.low_engagement_threshold {
            insights.push(LearningInsight {
                insight_type: InsightType::EngagementPattern,
                title: "Low Engagement Detected".to_string(),
                description: format!(
                    "Recent engagement level of {:.1}% indicates potential motivation issues.",
                    avg_engagement * 100.0
                ),
                confidence: 0.75,
                recommendations: vec![
                    "Introduce more interactive activities".to_string(),
                    "Vary learning strategies".to_string(),
                    "Set shorter, achievable goals".to_string(),
                    "Consider gamification elements".to_string(),
                ],
                supporting_metrics: [
                    ("avg_engagement".to_string(), avg_engagement),
                ].iter().cloned().collect(),
                generated_at: Utc::now(),
                priority: InsightPriority::High,
            });
        } else if avg_engagement > 0.8 {
            insights.push(LearningInsight {
                insight_type: InsightType::EngagementPattern,
                title: "High Engagement Level".to_string(),
                description: format!(
                    "Excellent engagement level of {:.1}% shows strong motivation.",
                    avg_engagement * 100.0
                ),
                confidence: 0.80,
                recommendations: vec![
                    "Maintain current approach".to_string(),
                    "Consider challenging projects".to_string(),
                ],
                supporting_metrics: [
                    ("avg_engagement".to_string(), avg_engagement),
                ].iter().cloned().collect(),
                generated_at: Utc::now(),
                priority: InsightPriority::Low,
            });
        }

        insights
    }

    /// Analyze time management patterns
    fn analyze_time_management(&self, progress_history: &[StudentProgress]) -> Vec<LearningInsight> {
        let mut insights = Vec::new();

        let session_times: Vec<u32> = progress_history.iter()
            .map(|p| p.time_spent)
            .collect();

        if session_times.is_empty() {
            return insights;
        }

        let avg_time = session_times.iter().sum::<u32>() as f64 / session_times.len() as f64;
        let max_recommended_time = self.config.max_session_duration as f64;

        if avg_time > max_recommended_time * 1.2 {
            insights.push(LearningInsight {
                insight_type: InsightType::TimeManagement,
                title: "Long Session Duration".to_string(),
                description: format!(
                    "Average session time of {:.1} minutes exceeds recommended duration.",
                    avg_time
                ),
                confidence: 0.70,
                recommendations: vec![
                    "Break sessions into shorter segments".to_string(),
                    "Take regular breaks during study".to_string(),
                    "Focus on efficiency over duration".to_string(),
                ],
                supporting_metrics: [
                    ("avg_session_time".to_string(), avg_time),
                    ("recommended_time".to_string(), max_recommended_time),
                ].iter().cloned().collect(),
                generated_at: Utc::now(),
                priority: InsightPriority::Medium,
            });
        }

        insights
    }

    /// Analyze difficulty progression patterns
    fn analyze_difficulty_progression(&self, progress_history: &[StudentProgress]) -> Vec<LearningInsight> {
        let mut insights = Vec::new();

        // Track difficulty levels over time
        let recent_difficulties: Vec<u8> = progress_history.iter()
            .rev()
            .take(10)
            .map(|p| p.difficulty_level)
            .collect();

        if recent_difficulties.len() < 5 {
            return insights;
        }

        // Check if student is stuck at same difficulty
        let mode_difficulty = self.find_mode(&recent_difficulties);
        let stuck_count = recent_difficulties.iter()
            .filter(|&&d| d == mode_difficulty)
            .count();

        if stuck_count >= 8 {
            insights.push(LearningInsight {
                insight_type: InsightType::DifficultyProgression,
                title: "Plateau at Current Difficulty".to_string(),
                description: format!(
                    "Staying at difficulty level {} for {} recent sessions may indicate readiness for advancement.",
                    mode_difficulty, stuck_count
                ),
                confidence: 0.65,
                recommendations: vec![
                    "Try slightly more challenging problems".to_string(),
                    "Review mastery of current level concepts".to_string(),
                    "Consider mixed difficulty practice".to_string(),
                ],
                supporting_metrics: [
                    ("current_difficulty".to_string(), mode_difficulty as f64),
                    ("sessions_at_level".to_string(), stuck_count as f64),
                ].iter().cloned().collect(),
                generated_at: Utc::now(),
                priority: InsightPriority::Medium,
            });
        }

        insights
    }

    /// Calculate linear trend from a series of values
    fn calculate_linear_trend(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f64;
        let x_sum: f64 = (0..values.len()).map(|i| i as f64).sum();
        let y_sum: f64 = values.iter().sum();
        let xy_sum: f64 = values.iter().enumerate()
            .map(|(i, &y)| i as f64 * y)
            .sum();
        let x_squared_sum: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();

        let slope = (n * xy_sum - x_sum * y_sum) / (n * x_squared_sum - x_sum.powi(2));
        slope
    }

    /// Find the most common value in a vector
    fn find_mode(&self, values: &[u8]) -> u8 {
        let mut counts = HashMap::new();
        for &value in values {
            *counts.entry(value).or_insert(0) += 1;
        }
        
        counts.into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(value, _)| value)
            .unwrap_or(0)
    }

    /// Create insight for insufficient data
    fn create_insufficient_data_insight(&self) -> LearningInsight {
        LearningInsight {
            insight_type: InsightType::PerformanceTrend,
            title: "More Data Needed".to_string(),
            description: "Insufficient learning data for comprehensive analysis. Continue practicing to unlock detailed insights.".to_string(),
            confidence: 1.0,
            recommendations: vec![
                "Complete a few more practice sessions".to_string(),
                "Try different types of problems".to_string(),
                "Maintain consistent learning schedule".to_string(),
            ],
            supporting_metrics: HashMap::new(),
            generated_at: Utc::now(),
            priority: InsightPriority::Info,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::education::student_model::LearningStyle;

    #[test]
    fn test_learning_analytics_creation() {
        let config = EducationConfig::default();
        let analytics = LearningAnalytics::new(&config);
        
        assert!(analytics.progress_history.is_empty());
        assert!(analytics.insights_cache.is_empty());
    }

    #[test]
    fn test_progress_recording() {
        let config = EducationConfig::default();
        let mut analytics = LearningAnalytics::new(&config);
        
        let progress = StudentProgress {
            timestamp: Utc::now(),
            topic: "algebra".to_string(),
            score: 0.85,
            time_spent: 30,
            problems_attempted: 10,
            problems_correct: 8,
            difficulty_level: 3,
            strategy_used: "guided_practice".to_string(),
            engagement_level: 0.9,
        };

        analytics.record_progress("student1", &progress);
        
        assert_eq!(analytics.progress_history.get("student1").unwrap().len(), 1);
    }

    #[test]
    fn test_trend_calculation() {
        let config = EducationConfig::default();
        let analytics = LearningAnalytics::new(&config);
        
        let increasing_trend = analytics.calculate_linear_trend(&[0.5, 0.6, 0.7, 0.8, 0.9]);
        let decreasing_trend = analytics.calculate_linear_trend(&[0.9, 0.8, 0.7, 0.6, 0.5]);
        
        assert!(increasing_trend > 0.0);
        assert!(decreasing_trend < 0.0);
    }

    #[test]
    fn test_insufficient_data_insight() {
        let config = EducationConfig::default();
        let analytics = LearningAnalytics::new(&config);
        let student_model = StudentModel::new(
            "test_student".to_string(),
            15,
            LearningStyle::Visual
        );

        let insights = analytics.analyze_student_progress(&student_model);
        
        assert_eq!(insights.len(), 1);
        assert_eq!(insights[0].title, "More Data Needed");
    }
}