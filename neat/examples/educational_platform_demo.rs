//! Educational Platform Demo
//!
//! This example demonstrates the comprehensive educational technology platform
//! built on top of the NEAT mathematical discovery system. It showcases:
//!
//! - Student registration and modeling
//! - Adaptive tutoring sessions with personalized strategies
//! - Assessment engine with difficulty calibration
//! - Personalized learning path generation
//! - Real-time learning analytics and insights
//! - Educational problem generation
//! - Step-by-step solution explanations
//!
//! Run with: `cargo run --example educational_platform_demo`

use neat::education::*;
use neat::education::student_model::{StudentProgress, LearningStyle};
use neat::error::Result;
use chrono::Utc;

fn main() -> Result<()> {
    println!("🎓 NEAT Educational Platform Demo");
    println!("==================================\n");

    // Create educational platform
    let config = EducationConfig::default();
    let mut platform = EducationalPlatform::new(config);
    println!("✅ Educational platform initialized");

    // Register diverse students with different learning styles
    register_students(&mut platform)?;

    // Demonstrate adaptive tutoring
    demonstrate_adaptive_tutoring(&mut platform)?;

    // Demonstrate assessment engine
    demonstrate_assessment_system(&mut platform)?;

    // Demonstrate learning path generation
    demonstrate_learning_paths(&mut platform)?;

    // Demonstrate problem generation
    demonstrate_problem_generation(&mut platform)?;

    // Demonstrate learning analytics
    demonstrate_learning_analytics(&mut platform)?;

    println!("\n🎯 Educational Platform Demo Complete!");
    println!("The NEAT educational platform successfully demonstrates:");
    println!("• Adaptive tutoring with personalized strategies");
    println!("• Comprehensive assessment and difficulty calibration");
    println!("• Personalized learning path generation");
    println!("• Real-time analytics and learning insights");
    println!("• Educational problem generation");
    println!("• Step-by-step solution explanations");
    
    Ok(())
}

fn register_students(platform: &mut EducationalPlatform) -> Result<()> {
    println!("👥 Registering Students");
    println!("----------------------");

    let students = vec![
        ("alice", 14, LearningStyle::Visual, "Strong visual learner"),
        ("bob", 16, LearningStyle::Kinesthetic, "Hands-on learner"),
        ("carol", 15, LearningStyle::Auditory, "Learns through listening"),
        ("david", 13, LearningStyle::ReadingWriting, "Text-based learner"),
        ("eve", 17, LearningStyle::Multimodal, "Mixed learning preferences"),
    ];

    for (name, age, style, description) in students {
        platform.register_student(name.to_string(), age, style)?;
        println!("✅ Registered {}: Age {}, {} - {}", name, age, format!("{:?}", style), description);
    }

    println!();
    Ok(())
}

fn demonstrate_adaptive_tutoring(platform: &mut EducationalPlatform) -> Result<()> {
    println!("🧠 Adaptive Tutoring System");
    println!("---------------------------");

    // Start tutoring session for Alice (Visual learner)
    let session = platform.start_tutoring_session("alice")?;
    println!("📚 Started tutoring session for Alice");
    println!("   Session ID: {}", session.session_id);
    println!("   Strategy: {:?}", session.strategy);
    println!("   Topic: {}", session.current_topic);
    println!("   Goals: {:?}", session.session_goals);

    // Start session for Bob (Kinesthetic learner) to show different strategy
    let session_bob = platform.start_tutoring_session("bob")?;
    println!("\n📚 Started tutoring session for Bob");
    println!("   Session ID: {}", session_bob.session_id);
    println!("   Strategy: {:?} (Different from Alice!)", session_bob.strategy);
    println!("   Topic: {}", session_bob.current_topic);

    println!();
    Ok(())
}

fn demonstrate_assessment_system(platform: &mut EducationalPlatform) -> Result<()> {
    println!("📊 Assessment Engine");
    println!("-------------------");

    // Conduct assessment for Carol
    let assessment = platform.conduct_assessment("carol", "basic_arithmetic")?;
    println!("📝 Assessment completed for Carol in basic_arithmetic");
    println!("   Overall Score: {:.1}%", assessment.overall_score * 100.0);
    println!("   Points: {}/{}", assessment.points_earned, assessment.total_points);
    println!("   Recommended Difficulty: {:?}", assessment.recommended_difficulty);
    
    if !assessment.strengths.is_empty() {
        println!("   Strengths: {:?}", assessment.strengths);
    }
    
    if !assessment.improvement_areas.is_empty() {
        println!("   Improvement Areas: {:?}", assessment.improvement_areas);
    }

    // Show adaptive difficulty adjustment
    let assessment_david = platform.conduct_assessment("david", "linear_equations")?;
    println!("\n📝 Assessment completed for David in linear_equations");
    println!("   Overall Score: {:.1}%", assessment_david.overall_score * 100.0);
    println!("   Recommended Difficulty: {:?}", assessment_david.recommended_difficulty);

    println!();
    Ok(())
}

fn demonstrate_learning_paths(platform: &mut EducationalPlatform) -> Result<()> {
    println!("🛤️  Personalized Learning Paths");
    println!("-------------------------------");

    // Generate learning path for Eve
    let learning_path = platform.generate_learning_path("eve")?;
    println!("🎯 Generated learning path for Eve");
    println!("   Path ID: {}", learning_path.path_id);
    println!("   Total Objectives: {}", learning_path.objectives.len());
    println!("   Estimated Time: {:.1} hours", learning_path.estimated_total_time);
    println!("   Progress: {:.1}%", learning_path.progress.completion_percentage * 100.0);

    // Show first few objectives
    println!("\n   First 3 Learning Objectives:");
    for (i, objective) in learning_path.objectives.iter().take(3).enumerate() {
        println!("   {}. {} ({})", i + 1, objective.title, format!("{:?}", objective.difficulty));
        println!("      Duration: {} min | Topic: {}", objective.estimated_duration, objective.topic);
    }

    // Get recommended next activity for Alice
    let next_activity = platform.get_next_activity("alice")?;
    println!("\n📋 Next recommended activity for Alice:");
    println!("   Activity: {}", next_activity.title);
    println!("   Description: {}", next_activity.description);
    println!("   Estimated Duration: {} minutes", next_activity.estimated_duration);

    println!();
    Ok(())
}

fn demonstrate_problem_generation(platform: &mut EducationalPlatform) -> Result<()> {
    println!("🧮 Educational Problem Generation");
    println!("--------------------------------");

    // Create problem generator
    let config = EducationConfig::default();
    let mut generator = EducationalProblemGenerator::new(&config);

    // Generate problems for different students
    let alice_model = StudentModel::new("alice".to_string(), 14, LearningStyle::Visual);
    let problem_alice = generator.generate_for_student(&alice_model, "linear_equations")?;
    
    println!("📐 Generated problem for Alice (Visual learner):");
    println!("   Type: {:?}", problem_alice.problem_type);
    println!("   Difficulty: {:?}", problem_alice.difficulty);
    println!("   Problem: {}", problem_alice.problem_statement);
    println!("   Expected Solution: {}", problem_alice.expected_solution);
    println!("   Hints: {}", problem_alice.hints.len());
    println!("   Estimated Time: {} minutes", problem_alice.estimated_time);

    // Generate different problem for Bob (Kinesthetic learner)
    let bob_model = StudentModel::new("bob".to_string(), 16, LearningStyle::Kinesthetic);
    let problem_bob = generator.generate_for_student(&bob_model, "basic_arithmetic")?;
    
    println!("\n🔢 Generated problem for Bob (Kinesthetic learner):");
    println!("   Type: {:?}", problem_bob.problem_type);
    println!("   Problem: {}", problem_bob.problem_statement);
    println!("   First Hint: {}", problem_bob.hints.get(0).unwrap_or(&"No hints".to_string()));

    // Generate pattern recognition problem
    let pattern_problem = generator.generate_problem(
        ProblemType::PatternRecognition,
        ProblemDifficulty::Medium,
        "patterns"
    )?;
    
    println!("\n🔍 Pattern Recognition Problem:");
    println!("   Problem: {}", pattern_problem.problem_statement);
    println!("   Answer: {}", pattern_problem.expected_solution);

    println!();
    Ok(())
}

fn demonstrate_learning_analytics(platform: &mut EducationalPlatform) -> Result<()> {
    println!("📈 Learning Analytics & Insights");
    println!("-------------------------------");

    // Simulate some learning progress for Alice
    let progress_data = vec![
        StudentProgress {
            timestamp: Utc::now(),
            topic: "basic_arithmetic".to_string(),
            score: 0.85,
            time_spent: 25,
            problems_attempted: 10,
            problems_correct: 8,
            difficulty_level: 2,
            strategy_used: "visual_scaffolding".to_string(),
            engagement_level: 0.9,
        },
        StudentProgress {
            timestamp: Utc::now(),
            topic: "linear_equations".to_string(),
            score: 0.75,
            time_spent: 35,
            problems_attempted: 8,
            problems_correct: 6,
            difficulty_level: 3,
            strategy_used: "visual_scaffolding".to_string(),
            engagement_level: 0.8,
        },
        StudentProgress {
            timestamp: Utc::now(),
            topic: "algebraic_expressions".to_string(),
            score: 0.90,
            time_spent: 20,
            problems_attempted: 12,
            problems_correct: 11,
            difficulty_level: 3,
            strategy_used: "guided_discovery".to_string(),
            engagement_level: 0.95,
        },
    ];

    // Update student progress
    for progress in &progress_data {
        platform.update_student_progress("alice", progress)?;
    }

    println!("📊 Updated progress data for Alice with {} sessions", progress_data.len());

    // Get analytics insights
    let insights = platform.get_student_analytics("alice")?;
    println!("\n🔍 Generated {} learning insights for Alice:", insights.len());

    for (i, insight) in insights.iter().enumerate() {
        println!("\n   Insight {}: {}", i + 1, insight.title);
        println!("   Type: {:?} | Priority: {:?}", insight.insight_type, insight.priority);
        println!("   Description: {}", insight.description);
        println!("   Confidence: {:.1}%", insight.confidence * 100.0);
        
        if !insight.recommendations.is_empty() {
            println!("   Recommendations:");
            for rec in &insight.recommendations {
                println!("   • {}", rec);
            }
        }
    }

    // Demonstrate explanation engine
    demonstrate_explanation_engine()?;

    println!();
    Ok(())
}

fn demonstrate_explanation_engine() -> Result<()> {
    println!("\n💡 Step-by-Step Explanation Engine");
    println!("----------------------------------");

    let config = EducationConfig::default();
    let mut explanation_engine = ExplanationEngine::new(&config);
    let student_model = StudentModel::new("alice".to_string(), 14, LearningStyle::Visual);

    // Create a problem to explain
    use neat::calculator::algebra::AlgebraProblem;
    let problem = AlgebraProblem::linear_equation(2.0, 3.0, 7.0); // 2x + 3 = 7

    // Generate step-by-step solution
    let solution = explanation_engine.explain_solution(&problem, &student_model)?;

    println!("📝 Step-by-step solution for: {}", solution.problem_statement);
    println!("Strategy: {}", solution.strategy_description);
    println!();

    for (i, step) in solution.steps.iter().enumerate() {
        println!("Step {}: {}", i + 1, step.title);
        println!("   Explanation: {}", step.explanation);
        println!("   Before: {} → After: {}", step.before_expression, step.after_expression);
        println!("   Reasoning: {}", step.reasoning);
        if !step.common_mistakes.is_empty() {
            println!("   ⚠️  Avoid: {}", step.common_mistakes[0]);
        }
        println!();
    }

    println!("✅ Final Answer: {}", solution.final_answer);
    println!("🔍 Verification: {}", solution.verification_method);

    // Demonstrate error analysis
    let error_analysis = explanation_engine.analyze_error("4", 2.0, &problem)?;
    println!("\n🚨 Error Analysis Example (student answered '4' instead of '2'):");
    println!("   Error: {}", error_analysis.error_description);
    println!("   Why: {}", error_analysis.error_reasoning);
    println!("   Correct Approach: {}", error_analysis.correct_approach);
    println!("   Prevention: {}", error_analysis.prevention_strategy);

    Ok(())
}