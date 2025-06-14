// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod frontend;

use serde::{Deserialize, Serialize};
use neat_edu::{generate_math_problem, validate_answer, create_network_visualization, NetworkVisualization};

#[derive(Debug, Serialize, Deserialize)]
pub struct ProblemRequest {
    pub topic: String,
    pub difficulty: String,
    pub problem_type: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProblemResponse {
    pub problem_text: String,
    pub expected_answer: String,
    pub explanation: String,
    pub hints: Vec<String>,
    pub network_data: NetworkVisualization,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SolutionRequest {
    pub problem: String,
    pub student_answer: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SolutionResponse {
    pub answer: String,
    pub explanation: String,
    pub steps: Vec<String>,
    pub network_data: NetworkVisualization,
    pub correct: bool,
}

// Initialize educational platform
#[tauri::command]
async fn initialize_educational_platform() -> Result<String, String> {
    Ok("NEAT Educational Platform initialized successfully".to_string())
}

// Generate a mathematical problem
#[tauri::command]
async fn generate_problem(request: ProblemRequest) -> Result<ProblemResponse, String> {
    let (problem_text, expected_answer) = generate_math_problem(&request.topic, &request.difficulty)?;
    let network_data = create_network_visualization(&request.topic);
    
    let response = ProblemResponse {
        problem_text,
        expected_answer,
        explanation: format!("This {} problem demonstrates how NEAT neural networks can reason about mathematical concepts", request.topic),
        hints: vec![
            "Think step by step".to_string(),
            "Apply the basic mathematical rules you know".to_string(),
            "Check your calculation carefully".to_string(),
        ],
        network_data,
    };
    
    Ok(response)
}

// Solve a mathematical problem
#[tauri::command]
async fn solve_problem(request: SolutionRequest) -> Result<SolutionResponse, String> {
    let is_correct = validate_answer(&request.problem, &request.student_answer);
    let network_data = create_network_visualization("general");
    
    let response = SolutionResponse {
        answer: request.student_answer.clone(),
        explanation: if is_correct {
            "Correct! Great job solving this problem. The neural network processed your solution and confirmed it matches the expected answer.".to_string()
        } else {
            "Not quite right. Let's work through this step by step. The neural network can help guide you to the correct solution.".to_string()
        },
        steps: vec![
            "Neural network analyzes the input problem".to_string(),
            "Mathematical reasoning patterns are applied".to_string(),
            "Solution confidence is calculated".to_string(),
            "Feedback is generated based on correctness".to_string(),
        ],
        network_data,
        correct: is_correct,
    };
    
    Ok(response)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            initialize_educational_platform,
            generate_problem,
            solve_problem
        ])
        .run(tauri::generate_context!())
        .expect("error while running NEAT Educational Platform");
}

fn main() {
    // Print welcome message for CLI usage
    println!("🧠 NEAT Educational Platform");
    println!("═══════════════════════════════════════");
    println!("Interactive mathematical learning with neural network visualization");
    println!();
    
    // Check if we should run in CLI mode or GUI mode
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() > 1 && args[1] == "--cli" {
        run_cli_mode();
    } else {
        println!("Starting GUI application...");
        run();
    }
}

fn run_cli_mode() {
    use std::io::{self, Write};
    use neat_edu::problem_generator::generate_math_problem;
    
    println!("CLI Mode - Interactive Problem Solver");
    println!("Available topics: arithmetic, algebra, calculus, trigonometry, statistics, discrete math");
    println!("Available difficulties: easy, medium, hard");
    println!("Type 'quit' to exit");
    println!();
    
    loop {
        print!("Enter topic (or 'quit'): ");
        io::stdout().flush().unwrap();
        
        let mut topic = String::new();
        io::stdin().read_line(&mut topic).unwrap();
        let topic = topic.trim();
        
        if topic == "quit" {
            println!("Goodbye!");
            break;
        }
        
        print!("Enter difficulty (easy/medium/hard): ");
        io::stdout().flush().unwrap();
        
        let mut difficulty = String::new();
        io::stdin().read_line(&mut difficulty).unwrap();
        let difficulty = difficulty.trim();
        
        match generate_math_problem(topic, difficulty) {
            Ok((problem, answer)) => {
                println!("\nProblem: {}", problem);
                print!("Your answer: ");
                io::stdout().flush().unwrap();
                
                let mut user_answer = String::new();
                io::stdin().read_line(&mut user_answer).unwrap();
                let user_answer = user_answer.trim();
                
                if user_answer == answer {
                    println!("✅ Correct! Well done!\n");
                } else {
                    println!("❌ Incorrect. The correct answer is: {}\n", answer);
                }
            }
            Err(e) => {
                println!("Error generating problem: {}\n", e);
            }
        }
    }
}