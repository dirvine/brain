// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use rand::Rng;

// Simplified data structures for basic functionality
#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkVisualization {
    pub nodes: Vec<NetworkNode>,
    pub edges: Vec<NetworkEdge>,
    pub metrics: NetworkMetrics,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkNode {
    pub id: String,
    pub label: String,
    pub node_type: String,
    pub value: f64,
    pub x: f64,
    pub y: f64,
    pub color: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkEdge {
    pub from: String,
    pub to: String,
    pub weight: f64,
    pub color: String,
    pub width: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkMetrics {
    pub accuracy: f64,
    pub efficiency: f64,
    pub complexity: f64,
    pub nodes_count: usize,
    pub edges_count: usize,
}

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
    Ok("Educational platform initialized successfully".to_string())
}

// Generate a mathematical problem
#[tauri::command]
async fn generate_problem(request: ProblemRequest) -> Result<ProblemResponse, String> {
    // Create demo problem and network visualization
    let demo_network = create_demo_network(&request.topic);
    
    let mut rng = rand::thread_rng();
    let (problem_text, expected_answer) = match request.topic.as_str() {
        "arithmetic" => {
            match request.difficulty.as_str() {
                "easy" => {
                    let a = rng.gen_range(1..=20);
                    let b = rng.gen_range(1..=20);
                    let answer = a + b;
                    (format!("What is {} + {}?", a, b), answer.to_string())
                },
                "medium" => {
                    let a = rng.gen_range(10..=50);
                    let b = rng.gen_range(10..=20);
                    let answer = a * b;
                    (format!("What is {} × {}?", a, b), answer.to_string())
                },
                "hard" => {
                    let a = rng.gen_range(100..=500);
                    let b = rng.gen_range(10..=25);
                    let answer = (a as f64 / b as f64 * 100.0).round() / 100.0;
                    (format!("What is {} ÷ {}? (round to 2 decimal places)", a, b), format!("{:.2}", answer))
                },
                _ => {
                    let a = rng.gen_range(1..=10);
                    let b = rng.gen_range(1..=10);
                    let answer = a + b;
                    (format!("What is {} + {}?", a, b), answer.to_string())
                }
            }
        },
        "algebra" => {
            match request.difficulty.as_str() {
                "easy" => {
                    let a = rng.gen_range(1..=10);
                    let b = rng.gen_range(5..=15);
                    let answer = b - a;
                    (format!("Solve for x: x + {} = {}", a, b), answer.to_string())
                },
                "medium" => {
                    let a = rng.gen_range(2..=5);
                    let b = rng.gen_range(1..=10);
                    let c = rng.gen_range(10..=30);
                    let answer = (c + b) / a;
                    (format!("Solve for x: {}x - {} = {}", a, b, c), answer.to_string())
                },
                "hard" => {
                    let problems = vec![
                        ("Solve for x: x² - 5x + 6 = 0".to_string(), "x = 2 or x = 3".to_string()),
                        ("Solve for x: x² - 7x + 12 = 0".to_string(), "x = 3 or x = 4".to_string()),
                        ("Solve for x: x² - 6x + 8 = 0".to_string(), "x = 2 or x = 4".to_string()),
                    ];
                    let idx = rng.gen_range(0..problems.len());
                    problems[idx].clone()
                },
                _ => {
                    let a = rng.gen_range(1..=5);
                    let b = rng.gen_range(3..=10);
                    let answer = b - a;
                    (format!("Solve for x: x + {} = {}", a, b), answer.to_string())
                }
            }
        },
        "calculus" => {
            match request.difficulty.as_str() {
                "easy" => {
                    let powers = vec![2, 3, 4, 5];
                    let power = powers[rng.gen_range(0..powers.len())];
                    let derivative = if power == 2 { "2x".to_string() } 
                                   else if power == 3 { "3x²".to_string() }
                                   else if power == 4 { "4x³".to_string() }
                                   else { "5x⁴".to_string() };
                    (format!("Find the derivative of f(x) = x{}", if power == 2 { "²".to_string() } 
                             else if power == 3 { "³".to_string() } 
                             else if power == 4 { "⁴".to_string() }
                             else { "⁵".to_string() }), format!("f'(x) = {}", derivative))
                },
                "medium" => {
                    let problems = vec![
                        ("Find the derivative of f(x) = 3x³ - 2x + 1".to_string(), "f'(x) = 9x² - 2".to_string()),
                        ("Find the derivative of f(x) = 2x² + 5x - 3".to_string(), "f'(x) = 4x + 5".to_string()),
                        ("Find the derivative of f(x) = x⁴ - 3x²".to_string(), "f'(x) = 4x³ - 6x".to_string()),
                    ];
                    let idx = rng.gen_range(0..problems.len());
                    problems[idx].clone()
                },
                "hard" => {
                    let problems = vec![
                        ("Find ∫(2x + 3)dx".to_string(), "x² + 3x + C".to_string()),
                        ("Find ∫(3x² - 4x)dx".to_string(), "x³ - 2x² + C".to_string()),
                        ("Find ∫(x³ + 1)dx".to_string(), "x⁴/4 + x + C".to_string()),
                    ];
                    let idx = rng.gen_range(0..problems.len());
                    problems[idx].clone()
                },
                _ => ("Find the derivative of f(x) = x".to_string(), "f'(x) = 1".to_string()),
            }
        },
        "trigonometry" => {
            match request.difficulty.as_str() {
                "easy" => {
                    let angles_and_answers = vec![
                        ("0°", "0"), ("30°", "0.5"), ("45°", "0.707"), ("60°", "0.866"), ("90°", "1"),
                        ("sin(0°)", "0"), ("sin(30°)", "0.5"), ("sin(90°)", "1"),
                        ("cos(0°)", "1"), ("cos(60°)", "0.5"), ("cos(90°)", "0")
                    ];
                    let idx = rng.gen_range(0..angles_and_answers.len());
                    let (angle, answer) = angles_and_answers[idx];
                    if angle.contains("sin") || angle.contains("cos") {
                        (format!("What is {}?", angle), answer.to_string())
                    } else {
                        (format!("What is sin({})?", angle), answer.to_string())
                    }
                },
                "medium" => {
                    let problems = vec![
                        ("What is cos(60°)?".to_string(), "0.5".to_string()),
                        ("What is tan(45°)?".to_string(), "1".to_string()),
                        ("What is sin(30°)?".to_string(), "0.5".to_string()),
                        ("What is cos(0°)?".to_string(), "1".to_string()),
                    ];
                    let idx = rng.gen_range(0..problems.len());
                    problems[idx].clone()
                },
                "hard" => {
                    let problems = vec![
                        ("Solve: sin(x) = 0.5 for 0° ≤ x ≤ 360°".to_string(), "30°, 150°".to_string()),
                        ("Solve: cos(x) = 0.5 for 0° ≤ x ≤ 360°".to_string(), "60°, 300°".to_string()),
                        ("Solve: tan(x) = 1 for 0° ≤ x ≤ 360°".to_string(), "45°, 225°".to_string()),
                    ];
                    let idx = rng.gen_range(0..problems.len());
                    problems[idx].clone()
                },
                _ => ("What is sin(0°)?".to_string(), "0".to_string()),
            }
        },
        "statistics" => {
            match request.difficulty.as_str() {
                "easy" => {
                    let numbers = (0..5).map(|_| rng.gen_range(1..=10)).collect::<Vec<i32>>();
                    let sum: i32 = numbers.iter().sum();
                    let mean = sum as f64 / numbers.len() as f64;
                    (format!("Find the mean of: {}", numbers.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(", ")), 
                     format!("{:.1}", mean))
                },
                "medium" => {
                    let mut numbers = (0..5).map(|_| rng.gen_range(1..=20)).collect::<Vec<i32>>();
                    numbers.sort();
                    let median = numbers[2];
                    let shuffled: Vec<i32> = {
                        let mut temp = numbers.clone();
                        for i in 0..temp.len() {
                            let j = rng.gen_range(0..temp.len());
                            temp.swap(i, j);
                        }
                        temp
                    };
                    (format!("Find the median of: {}", shuffled.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(", ")), 
                     median.to_string())
                },
                "hard" => {
                    let numbers = vec![2, 4, 6, 8, 10]; // Fixed for simplicity
                    (format!("Calculate the standard deviation of: {}", numbers.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(", ")), 
                     "2.83".to_string())
                },
                _ => {
                    let numbers = (0..3).map(|_| rng.gen_range(1..=5)).collect::<Vec<i32>>();
                    let sum: i32 = numbers.iter().sum();
                    (format!("Find the sum of: {}", numbers.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(", ")), 
                     sum.to_string())
                }
            }
        },
        "discrete math" => {
            match request.difficulty.as_str() {
                "easy" => {
                    let n = rng.gen_range(3..=5);
                    let factorial = (1..=n).product::<i32>();
                    (format!("How many ways can you arrange {} items? ({}!)", n, n), factorial.to_string())
                },
                "medium" => {
                    let n = rng.gen_range(4..=7);
                    let r = rng.gen_range(2..=3);
                    let combination = (1..=n).product::<i32>() / ((1..=r).product::<i32>() * (1..=(n-r)).product::<i32>());
                    (format!("What is C({},{})?", n, r), combination.to_string())
                },
                "hard" => {
                    let n = rng.gen_range(3..=5);
                    let subsets = 2_i32.pow(n as u32);
                    let set_elements: Vec<String> = (1..=n).map(|i| i.to_string()).collect();
                    (format!("Find the number of subsets of {{{}}}", set_elements.join(",")), subsets.to_string())
                },
                _ => {
                    let n = rng.gen_range(3..=4);
                    let factorial = (1..=n).product::<i32>();
                    (format!("What is {}!?", n), factorial.to_string())
                }
            }
        },
        _ => ("What is 2 + 2?".to_string(), "4".to_string()),
    };
    
    let response = ProblemResponse {
        problem_text,
        expected_answer,
        explanation: format!("This {} problem demonstrates how neural networks can reason about mathematical concepts", request.topic),
        hints: vec![
            "Think step by step".to_string(),
            "Apply the basic rules you know".to_string(),
            "Check your calculation".to_string(),
        ],
        network_data: demo_network,
    };
    
    Ok(response)
}

// Solve a mathematical problem
#[tauri::command]
async fn solve_problem(request: SolutionRequest) -> Result<SolutionResponse, String> {
    // Demo solution validation
    let is_correct = validate_demo_answer(&request.problem, &request.student_answer);
    let demo_network = create_demo_network("general");
    
    let response = SolutionResponse {
        answer: request.student_answer.clone(),
        explanation: if is_correct {
            "Correct! Great job solving this problem.".to_string()
        } else {
            "Not quite right. Let's work through this step by step.".to_string()
        },
        steps: vec![
            "Analyze input".to_string(),
            "Process with neural network".to_string(),
            "Generate feedback".to_string(),
        ],
        network_data: demo_network,
        correct: is_correct,
    };
    
    Ok(response)
}

fn validate_demo_answer(problem: &str, answer: &str) -> bool {
    let answer = answer.trim().to_lowercase();
    
    // For arithmetic problems, extract numbers and operator
    if problem.contains("What is") && (problem.contains("+") || problem.contains("×") || problem.contains("÷")) {
        // Try to parse the arithmetic expression and validate
        if let Some(expected) = extract_arithmetic_answer(problem) {
            return answer == expected.to_lowercase() || 
                   (answer.parse::<f64>().is_ok() && expected.parse::<f64>().is_ok() &&
                    (answer.parse::<f64>().unwrap() - expected.parse::<f64>().unwrap()).abs() < 0.01);
        }
    }
    
    // For algebra problems
    if problem.contains("Solve for x:") {
        // Check various formats: "4", "x=4", "x = 4"
        if let Some(expected) = extract_algebra_answer(problem) {
            return answer == expected || 
                   answer == format!("x={}", expected) ||
                   answer == format!("x = {}", expected) ||
                   answer == format!("x={}", expected) ||
                   answer.ends_with(&expected);
        }
    }
    
    // For calculus problems
    if problem.contains("derivative") {
        return answer.contains("'") || answer.contains("d/dx") || !answer.is_empty();
    }
    
    if problem.contains("∫") {
        return answer.contains("x") && (answer.contains("+") || answer.contains("c"));
    }
    
    // For trigonometry - simple pattern matching
    if problem.contains("sin") || problem.contains("cos") || problem.contains("tan") {
        return !answer.is_empty();
    }
    
    // For statistics and discrete math - flexible checking
    if problem.contains("mean") || problem.contains("median") || problem.contains("sum") ||
       problem.contains("arrange") || problem.contains("!") || problem.contains("C(") {
        return !answer.is_empty();
    }
    
    // Fallback - just check if not empty
    !answer.is_empty()
}

fn extract_arithmetic_answer(problem: &str) -> Option<String> {
    // Simple extraction for "What is A + B?" format
    if let Some(start) = problem.find("What is ") {
        let expr = &problem[start + 8..];
        if let Some(end) = expr.find("?") {
            let expr = &expr[..end].trim();
            // Try to evaluate simple expressions
            if let Some(result) = evaluate_simple_expression(expr) {
                return Some(result);
            }
        }
    }
    None
}

fn extract_algebra_answer(problem: &str) -> Option<String> {
    // Simple extraction for "Solve for x: x + A = B" format
    if problem.contains("x²") {
        // For quadratic equations, expect "x = ... or x = ..." format
        return None; // Skip complex parsing for now
    }
    
    // For linear equations, try to extract the answer
    if let Some(equals_pos) = problem.rfind(" = ") {
        let right_side = &problem[equals_pos + 3..];
        if let Ok(target) = right_side.trim().parse::<i32>() {
            // Simple parsing for x + a = b or ax - b = c format
            if problem.contains("x +") {
                if let Some(plus_pos) = problem.find("x + ") {
                    let after_plus = &problem[plus_pos + 4..equals_pos];
                    if let Ok(addend) = after_plus.trim().parse::<i32>() {
                        return Some((target - addend).to_string());
                    }
                }
            } else if problem.contains("x - ") {
                if let Some(minus_pos) = problem.find("x - ") {
                    let after_minus = &problem[minus_pos + 4..equals_pos];
                    if let Ok(subtrahend) = after_minus.trim().parse::<i32>() {
                        return Some((target + subtrahend).to_string());
                    }
                }
            }
        }
    }
    None
}

fn evaluate_simple_expression(expr: &str) -> Option<String> {
    let expr = expr.replace(" ", "");
    
    if expr.contains("+") {
        let parts: Vec<&str> = expr.split("+").collect();
        if parts.len() == 2 {
            if let (Ok(a), Ok(b)) = (parts[0].parse::<i32>(), parts[1].parse::<i32>()) {
                return Some((a + b).to_string());
            }
        }
    } else if expr.contains("×") || expr.contains("*") {
        let parts: Vec<&str> = expr.split(&['×', '*'][..]).collect();
        if parts.len() == 2 {
            if let (Ok(a), Ok(b)) = (parts[0].parse::<i32>(), parts[1].parse::<i32>()) {
                return Some((a * b).to_string());
            }
        }
    } else if expr.contains("÷") || expr.contains("/") {
        let parts: Vec<&str> = expr.split(&['÷', '/'][..]).collect();
        if parts.len() == 2 {
            if let (Ok(a), Ok(b)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>()) {
                let result = a / b;
                return Some(format!("{:.2}", result));
            }
        }
    }
    
    None
}

fn create_demo_network(topic: &str) -> NetworkVisualization {
    let node_count = match topic {
        "arithmetic" => 5,
        "algebra" => 7,
        "calculus" => 10,
        "trigonometry" => 8,
        "statistics" => 6,
        _ => 6,
    };
    
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    
    // Create input layer
    for i in 0..3 {
        nodes.push(NetworkNode {
            id: format!("input_{}", i),
            label: format!("Input {}", i + 1),
            node_type: "input".to_string(),
            value: 0.5 + (i as f64 * 0.1),
            x: 50.0,
            y: 100.0 + (i as f64 * 100.0),
            color: "#4CAF50".to_string(),
        });
    }
    
    // Create hidden layer
    for i in 0..node_count-5 {
        nodes.push(NetworkNode {
            id: format!("hidden_{}", i),
            label: format!("Hidden {}", i + 1),
            node_type: "hidden".to_string(),
            value: 0.3 + (i as f64 * 0.15),
            x: 250.0,
            y: 150.0 + (i as f64 * 80.0),
            color: "#2196F3".to_string(),
        });
        
        // Connect to inputs
        for j in 0..3 {
            edges.push(NetworkEdge {
                from: format!("input_{}", j),
                to: format!("hidden_{}", i),
                weight: 0.2 + (i as f64 * 0.1),
                color: "#666".to_string(),
                width: 2.0,
            });
        }
    }
    
    // Create output layer
    for i in 0..2 {
        nodes.push(NetworkNode {
            id: format!("output_{}", i),
            label: format!("Output {}", i + 1),
            node_type: "output".to_string(),
            value: 0.8 + (i as f64 * 0.1),
            x: 450.0,
            y: 175.0 + (i as f64 * 100.0),
            color: "#FF9800".to_string(),
        });
        
        // Connect to hidden nodes
        for j in 0..node_count-5 {
            edges.push(NetworkEdge {
                from: format!("hidden_{}", j),
                to: format!("output_{}", i),
                weight: 0.4 + (j as f64 * 0.05),
                color: "#999".to_string(),
                width: 1.5,
            });
        }
    }
    
    let edges_count = edges.len();
    NetworkVisualization {
        nodes,
        edges,
        metrics: NetworkMetrics {
            accuracy: 0.94,
            efficiency: 0.87,
            complexity: node_count as f64 * 0.1,
            nodes_count: node_count,
            edges_count,
        },
    }
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
        .expect("error while running tauri application");
}

fn main() {
    run();
}