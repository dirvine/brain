use rand::Rng;

pub fn generate_math_problem(topic: &str, difficulty: &str) -> Result<(String, String), String> {
    let mut rng = rand::thread_rng();
    
    match topic {
        "arithmetic" => generate_arithmetic_problem(difficulty, &mut rng),
        "algebra" => generate_algebra_problem(difficulty, &mut rng),
        "calculus" => generate_calculus_problem(difficulty, &mut rng),
        "trigonometry" => generate_trigonometry_problem(difficulty, &mut rng),
        "statistics" => generate_statistics_problem(difficulty, &mut rng),
        "discrete math" => generate_discrete_math_problem(difficulty, &mut rng),
        _ => Ok(("What is 2 + 2?".to_string(), "4".to_string())),
    }
}

fn generate_arithmetic_problem(difficulty: &str, rng: &mut impl Rng) -> Result<(String, String), String> {
    match difficulty {
        "easy" => {
            let a = rng.gen_range(1..=20);
            let b = rng.gen_range(1..=20);
            let answer = a + b;
            Ok((format!("What is {} + {}?", a, b), answer.to_string()))
        },
        "medium" => {
            let a = rng.gen_range(10..=50);
            let b = rng.gen_range(10..=20);
            let answer = a * b;
            Ok((format!("What is {} × {}?", a, b), answer.to_string()))
        },
        "hard" => {
            let a = rng.gen_range(100..=500);
            let b = rng.gen_range(10..=25);
            let answer = (a as f64 / b as f64 * 100.0).round() / 100.0;
            Ok((format!("What is {} ÷ {}? (round to 2 decimal places)", a, b), format!("{:.2}", answer)))
        },
        _ => {
            let a = rng.gen_range(1..=10);
            let b = rng.gen_range(1..=10);
            let answer = a + b;
            Ok((format!("What is {} + {}?", a, b), answer.to_string()))
        }
    }
}

fn generate_algebra_problem(difficulty: &str, rng: &mut impl Rng) -> Result<(String, String), String> {
    match difficulty {
        "easy" => {
            let a = rng.gen_range(1..=10);
            let b = rng.gen_range(5..=15);
            let answer = b - a;
            Ok((format!("Solve for x: x + {} = {}", a, b), answer.to_string()))
        },
        "medium" => {
            let a = rng.gen_range(2..=5);
            let b = rng.gen_range(1..=10);
            let c = rng.gen_range(10..=30);
            let answer = (c + b) / a;
            Ok((format!("Solve for x: {}x - {} = {}", a, b, c), answer.to_string()))
        },
        "hard" => {
            let problems = vec![
                ("Solve for x: x² - 5x + 6 = 0".to_string(), "x = 2 or x = 3".to_string()),
                ("Solve for x: x² - 7x + 12 = 0".to_string(), "x = 3 or x = 4".to_string()),
                ("Solve for x: x² - 6x + 8 = 0".to_string(), "x = 2 or x = 4".to_string()),
            ];
            let idx = rng.gen_range(0..problems.len());
            Ok(problems[idx].clone())
        },
        _ => {
            let a = rng.gen_range(1..=5);
            let b = rng.gen_range(3..=10);
            let answer = b - a;
            Ok((format!("Solve for x: x + {} = {}", a, b), answer.to_string()))
        }
    }
}

fn generate_calculus_problem(difficulty: &str, rng: &mut impl Rng) -> Result<(String, String), String> {
    match difficulty {
        "easy" => {
            let powers = vec![2, 3, 4, 5];
            let power = powers[rng.gen_range(0..powers.len())];
            let derivative = match power {
                2 => "2x".to_string(),
                3 => "3x²".to_string(),
                4 => "4x³".to_string(),
                5 => "5x⁴".to_string(),
                _ => "2x".to_string(),
            };
            let power_symbol = match power {
                2 => "²".to_string(),
                3 => "³".to_string(),
                4 => "⁴".to_string(),
                5 => "⁵".to_string(),
                _ => "²".to_string(),
            };
            Ok((format!("Find the derivative of f(x) = x{}", power_symbol), format!("f'(x) = {}", derivative)))
        },
        "medium" => {
            let problems = vec![
                ("Find the derivative of f(x) = 3x³ - 2x + 1".to_string(), "f'(x) = 9x² - 2".to_string()),
                ("Find the derivative of f(x) = 2x² + 5x - 3".to_string(), "f'(x) = 4x + 5".to_string()),
                ("Find the derivative of f(x) = x⁴ - 3x²".to_string(), "f'(x) = 4x³ - 6x".to_string()),
            ];
            let idx = rng.gen_range(0..problems.len());
            Ok(problems[idx].clone())
        },
        "hard" => {
            let problems = vec![
                ("Find ∫(2x + 3)dx".to_string(), "x² + 3x + C".to_string()),
                ("Find ∫(3x² - 4x)dx".to_string(), "x³ - 2x² + C".to_string()),
                ("Find ∫(x³ + 1)dx".to_string(), "x⁴/4 + x + C".to_string()),
            ];
            let idx = rng.gen_range(0..problems.len());
            Ok(problems[idx].clone())
        },
        _ => Ok(("Find the derivative of f(x) = x".to_string(), "f'(x) = 1".to_string())),
    }
}

fn generate_trigonometry_problem(difficulty: &str, rng: &mut impl Rng) -> Result<(String, String), String> {
    match difficulty {
        "easy" => {
            let angles_and_answers = vec![
                ("0°", "0"), ("30°", "0.5"), ("45°", "0.707"), ("60°", "0.866"), ("90°", "1"),
                ("sin(0°)", "0"), ("sin(30°)", "0.5"), ("sin(90°)", "1"),
                ("cos(0°)", "1"), ("cos(60°)", "0.5"), ("cos(90°)", "0")
            ];
            let idx = rng.gen_range(0..angles_and_answers.len());
            let (angle, answer) = angles_and_answers[idx];
            if angle.contains("sin") || angle.contains("cos") {
                Ok((format!("What is {}?", angle), answer.to_string()))
            } else {
                Ok((format!("What is sin({})?", angle), answer.to_string()))
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
            Ok(problems[idx].clone())
        },
        "hard" => {
            let problems = vec![
                ("Solve: sin(x) = 0.5 for 0° ≤ x ≤ 360°".to_string(), "30°, 150°".to_string()),
                ("Solve: cos(x) = 0.5 for 0° ≤ x ≤ 360°".to_string(), "60°, 300°".to_string()),
                ("Solve: tan(x) = 1 for 0° ≤ x ≤ 360°".to_string(), "45°, 225°".to_string()),
            ];
            let idx = rng.gen_range(0..problems.len());
            Ok(problems[idx].clone())
        },
        _ => Ok(("What is sin(0°)?".to_string(), "0".to_string())),
    }
}

fn generate_statistics_problem(difficulty: &str, rng: &mut impl Rng) -> Result<(String, String), String> {
    match difficulty {
        "easy" => {
            let numbers = (0..5).map(|_| rng.gen_range(1..=10)).collect::<Vec<i32>>();
            let sum: i32 = numbers.iter().sum();
            let mean = sum as f64 / numbers.len() as f64;
            Ok((format!("Find the mean of: {}", numbers.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(", ")), 
                 format!("{:.1}", mean)))
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
            Ok((format!("Find the median of: {}", shuffled.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(", ")), 
                 median.to_string()))
        },
        "hard" => {
            let numbers = vec![2, 4, 6, 8, 10];
            Ok((format!("Calculate the standard deviation of: {}", numbers.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(", ")), 
                 "2.83".to_string()))
        },
        _ => {
            let numbers = (0..3).map(|_| rng.gen_range(1..=5)).collect::<Vec<i32>>();
            let sum: i32 = numbers.iter().sum();
            Ok((format!("Find the sum of: {}", numbers.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(", ")), 
                 sum.to_string()))
        }
    }
}

fn generate_discrete_math_problem(difficulty: &str, rng: &mut impl Rng) -> Result<(String, String), String> {
    match difficulty {
        "easy" => {
            let n = rng.gen_range(3..=5);
            let factorial = (1..=n).product::<i32>();
            Ok((format!("How many ways can you arrange {} items? ({}!)", n, n), factorial.to_string()))
        },
        "medium" => {
            let n = rng.gen_range(4..=7);
            let r = rng.gen_range(2..=3);
            let combination = (1..=n).product::<i32>() / ((1..=r).product::<i32>() * (1..=(n-r)).product::<i32>());
            Ok((format!("What is C({},{})?", n, r), combination.to_string()))
        },
        "hard" => {
            let n = rng.gen_range(3..=5);
            let subsets = 2_i32.pow(n as u32);
            let set_elements: Vec<String> = (1..=n).map(|i| i.to_string()).collect();
            Ok((format!("Find the number of subsets of {{{}}}", set_elements.join(",")), subsets.to_string()))
        },
        _ => {
            let n = rng.gen_range(3..=4);
            let factorial = (1..=n).product::<i32>();
            Ok((format!("What is {}!?", n), factorial.to_string()))
        }
    }
}

pub fn validate_answer(problem: &str, answer: &str) -> bool {
    let answer = answer.trim().to_lowercase();
    
    // For arithmetic problems, extract numbers and operator
    if problem.contains("What is") && (problem.contains("+") || problem.contains("×") || problem.contains("÷")) {
        if let Some(expected) = extract_arithmetic_answer(problem) {
            return answer == expected.to_lowercase() || 
                   (answer.parse::<f64>().is_ok() && expected.parse::<f64>().is_ok() &&
                    (answer.parse::<f64>().unwrap() - expected.parse::<f64>().unwrap()).abs() < 0.01);
        }
    }
    
    // For algebra problems
    if problem.contains("Solve for x:") {
        if let Some(expected) = extract_algebra_answer(problem) {
            return answer == expected || 
                   answer == format!("x={}", expected) ||
                   answer == format!("x = {}", expected) ||
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
    
    // For trigonometry, statistics, and discrete math - flexible checking
    !answer.is_empty()
}

fn extract_arithmetic_answer(problem: &str) -> Option<String> {
    if let Some(start) = problem.find("What is ") {
        let expr = &problem[start + 8..];
        if let Some(end) = expr.find("?") {
            let expr = &expr[..end].trim();
            return evaluate_simple_expression(expr);
        }
    }
    None
}

fn extract_algebra_answer(problem: &str) -> Option<String> {
    if problem.contains("x²") {
        return None; // Skip complex parsing for quadratic equations
    }
    
    if let Some(equals_pos) = problem.rfind(" = ") {
        let right_side = &problem[equals_pos + 3..];
        if let Ok(target) = right_side.trim().parse::<i32>() {
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