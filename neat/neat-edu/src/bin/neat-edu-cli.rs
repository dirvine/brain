use std::io::{self, Write};
use neat_edu::problem_generator::generate_math_problem;

fn main() {
    println!("🧠 NEAT Educational Platform - CLI Mode");
    println!("═══════════════════════════════════════");
    println!("Interactive mathematical learning");
    println!();
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