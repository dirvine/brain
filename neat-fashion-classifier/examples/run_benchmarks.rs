//! Example: Running NEAT Benchmarks
//!
//! This example demonstrates how to use the comprehensive benchmarking suite
//! to evaluate NEAT algorithm performance across different problem domains.

use neat_fashion_classifier::benchmarks::{
    BenchmarkSuite, BenchmarkConfig, run_quick_benchmark, test_runner
};
use std::time::Duration;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    println!("NEAT Benchmarking Suite Example");
    println!("================================");
    
    // Example 1: Quick benchmark validation
    println!("\n1. Running quick benchmark validation...");
    test_runner::run_validation_tests()?;
    
    // Example 2: Classic problems only
    println!("\n2. Running classic problems benchmark...");
    test_runner::run_classic_problems_only()?;
    
    // Example 3: Quick benchmark with results
    println!("\n3. Running comprehensive quick benchmark...");
    let quick_results = run_quick_benchmark()?;
    println!("Quick benchmark completed:");
    println!("  Total tests: {}", quick_results.summary.total_tests);
    println!("  Tests passed: {}", quick_results.summary.tests_passed);
    println!("  Success rate: {:.1}%", 
            (quick_results.summary.tests_passed as f64 / quick_results.summary.total_tests as f64) * 100.0);
    println!("  Total time: {:.2}s", quick_results.summary.total_time.as_secs_f64());
    
    // Example 4: Custom benchmark configuration
    println!("\n4. Running custom benchmark configuration...");
    let config = BenchmarkConfig {
        runs_per_test: 2,
        test_timeout: Duration::from_secs(60),
        include_expensive: false, // Set to true for full performance tests
        random_seed: Some(12345),
        output_dir: Some(PathBuf::from("benchmark_results")),
    };
    
    let mut suite = BenchmarkSuite::new(config);
    let results = suite.run_all()?;
    
    // Print detailed summary
    suite.print_summary(&results);
    
    println!("\nBenchmark suite completed successfully!");
    println!("Results saved to: benchmark_results/");
    
    Ok(())
}