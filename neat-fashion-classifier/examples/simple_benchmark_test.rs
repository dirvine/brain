//! Simple benchmark test to verify the framework works

use neat_fashion_classifier::benchmarks::{BenchmarkConfig, run_quick_benchmark};
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing NEAT Benchmarking Suite...");
    
    // Run a very quick benchmark
    let results = run_quick_benchmark()?;
    
    println!("✅ Benchmark completed successfully!");
    println!("Total tests: {}", results.summary.total_tests);
    println!("Tests passed: {}", results.summary.tests_passed);
    println!("Total time: {:.2}s", results.summary.total_time.as_secs_f64());
    
    // Show basic result details
    for result in &results.test_results {
        let status = if result.passed { "✓" } else { "✗" };
        println!("  {} {} - {:.3} fitness", status, result.name, result.metrics.final_fitness);
    }
    
    Ok(())
}