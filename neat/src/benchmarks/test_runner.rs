//! Simple test runner for the benchmarking suite
//!
//! This module provides a simple way to run and test the benchmarking
//! framework without running the full expensive benchmark suite.

use crate::benchmarks::{BenchmarkSuite, BenchmarkConfig, run_quick_benchmark};
use crate::error::Result;
use std::time::Duration;

/// Run a quick validation of the benchmark framework
pub fn run_validation_tests() -> Result<()> {
    println!("Running benchmark framework validation...");
    
    // Test quick benchmark
    let config = BenchmarkConfig {
        runs_per_test: 1,
        test_timeout: Duration::from_secs(30),
        include_expensive: false,
        random_seed: Some(42),
        output_dir: None,
    };
    
    let mut suite = BenchmarkSuite::new(config);
    
    println!("Testing basic benchmark execution...");
    let results = suite.run_all()?;
    
    println!("Benchmark validation completed successfully!");
    println!("Total tests run: {}", results.test_results.len());
    println!("Tests passed: {}", results.test_results.iter().filter(|t| t.passed).count());
    
    suite.print_summary(&results);
    
    Ok(())
}

/// Run only the classic problems benchmarks for quick testing
pub fn run_classic_problems_only() -> Result<()> {
    use crate::benchmarks::classic_problems;
    
    println!("Running classic problems benchmark validation...");
    
    let config = BenchmarkConfig {
        runs_per_test: 1,
        test_timeout: Duration::from_secs(60),
        include_expensive: false,
        random_seed: Some(42),
        output_dir: None,
    };
    
    let results = classic_problems::run_all_tests(&config)?;
    
    println!("Classic problems validation completed!");
    println!("Problems tested: {}", results.len());
    
    for result in &results {
        let status = if result.passed { "✓ PASS" } else { "✗ FAIL" };
        println!("  {} {}: {:.3} fitness in {:.2}s", 
                status, result.name, result.metrics.final_fitness, 
                result.metrics.execution_time.as_secs_f64());
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_benchmark_framework_basic() -> Result<()> {
        // Very basic test to ensure the framework can be instantiated
        let config = BenchmarkConfig {
            runs_per_test: 1,
            test_timeout: Duration::from_secs(10),
            include_expensive: false,
            random_seed: Some(42),
            output_dir: None,
        };
        
        let _suite = BenchmarkSuite::new(config);
        Ok(())
    }
    
    #[test]
    fn test_quick_benchmark() -> Result<()> {
        // Test the convenience quick benchmark function
        let results = run_quick_benchmark()?;
        
        // Should have some results
        assert!(!results.test_results.is_empty());
        
        // Should have timing information
        assert!(results.summary.total_time > Duration::default());
        
        Ok(())
    }
}