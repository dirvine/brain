//! Comprehensive benchmarking suite for NEAT implementation
//!
//! This module provides standardized benchmarks for evaluating NEAT performance
//! across different problem domains, comparing against reference implementations,
//! and monitoring performance regressions.

pub mod classic_problems;
pub mod performance_tests;
pub mod regression_tests;
pub mod comparison_suite;
pub mod test_runner;

use crate::neat::{NEATTrainer, TrainingResult, FitnessEvaluator};
use crate::config::NEATConfig;
use crate::error::Result;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

/// Comprehensive benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Benchmark metadata
    pub metadata: BenchmarkMetadata,
    /// Individual test results
    pub test_results: Vec<TestResult>,
    /// Overall performance summary
    pub summary: PerformanceSummary,
    /// System information
    pub system_info: SystemInfo,
}

/// Metadata about the benchmark run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetadata {
    /// Benchmark suite version
    pub version: String,
    /// Date and time of benchmark
    pub timestamp: String,
    /// Environment description
    pub environment: String,
    /// Configuration used
    pub config: NEATConfig,
}

/// Result of a single benchmark test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Test name
    pub name: String,
    /// Test category
    pub category: String,
    /// Whether test passed
    pub passed: bool,
    /// Performance metrics
    pub metrics: TestMetrics,
    /// Detailed results
    pub details: TestDetails,
}

/// Performance metrics for a test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestMetrics {
    /// Total execution time
    pub execution_time: Duration,
    /// Final fitness achieved
    pub final_fitness: f64,
    /// Target fitness (if applicable)
    pub target_fitness: Option<f64>,
    /// Generations to convergence
    pub generations_to_convergence: Option<usize>,
    /// Success rate (for multiple runs)
    pub success_rate: f64,
    /// Memory usage (peak)
    pub peak_memory_mb: f64,
    /// Throughput (evaluations per second)
    pub throughput: f64,
}

/// Detailed test results and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestDetails {
    /// Fitness progression over generations
    pub fitness_history: Vec<f64>,
    /// Population statistics
    pub population_stats: Vec<PopulationSnapshot>,
    /// Error messages (if any)
    pub errors: Vec<String>,
    /// Additional data
    pub metadata: std::collections::HashMap<String, String>,
}

/// Snapshot of population state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationSnapshot {
    /// Generation number
    pub generation: usize,
    /// Average fitness
    pub avg_fitness: f64,
    /// Best fitness
    pub best_fitness: f64,
    /// Number of species
    pub species_count: usize,
    /// Average complexity
    pub avg_complexity: f64,
}

/// Overall performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    /// Total tests run
    pub total_tests: usize,
    /// Tests passed
    pub tests_passed: usize,
    /// Total benchmark time
    pub total_time: Duration,
    /// Average performance score
    pub avg_performance_score: f64,
    /// Performance compared to baseline
    pub relative_performance: f64,
    /// Performance trend
    pub trend: PerformanceTrend,
}

/// Performance trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceTrend {
    /// Performance improved
    Improving(f64),
    /// Performance declined
    Declining(f64),
    /// Performance stable
    Stable,
    /// Insufficient data
    Unknown,
}

/// System information for benchmark context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// Available memory (GB)
    pub memory_gb: f64,
    /// Operating system
    pub os: String,
    /// Rust version
    pub rust_version: String,
    /// Compilation flags
    pub compile_flags: Vec<String>,
}

/// Main benchmark coordinator
pub struct BenchmarkSuite {
    /// Configuration for benchmarks
    config: BenchmarkConfig,
    /// Historical results for comparison
    baseline_results: Option<BenchmarkResults>,
}

/// Configuration for benchmark execution
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of runs per test
    pub runs_per_test: usize,
    /// Timeout per test
    pub test_timeout: Duration,
    /// Whether to run expensive tests
    pub include_expensive: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Output directory for results
    pub output_dir: Option<std::path::PathBuf>,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            baseline_results: None,
        }
    }
    
    /// Load baseline results for comparison
    pub fn with_baseline<P: AsRef<std::path::Path>>(mut self, path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        self.baseline_results = Some(serde_json::from_str(&content)?);
        Ok(self)
    }
    
    /// Run all benchmarks
    pub fn run_all(&mut self) -> Result<BenchmarkResults> {
        let start_time = Instant::now();
        let mut test_results = Vec::new();
        
        println!("Starting NEAT benchmark suite...");
        
        // Run classic problem benchmarks
        test_results.extend(self.run_classic_problems()?);
        
        // Run performance tests
        test_results.extend(self.run_performance_tests()?);
        
        // Run regression tests
        test_results.extend(self.run_regression_tests()?);
        
        // Run comparison tests (if enabled)
        if self.config.include_expensive {
            test_results.extend(self.run_comparison_tests()?);
        }
        
        let total_time = start_time.elapsed();
        
        // Generate summary
        let summary = self.generate_summary(&test_results, total_time);
        
        let results = BenchmarkResults {
            metadata: BenchmarkMetadata {
                version: env!("CARGO_PKG_VERSION").to_string(),
                timestamp: chrono::Utc::now().to_rfc3339(),
                environment: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
                config: NEATConfig::default(),
            },
            test_results,
            summary,
            system_info: SystemInfo::collect()?,
        };
        
        // Save results if output directory specified
        if let Some(output_dir) = &self.config.output_dir {
            self.save_results(&results, output_dir)?;
        }
        
        Ok(results)
    }
    
    /// Run classic problem benchmarks
    fn run_classic_problems(&self) -> Result<Vec<TestResult>> {
        println!("Running classic problem benchmarks...");
        classic_problems::run_all_tests(&self.config)
    }
    
    /// Run performance tests
    fn run_performance_tests(&self) -> Result<Vec<TestResult>> {
        println!("Running performance tests...");
        performance_tests::run_all_tests(&self.config)
    }
    
    /// Run regression tests
    fn run_regression_tests(&self) -> Result<Vec<TestResult>> {
        println!("Running regression tests...");
        regression_tests::run_all_tests(&self.config)
    }
    
    /// Run comparison tests
    fn run_comparison_tests(&self) -> Result<Vec<TestResult>> {
        println!("Running comparison tests...");
        comparison_suite::run_all_tests(&self.config)
    }
    
    /// Generate performance summary
    fn generate_summary(&self, test_results: &[TestResult], total_time: Duration) -> PerformanceSummary {
        let total_tests = test_results.len();
        let tests_passed = test_results.iter().filter(|r| r.passed).count();
        
        let avg_performance_score = if !test_results.is_empty() {
            test_results.iter()
                .map(|r| self.calculate_performance_score(r))
                .sum::<f64>() / test_results.len() as f64
        } else {
            0.0
        };
        
        let relative_performance = self.calculate_relative_performance(test_results);
        let trend = self.analyze_performance_trend(test_results);
        
        PerformanceSummary {
            total_tests,
            tests_passed,
            total_time,
            avg_performance_score,
            relative_performance,
            trend,
        }
    }
    
    /// Calculate performance score for a test
    fn calculate_performance_score(&self, test_result: &TestResult) -> f64 {
        let fitness_score = test_result.metrics.final_fitness;
        let time_score = 1.0 / (1.0 + test_result.metrics.execution_time.as_secs_f64());
        let success_score = if test_result.passed { 1.0 } else { 0.0 };
        
        // Weighted average
        (fitness_score * 0.5) + (time_score * 0.3) + (success_score * 0.2)
    }
    
    /// Calculate relative performance compared to baseline
    fn calculate_relative_performance(&self, test_results: &[TestResult]) -> f64 {
        if let Some(baseline) = &self.baseline_results {
            // Compare against baseline results
            let current_avg = test_results.iter()
                .map(|r| self.calculate_performance_score(r))
                .sum::<f64>() / test_results.len() as f64;
                
            let baseline_avg = baseline.summary.avg_performance_score;
            
            if baseline_avg > 0.0 {
                current_avg / baseline_avg
            } else {
                1.0
            }
        } else {
            1.0 // No baseline to compare against
        }
    }
    
    /// Analyze performance trend
    fn analyze_performance_trend(&self, _test_results: &[TestResult]) -> PerformanceTrend {
        // Simplified trend analysis
        if let Some(_baseline) = &self.baseline_results {
            let improvement = self.calculate_relative_performance(_test_results) - 1.0;
            if improvement > 0.05 {
                PerformanceTrend::Improving(improvement)
            } else if improvement < -0.05 {
                PerformanceTrend::Declining(-improvement)
            } else {
                PerformanceTrend::Stable
            }
        } else {
            PerformanceTrend::Unknown
        }
    }
    
    /// Save benchmark results
    fn save_results(&self, results: &BenchmarkResults, output_dir: &std::path::Path) -> Result<()> {
        std::fs::create_dir_all(output_dir)?;
        
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let filename = format!("benchmark_results_{}.json", timestamp);
        let filepath = output_dir.join(filename);
        
        let json = serde_json::to_string_pretty(results)?;
        std::fs::write(filepath, json)?;
        
        // Also save a "latest" copy
        let latest_path = output_dir.join("benchmark_results_latest.json");
        let latest_json = serde_json::to_string_pretty(results)?;
        std::fs::write(latest_path, latest_json)?;
        
        println!("Benchmark results saved to: {:?}", output_dir);
        
        Ok(())
    }
    
    /// Print benchmark summary
    pub fn print_summary(&self, results: &BenchmarkResults) {
        println!("\n=== NEAT Benchmark Results ===");
        println!("Total tests: {}", results.summary.total_tests);
        println!("Tests passed: {}", results.summary.tests_passed);
        println!("Success rate: {:.1}%", 
                (results.summary.tests_passed as f64 / results.summary.total_tests as f64) * 100.0);
        println!("Total time: {:.2}s", results.summary.total_time.as_secs_f64());
        println!("Avg performance score: {:.3}", results.summary.avg_performance_score);
        println!("Relative performance: {:.2}x", results.summary.relative_performance);
        
        match &results.summary.trend {
            PerformanceTrend::Improving(delta) => 
                println!("Trend: Improving (+{:.1}%)", delta * 100.0),
            PerformanceTrend::Declining(delta) => 
                println!("Trend: Declining (-{:.1}%)", delta * 100.0),
            PerformanceTrend::Stable => 
                println!("Trend: Stable"),
            PerformanceTrend::Unknown => 
                println!("Trend: Unknown (no baseline)"),
        }
        
        println!("\nSystem Info:");
        println!("  CPU cores: {}", results.system_info.cpu_cores);
        println!("  Memory: {:.1} GB", results.system_info.memory_gb);
        println!("  OS: {}", results.system_info.os);
        
        println!("\nDetailed Results:");
        for result in &results.test_results {
            let status = if result.passed { "✓" } else { "✗" };
            println!("  {} {} - Fitness: {:.3}, Time: {:.2}s", 
                    status, result.name, result.metrics.final_fitness, 
                    result.metrics.execution_time.as_secs_f64());
        }
    }
}

impl SystemInfo {
    /// Collect current system information
    fn collect() -> Result<Self> {
        Ok(Self {
            cpu_cores: num_cpus::get(),
            memory_gb: Self::get_memory_gb(),
            os: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
            rust_version: "1.75.0".to_string(), // Simplified for now
            compile_flags: Self::get_compile_flags(),
        })
    }
    
    /// Get available memory in GB (simplified)
    fn get_memory_gb() -> f64 {
        // Simplified memory detection
        // In a real implementation, you'd use system APIs
        8.0 // Default assumption
    }
    
    /// Get compilation flags
    fn get_compile_flags() -> Vec<String> {
        let mut flags = Vec::new();
        
        if cfg!(debug_assertions) {
            flags.push("debug".to_string());
        } else {
            flags.push("release".to_string());
        }
        
        if cfg!(feature = "std") {
            flags.push("std".to_string());
        }
        
        flags
    }
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            runs_per_test: 5,
            test_timeout: Duration::from_secs(300), // 5 minutes per test
            include_expensive: false,
            random_seed: Some(42),
            output_dir: None,
        }
    }
}

/// Convenience function to run quick benchmarks
pub fn run_quick_benchmark() -> Result<BenchmarkResults> {
    let config = BenchmarkConfig {
        runs_per_test: 3,
        test_timeout: Duration::from_secs(60),
        include_expensive: false,
        random_seed: Some(42),
        output_dir: None,
    };
    
    let mut suite = BenchmarkSuite::new(config);
    suite.run_all()
}

/// Convenience function to run comprehensive benchmarks
pub fn run_comprehensive_benchmark<P: AsRef<std::path::Path>>(output_dir: P) -> Result<BenchmarkResults> {
    let config = BenchmarkConfig {
        runs_per_test: 10,
        test_timeout: Duration::from_secs(600),
        include_expensive: true,
        random_seed: Some(42),
        output_dir: Some(output_dir.as_ref().to_path_buf()),
    };
    
    let mut suite = BenchmarkSuite::new(config);
    suite.run_all()
}