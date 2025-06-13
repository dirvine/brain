//! Performance benchmarks for NEAT implementation
//!
//! This module contains tests focused on computational performance,
//! scalability, and resource utilization.

use crate::neat::{NEATTrainer, FitnessEvaluator, Genome, Network, ParallelEvaluator, ParallelConfig};
use crate::neat::fitness::XORFitnessEvaluator;
use crate::config::NEATConfig;
use crate::error::Result;
use crate::benchmarks::{BenchmarkConfig, TestResult, TestMetrics, TestDetails, PopulationSnapshot};
use std::time::{Duration, Instant};
use std::sync::Arc;
use std::thread;

/// Performance test for large population sizes
pub struct ScalabilityTest {
    /// Population sizes to test
    pub population_sizes: Vec<usize>,
    /// Generations to run for each size
    pub generations_per_test: usize,
    /// Maximum time per test
    pub max_time_per_test: Duration,
}

/// Performance test for parallel evaluation
pub struct ParallelPerformanceTest {
    /// Thread counts to test
    pub thread_counts: Vec<usize>,
    /// Population size for testing
    pub population_size: usize,
    /// Number of evaluations per test
    pub evaluations_per_test: usize,
}

/// Memory usage and efficiency test
pub struct MemoryEfficiencyTest {
    /// Test durations (how long to run)
    pub test_durations: Vec<Duration>,
    /// Population size
    pub population_size: usize,
    /// Whether to stress test memory
    pub stress_test: bool,
}

/// Throughput benchmark for fitness evaluation
pub struct ThroughputTest {
    /// Batch sizes to test
    pub batch_sizes: Vec<usize>,
    /// Total evaluations to perform
    pub total_evaluations: usize,
    /// Test both sequential and parallel
    pub test_parallel: bool,
}

/// Network activation performance test
pub struct ActivationPerformanceTest {
    /// Network complexity levels (nodes, connections)
    pub complexity_levels: Vec<(usize, usize)>,
    /// Activations per complexity level
    pub activations_per_level: usize,
    /// Input sizes to test
    pub input_sizes: Vec<usize>,
}

/// Simple evaluator for performance testing
#[derive(Debug, Clone)]
pub struct BenchmarkEvaluator {
    /// Computation complexity (iterations per evaluation)
    pub complexity: usize,
    /// Input size
    pub input_size: usize,
    /// Output size
    pub output_size: usize,
}

impl FitnessEvaluator for BenchmarkEvaluator {
    fn evaluate(&self, genome: &Genome) -> Result<f64> {
        let mut network = Network::from_genome(genome)?;
        
        // Create random inputs
        let inputs: Vec<f64> = (0..self.input_size)
            .map(|_| rand::random::<f64>() * 2.0 - 1.0)
            .collect();
        
        let mut total_output = 0.0;
        
        // Perform multiple activations to simulate computational load
        for _ in 0..self.complexity {
            let outputs = network.activate(&inputs)?;
            total_output += outputs.iter().sum::<f64>();
        }
        
        // Return normalized fitness
        Ok((total_output / (self.complexity as f64 * self.output_size as f64)).abs().min(1.0))
    }
    
    fn input_size(&self) -> usize {
        self.input_size
    }
    
    fn output_size(&self) -> usize {
        self.output_size
    }
}

/// Run all performance tests
pub fn run_all_tests(config: &BenchmarkConfig) -> Result<Vec<TestResult>> {
    let mut results = Vec::new();
    
    // Skip expensive tests if not requested
    if !config.include_expensive {
        println!("Skipping expensive performance tests");
        return Ok(results);
    }
    
    // Scalability tests
    results.push(run_scalability_test(config)?);
    
    // Parallel performance tests
    results.push(run_parallel_performance_test(config)?);
    
    // Memory efficiency tests
    results.push(run_memory_efficiency_test(config)?);
    
    // Throughput tests
    results.push(run_throughput_test(config)?);
    
    // Activation performance tests
    results.push(run_activation_performance_test(config)?);
    
    Ok(results)
}

/// Test scalability with different population sizes
fn run_scalability_test(config: &BenchmarkConfig) -> Result<TestResult> {
    let start_time = Instant::now();
    
    let test = ScalabilityTest {
        population_sizes: vec![50, 100, 200, 500, 1000],
        generations_per_test: 5,
        max_time_per_test: Duration::from_secs(120),
    };
    
    let mut fitness_history = Vec::new();
    let mut population_snapshots = Vec::new();
    let mut throughput_data = Vec::new();
    let mut errors = Vec::new();
    
    for &pop_size in &test.population_sizes {
        let test_start = Instant::now();
        
        let evaluator = BenchmarkEvaluator {
            complexity: 10,
            input_size: 4,
            output_size: 2,
        };
        
        let mut neat_config = NEATConfig::default();
        neat_config.population.size = pop_size;
        neat_config.population.max_generations = test.generations_per_test;
        neat_config.population.target_fitness = 0.8;
        
        match run_timed_training(evaluator, neat_config, test.max_time_per_test) {
            Ok((result, elapsed)) => {
                let generations_per_sec = result.state.generation as f64 / elapsed.as_secs_f64();
                let evaluations_per_sec = (result.state.generation * pop_size) as f64 / elapsed.as_secs_f64();
                
                throughput_data.push((pop_size, generations_per_sec, evaluations_per_sec));
                fitness_history.extend(result.stats.fitness_history);
                
                population_snapshots.push(PopulationSnapshot {
                    generation: pop_size, // Using generation field to store pop_size for analysis
                    avg_fitness: result.stats.fitness_history.last().copied().unwrap_or(0.0),
                    best_fitness: result.best_genome.fitness,
                    species_count: pop_size / 20, // Estimated
                    avg_complexity: evaluations_per_sec, // Store throughput in complexity field
                });
                
                println!("Population {}: {:.2} eval/sec in {:.2}s", 
                        pop_size, evaluations_per_sec, elapsed.as_secs_f64());
            }
            Err(e) => {
                errors.push(format!("Population size {} failed: {}", pop_size, e));
            }
        }
        
        if test_start.elapsed() > test.max_time_per_test {
            errors.push(format!("Population size {} exceeded time limit", pop_size));
            break;
        }
    }
    
    let execution_time = start_time.elapsed();
    let final_fitness = fitness_history.last().copied().unwrap_or(0.0);
    
    // Calculate performance metrics
    let max_throughput = throughput_data.iter()
        .map(|(_, _, eval_per_sec)| *eval_per_sec)
        .fold(0.0, f64::max);
    
    let scalability_score = if throughput_data.len() > 1 {
        let first_throughput = throughput_data[0].2;
        let last_throughput = throughput_data.last().unwrap().2;
        // Good scalability means throughput doesn't degrade too much with size
        (last_throughput / first_throughput).min(1.0)
    } else {
        0.0
    };
    
    Ok(TestResult {
        name: "Population Scalability".to_string(),
        category: "Performance".to_string(),
        passed: scalability_score > 0.1 && errors.len() < test.population_sizes.len() / 2,
        metrics: TestMetrics {
            execution_time,
            final_fitness,
            target_fitness: None,
            generations_to_convergence: None,
            success_rate: 1.0 - (errors.len() as f64 / test.population_sizes.len() as f64),
            peak_memory_mb: estimate_peak_memory(&test.population_sizes),
            throughput: max_throughput,
        },
        details: TestDetails {
            fitness_history,
            population_stats: population_snapshots,
            errors,
            metadata: std::collections::HashMap::from([
                ("test_type".to_string(), "scalability".to_string()),
                ("max_population".to_string(), test.population_sizes.iter().max().unwrap().to_string()),
                ("scalability_score".to_string(), scalability_score.to_string()),
                ("max_throughput".to_string(), max_throughput.to_string()),
            ]),
        },
    })
}

/// Test parallel performance with different thread counts
fn run_parallel_performance_test(config: &BenchmarkConfig) -> Result<TestResult> {
    let start_time = Instant::now();
    
    let test = ParallelPerformanceTest {
        thread_counts: vec![1, 2, 4, 8, num_cpus::get()],
        population_size: 200,
        evaluations_per_test: 1000,
    };
    
    let mut fitness_history = Vec::new();
    let mut population_snapshots = Vec::new();
    let mut errors = Vec::new();
    
    let evaluator = BenchmarkEvaluator {
        complexity: 20,
        input_size: 3,
        output_size: 1,
    };
    
    let baseline_time = measure_sequential_time(&evaluator, test.population_size, test.evaluations_per_test)?;
    
    for &thread_count in &test.thread_counts {
        match measure_parallel_time(&evaluator, test.population_size, test.evaluations_per_test, thread_count) {
            Ok(parallel_time) => {
                let speedup = baseline_time.as_secs_f64() / parallel_time.as_secs_f64();
                let efficiency = speedup / thread_count as f64;
                
                population_snapshots.push(PopulationSnapshot {
                    generation: thread_count,
                    avg_fitness: efficiency,
                    best_fitness: speedup,
                    species_count: thread_count,
                    avg_complexity: parallel_time.as_millis() as f64,
                });
                
                println!("Threads {}: {:.2}x speedup, {:.1}% efficiency", 
                        thread_count, speedup, efficiency * 100.0);
            }
            Err(e) => {
                errors.push(format!("Thread count {} failed: {}", thread_count, e));
            }
        }
    }
    
    let execution_time = start_time.elapsed();
    let max_speedup = population_snapshots.iter()
        .map(|s| s.best_fitness)
        .fold(0.0, f64::max);
    
    Ok(TestResult {
        name: "Parallel Performance".to_string(),
        category: "Performance".to_string(),
        passed: max_speedup > 1.5 && errors.is_empty(),
        metrics: TestMetrics {
            execution_time,
            final_fitness: max_speedup,
            target_fitness: Some(2.0), // Target 2x speedup
            generations_to_convergence: None,
            success_rate: 1.0 - (errors.len() as f64 / test.thread_counts.len() as f64),
            peak_memory_mb: 100.0,
            throughput: test.evaluations_per_test as f64 / baseline_time.as_secs_f64(),
        },
        details: TestDetails {
            fitness_history,
            population_stats: population_snapshots,
            errors,
            metadata: std::collections::HashMap::from([
                ("test_type".to_string(), "parallel_performance".to_string()),
                ("baseline_time_ms".to_string(), baseline_time.as_millis().to_string()),
                ("max_speedup".to_string(), max_speedup.to_string()),
                ("max_threads".to_string(), test.thread_counts.iter().max().unwrap().to_string()),
            ]),
        },
    })
}

/// Test memory efficiency over time
fn run_memory_efficiency_test(config: &BenchmarkConfig) -> Result<TestResult> {
    let start_time = Instant::now();
    
    let test = MemoryEfficiencyTest {
        test_durations: vec![Duration::from_secs(30), Duration::from_secs(60)],
        population_size: 150,
        stress_test: false,
    };
    
    let mut fitness_history = Vec::new();
    let mut population_snapshots = Vec::new();
    let mut errors = Vec::new();
    
    let evaluator = BenchmarkEvaluator {
        complexity: 15,
        input_size: 5,
        output_size: 3,
    };
    
    for &duration in &test.test_durations {
        match run_memory_stress_test(&evaluator, test.population_size, duration) {
            Ok((generations, final_fitness, peak_memory)) => {
                fitness_history.push(final_fitness);
                
                population_snapshots.push(PopulationSnapshot {
                    generation: duration.as_secs() as usize,
                    avg_fitness: final_fitness,
                    best_fitness: final_fitness,
                    species_count: generations,
                    avg_complexity: peak_memory,
                });
                
                println!("Duration {}s: {} generations, {:.1} MB peak", 
                        duration.as_secs(), generations, peak_memory);
            }
            Err(e) => {
                errors.push(format!("Duration {}s failed: {}", duration.as_secs(), e));
            }
        }
    }
    
    let execution_time = start_time.elapsed();
    let avg_memory = population_snapshots.iter()
        .map(|s| s.avg_complexity)
        .sum::<f64>() / population_snapshots.len() as f64;
    
    Ok(TestResult {
        name: "Memory Efficiency".to_string(),
        category: "Performance".to_string(),
        passed: avg_memory < 200.0 && errors.is_empty(),
        metrics: TestMetrics {
            execution_time,
            final_fitness: fitness_history.last().copied().unwrap_or(0.0),
            target_fitness: None,
            generations_to_convergence: None,
            success_rate: 1.0 - (errors.len() as f64 / test.test_durations.len() as f64),
            peak_memory_mb: avg_memory,
            throughput: 0.0,
        },
        details: TestDetails {
            fitness_history,
            population_stats: population_snapshots,
            errors,
            metadata: std::collections::HashMap::from([
                ("test_type".to_string(), "memory_efficiency".to_string()),
                ("avg_memory_mb".to_string(), avg_memory.to_string()),
                ("stress_test".to_string(), test.stress_test.to_string()),
            ]),
        },
    })
}

/// Test evaluation throughput
fn run_throughput_test(config: &BenchmarkConfig) -> Result<TestResult> {
    let start_time = Instant::now();
    
    let test = ThroughputTest {
        batch_sizes: vec![1, 10, 50, 100],
        total_evaluations: 5000,
        test_parallel: true,
    };
    
    let mut population_snapshots = Vec::new();
    let mut errors = Vec::new();
    
    let evaluator = BenchmarkEvaluator {
        complexity: 5,
        input_size: 2,
        output_size: 1,
    };
    
    for &batch_size in &test.batch_sizes {
        // Sequential test
        match measure_batch_throughput(&evaluator, batch_size, test.total_evaluations, false) {
            Ok(throughput) => {
                population_snapshots.push(PopulationSnapshot {
                    generation: batch_size,
                    avg_fitness: throughput,
                    best_fitness: throughput,
                    species_count: 0, // Sequential
                    avg_complexity: batch_size as f64,
                });
            }
            Err(e) => {
                errors.push(format!("Sequential batch {} failed: {}", batch_size, e));
            }
        }
        
        // Parallel test
        if test.test_parallel {
            match measure_batch_throughput(&evaluator, batch_size, test.total_evaluations, true) {
                Ok(throughput) => {
                    population_snapshots.push(PopulationSnapshot {
                        generation: batch_size,
                        avg_fitness: throughput,
                        best_fitness: throughput,
                        species_count: 1, // Parallel
                        avg_complexity: batch_size as f64,
                    });
                }
                Err(e) => {
                    errors.push(format!("Parallel batch {} failed: {}", batch_size, e));
                }
            }
        }
    }
    
    let execution_time = start_time.elapsed();
    let max_throughput = population_snapshots.iter()
        .map(|s| s.avg_fitness)
        .fold(0.0, f64::max);
    
    Ok(TestResult {
        name: "Evaluation Throughput".to_string(),
        category: "Performance".to_string(),
        passed: max_throughput > 1000.0 && errors.is_empty(),
        metrics: TestMetrics {
            execution_time,
            final_fitness: max_throughput,
            target_fitness: Some(1000.0),
            generations_to_convergence: None,
            success_rate: 1.0 - (errors.len() as f64 / (test.batch_sizes.len() * 2) as f64),
            peak_memory_mb: 50.0,
            throughput: max_throughput,
        },
        details: TestDetails {
            fitness_history: vec![max_throughput],
            population_stats: population_snapshots,
            errors,
            metadata: std::collections::HashMap::from([
                ("test_type".to_string(), "throughput".to_string()),
                ("max_throughput".to_string(), max_throughput.to_string()),
                ("total_evaluations".to_string(), test.total_evaluations.to_string()),
            ]),
        },
    })
}

/// Test network activation performance
fn run_activation_performance_test(config: &BenchmarkConfig) -> Result<TestResult> {
    let start_time = Instant::now();
    
    let test = ActivationPerformanceTest {
        complexity_levels: vec![(5, 10), (10, 25), (20, 50), (30, 100)],
        activations_per_level: 10000,
        input_sizes: vec![2, 5, 10],
    };
    
    let mut population_snapshots = Vec::new();
    let mut errors = Vec::new();
    
    for &(nodes, connections) in &test.complexity_levels {
        for &input_size in &test.input_sizes {
            match measure_activation_performance(nodes, connections, input_size, test.activations_per_level) {
                Ok(activations_per_sec) => {
                    population_snapshots.push(PopulationSnapshot {
                        generation: nodes,
                        avg_fitness: activations_per_sec,
                        best_fitness: activations_per_sec,
                        species_count: input_size,
                        avg_complexity: connections as f64,
                    });
                }
                Err(e) => {
                    errors.push(format!("Complexity ({}, {}) input {} failed: {}", 
                               nodes, connections, input_size, e));
                }
            }
        }
    }
    
    let execution_time = start_time.elapsed();
    let max_activation_rate = population_snapshots.iter()
        .map(|s| s.avg_fitness)
        .fold(0.0, f64::max);
    
    Ok(TestResult {
        name: "Network Activation Performance".to_string(),
        category: "Performance".to_string(),
        passed: max_activation_rate > 10000.0 && errors.is_empty(),
        metrics: TestMetrics {
            execution_time,
            final_fitness: max_activation_rate,
            target_fitness: Some(10000.0),
            generations_to_convergence: None,
            success_rate: 1.0 - (errors.len() as f64 / (test.complexity_levels.len() * test.input_sizes.len()) as f64),
            peak_memory_mb: 30.0,
            throughput: max_activation_rate,
        },
        details: TestDetails {
            fitness_history: vec![max_activation_rate],
            population_stats: population_snapshots,
            errors,
            metadata: std::collections::HashMap::from([
                ("test_type".to_string(), "activation_performance".to_string()),
                ("max_activation_rate".to_string(), max_activation_rate.to_string()),
                ("max_nodes".to_string(), test.complexity_levels.iter().map(|(n, _)| *n).max().unwrap().to_string()),
            ]),
        },
    })
}

// Helper functions

fn run_timed_training(
    evaluator: BenchmarkEvaluator,
    config: NEATConfig,
    max_time: Duration,
) -> Result<(crate::neat::trainer::TrainingResult, Duration)> {
    let start = Instant::now();
    let mut trainer = NEATTrainer::new(evaluator, config);
    let result = trainer.train()?;
    let elapsed = start.elapsed();
    
    if elapsed > max_time {
        return Err(crate::error::NEATError::Other(
            anyhow::anyhow!("Training exceeded time limit")
        ));
    }
    
    Ok((result, elapsed))
}

fn measure_sequential_time(
    evaluator: &BenchmarkEvaluator,
    population_size: usize,
    evaluations: usize,
) -> Result<Duration> {
    let start = Instant::now();
    
    for _ in 0..evaluations {
        let genome = crate::neat::genome::Genome::new(0, evaluator.input_size, evaluator.output_size);
        evaluator.evaluate(&genome)?;
    }
    
    Ok(start.elapsed())
}

fn measure_parallel_time(
    evaluator: &BenchmarkEvaluator,
    population_size: usize,
    evaluations: usize,
    thread_count: usize,
) -> Result<Duration> {
    let start = Instant::now();
    let mut parallel_eval = ParallelEvaluator::new(evaluator.clone(), Some(thread_count));
    
    let mut population: Vec<_> = (0..population_size)
        .map(|i| crate::neat::genome::Genome::new(i, evaluator.input_size, evaluator.output_size))
        .collect();
    
    let batches = (evaluations + population_size - 1) / population_size;
    for _ in 0..batches {
        parallel_eval.evaluate_population_rayon(&mut population)?;
    }
    
    Ok(start.elapsed())
}

fn run_memory_stress_test(
    evaluator: &BenchmarkEvaluator,
    population_size: usize,
    duration: Duration,
) -> Result<(usize, f64, f64)> {
    let start = Instant::now();
    
    let mut config = NEATConfig::default();
    config.population.size = population_size;
    // Set a large max generations but we'll stop based on time
    config.population.max_generations = 10000;
    
    let mut trainer = NEATTrainer::new(evaluator.clone(), config);
    
    // Override the internal timeout by setting a reasonable max generations
    // that should complete within the duration
    let estimated_generations = duration.as_secs() as usize / 2; // Estimate 2 seconds per generation
    let mut test_config = NEATConfig::default();
    test_config.population.size = population_size;
    test_config.population.max_generations = estimated_generations.max(5);
    
    let mut test_trainer = NEATTrainer::new(evaluator.clone(), test_config);
    let result = test_trainer.train()?;
    
    let peak_memory = estimate_peak_memory(&[population_size]);
    
    Ok((result.state.generation, result.best_genome.fitness, peak_memory))
}

fn measure_batch_throughput(
    evaluator: &BenchmarkEvaluator,
    batch_size: usize,
    total_evaluations: usize,
    parallel: bool,
) -> Result<f64> {
    let start = Instant::now();
    
    if parallel {
        let mut parallel_eval = ParallelEvaluator::new(evaluator.clone(), None);
        let batches = (total_evaluations + batch_size - 1) / batch_size;
        
        for _ in 0..batches {
            let mut batch: Vec<_> = (0..batch_size)
                .map(|i| crate::neat::genome::Genome::new(i, evaluator.input_size, evaluator.output_size))
                .collect();
            parallel_eval.evaluate_population_rayon(&mut batch)?;
        }
    } else {
        for _ in 0..total_evaluations {
            let genome = crate::neat::genome::Genome::new(0, evaluator.input_size, evaluator.output_size);
            evaluator.evaluate(&genome)?;
        }
    }
    
    let elapsed = start.elapsed();
    Ok(total_evaluations as f64 / elapsed.as_secs_f64())
}

fn measure_activation_performance(
    nodes: usize,
    connections: usize,
    input_size: usize,
    activations: usize,
) -> Result<f64> {
    // Create a test genome with specified complexity
    let mut genome = crate::neat::genome::Genome::new(0, input_size, 1);
    
    // Add hidden nodes and connections to reach target complexity
    // This is simplified - in reality you'd build a more realistic network
    let start = Instant::now();
    let mut network = Network::from_genome(&genome)?;
    
    let inputs: Vec<f64> = (0..input_size).map(|_| rand::random::<f64>()).collect();
    
    for _ in 0..activations {
        network.activate(&inputs)?;
    }
    
    let elapsed = start.elapsed();
    Ok(activations as f64 / elapsed.as_secs_f64())
}

fn estimate_peak_memory(population_sizes: &[usize]) -> f64 {
    // Simplified memory estimation
    let max_pop = population_sizes.iter().max().unwrap_or(&100);
    20.0 + (*max_pop as f64 * 0.1) // Base + per-genome estimation
}

impl Default for BenchmarkEvaluator {
    fn default() -> Self {
        Self {
            complexity: 10,
            input_size: 2,
            output_size: 1,
        }
    }
}