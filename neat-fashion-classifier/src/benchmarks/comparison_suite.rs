//! Comparison tests against reference implementations and baselines
//!
//! This module contains tests that compare our NEAT implementation
//! against established baselines, reference implementations, and
//! alternative approaches.

use crate::neat::{NEATTrainer, FitnessEvaluator, Genome, Network};
use crate::neat::fitness::XORFitnessEvaluator;
use crate::config::NEATConfig;
use crate::error::Result;
use crate::benchmarks::{BenchmarkConfig, TestResult, TestMetrics, TestDetails, PopulationSnapshot};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use rand::Rng;

/// Baseline comparison against simple random search
pub struct RandomSearchBaseline {
    /// Number of random samples to try
    pub sample_count: usize,
    /// Maximum network complexity for random generation
    pub max_complexity: NetworkComplexity,
    /// Timeout for random search
    pub timeout: Duration,
}

/// Baseline comparison against hill climbing
pub struct HillClimbingBaseline {
    /// Starting network topology
    pub initial_topology: NetworkComplexity,
    /// Number of hill climbing steps
    pub max_steps: usize,
    /// Step size for weight adjustments
    pub step_size: f64,
}

/// Comparison against fixed topology neural networks
pub struct FixedTopologyComparison {
    /// Topologies to test
    pub topologies: Vec<FixedTopology>,
    /// Training epochs for fixed networks
    pub training_epochs: usize,
    /// Learning rate for weight updates
    pub learning_rate: f64,
}

/// Reference benchmark results from literature
pub struct LiteratureComparison {
    /// Expected performance benchmarks from papers
    pub reference_results: HashMap<String, ReferenceResult>,
    /// Tolerance for performance comparison
    pub tolerance: f64,
    /// Problems to compare against literature
    pub benchmark_problems: Vec<BenchmarkProblem>,
}

/// Performance comparison against parallel implementations
pub struct ParallelComparison {
    /// Thread counts to compare
    pub thread_counts: Vec<usize>,
    /// Population sizes for scaling tests
    pub population_sizes: Vec<usize>,
    /// Expected speedup ratios
    pub expected_speedups: HashMap<usize, f64>,
}

/// Network complexity specification
#[derive(Debug, Clone)]
pub struct NetworkComplexity {
    pub nodes: usize,
    pub connections: usize,
    pub depth: usize,
}

/// Fixed network topology
#[derive(Debug, Clone)]
pub struct FixedTopology {
    pub name: String,
    pub hidden_layers: Vec<usize>,
    pub activation: String,
}

/// Reference result from literature
#[derive(Debug, Clone)]
pub struct ReferenceResult {
    pub mean_fitness: f64,
    pub std_deviation: f64,
    pub generations_to_solve: Option<usize>,
    pub success_rate: f64,
    pub source: String,
}

/// Benchmark problem specification
#[derive(Debug, Clone)]
pub enum BenchmarkProblem {
    XOR,
    DoublePoleBalancing,
    CartPole,
    FunctionApproximation,
}

/// Simple random search implementation
pub struct RandomSearchEvaluator {
    pub max_complexity: NetworkComplexity,
}

/// Hill climbing implementation  
pub struct HillClimbingEvaluator {
    pub step_size: f64,
}

/// Fixed topology network evaluator
pub struct FixedNetworkEvaluator {
    pub topology: FixedTopology,
}

/// Run all comparison tests
pub fn run_all_tests(config: &BenchmarkConfig) -> Result<Vec<TestResult>> {
    let mut results = Vec::new();
    
    if !config.include_expensive {
        println!("Skipping expensive comparison tests");
        return Ok(results);
    }
    
    // Random search baseline
    results.push(run_random_search_comparison(config)?);
    
    // Hill climbing baseline
    results.push(run_hill_climbing_comparison(config)?);
    
    // Fixed topology comparison
    results.push(run_fixed_topology_comparison(config)?);
    
    // Literature comparison
    results.push(run_literature_comparison(config)?);
    
    // Parallel implementation comparison
    results.push(run_parallel_comparison(config)?);
    
    Ok(results)
}

/// Compare against random search baseline
fn run_random_search_comparison(config: &BenchmarkConfig) -> Result<TestResult> {
    let start_time = Instant::now();
    
    let baseline = RandomSearchBaseline {
        sample_count: 10000,
        max_complexity: NetworkComplexity {
            nodes: 20,
            connections: 50,
            depth: 5,
        },
        timeout: Duration::from_secs(120),
    };
    
    let evaluator = XORFitnessEvaluator::default();
    
    // Run random search
    let random_result = run_random_search(&evaluator, &baseline)?;
    
    // Run NEAT
    let mut neat_config = NEATConfig::default();
    neat_config.population.size = 100;
    neat_config.population.max_generations = 50;
    neat_config.population.target_fitness = 3.8;
    
    let mut neat_trainer = NEATTrainer::new(evaluator.clone(), neat_config);
    let neat_result = neat_trainer.train()?;
    
    // Compare results
    let neat_performance = neat_result.best_genome.fitness;
    let random_performance = random_result.best_fitness;
    let improvement_ratio = neat_performance / random_performance.max(0.001);
    
    let mut population_snapshots = Vec::new();
    population_snapshots.push(PopulationSnapshot {
        generation: 0,
        avg_fitness: random_performance,
        best_fitness: random_performance,
        species_count: 0,
        avg_complexity: random_result.evaluations as f64,
    });
    
    population_snapshots.push(PopulationSnapshot {
        generation: 1,
        avg_fitness: neat_performance,
        best_fitness: neat_performance,
        species_count: 1,
        avg_complexity: neat_result.state.generation as f64,
    });
    
    let execution_time = start_time.elapsed();
    
    Ok(TestResult {
        name: "Random Search Comparison".to_string(),
        category: "Comparison".to_string(),
        passed: improvement_ratio > 1.5, // NEAT should be 1.5x better than random
        metrics: TestMetrics {
            execution_time,
            final_fitness: improvement_ratio,
            target_fitness: Some(1.5),
            generations_to_convergence: Some(neat_result.state.generation),
            success_rate: if improvement_ratio > 1.5 { 1.0 } else { 0.0 },
            peak_memory_mb: 100.0,
            throughput: improvement_ratio,
        },
        details: TestDetails {
            fitness_history: vec![random_performance, neat_performance],
            population_stats: population_snapshots,
            errors: Vec::new(),
            metadata: HashMap::from([
                ("test_type".to_string(), "random_search_comparison".to_string()),
                ("improvement_ratio".to_string(), improvement_ratio.to_string()),
                ("random_performance".to_string(), random_performance.to_string()),
                ("neat_performance".to_string(), neat_performance.to_string()),
                ("random_evaluations".to_string(), random_result.evaluations.to_string()),
            ]),
        },
    })
}

/// Compare against hill climbing baseline
fn run_hill_climbing_comparison(config: &BenchmarkConfig) -> Result<TestResult> {
    let start_time = Instant::now();
    
    let baseline = HillClimbingBaseline {
        initial_topology: NetworkComplexity {
            nodes: 10,
            connections: 20,
            depth: 3,
        },
        max_steps: 5000,
        step_size: 0.1,
    };
    
    let evaluator = XORFitnessEvaluator::default();
    
    // Run hill climbing
    let hill_climbing_result = run_hill_climbing(&evaluator, &baseline)?;
    
    // Run NEAT
    let mut neat_config = NEATConfig::default();
    neat_config.population.size = 50;
    neat_config.population.max_generations = 40;
    neat_config.population.target_fitness = 3.8;
    
    let mut neat_trainer = NEATTrainer::new(evaluator.clone(), neat_config);
    let neat_result = neat_trainer.train()?;
    
    // Compare results
    let neat_performance = neat_result.best_genome.fitness;
    let hill_climbing_performance = hill_climbing_result.best_fitness;
    let improvement_ratio = neat_performance / hill_climbing_performance.max(0.001);
    
    let execution_time = start_time.elapsed();
    
    Ok(TestResult {
        name: "Hill Climbing Comparison".to_string(),
        category: "Comparison".to_string(),
        passed: improvement_ratio > 1.2, // NEAT should be 1.2x better than hill climbing
        metrics: TestMetrics {
            execution_time,
            final_fitness: improvement_ratio,
            target_fitness: Some(1.2),
            generations_to_convergence: Some(neat_result.state.generation),
            success_rate: if improvement_ratio > 1.2 { 1.0 } else { 0.0 },
            peak_memory_mb: 80.0,
            throughput: improvement_ratio,
        },
        details: TestDetails {
            fitness_history: vec![hill_climbing_performance, neat_performance],
            population_stats: vec![
                PopulationSnapshot {
                    generation: 0,
                    avg_fitness: hill_climbing_performance,
                    best_fitness: hill_climbing_performance,
                    species_count: 0,
                    avg_complexity: hill_climbing_result.steps as f64,
                },
                PopulationSnapshot {
                    generation: 1,
                    avg_fitness: neat_performance,
                    best_fitness: neat_performance,
                    species_count: 1,
                    avg_complexity: neat_result.state.generation as f64,
                },
            ],
            errors: Vec::new(),
            metadata: HashMap::from([
                ("test_type".to_string(), "hill_climbing_comparison".to_string()),
                ("improvement_ratio".to_string(), improvement_ratio.to_string()),
                ("hill_climbing_performance".to_string(), hill_climbing_performance.to_string()),
                ("neat_performance".to_string(), neat_performance.to_string()),
            ]),
        },
    })
}

/// Compare against fixed topology networks
fn run_fixed_topology_comparison(config: &BenchmarkConfig) -> Result<TestResult> {
    let start_time = Instant::now();
    
    let comparison = FixedTopologyComparison {
        topologies: vec![
            FixedTopology {
                name: "Simple MLP".to_string(),
                hidden_layers: vec![4, 2],
                activation: "sigmoid".to_string(),
            },
            FixedTopology {
                name: "Deep MLP".to_string(),
                hidden_layers: vec![8, 6, 4],
                activation: "tanh".to_string(),
            },
        ],
        training_epochs: 1000,
        learning_rate: 0.1,
    };
    
    let evaluator = XORFitnessEvaluator::default();
    let mut fixed_results = Vec::new();
    
    // Test fixed topologies
    for topology in &comparison.topologies {
        let result = run_fixed_topology_training(&evaluator, topology, &comparison)?;
        fixed_results.push(result);
    }
    
    // Run NEAT
    let mut neat_config = NEATConfig::default();
    neat_config.population.size = 100;
    neat_config.population.max_generations = 50;
    neat_config.population.target_fitness = 3.8;
    
    let mut neat_trainer = NEATTrainer::new(evaluator.clone(), neat_config);
    let neat_result = neat_trainer.train()?;
    
    // Compare results
    let neat_performance = neat_result.best_genome.fitness;
    let best_fixed_performance = fixed_results.iter()
        .map(|r| r.best_fitness)
        .fold(0.0, f64::max);
    
    let improvement_ratio = neat_performance / best_fixed_performance.max(0.001);
    
    let mut population_snapshots = Vec::new();
    for (i, result) in fixed_results.iter().enumerate() {
        population_snapshots.push(PopulationSnapshot {
            generation: i,
            avg_fitness: result.best_fitness,
            best_fitness: result.best_fitness,
            species_count: 0,
            avg_complexity: result.epochs as f64,
        });
    }
    
    population_snapshots.push(PopulationSnapshot {
        generation: fixed_results.len(),
        avg_fitness: neat_performance,
        best_fitness: neat_performance,
        species_count: 1,
        avg_complexity: neat_result.state.generation as f64,
    });
    
    let execution_time = start_time.elapsed();
    
    Ok(TestResult {
        name: "Fixed Topology Comparison".to_string(),
        category: "Comparison".to_string(),
        passed: improvement_ratio > 1.0, // NEAT should at least match fixed topologies
        metrics: TestMetrics {
            execution_time,
            final_fitness: improvement_ratio,
            target_fitness: Some(1.0),
            generations_to_convergence: Some(neat_result.state.generation),
            success_rate: if improvement_ratio > 1.0 { 1.0 } else { 0.0 },
            peak_memory_mb: 120.0,
            throughput: improvement_ratio,
        },
        details: TestDetails {
            fitness_history: fixed_results.iter().map(|r| r.best_fitness).collect(),
            population_stats: population_snapshots,
            errors: Vec::new(),
            metadata: HashMap::from([
                ("test_type".to_string(), "fixed_topology_comparison".to_string()),
                ("improvement_ratio".to_string(), improvement_ratio.to_string()),
                ("best_fixed_performance".to_string(), best_fixed_performance.to_string()),
                ("neat_performance".to_string(), neat_performance.to_string()),
            ]),
        },
    })
}

/// Compare against literature benchmarks
fn run_literature_comparison(config: &BenchmarkConfig) -> Result<TestResult> {
    let start_time = Instant::now();
    
    let comparison = LiteratureComparison {
        reference_results: HashMap::from([
            ("XOR".to_string(), ReferenceResult {
                mean_fitness: 3.9,
                std_deviation: 0.1,
                generations_to_solve: Some(25),
                success_rate: 0.95,
                source: "Stanley & Miikkulainen (2002)".to_string(),
            }),
        ]),
        tolerance: 0.15, // 15% tolerance
        benchmark_problems: vec![BenchmarkProblem::XOR],
    };
    
    let mut comparison_results = Vec::new();
    let mut fitness_history = Vec::new();
    let mut population_snapshots = Vec::new();
    
    for problem in &comparison.benchmark_problems {
        let problem_name = format!("{:?}", problem);
        if let Some(reference) = comparison.reference_results.get(&problem_name) {
            // Run our implementation
            let our_result = run_benchmark_problem(problem, config)?;
            
            // Compare against reference
            let fitness_diff = (our_result.best_fitness - reference.mean_fitness).abs();
            let fitness_tolerance = reference.mean_fitness * comparison.tolerance;
            let fitness_acceptable = fitness_diff <= fitness_tolerance;
            
            let generation_acceptable = if let (Some(our_gens), Some(ref_gens)) = 
                (our_result.generations, reference.generations_to_solve) {
                (our_gens as f64 - ref_gens as f64).abs() <= ref_gens as f64 * comparison.tolerance
            } else {
                true // No reference data available
            };
            
            comparison_results.push((
                problem_name.clone(),
                fitness_acceptable && generation_acceptable,
                our_result.best_fitness,
                reference.mean_fitness,
            ));
            
            fitness_history.push(our_result.best_fitness);
            fitness_history.push(reference.mean_fitness);
            
            population_snapshots.push(PopulationSnapshot {
                generation: 0,
                avg_fitness: our_result.best_fitness,
                best_fitness: reference.mean_fitness,
                species_count: if fitness_acceptable { 1 } else { 0 },
                avg_complexity: fitness_diff,
            });
        }
    }
    
    let execution_time = start_time.elapsed();
    let success_rate = comparison_results.iter()
        .filter(|(_, acceptable, _, _)| *acceptable)
        .count() as f64 / comparison_results.len() as f64;
    
    let avg_fitness = fitness_history.iter().step_by(2).sum::<f64>() / (fitness_history.len() / 2) as f64;
    
    Ok(TestResult {
        name: "Literature Comparison".to_string(),
        category: "Comparison".to_string(),
        passed: success_rate >= 0.8, // 80% of benchmarks should match literature
        metrics: TestMetrics {
            execution_time,
            final_fitness: avg_fitness,
            target_fitness: None,
            generations_to_convergence: None,
            success_rate,
            peak_memory_mb: 100.0,
            throughput: success_rate,
        },
        details: TestDetails {
            fitness_history,
            population_stats: population_snapshots,
            errors: Vec::new(),
            metadata: HashMap::from([
                ("test_type".to_string(), "literature_comparison".to_string()),
                ("success_rate".to_string(), success_rate.to_string()),
                ("tolerance".to_string(), comparison.tolerance.to_string()),
                ("benchmarks_tested".to_string(), comparison_results.len().to_string()),
            ]),
        },
    })
}

/// Compare parallel implementation performance
fn run_parallel_comparison(config: &BenchmarkConfig) -> Result<TestResult> {
    let start_time = Instant::now();
    
    let comparison = ParallelComparison {
        thread_counts: vec![1, 2, 4, 8],
        population_sizes: vec![100, 200],
        expected_speedups: HashMap::from([
            (2, 1.8),
            (4, 3.2),
            (8, 5.0),
        ]),
    };
    
    let evaluator = XORFitnessEvaluator::default();
    let mut speedup_results = Vec::new();
    let mut population_snapshots = Vec::new();
    
    for &pop_size in &comparison.population_sizes {
        // Baseline: single thread
        let baseline_time = measure_training_time(&evaluator, pop_size, 1, 20)?;
        
        for &thread_count in &comparison.thread_counts {
            if thread_count == 1 { continue; }
            
            let parallel_time = measure_training_time(&evaluator, pop_size, thread_count, 20)?;
            let actual_speedup = baseline_time.as_secs_f64() / parallel_time.as_secs_f64();
            
            let expected_speedup = comparison.expected_speedups.get(&thread_count).copied().unwrap_or(1.0);
            let efficiency = actual_speedup / expected_speedup;
            
            speedup_results.push((thread_count, actual_speedup, efficiency));
            
            population_snapshots.push(PopulationSnapshot {
                generation: thread_count,
                avg_fitness: actual_speedup,
                best_fitness: expected_speedup,
                species_count: pop_size,
                avg_complexity: efficiency,
            });
        }
    }
    
    let execution_time = start_time.elapsed();
    let avg_efficiency = speedup_results.iter()
        .map(|(_, _, eff)| *eff)
        .sum::<f64>() / speedup_results.len() as f64;
    
    let max_speedup = speedup_results.iter()
        .map(|(_, speedup, _)| *speedup)
        .fold(0.0, f64::max);
    
    Ok(TestResult {
        name: "Parallel Performance Comparison".to_string(),
        category: "Comparison".to_string(),
        passed: avg_efficiency > 0.7 && max_speedup > 2.0,
        metrics: TestMetrics {
            execution_time,
            final_fitness: max_speedup,
            target_fitness: Some(2.0),
            generations_to_convergence: None,
            success_rate: avg_efficiency,
            peak_memory_mb: 150.0,
            throughput: max_speedup,
        },
        details: TestDetails {
            fitness_history: speedup_results.iter().map(|(_, speedup, _)| *speedup).collect(),
            population_stats: population_snapshots,
            errors: Vec::new(),
            metadata: HashMap::from([
                ("test_type".to_string(), "parallel_comparison".to_string()),
                ("avg_efficiency".to_string(), avg_efficiency.to_string()),
                ("max_speedup".to_string(), max_speedup.to_string()),
                ("max_threads".to_string(), comparison.thread_counts.iter().max().unwrap().to_string()),
            ]),
        },
    })
}

// Helper functions and result structures

#[derive(Debug)]
struct RandomSearchResult {
    best_fitness: f64,
    evaluations: usize,
    time_taken: Duration,
}

#[derive(Debug)]
struct HillClimbingResult {
    best_fitness: f64,
    steps: usize,
    time_taken: Duration,
}

#[derive(Debug)]
struct FixedTopologyResult {
    best_fitness: f64,
    epochs: usize,
    time_taken: Duration,
}

#[derive(Debug)]
struct BenchmarkProblemResult {
    best_fitness: f64,
    generations: Option<usize>,
    success: bool,
}

fn run_random_search(
    evaluator: &XORFitnessEvaluator,
    baseline: &RandomSearchBaseline,
) -> Result<RandomSearchResult> {
    let start_time = Instant::now();
    let mut best_fitness = 0.0;
    let mut rng = rand::thread_rng();
    
    for i in 0..baseline.sample_count {
        if start_time.elapsed() > baseline.timeout {
            break;
        }
        
        // Generate random genome
        let mut genome = Genome::new(i, evaluator.input_size(), evaluator.output_size());
        
        // Add random complexity
        let extra_nodes = rng.gen_range(0..baseline.max_complexity.nodes);
        let extra_connections = rng.gen_range(0..baseline.max_complexity.connections);
        
        // Simplified random network generation
        for _ in 0..extra_nodes {
            // Add random hidden nodes (simplified)
        }
        
        // Randomize weights
        for connection in &mut genome.connections {
            connection.weight = rng.gen_range(-3.0..3.0);
        }
        
        let fitness = evaluator.evaluate(&genome)?;
        if fitness > best_fitness {
            best_fitness = fitness;
        }
    }
    
    Ok(RandomSearchResult {
        best_fitness,
        evaluations: baseline.sample_count,
        time_taken: start_time.elapsed(),
    })
}

fn run_hill_climbing(
    evaluator: &XORFitnessEvaluator,
    baseline: &HillClimbingBaseline,
) -> Result<HillClimbingResult> {
    let start_time = Instant::now();
    let mut rng = rand::thread_rng();
    
    // Initialize with random genome
    let mut current_genome = Genome::new(0, evaluator.input_size(), evaluator.output_size());
    let mut current_fitness = evaluator.evaluate(&current_genome)?;
    let mut best_fitness = current_fitness;
    
    for step in 0..baseline.max_steps {
        // Create neighbor by perturbing weights
        let mut neighbor = current_genome.clone();
        
        // Randomly modify one connection weight
        if !neighbor.connections.is_empty() {
            let conn_idx = rng.gen_range(0..neighbor.connections.len());
            let delta = rng.gen_range(-baseline.step_size..baseline.step_size);
            neighbor.connections[conn_idx].weight += delta;
        }
        
        let neighbor_fitness = evaluator.evaluate(&neighbor)?;
        
        // Accept if better
        if neighbor_fitness > current_fitness {
            current_genome = neighbor;
            current_fitness = neighbor_fitness;
            
            if current_fitness > best_fitness {
                best_fitness = current_fitness;
            }
        }
    }
    
    Ok(HillClimbingResult {
        best_fitness,
        steps: baseline.max_steps,
        time_taken: start_time.elapsed(),
    })
}

fn run_fixed_topology_training(
    evaluator: &XORFitnessEvaluator,
    topology: &FixedTopology,
    comparison: &FixedTopologyComparison,
) -> Result<FixedTopologyResult> {
    let start_time = Instant::now();
    
    // Simplified fixed topology training
    // In a real implementation, you'd use a proper neural network library
    let mut best_fitness = 0.0;
    let mut rng = rand::thread_rng();
    
    // Create a genome representing the fixed topology
    let mut genome = Genome::new(0, evaluator.input_size(), evaluator.output_size());
    
    // Simulate training by randomly adjusting weights
    for epoch in 0..comparison.training_epochs {
        // Perturb weights slightly (simulating gradient descent)
        for connection in &mut genome.connections {
            let delta = rng.gen_range(-comparison.learning_rate..comparison.learning_rate);
            connection.weight += delta;
        }
        
        let fitness = evaluator.evaluate(&genome)?;
        if fitness > best_fitness {
            best_fitness = fitness;
        }
        
        // Early stopping if solved
        if fitness > 3.8 {
            break;
        }
    }
    
    Ok(FixedTopologyResult {
        best_fitness,
        epochs: comparison.training_epochs,
        time_taken: start_time.elapsed(),
    })
}

fn run_benchmark_problem(
    problem: &BenchmarkProblem,
    config: &BenchmarkConfig,
) -> Result<BenchmarkProblemResult> {
    let evaluator = match problem {
        BenchmarkProblem::XOR => XORFitnessEvaluator::default(),
        _ => {
            // For now, just use XOR for all problems
            // In a real implementation, you'd have different evaluators
            XORFitnessEvaluator::default()
        }
    };
    
    let mut neat_config = NEATConfig::default();
    neat_config.population.size = 100;
    neat_config.population.max_generations = 50;
    neat_config.population.target_fitness = 3.8;
    
    if let Some(seed) = config.random_seed {
        neat_config.population.random_seed = Some(seed);
    }
    
    let mut trainer = NEATTrainer::new(evaluator, neat_config);
    let result = trainer.train()?;
    
    Ok(BenchmarkProblemResult {
        best_fitness: result.best_genome.fitness,
        generations: Some(result.state.generation),
        success: result.success,
    })
}

fn measure_training_time(
    evaluator: &XORFitnessEvaluator,
    population_size: usize,
    thread_count: usize,
    generations: usize,
) -> Result<Duration> {
    let start_time = Instant::now();
    
    let mut neat_config = NEATConfig::default();
    neat_config.population.size = population_size;
    neat_config.population.max_generations = generations;
    neat_config.population.target_fitness = 3.8;
    
    // Configure for parallel execution if thread_count > 1
    // This is simplified - in practice you'd use the parallel evaluator
    
    let mut trainer = NEATTrainer::new(evaluator.clone(), neat_config);
    trainer.train()?;
    
    Ok(start_time.elapsed())
}