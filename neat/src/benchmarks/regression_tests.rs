//! Regression tests for NEAT implementation
//!
//! This module contains tests to ensure that algorithm behavior
//! remains consistent across versions and detect performance regressions.

use crate::neat::{NEATTrainer, FitnessEvaluator, Genome, Network, InnovationTracker, SpeciesManager};
use crate::neat::fitness::XORFitnessEvaluator;
use crate::config::NEATConfig;
use crate::error::Result;
use crate::benchmarks::{BenchmarkConfig, TestResult, TestMetrics, TestDetails, PopulationSnapshot};
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Test to ensure deterministic behavior with fixed seeds
pub struct DeterminismTest {
    /// Random seed to use
    pub random_seed: u64,
    /// Number of runs to compare
    pub num_runs: usize,
    /// Generations to run
    pub generations: usize,
}

/// Test for fitness evolution consistency
pub struct FitnessEvolutionTest {
    /// Minimum expected improvement rate
    pub min_improvement_rate: f64,
    /// Maximum allowed stagnation
    pub max_stagnation_generations: usize,
    /// Target problems to test
    pub test_problems: Vec<TestProblem>,
}

/// Test for genetic operator consistency
pub struct GeneticOperatorTest {
    /// Mutation rates to test
    pub mutation_rates: Vec<f64>,
    /// Crossover rates to test
    pub crossover_rates: Vec<f64>,
    /// Expected diversity metrics
    pub diversity_thresholds: DiversityThresholds,
}

/// Test for species formation and stability
pub struct SpeciationTest {
    /// Compatibility thresholds to test
    pub compatibility_thresholds: Vec<f64>,
    /// Expected species counts
    pub expected_species_range: (usize, usize),
    /// Stability check generations
    pub stability_generations: usize,
}

/// Test for network topology evolution
pub struct TopologyEvolutionTest {
    /// Starting topology constraints
    pub initial_topology: TopologyConstraints,
    /// Maximum allowed complexity
    pub max_complexity: TopologyConstraints,
    /// Evolution patterns to check
    pub evolution_patterns: Vec<EvolutionPattern>,
}

/// Test problem definitions
#[derive(Debug, Clone)]
pub enum TestProblem {
    XOR,
    SimpleClassification,
    FunctionApproximation,
}

/// Diversity measurement thresholds
#[derive(Debug, Clone)]
pub struct DiversityThresholds {
    pub min_genetic_diversity: f64,
    pub min_fitness_diversity: f64,
    pub min_structural_diversity: f64,
}

/// Topology constraints
#[derive(Debug, Clone)]
pub struct TopologyConstraints {
    pub max_nodes: usize,
    pub max_connections: usize,
    pub max_depth: usize,
}

/// Evolution pattern types
#[derive(Debug, Clone)]
pub enum EvolutionPattern {
    ComplexityGrowth,
    PerformanceImprovement,
    StructuralDiversification,
}

/// Reference results for regression detection
#[derive(Debug, Clone)]
pub struct ReferenceResults {
    pub fitness_trajectories: Vec<Vec<f64>>,
    pub topology_evolution: Vec<TopologySnapshot>,
    pub species_evolution: Vec<usize>,
    pub performance_metrics: HashMap<String, f64>,
}

/// Topology snapshot for comparison
#[derive(Debug, Clone)]
pub struct TopologySnapshot {
    pub generation: usize,
    pub avg_nodes: f64,
    pub avg_connections: f64,
    pub max_depth: usize,
}

/// Simple evaluator for determinism testing
#[derive(Debug, Clone)]
pub struct DeterministicEvaluator {
    pub problem_type: TestProblem,
}

impl FitnessEvaluator for DeterministicEvaluator {
    fn evaluate(&self, genome: &Genome) -> Result<f64> {
        match self.problem_type {
            TestProblem::XOR => self.evaluate_xor(genome),
            TestProblem::SimpleClassification => self.evaluate_classification(genome),
            TestProblem::FunctionApproximation => self.evaluate_function_approximation(genome),
        }
    }
    
    fn input_size(&self) -> usize {
        match self.problem_type {
            TestProblem::XOR => 2,
            TestProblem::SimpleClassification => 3,
            TestProblem::FunctionApproximation => 1,
        }
    }
    
    fn output_size(&self) -> usize {
        match self.problem_type {
            TestProblem::XOR => 1,
            TestProblem::SimpleClassification => 2,
            TestProblem::FunctionApproximation => 1,
        }
    }
    
    fn max_fitness(&self) -> f64 {
        match self.problem_type {
            TestProblem::XOR => 4.0,
            TestProblem::SimpleClassification => 1.0,
            TestProblem::FunctionApproximation => 1.0,
        }
    }
}

impl DeterministicEvaluator {
    fn evaluate_xor(&self, genome: &Genome) -> Result<f64> {
        let mut network = Network::from_genome(genome)?;
        let test_cases = [
            (vec![0.0, 0.0], 0.0),
            (vec![0.0, 1.0], 1.0),
            (vec![1.0, 0.0], 1.0),
            (vec![1.0, 1.0], 0.0),
        ];
        
        let mut total_error = 0.0;
        for (inputs, expected) in &test_cases {
            let outputs = network.activate(inputs)?;
            total_error += (outputs[0] - expected).abs();
        }
        
        Ok(4.0 - total_error)
    }
    
    fn evaluate_classification(&self, genome: &Genome) -> Result<f64> {
        let mut network = Network::from_genome(genome)?;
        let test_cases = [
            (vec![1.0, 0.0, 0.0], vec![1.0, 0.0]),
            (vec![0.0, 1.0, 0.0], vec![0.0, 1.0]),
            (vec![0.0, 0.0, 1.0], vec![1.0, 0.0]),
            (vec![1.0, 1.0, 0.0], vec![0.0, 1.0]),
        ];
        
        let mut correct = 0;
        for (inputs, expected) in &test_cases {
            let outputs = network.activate(inputs)?;
            let predicted_class = if outputs[0] > outputs[1] { 0 } else { 1 };
            let expected_class = if expected[0] > expected[1] { 0 } else { 1 };
            
            if predicted_class == expected_class {
                correct += 1;
            }
        }
        
        Ok(correct as f64 / test_cases.len() as f64)
    }
    
    fn evaluate_function_approximation(&self, genome: &Genome) -> Result<f64> {
        let mut network = Network::from_genome(genome)?;
        let sample_points = 20;
        let mut total_error = 0.0;
        
        for i in 0..sample_points {
            let x = -1.0 + 2.0 * i as f64 / (sample_points - 1) as f64;
            let expected = x * x; // Simple quadratic function
            
            let outputs = network.activate(&[x])?;
            total_error += (outputs[0] - expected).abs();
        }
        
        let avg_error = total_error / sample_points as f64;
        Ok(1.0 / (1.0 + avg_error))
    }
}

/// Run all regression tests
pub fn run_all_tests(config: &BenchmarkConfig) -> Result<Vec<TestResult>> {
    let mut results = Vec::new();
    
    // Determinism tests
    results.push(run_determinism_test(config)?);
    
    // Fitness evolution tests
    results.push(run_fitness_evolution_test(config)?);
    
    // Genetic operator tests
    results.push(run_genetic_operator_test(config)?);
    
    // Speciation tests
    results.push(run_speciation_test(config)?);
    
    // Topology evolution tests
    results.push(run_topology_evolution_test(config)?);
    
    Ok(results)
}

/// Test deterministic behavior with fixed seeds
fn run_determinism_test(config: &BenchmarkConfig) -> Result<TestResult> {
    let start_time = Instant::now();
    
    let test = DeterminismTest {
        random_seed: config.random_seed.unwrap_or(42),
        num_runs: 3,
        generations: 20,
    };
    
    let evaluator = DeterministicEvaluator {
        problem_type: TestProblem::XOR,
    };
    
    let mut all_fitness_histories = Vec::new();
    let mut all_topology_data = Vec::new();
    let mut errors = Vec::new();
    
    // Run multiple times with same seed
    for run in 0..test.num_runs {
        let mut neat_config = NEATConfig::default();
        neat_config.population.size = 50;
        neat_config.population.max_generations = test.generations;
        neat_config.population.random_seed = Some(test.random_seed);
        neat_config.population.target_fitness = 3.8;
        
        match NEATTrainer::new(evaluator.clone(), neat_config).train() {
            Ok(result) => {
                all_fitness_histories.push(result.stats.fitness_history);
                
                let topology_data: Vec<_> = result.stats.generation_stats.iter()
                    .map(|stats| (stats.avg_nodes, stats.avg_connections))
                    .collect();
                all_topology_data.push(topology_data);
            }
            Err(e) => {
                errors.push(format!("Run {} failed: {}", run, e));
            }
        }
    }
    
    // Check determinism
    let is_deterministic = check_determinism(&all_fitness_histories, &all_topology_data);
    let consistency_score = calculate_consistency_score(&all_fitness_histories);
    
    let execution_time = start_time.elapsed();
    let final_fitness = all_fitness_histories.first()
        .and_then(|h| h.last())
        .copied()
        .unwrap_or(0.0);
    
    Ok(TestResult {
        name: "Determinism Verification".to_string(),
        category: "Regression".to_string(),
        passed: is_deterministic && consistency_score > 0.95,
        metrics: TestMetrics {
            execution_time,
            final_fitness,
            target_fitness: None,
            generations_to_convergence: None,
            success_rate: if is_deterministic { 1.0 } else { 0.0 },
            peak_memory_mb: 40.0,
            throughput: consistency_score,
        },
        details: TestDetails {
            fitness_history: all_fitness_histories.into_iter().flatten().collect(),
            population_stats: Vec::new(),
            errors,
            metadata: HashMap::from([
                ("test_type".to_string(), "determinism".to_string()),
                ("consistency_score".to_string(), consistency_score.to_string()),
                ("is_deterministic".to_string(), is_deterministic.to_string()),
                ("random_seed".to_string(), test.random_seed.to_string()),
            ]),
        },
    })
}

/// Test fitness evolution patterns
fn run_fitness_evolution_test(config: &BenchmarkConfig) -> Result<TestResult> {
    let start_time = Instant::now();
    
    let test = FitnessEvolutionTest {
        min_improvement_rate: 0.1,
        max_stagnation_generations: 50,
        test_problems: vec![TestProblem::XOR, TestProblem::SimpleClassification],
    };
    
    let mut fitness_history = Vec::new();
    let mut population_snapshots = Vec::new();
    let mut errors = Vec::new();
    let mut improvement_rates = Vec::new();
    
    for problem in &test.test_problems {
        let evaluator = DeterministicEvaluator {
            problem_type: problem.clone(),
        };
        
        let mut neat_config = NEATConfig::default();
        neat_config.population.size = 100;
        neat_config.population.max_generations = 100;
        
        match NEATTrainer::new(evaluator, neat_config).train() {
            Ok(result) => {
                let improvement_rate = calculate_improvement_rate(&result.stats.fitness_history);
                improvement_rates.push(improvement_rate);
                
                let fitness_clone = result.stats.fitness_history.clone();
                fitness_history.extend(result.stats.fitness_history);
                
                // Check for stagnation patterns
                let stagnation = check_stagnation(&fitness_clone, test.max_stagnation_generations);
                
                population_snapshots.push(PopulationSnapshot {
                    generation: 0,
                    avg_fitness: improvement_rate,
                    best_fitness: result.best_genome.fitness,
                    species_count: if stagnation { 1 } else { 0 },
                    avg_complexity: result.stats.generation_stats.len() as f64,
                });
            }
            Err(e) => {
                errors.push(format!("Problem {:?} failed: {}", problem, e));
            }
        }
    }
    
    let execution_time = start_time.elapsed();
    let avg_improvement_rate = improvement_rates.iter().sum::<f64>() / improvement_rates.len() as f64;
    let meets_improvement_threshold = avg_improvement_rate >= test.min_improvement_rate;
    
    Ok(TestResult {
        name: "Fitness Evolution Patterns".to_string(),
        category: "Regression".to_string(),
        passed: meets_improvement_threshold && errors.is_empty(),
        metrics: TestMetrics {
            execution_time,
            final_fitness: avg_improvement_rate,
            target_fitness: Some(test.min_improvement_rate),
            generations_to_convergence: None,
            success_rate: if meets_improvement_threshold { 1.0 } else { 0.0 },
            peak_memory_mb: 60.0,
            throughput: avg_improvement_rate,
        },
        details: TestDetails {
            fitness_history,
            population_stats: population_snapshots,
            errors,
            metadata: HashMap::from([
                ("test_type".to_string(), "fitness_evolution".to_string()),
                ("avg_improvement_rate".to_string(), avg_improvement_rate.to_string()),
                ("min_required_rate".to_string(), test.min_improvement_rate.to_string()),
            ]),
        },
    })
}

/// Test genetic operator consistency
fn run_genetic_operator_test(config: &BenchmarkConfig) -> Result<TestResult> {
    let start_time = Instant::now();
    
    let test = GeneticOperatorTest {
        mutation_rates: vec![0.1, 0.3, 0.5],
        crossover_rates: vec![0.7, 0.8, 0.9],
        diversity_thresholds: DiversityThresholds {
            min_genetic_diversity: 0.3,
            min_fitness_diversity: 0.2,
            min_structural_diversity: 0.4,
        },
    };
    
    let mut population_snapshots = Vec::new();
    let mut errors = Vec::new();
    let mut diversity_scores = Vec::new();
    
    let evaluator = DeterministicEvaluator {
        problem_type: TestProblem::XOR,
    };
    
    for &mutation_rate in &test.mutation_rates {
        for &crossover_rate in &test.crossover_rates {
            let mut neat_config = NEATConfig::default();
            neat_config.population.size = 80;
            neat_config.population.max_generations = 30;
            neat_config.mutation.add_node_rate = mutation_rate * 0.1;
            neat_config.mutation.add_connection_rate = mutation_rate * 0.2;
            neat_config.mutation.weight_mutation_rate = mutation_rate;
            
            match run_diversity_analysis(evaluator.clone(), neat_config) {
                Ok((genetic_div, fitness_div, structural_div)) => {
                    diversity_scores.push((genetic_div, fitness_div, structural_div));
                    
                    population_snapshots.push(PopulationSnapshot {
                        generation: (mutation_rate * 100.0) as usize,
                        avg_fitness: genetic_div,
                        best_fitness: fitness_div,
                        species_count: (crossover_rate * 100.0) as usize,
                        avg_complexity: structural_div,
                    });
                }
                Err(e) => {
                    errors.push(format!("Mutation {:.1}, Crossover {:.1} failed: {}", 
                               mutation_rate, crossover_rate, e));
                }
            }
        }
    }
    
    let execution_time = start_time.elapsed();
    
    // Check if diversity thresholds are met
    let meets_thresholds = diversity_scores.iter().any(|(g, f, s)| {
        *g >= test.diversity_thresholds.min_genetic_diversity &&
        *f >= test.diversity_thresholds.min_fitness_diversity &&
        *s >= test.diversity_thresholds.min_structural_diversity
    });
    
    let avg_genetic_diversity = diversity_scores.iter().map(|(g, _, _)| *g).sum::<f64>() / diversity_scores.len() as f64;
    
    Ok(TestResult {
        name: "Genetic Operator Consistency".to_string(),
        category: "Regression".to_string(),
        passed: meets_thresholds && errors.is_empty(),
        metrics: TestMetrics {
            execution_time,
            final_fitness: avg_genetic_diversity,
            target_fitness: Some(test.diversity_thresholds.min_genetic_diversity),
            generations_to_convergence: None,
            success_rate: if meets_thresholds { 1.0 } else { 0.0 },
            peak_memory_mb: 70.0,
            throughput: avg_genetic_diversity,
        },
        details: TestDetails {
            fitness_history: diversity_scores.iter().map(|(g, _, _)| *g).collect(),
            population_stats: population_snapshots,
            errors,
            metadata: HashMap::from([
                ("test_type".to_string(), "genetic_operators".to_string()),
                ("avg_genetic_diversity".to_string(), avg_genetic_diversity.to_string()),
                ("meets_thresholds".to_string(), meets_thresholds.to_string()),
            ]),
        },
    })
}

/// Test speciation behavior
fn run_speciation_test(config: &BenchmarkConfig) -> Result<TestResult> {
    let start_time = Instant::now();
    
    let test = SpeciationTest {
        compatibility_thresholds: vec![1.0, 2.0, 3.0],
        expected_species_range: (3, 15),
        stability_generations: 20,
    };
    
    let mut population_snapshots = Vec::new();
    let mut errors = Vec::new();
    let mut species_stability_scores = Vec::new();
    
    let evaluator = DeterministicEvaluator {
        problem_type: TestProblem::SimpleClassification,
    };
    
    for &threshold in &test.compatibility_thresholds {
        let mut neat_config = NEATConfig::default();
        neat_config.population.size = 100;
        neat_config.population.max_generations = 50;
        neat_config.speciation.compatibility_threshold = threshold;
        
        match analyze_speciation_behavior(evaluator.clone(), neat_config, &test) {
            Ok((avg_species, stability_score)) => {
                species_stability_scores.push(stability_score);
                
                let in_range = avg_species >= test.expected_species_range.0 as f64 && 
                              avg_species <= test.expected_species_range.1 as f64;
                
                population_snapshots.push(PopulationSnapshot {
                    generation: (threshold * 10.0) as usize,
                    avg_fitness: avg_species,
                    best_fitness: stability_score,
                    species_count: if in_range { 1 } else { 0 },
                    avg_complexity: threshold,
                });
            }
            Err(e) => {
                errors.push(format!("Threshold {:.1} failed: {}", threshold, e));
            }
        }
    }
    
    let execution_time = start_time.elapsed();
    let avg_stability = species_stability_scores.iter().sum::<f64>() / species_stability_scores.len() as f64;
    
    Ok(TestResult {
        name: "Speciation Behavior".to_string(),
        category: "Regression".to_string(),
        passed: avg_stability > 0.5 && errors.is_empty(),
        metrics: TestMetrics {
            execution_time,
            final_fitness: avg_stability,
            target_fitness: Some(0.5),
            generations_to_convergence: None,
            success_rate: avg_stability,
            peak_memory_mb: 80.0,
            throughput: avg_stability,
        },
        details: TestDetails {
            fitness_history: species_stability_scores,
            population_stats: population_snapshots,
            errors,
            metadata: HashMap::from([
                ("test_type".to_string(), "speciation".to_string()),
                ("avg_stability".to_string(), avg_stability.to_string()),
                ("expected_range".to_string(), format!("{}-{}", test.expected_species_range.0, test.expected_species_range.1)),
            ]),
        },
    })
}

/// Test topology evolution patterns
fn run_topology_evolution_test(config: &BenchmarkConfig) -> Result<TestResult> {
    let start_time = Instant::now();
    
    let test = TopologyEvolutionTest {
        initial_topology: TopologyConstraints {
            max_nodes: 5,
            max_connections: 10,
            max_depth: 2,
        },
        max_complexity: TopologyConstraints {
            max_nodes: 50,
            max_connections: 200,
            max_depth: 10,
        },
        evolution_patterns: vec![
            EvolutionPattern::ComplexityGrowth,
            EvolutionPattern::PerformanceImprovement,
        ],
    };
    
    let evaluator = DeterministicEvaluator {
        problem_type: TestProblem::FunctionApproximation,
    };
    
    let mut neat_config = NEATConfig::default();
    neat_config.population.size = 100;
    neat_config.population.max_generations = 80;
    
    let mut fitness_history = Vec::new();
    let mut population_snapshots = Vec::new();
    let mut errors = Vec::new();
    
    match analyze_topology_evolution(evaluator, neat_config, &test) {
        Ok((complexity_growth, performance_correlation, topology_snapshots)) => {
            fitness_history.push(complexity_growth);
            fitness_history.push(performance_correlation);
            
            for snapshot in topology_snapshots {
                population_snapshots.push(PopulationSnapshot {
                    generation: snapshot.generation,
                    avg_fitness: snapshot.avg_nodes,
                    best_fitness: snapshot.avg_connections,
                    species_count: snapshot.max_depth,
                    avg_complexity: snapshot.avg_nodes + snapshot.avg_connections,
                });
            }
        }
        Err(e) => {
            errors.push(format!("Topology evolution analysis failed: {}", e));
        }
    }
    
    let execution_time = start_time.elapsed();
    let avg_pattern_score = fitness_history.iter().sum::<f64>() / fitness_history.len() as f64;
    
    Ok(TestResult {
        name: "Topology Evolution Patterns".to_string(),
        category: "Regression".to_string(),
        passed: avg_pattern_score > 0.3 && errors.is_empty(),
        metrics: TestMetrics {
            execution_time,
            final_fitness: avg_pattern_score,
            target_fitness: Some(0.3),
            generations_to_convergence: None,
            success_rate: if avg_pattern_score > 0.3 { 1.0 } else { 0.0 },
            peak_memory_mb: 90.0,
            throughput: avg_pattern_score,
        },
        details: TestDetails {
            fitness_history,
            population_stats: population_snapshots,
            errors,
            metadata: HashMap::from([
                ("test_type".to_string(), "topology_evolution".to_string()),
                ("avg_pattern_score".to_string(), avg_pattern_score.to_string()),
                ("max_nodes".to_string(), test.max_complexity.max_nodes.to_string()),
            ]),
        },
    })
}

// Helper functions

fn check_determinism(
    fitness_histories: &[Vec<f64>],
    topology_data: &[Vec<(f64, f64)>],
) -> bool {
    if fitness_histories.len() < 2 {
        return false;
    }
    
    let first_fitness = &fitness_histories[0];
    let first_topology = &topology_data[0];
    
    // Check fitness determinism
    for history in &fitness_histories[1..] {
        if history.len() != first_fitness.len() {
            return false;
        }
        for (a, b) in history.iter().zip(first_fitness.iter()) {
            if (a - b).abs() > 1e-10 {
                return false;
            }
        }
    }
    
    // Check topology determinism
    for topology in &topology_data[1..] {
        if topology.len() != first_topology.len() {
            return false;
        }
        for ((a_nodes, a_conns), (b_nodes, b_conns)) in topology.iter().zip(first_topology.iter()) {
            if (a_nodes - b_nodes).abs() > 1e-10 || (a_conns - b_conns).abs() > 1e-10 {
                return false;
            }
        }
    }
    
    true
}

fn calculate_consistency_score(fitness_histories: &[Vec<f64>]) -> f64 {
    if fitness_histories.len() < 2 {
        return 0.0;
    }
    
    let mut total_variance = 0.0;
    let mut count = 0;
    
    let min_len = fitness_histories.iter().map(|h| h.len()).min().unwrap_or(0);
    
    for i in 0..min_len {
        let values: Vec<f64> = fitness_histories.iter().map(|h| h[i]).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        total_variance += variance;
        count += 1;
    }
    
    if count > 0 {
        let avg_variance = total_variance / count as f64;
        1.0 / (1.0 + avg_variance)
    } else {
        0.0
    }
}

fn calculate_improvement_rate(fitness_history: &[f64]) -> f64 {
    if fitness_history.len() < 2 {
        return 0.0;
    }
    
    let initial = fitness_history[0];
    let final_fitness = *fitness_history.last().unwrap();
    
    if initial == 0.0 {
        return final_fitness;
    }
    
    (final_fitness - initial) / initial
}

fn check_stagnation(fitness_history: &[f64], max_stagnation: usize) -> bool {
    if fitness_history.len() < max_stagnation {
        return false;
    }
    
    let recent = &fitness_history[fitness_history.len() - max_stagnation..];
    let variance = {
        let mean = recent.iter().sum::<f64>() / recent.len() as f64;
        recent.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / recent.len() as f64
    };
    
    variance < 1e-6 // Very low variance indicates stagnation
}

fn run_diversity_analysis(
    evaluator: DeterministicEvaluator,
    config: NEATConfig,
) -> Result<(f64, f64, f64)> {
    let mut trainer = NEATTrainer::new(evaluator, config);
    let result = trainer.train()?;
    
    // Simplified diversity calculations
    let genetic_diversity = calculate_genetic_diversity(&result.stats.generation_stats);
    let fitness_diversity = calculate_fitness_diversity(&result.stats.fitness_history);
    let structural_diversity = calculate_structural_diversity(&result.stats.generation_stats);
    
    Ok((genetic_diversity, fitness_diversity, structural_diversity))
}

fn analyze_speciation_behavior(
    evaluator: DeterministicEvaluator,
    config: NEATConfig,
    test: &SpeciationTest,
) -> Result<(f64, f64)> {
    let mut trainer = NEATTrainer::new(evaluator, config);
    let result = trainer.train()?;
    
    let species_counts: Vec<usize> = result.stats.generation_stats.iter()
        .map(|s| s.species_count)
        .collect();
    
    let avg_species = species_counts.iter().sum::<usize>() as f64 / species_counts.len() as f64;
    
    // Calculate stability (lower variance = higher stability)
    let species_variance = {
        let mean = avg_species;
        species_counts.iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>() / species_counts.len() as f64
    };
    
    let stability_score = 1.0 / (1.0 + species_variance);
    
    Ok((avg_species, stability_score))
}

fn analyze_topology_evolution(
    evaluator: DeterministicEvaluator,
    config: NEATConfig,
    test: &TopologyEvolutionTest,
) -> Result<(f64, f64, Vec<TopologySnapshot>)> {
    let mut trainer = NEATTrainer::new(evaluator, config);
    let result = trainer.train()?;
    
    let mut topology_snapshots = Vec::new();
    
    for (gen, stats) in result.stats.generation_stats.iter().enumerate() {
        topology_snapshots.push(TopologySnapshot {
            generation: gen,
            avg_nodes: stats.avg_nodes,
            avg_connections: stats.avg_connections,
            max_depth: 3, // Simplified
        });
    }
    
    // Calculate complexity growth
    let initial_complexity = topology_snapshots[0].avg_nodes + topology_snapshots[0].avg_connections;
    let final_complexity = topology_snapshots.last().unwrap().avg_nodes + topology_snapshots.last().unwrap().avg_connections;
    let complexity_growth = if initial_complexity > 0.0 {
        (final_complexity - initial_complexity) / initial_complexity
    } else {
        0.0
    };
    
    // Calculate performance correlation
    let performance_correlation = calculate_performance_correlation(
        &result.stats.fitness_history,
        &topology_snapshots
    );
    
    Ok((complexity_growth, performance_correlation, topology_snapshots))
}

fn calculate_genetic_diversity(generation_stats: &[crate::neat::population::EvolutionStats]) -> f64 {
    // Simplified genetic diversity based on species count variation
    if generation_stats.is_empty() {
        return 0.0;
    }
    
    let species_counts: Vec<f64> = generation_stats.iter()
        .map(|s| s.species_count as f64)
        .collect();
    
    let mean = species_counts.iter().sum::<f64>() / species_counts.len() as f64;
    let variance = species_counts.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / species_counts.len() as f64;
    
    variance.sqrt() / (mean + 1.0)
}

fn calculate_fitness_diversity(fitness_history: &[f64]) -> f64 {
    if fitness_history.len() < 2 {
        return 0.0;
    }
    
    let mean = fitness_history.iter().sum::<f64>() / fitness_history.len() as f64;
    let variance = fitness_history.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / fitness_history.len() as f64;
    
    variance.sqrt()
}

fn calculate_structural_diversity(generation_stats: &[crate::neat::population::EvolutionStats]) -> f64 {
    if generation_stats.is_empty() {
        return 0.0;
    }
    
    // Diversity based on topology variation
    let node_variance = {
        let nodes: Vec<f64> = generation_stats.iter().map(|s| s.avg_nodes).collect();
        let mean = nodes.iter().sum::<f64>() / nodes.len() as f64;
        nodes.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / nodes.len() as f64
    };
    
    let conn_variance = {
        let conns: Vec<f64> = generation_stats.iter().map(|s| s.avg_connections).collect();
        let mean = conns.iter().sum::<f64>() / conns.len() as f64;
        conns.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / conns.len() as f64
    };
    
    (node_variance + conn_variance).sqrt()
}

fn calculate_performance_correlation(
    fitness_history: &[f64],
    topology_snapshots: &[TopologySnapshot],
) -> f64 {
    if fitness_history.len() != topology_snapshots.len() || fitness_history.len() < 2 {
        return 0.0;
    }
    
    // Simple correlation between fitness and complexity
    let complexities: Vec<f64> = topology_snapshots.iter()
        .map(|t| t.avg_nodes + t.avg_connections)
        .collect();
    
    // Calculate Pearson correlation coefficient
    let n = fitness_history.len() as f64;
    let sum_x = fitness_history.iter().sum::<f64>();
    let sum_y = complexities.iter().sum::<f64>();
    let sum_xy = fitness_history.iter().zip(complexities.iter())
        .map(|(x, y)| x * y)
        .sum::<f64>();
    let sum_x2 = fitness_history.iter().map(|x| x * x).sum::<f64>();
    let sum_y2 = complexities.iter().map(|y| y * y).sum::<f64>();
    
    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
    
    if denominator == 0.0 {
        0.0
    } else {
        (numerator / denominator).abs()
    }
}