//! Parallel fitness evaluation and performance optimization
//!
//! This module provides parallel processing capabilities for NEAT algorithm
//! components, including multi-threaded fitness evaluation and optimized
//! population processing.

use crate::neat::genome::Genome;
use crate::neat::fitness::FitnessEvaluator;
use crate::neat::population::{PopulationManager, EvolutionStats};
use crate::config::NEATConfig;
use crate::error::Result;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use rayon::prelude::*;

/// Parallel fitness evaluation coordinator
pub struct ParallelEvaluator<E: FitnessEvaluator + Send + Sync + 'static> {
    /// Fitness evaluator
    evaluator: Arc<E>,
    /// Number of threads to use
    num_threads: usize,
    /// Evaluation statistics
    stats: EvaluationStats,
}

/// Statistics for parallel evaluation performance
#[derive(Debug, Clone, Default)]
pub struct EvaluationStats {
    /// Total genomes evaluated
    pub total_genomes: usize,
    /// Total evaluation time
    pub total_time: Duration,
    /// Average time per genome
    pub avg_time_per_genome: Duration,
    /// Parallel efficiency (speedup ratio)
    pub parallel_efficiency: f64,
    /// Thread utilization statistics
    pub thread_stats: Vec<ThreadStats>,
}

/// Statistics for individual thread performance
#[derive(Debug, Clone)]
pub struct ThreadStats {
    /// Thread ID
    pub thread_id: usize,
    /// Number of genomes evaluated by this thread
    pub genomes_evaluated: usize,
    /// Total time spent evaluating
    pub evaluation_time: Duration,
    /// Thread utilization percentage
    pub utilization: f64,
}

/// Parallel population processor for optimized evolution
pub struct ParallelPopulationProcessor<E: FitnessEvaluator + Send + Sync + 'static> {
    /// Parallel evaluator
    evaluator: ParallelEvaluator<E>,
    /// Processing configuration
    config: ParallelConfig,
    /// Performance metrics
    metrics: ProcessingMetrics,
}

/// Configuration for parallel processing
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of threads for fitness evaluation
    pub fitness_threads: usize,
    /// Batch size for parallel processing
    pub batch_size: usize,
    /// Whether to use work stealing
    pub work_stealing: bool,
    /// Thread pool configuration
    pub thread_pool: ThreadPoolConfig,
    /// Memory optimization settings
    pub memory_opts: MemoryOptimization,
}

/// Thread pool configuration
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    /// Stack size per thread
    pub stack_size: Option<usize>,
    /// Thread naming prefix
    pub thread_name_prefix: String,
    /// Thread priority (if supported)
    pub priority: Option<i32>,
}

/// Memory optimization settings
#[derive(Debug, Clone)]
pub struct MemoryOptimization {
    /// Whether to use memory pooling for networks
    pub network_pooling: bool,
    /// Maximum memory usage per thread (bytes)
    pub max_memory_per_thread: Option<usize>,
    /// Garbage collection frequency
    pub gc_frequency: usize,
}

/// Performance metrics for parallel processing
#[derive(Debug, Clone, Default)]
pub struct ProcessingMetrics {
    /// Evaluation performance
    pub evaluation: EvaluationStats,
    /// Memory usage statistics
    pub memory_usage: MemoryStats,
    /// Thread pool performance
    pub thread_pool: ThreadPoolStats,
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Peak memory usage (bytes)
    pub peak_memory: usize,
    /// Current memory usage (bytes)
    pub current_memory: usize,
    /// Memory allocations per second
    pub allocations_per_sec: f64,
    /// Memory efficiency (useful work / total memory)
    pub efficiency: f64,
}

/// Thread pool performance statistics
#[derive(Debug, Clone, Default)]
pub struct ThreadPoolStats {
    /// Number of active threads
    pub active_threads: usize,
    /// Queue length
    pub queue_length: usize,
    /// Task completion rate
    pub completion_rate: f64,
    /// Load balancing efficiency
    pub load_balance_efficiency: f64,
}

impl<E: FitnessEvaluator + Send + Sync + 'static> ParallelEvaluator<E> {
    /// Create a new parallel evaluator
    pub fn new(evaluator: E, num_threads: Option<usize>) -> Self {
        let num_threads = num_threads.unwrap_or_else(|| {
            // Use number of logical CPUs, but cap at reasonable limit
            num_cpus::get().min(16)
        });
        
        Self {
            evaluator: Arc::new(evaluator),
            num_threads,
            stats: EvaluationStats::default(),
        }
    }
    
    /// Evaluate population fitness in parallel using rayon
    pub fn evaluate_population_rayon(&mut self, population: &mut [Genome]) -> Result<Vec<f64>> {
        let start_time = Instant::now();
        
        // Configure rayon thread pool if needed
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.num_threads)
            .build()
            .map_err(|e| crate::error::NEATError::Other(e.into()))?;
        
        // Parallel evaluation using rayon
        let fitness_values: Result<Vec<f64>> = pool.install(|| {
            population
                .par_iter()
                .map(|genome| self.evaluator.evaluate(genome))
                .collect()
        });
        
        let fitness_values = fitness_values?;
        
        // Update genome fitness values
        for (genome, fitness) in population.iter_mut().zip(fitness_values.iter()) {
            genome.fitness = *fitness;
        }
        
        // Update statistics
        let evaluation_time = start_time.elapsed();
        self.update_stats(population.len(), evaluation_time);
        
        Ok(fitness_values)
    }
    
    /// Evaluate population fitness in parallel using custom thread pool
    pub fn evaluate_population_custom(&mut self, population: &mut [Genome]) -> Result<Vec<f64>> {
        let start_time = Instant::now();
        let population_size = population.len();
        
        if population_size == 0 {
            return Ok(Vec::new());
        }
        
        // Calculate chunk size for work distribution
        let chunk_size = (population_size + self.num_threads - 1) / self.num_threads;
        let chunk_size = chunk_size.max(1);
        
        // Shared results vector
        let results = Arc::new(Mutex::new(vec![0.0; population_size]));
        let evaluator = Arc::clone(&self.evaluator);
        
        // Create thread handles
        let mut handles = Vec::new();
        
        for (thread_id, chunk) in population.chunks(chunk_size).enumerate() {
            let chunk_evaluator = Arc::clone(&evaluator);
            let chunk_results = Arc::clone(&results);
            let chunk_start_idx = thread_id * chunk_size;
            let chunk_genomes: Vec<Genome> = chunk.to_vec();
            
            let handle = thread::spawn(move || {
                let thread_start = Instant::now();
                let mut chunk_fitness = Vec::new();
                
                for genome in &chunk_genomes {
                    match chunk_evaluator.evaluate(genome) {
                        Ok(fitness) => chunk_fitness.push(fitness),
                        Err(_) => chunk_fitness.push(0.0), // Default fitness on error
                    }
                }
                
                // Update shared results
                {
                    let mut results_guard = chunk_results.lock().unwrap();
                    for (i, fitness) in chunk_fitness.iter().enumerate() {
                        if chunk_start_idx + i < results_guard.len() {
                            results_guard[chunk_start_idx + i] = *fitness;
                        }
                    }
                }
                
                ThreadStats {
                    thread_id,
                    genomes_evaluated: chunk_genomes.len(),
                    evaluation_time: thread_start.elapsed(),
                    utilization: 100.0, // Simplified calculation
                }
            });
            
            handles.push(handle);
        }
        
        // Collect thread statistics
        let mut thread_stats = Vec::new();
        for handle in handles {
            match handle.join() {
                Ok(stats) => thread_stats.push(stats),
                Err(_) => {
                    // Thread panicked, create default stats
                    thread_stats.push(ThreadStats {
                        thread_id: thread_stats.len(),
                        genomes_evaluated: 0,
                        evaluation_time: Duration::default(),
                        utilization: 0.0,
                    });
                }
            }
        }
        
        // Extract results
        let fitness_values = {
            let results_guard = results.lock().unwrap();
            results_guard.clone()
        };
        
        // Update genome fitness values
        for (genome, fitness) in population.iter_mut().zip(fitness_values.iter()) {
            genome.fitness = *fitness;
        }
        
        // Update statistics
        let evaluation_time = start_time.elapsed();
        self.stats.thread_stats = thread_stats;
        self.update_stats(population_size, evaluation_time);
        
        Ok(fitness_values)
    }
    
    /// Update evaluation statistics
    fn update_stats(&mut self, genomes_evaluated: usize, evaluation_time: Duration) {
        self.stats.total_genomes += genomes_evaluated;
        self.stats.total_time += evaluation_time;
        
        if self.stats.total_genomes > 0 {
            self.stats.avg_time_per_genome = self.stats.total_time / self.stats.total_genomes as u32;
        }
        
        // Calculate parallel efficiency (rough estimate)
        if self.stats.total_genomes > 0 && evaluation_time.as_secs_f64() > 0.0 {
            let sequential_estimate = self.stats.avg_time_per_genome.as_secs_f64() * genomes_evaluated as f64;
            let parallel_actual = evaluation_time.as_secs_f64();
            self.stats.parallel_efficiency = sequential_estimate / (parallel_actual * self.num_threads as f64);
        }
    }
    
    /// Get evaluation statistics
    pub fn get_stats(&self) -> &EvaluationStats {
        &self.stats
    }
    
    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = EvaluationStats::default();
    }
    
    /// Get optimal batch size for this configuration
    pub fn get_optimal_batch_size(&self, population_size: usize) -> usize {
        // Heuristic: aim for 2-4 batches per thread
        let target_batches = self.num_threads * 3;
        (population_size + target_batches - 1) / target_batches
    }
}

impl<E: FitnessEvaluator + Send + Sync + 'static> ParallelPopulationProcessor<E> {
    /// Create a new parallel population processor
    pub fn new(evaluator: E, config: ParallelConfig) -> Self {
        let parallel_evaluator = ParallelEvaluator::new(evaluator, Some(config.fitness_threads));
        
        Self {
            evaluator: parallel_evaluator,
            config,
            metrics: ProcessingMetrics::default(),
        }
    }
    
    /// Process a complete generation with parallel optimization
    pub fn process_generation(
        &mut self,
        population_manager: &mut PopulationManager,
        neat_config: &NEATConfig,
    ) -> Result<EvolutionStats> {
        let start_time = Instant::now();
        
        // Parallel fitness evaluation
        let population = population_manager.get_population_mut();
        self.evaluator.evaluate_population_rayon(population)?;
        
        // Continue with standard evolution
        let evolution_stats = population_manager.evolve_generation(&*self.evaluator.evaluator, neat_config)?;
        
        // Update processing metrics
        self.update_processing_metrics(start_time.elapsed());
        
        Ok(evolution_stats)
    }
    
    /// Update processing metrics
    fn update_processing_metrics(&mut self, processing_time: Duration) {
        self.metrics.evaluation = self.evaluator.get_stats().clone();
        
        // Update memory stats (simplified)
        self.metrics.memory_usage.current_memory = self.estimate_memory_usage();
        if self.metrics.memory_usage.current_memory > self.metrics.memory_usage.peak_memory {
            self.metrics.memory_usage.peak_memory = self.metrics.memory_usage.current_memory;
        }
        
        // Update thread pool stats
        self.metrics.thread_pool.active_threads = self.config.fitness_threads;
        self.metrics.thread_pool.completion_rate = 1.0 / processing_time.as_secs_f64();
    }
    
    /// Estimate current memory usage
    fn estimate_memory_usage(&self) -> usize {
        // Simplified memory estimation
        // In a real implementation, you'd use system APIs
        std::mem::size_of::<Genome>() * 1000 // Rough estimate
    }
    
    /// Get processing metrics
    pub fn get_metrics(&self) -> &ProcessingMetrics {
        &self.metrics
    }
    
    /// Optimize configuration based on runtime performance
    pub fn auto_tune_config(&mut self, target_performance: f64) {
        let current_efficiency = self.metrics.evaluation.parallel_efficiency;
        
        if current_efficiency < target_performance {
            // Adjust thread count or batch size
            if self.config.fitness_threads < num_cpus::get() {
                self.config.fitness_threads += 1;
            } else if self.config.batch_size > 1 {
                self.config.batch_size = (self.config.batch_size as f64 * 0.8) as usize;
            }
        }
    }
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            fitness_threads: num_cpus::get().min(8),
            batch_size: 32,
            work_stealing: true,
            thread_pool: ThreadPoolConfig::default(),
            memory_opts: MemoryOptimization::default(),
        }
    }
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            stack_size: Some(2 * 1024 * 1024), // 2MB stack size
            thread_name_prefix: "neat-worker".to_string(),
            priority: None,
        }
    }
}

impl Default for MemoryOptimization {
    fn default() -> Self {
        Self {
            network_pooling: true,
            max_memory_per_thread: Some(256 * 1024 * 1024), // 256MB per thread
            gc_frequency: 100,
        }
    }
}

impl EvaluationStats {
    /// Calculate throughput (genomes per second)
    pub fn throughput(&self) -> f64 {
        if self.total_time.as_secs_f64() > 0.0 {
            self.total_genomes as f64 / self.total_time.as_secs_f64()
        } else {
            0.0
        }
    }
    
    /// Calculate speedup compared to sequential execution
    pub fn speedup_ratio(&self, num_threads: usize) -> f64 {
        self.parallel_efficiency * num_threads as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neat::fitness::XORFitnessEvaluator;

    #[test]
    fn test_parallel_evaluator_creation() {
        let evaluator = XORFitnessEvaluator::default();
        let parallel_eval = ParallelEvaluator::new(evaluator, Some(4));
        
        assert_eq!(parallel_eval.num_threads, 4);
        assert_eq!(parallel_eval.stats.total_genomes, 0);
    }
    
    #[test]
    fn test_parallel_config() {
        let config = ParallelConfig::default();
        
        assert!(config.fitness_threads > 0);
        assert!(config.batch_size > 0);
        assert!(config.work_stealing);
        assert!(config.memory_opts.network_pooling);
    }
    
    #[test]
    fn test_evaluation_stats() {
        let mut stats = EvaluationStats::default();
        
        assert_eq!(stats.throughput(), 0.0);
        assert_eq!(stats.speedup_ratio(4), 0.0);
        
        stats.total_genomes = 100;
        stats.total_time = Duration::from_secs(10);
        
        assert_eq!(stats.throughput(), 10.0); // 100 genomes / 10 seconds
        
        stats.parallel_efficiency = 0.8;
        assert_eq!(stats.speedup_ratio(4), 3.2); // 0.8 * 4
    }
    
    #[test]
    fn test_parallel_evaluation_rayon() -> Result<()> {
        let evaluator = XORFitnessEvaluator::default();
        let mut parallel_eval = ParallelEvaluator::new(evaluator, Some(2));
        
        // Create test population
        let mut population = vec![
            crate::neat::genome::Genome::new(0, 2, 1),
            crate::neat::genome::Genome::new(1, 2, 1),
            crate::neat::genome::Genome::new(2, 2, 1),
            crate::neat::genome::Genome::new(3, 2, 1),
        ];
        
        let fitness_values = parallel_eval.evaluate_population_rayon(&mut population)?;
        
        assert_eq!(fitness_values.len(), 4);
        assert!(fitness_values.iter().all(|&f| f >= 0.0 && f <= 1.0));
        
        // Check that genome fitness was updated
        for (genome, fitness) in population.iter().zip(fitness_values.iter()) {
            assert_eq!(genome.fitness, *fitness);
        }
        
        // Check statistics were updated
        let stats = parallel_eval.get_stats();
        assert_eq!(stats.total_genomes, 4);
        assert!(stats.total_time > Duration::default());
        
        Ok(())
    }
    
    #[test]
    fn test_parallel_evaluation_custom() -> Result<()> {
        let evaluator = XORFitnessEvaluator::default();
        let mut parallel_eval = ParallelEvaluator::new(evaluator, Some(2));
        
        // Create test population
        let mut population = vec![
            crate::neat::genome::Genome::new(0, 2, 1),
            crate::neat::genome::Genome::new(1, 2, 1),
        ];
        
        let fitness_values = parallel_eval.evaluate_population_custom(&mut population)?;
        
        assert_eq!(fitness_values.len(), 2);
        assert!(fitness_values.iter().all(|&f| f >= 0.0 && f <= 1.0));
        
        // Check thread statistics
        let stats = parallel_eval.get_stats();
        assert!(!stats.thread_stats.is_empty());
        
        Ok(())
    }
    
    #[test]
    fn test_parallel_processor() -> Result<()> {
        let evaluator = XORFitnessEvaluator::default();
        let config = ParallelConfig {
            fitness_threads: 2,
            ..Default::default()
        };
        
        let processor = ParallelPopulationProcessor::new(evaluator, config);
        
        assert_eq!(processor.config.fitness_threads, 2);
        assert_eq!(processor.metrics.thread_pool.active_threads, 0); // Not processed yet
        
        Ok(())
    }
    
    #[test]
    fn test_optimal_batch_size() {
        let evaluator = XORFitnessEvaluator::default();
        let parallel_eval = ParallelEvaluator::new(evaluator, Some(4));
        
        let batch_size = parallel_eval.get_optimal_batch_size(100);
        
        // Should aim for ~3 batches per thread (4 threads = 12 batches)
        // 100 / 12 ≈ 8-9
        assert!(batch_size >= 8 && batch_size <= 9);
    }
    
    #[test]
    fn test_stats_reset() {
        let evaluator = XORFitnessEvaluator::default();
        let mut parallel_eval = ParallelEvaluator::new(evaluator, Some(2));
        
        // Simulate some statistics
        parallel_eval.stats.total_genomes = 100;
        parallel_eval.stats.total_time = Duration::from_secs(10);
        
        assert_eq!(parallel_eval.stats.total_genomes, 100);
        
        parallel_eval.reset_stats();
        
        assert_eq!(parallel_eval.stats.total_genomes, 0);
        assert_eq!(parallel_eval.stats.total_time, Duration::default());
    }
    
    #[test]
    fn test_auto_tune_config() {
        let evaluator = XORFitnessEvaluator::default();
        let config = ParallelConfig {
            fitness_threads: 2,
            batch_size: 32,
            ..Default::default()
        };
        
        let mut processor = ParallelPopulationProcessor::new(evaluator, config);
        
        // Simulate low efficiency
        processor.metrics.evaluation.parallel_efficiency = 0.5;
        let initial_threads = processor.config.fitness_threads;
        
        processor.auto_tune_config(0.8); // Target 80% efficiency
        
        // Should increase thread count or adjust batch size
        assert!(
            processor.config.fitness_threads > initial_threads || 
            processor.config.batch_size < 32
        );
    }
}