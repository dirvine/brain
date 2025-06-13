//! Benchmarks for NEAT algorithm components
//!
//! This file contains performance benchmarks for critical NEAT operations
//! to ensure we maintain good performance throughout development.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use neat_fashion_classifier::neat::{Genome, InnovationTracker};
use neat_fashion_classifier::config::NEATConfig;

fn benchmark_genome_creation(c: &mut Criterion) {
    c.bench_function("genome_creation_fashion_mnist", |b| {
        b.iter(|| {
            let genome = Genome::new(black_box(0), black_box(784), black_box(10));
            black_box(genome)
        })
    });
}

fn benchmark_genome_validation(c: &mut Criterion) {
    let genome = Genome::new(0, 784, 10);
    
    c.bench_function("genome_validation", |b| {
        b.iter(|| {
            let result = black_box(&genome).validate();
            black_box(result)
        })
    });
}

fn benchmark_innovation_tracking(c: &mut Criterion) {
    c.bench_function("innovation_tracker_get_id", |b| {
        let mut tracker = InnovationTracker::new();
        let mut counter = 0;
        
        b.iter(|| {
            let id = tracker.get_innovation_id(
                black_box(counter % 100), 
                black_box((counter + 1) % 100)
            );
            counter += 1;
            black_box(id)
        })
    });
}

fn benchmark_config_validation(c: &mut Criterion) {
    let config = NEATConfig::default();
    
    c.bench_function("config_validation", |b| {
        b.iter(|| {
            let result = black_box(&config).validate();
            black_box(result)
        })
    });
}

criterion_group!(
    benches,
    benchmark_genome_creation,
    benchmark_genome_validation,
    benchmark_innovation_tracking,
    benchmark_config_validation
);
criterion_main!(benches);