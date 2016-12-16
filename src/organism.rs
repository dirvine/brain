use genome::Genome;
use network::Network;
use novelty_item::NoveltyItem;
use parameters::Params;
use species::Species;
use std::time::{Duration, Instant};

///   Organisms are Genomes and Networks with fitness
///   information
///   i.e. The genotype and phenotype together
#[derive(Clone)]
pub struct Organism {
    params: Params,
    fitness: f64, // A measure of fitness for the Organism
    orig_fitness: f64, // A fitness measure that won't change during adjustments
    error: f64, // Used just for reporting purposes
    winner: bool, // Win marker (if needed for a particular task)
    net: Network, // The Organism's phenotype
    genome: Genome, // The Organism's genotype
    species: Species, // The Organism's Species
    noveltypoint: NoveltyItem, // The Organism's Novelty Point
    expected_offspring: f64, // Number of children this Organism may have
    generation: u32, // Tells which generation this Organism is from
    eliminate: bool, // Marker for destruction of inferior Organisms
    champion: bool, // Marks the species champ
    super_champ_offspring: u32, // Number of reserved offspring for a population leader
    pop_champ: bool, // Marks the best in population
    pop_champ_child: bool, // Marks the duplicate child of a champion (for tracking purposes)
    high_fit: f64, // DEBUG variable- high fitness of champ
    created_at: Instant, // When playing in real-time allows knowing the maturity of an individual
    // Track its origin- for debugging or analysis- we can tell how the organism was born
    mut_struct_baby: bool,
    mate_baby: bool,

    // MetaData for the object
    metadata: [u8; 32],
    modified: bool,
}

impl Organism {
    /// TODO
    pub fn new(fitness: f64, genome: Genome, generation: u32) -> Organism {
        unimplemented!()
    }
    /// Is this organism counted as "alive" (lived at least long enough)
    pub fn old_enough(&self) -> bool {
        self.params.time_alive_min() >= self.created_at.elapsed()
    }
    pub fn fitness(&self) -> f64 {
        self.fitness
    }
    pub fn genome(&self) -> &Genome {
        &self.genome
    }
    pub fn species(&mut self) -> &mut Species {
        &mut self.species
    }
}

// Regenerate the network based on a change in the genotype
// void update_phenotype();
//
// Print the Organism's genome to a file preceded by a comment detailing the organism's species, number, and fitness
// bool print_to_file(char *filename);
// bool write_to_file(std::ostream &outFile);
//
// Organism(double fit, Genome *g, int gen, const char* md = 0);
// Organism(const Organism& org);	// Copy Constructor
//
