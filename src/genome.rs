use link_gene::LinkGene;
use neuron_gene::NeuronGene;

struct Genome {
    id: u32,
    neurons: Vec<NeuronGene>,
    synapses: Vec<LinkGene>,
    num_inputs: u32,
    num_outputs: u32,
    fitness: f64,
    adjusted_fitness: f64,
    depth: u32,
    offspring_limit: u32,
}
