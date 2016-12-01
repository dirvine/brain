struct Genome {
    id: u32,
    neurons: Vec::<Neuron>,
    synapses: Vec<Synapse>,
    num_inputs : u32,
    num_outputs: u32,
    fitness: f64,
    adjusted_fitness: f64,
    depth: u32,
    offspring_limit: u32,
}


